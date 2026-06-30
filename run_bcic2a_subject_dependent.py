"""Run E2EG2G on BCIC IV-2a with the official subject-dependent split."""

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from model import E2EG2G, SupConLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def load_bcic_subject(data_path, prefix, subject):
    train = scipy.io.loadmat(os.path.join(data_path, f"{prefix}0{subject}T.mat"))
    test = scipy.io.loadmat(os.path.join(data_path, f"{prefix}0{subject}E.mat"))
    return train["data"], train["label"].squeeze(), test["data"], test["label"].squeeze()


def load_bcic_protocol(data_path, prefix, subject, n_subjects, protocol):
    if protocol == "dependent":
        return load_bcic_subject(data_path, prefix, subject)
    train_x, train_y = [], []
    test_x, test_y = None, None
    for sid in range(1, n_subjects + 1):
        x_tr, y_tr, x_te, y_te = load_bcic_subject(data_path, prefix, sid)
        x = np.concatenate([x_tr, x_te], axis=0)
        y = np.concatenate([y_tr, y_te], axis=0)
        if sid == subject:
            test_x, test_y = x, y
        else:
            train_x.append(x)
            train_y.append(y)
    return np.concatenate(train_x), np.concatenate(train_y), test_x, test_y


def split_train_valid(x, y, val_rate, seed):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_rate, random_state=seed)
    train_idx, valid_idx = next(splitter.split(np.arange(len(y)), y))
    return x[train_idx], y[train_idx], x[valid_idx], y[valid_idx]


def segment_augment(x_source, y_source, n_aug, batch_size, n_class, n_seg, n_channel, input_time):
    if n_aug <= 0:
        return None, None
    seg_len = input_time // n_seg
    per_class = max(1, n_aug * int(batch_size / n_class))
    aug_x, aug_y = [], []
    x_np = np.asarray(x_source)
    y_np = np.asarray(y_source)
    for cls in range(n_class):
        cls_data = x_np[y_np == cls]
        if len(cls_data) == 0:
            continue
        out = np.zeros((per_class, 1, n_channel, input_time), dtype=x_np.dtype)
        for row in range(per_class):
            picks = np.random.randint(0, len(cls_data), size=n_seg)
            for seg in range(n_seg):
                s, e = seg * seg_len, (seg + 1) * seg_len
                out[row, :, :, s:e] = cls_data[picks[seg], :, :, s:e]
        aug_x.append(out)
        aug_y.append(np.full(per_class, cls, dtype=np.int64))
    if not aug_x:
        return None, None
    aug_x = np.concatenate(aug_x)
    aug_y = np.concatenate(aug_y)
    order = np.random.permutation(len(aug_y))
    return torch.from_numpy(aug_x[order]), torch.from_numpy(aug_y[order])


def make_loader(x, y, batch_size, shuffle, generator=None):
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


def metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred) * 100.0,
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0,
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device).float())
            pred = out["logits"].argmax(dim=1).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y.numpy())
    return metrics(np.concatenate(all_true), np.concatenate(all_pred))


def run_bcic_dataset(args, *, dataset_name, prefix, n_channels, n_classes, default_data_path, protocol):
    if args.data_path is None:
        args.data_path = default_data_path
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.log_root) / f"{dataset_name}_{protocol}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    generator = set_seed(args.seed)

    for subject in range(1, args.n_subjects + 1):
        x_tr, y_tr, x_te, y_te = load_bcic_protocol(args.data_path, prefix, subject, args.n_subjects, protocol)
        x_tr = np.expand_dims(x_tr.astype(np.float32), 1)
        x_te = np.expand_dims(x_te.astype(np.float32), 1)
        y_tr = y_tr.astype(np.int64) - 1
        y_te = y_te.astype(np.int64) - 1
        mean, std = x_tr.mean(), x_tr.std() + 1e-6
        x_tr = (x_tr - mean) / std
        x_te = (x_te - mean) / std
        x_train, y_train, x_valid, y_valid = split_train_valid(x_tr, y_tr, args.val_rate, args.seed + subject)

        train_loader = make_loader(x_train, y_train, args.batch_size, True, generator)
        valid_loader = make_loader(x_valid, y_valid, args.batch_size, False)
        test_loader = make_loader(x_te, y_te, args.batch_size, False)
        model = E2EG2G(
            n_channels=n_channels,
            n_classes=n_classes,
            input_time=args.input_time,
            dropout=args.dropout,
            temporal_kernel=args.temporal_kernel,
            contrastive_target=args.contrastive_target,
            use_lng_residual=True,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        ce = nn.CrossEntropyLoss()
        scl = SupConLoss(args.supcon_temp)
        best_state, best = None, {"val_loss": float("inf"), "epoch": 0}

        for epoch in range(1, args.epochs + 1):
            model.train()
            for x, y in train_loader:
                x_aug, y_aug = segment_augment(x_train, y_train, args.n_aug, args.batch_size, n_classes, args.n_seg, n_channels, args.input_time)
                if x_aug is not None:
                    x = torch.cat([x, x_aug], dim=0)
                    y = torch.cat([y, y_aug], dim=0)
                x = x.to(device).float()
                y = y.to(device).long()
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss = ce(out["logits"], y)
                for z in out["z"].values():
                    loss = loss + args.alpha * scl(z, y)
                loss.backward()
                optimizer.step()

            model.eval()
            losses = []
            with torch.no_grad():
                for x, y in valid_loader:
                    logits = model(x.to(device).float())["logits"]
                    losses.append(float(ce(logits, y.to(device)).cpu()))
            val_loss = float(np.mean(losses))
            if val_loss < best["val_loss"]:
                best = {"val_loss": val_loss, "epoch": epoch}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if epoch % args.print_every == 0:
                print(f"{dataset_name} {protocol} subject={subject:02d} epoch={epoch} val_loss={val_loss:.4f}", flush=True)

        model.load_state_dict(best_state)
        row = {"subject": subject, "best_epoch": best["epoch"], **evaluate(model, test_loader, device)}
        rows.append(row)
        print(row, flush=True)

    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {out_dir / 'results.csv'}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path", default=None)
    p.add_argument("--log_root", default="logs")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=222)
    p.add_argument("--n_subjects", type=int, default=9)
    p.add_argument("--input_time", type=int, default=1000)
    p.add_argument("--val_rate", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=0.004)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--temporal_kernel", type=int, default=64)
    p.add_argument("--n_aug", type=int, default=3)
    p.add_argument("--n_seg", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--supcon_temp", type=float, default=0.1)
    p.add_argument("--contrastive_target", choices=["lng", "rgm", "both"], default="rgm")
    p.add_argument("--print_every", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    run_bcic_dataset(
        parse_args(),
        dataset_name="bcic2a",
        prefix="A",
        n_channels=22,
        n_classes=4,
        default_data_path="/home/ming/data/BCIC2A/",
        protocol="dependent",
    )

