"""Run E2EG2G on BCIC IV-2b with the official subject-dependent split.

Defaults are fixed to the best BCIC IV-2b subject-dependent reproduction
configuration from the release checks: compact latent nodes, LNG contrastive
regularization, and segment augmentation sampled from the full training session.
"""

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
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def load_subject(data_path, subject):
    train = scipy.io.loadmat(os.path.join(data_path, f"B0{subject}T.mat"))
    test = scipy.io.loadmat(os.path.join(data_path, f"B0{subject}E.mat"))
    x_train = train["data"].astype(np.float32)
    y_train = train["label"].squeeze().astype(np.int64) - 1
    x_test = test["data"].astype(np.float32)
    y_test = test["label"].squeeze().astype(np.int64) - 1
    return x_train, y_train, x_test, y_test


def split_train_valid(x, y, val_rate, seed):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_rate, random_state=seed)
    train_idx, valid_idx = next(splitter.split(np.arange(len(y)), y))
    return x[train_idx], y[train_idx], x[valid_idx], y[valid_idx]


def segment_augment(x_source, y_source, n_aug, batch_size, n_class, n_seg, n_channel, input_time):
    if n_aug <= 0:
        return None, None
    seg_len = input_time // n_seg
    per_class = max(1, n_aug * int(batch_size / n_class))
    x_np = np.asarray(x_source)
    y_np = np.asarray(y_source)
    aug_x, aug_y = [], []
    for cls in range(n_class):
        cls_data = x_np[y_np == cls]
        if len(cls_data) == 0:
            continue
        out = np.zeros((per_class, 1, n_channel, input_time), dtype=x_np.dtype)
        for row in range(per_class):
            picks = np.random.randint(0, len(cls_data), size=n_seg)
            for seg in range(n_seg):
                start, end = seg * seg_len, (seg + 1) * seg_len
                out[row, :, :, start:end] = cls_data[picks[seg], :, :, start:end]
        aug_x.append(out)
        aug_y.append(np.full(per_class, cls, dtype=np.int64))
    if not aug_x:
        return None, None
    aug_x = np.concatenate(aug_x)
    aug_y = np.concatenate(aug_y)
    order = np.random.permutation(len(aug_y))
    return torch.from_numpy(aug_x[order]), torch.from_numpy(aug_y[order])


def make_loader(x, y, batch_size, shuffle, generator=None):
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def compute_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred) * 100.0,
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0) * 100.0,
        "kappa": cohen_kappa_score(y_true, y_pred) * 100.0,
    }


def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device).float())
            all_pred.append(out["logits"].argmax(dim=1).cpu().numpy())
            all_true.append(y.numpy())
    return compute_metrics(np.concatenate(all_true), np.concatenate(all_pred))


def prepare_subject(args, subject):
    x_train_all, y_train_all, x_test, y_test = load_subject(args.data_path, subject)
    mean, std = x_train_all.mean(), x_train_all.std() + 1e-6
    x_train_all = ((x_train_all - mean) / std)[:, None, :, :]
    x_test = ((x_test - mean) / std)[:, None, :, :]
    x_train, y_train, x_valid, y_valid = split_train_valid(
        x_train_all, y_train_all, args.val_rate, args.seed + subject
    )
    return x_train_all, y_train_all, x_train, y_train, x_valid, y_valid, x_test, y_test


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_name = (
        f"bcic2b_dependent_L{args.latent_nodes}_f{args.lng_f1}_"
        f"aug{args.n_aug}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(args.log_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = set_seed(args.seed)
    rows = []

    for subject in range(1, args.n_subjects + 1):
        x_aug_source, y_aug_source, x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_subject(args, subject)
        train_loader = make_loader(x_train, y_train, args.batch_size, True, generator)
        valid_loader = make_loader(x_valid, y_valid, args.batch_size, False)
        test_loader = make_loader(x_test, y_test, args.batch_size, False)

        model = E2EG2G(
            n_channels=3,
            n_classes=2,
            input_time=args.input_time,
            latent_nodes=args.latent_nodes,
            lng_f1=args.lng_f1,
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
                x_aug, y_aug = segment_augment(
                    x_aug_source, y_aug_source, args.n_aug, args.batch_size, 2, args.n_seg, 3, args.input_time
                )
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
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            losses = []
            model.eval()
            with torch.no_grad():
                for x, y in valid_loader:
                    losses.append(float(ce(model(x.to(device).float())["logits"], y.to(device)).cpu()))
            val_loss = float(np.mean(losses))
            if val_loss < best["val_loss"]:
                best = {"val_loss": val_loss, "epoch": epoch}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if epoch % args.print_every == 0:
                print(f"BCIC2b dependent subject={subject:02d} epoch={epoch} val_loss={val_loss:.4f}", flush=True)

        model.load_state_dict(best_state)
        row = {"subject": subject, "best_epoch": best["epoch"], **evaluate(model, test_loader, device)}
        rows.append(row)
        print(row, flush=True)

    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: float(np.mean([row[key] for row in rows])) for key in ("acc", "f1", "kappa")}
    summary.update({f"{key}_std": float(np.std([row[key] for row in rows], ddof=1)) for key in ("acc", "f1", "kappa")})
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(f"Saved results to {out_dir / 'results.csv'}")
    print(summary)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", default="/home/ming/data/BCI2B/")
    parser.add_argument("--log_root", default="paper_logs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--n_subjects", type=int, default=9)
    parser.add_argument("--input_time", type=int, default=1000)
    parser.add_argument("--val_rate", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=72)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--temporal_kernel", type=int, default=64)
    parser.add_argument("--latent_nodes", type=int, default=8)
    parser.add_argument("--lng_f1", type=int, default=8)
    parser.add_argument("--n_aug", type=int, default=5)
    parser.add_argument("--n_seg", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--supcon_temp", type=float, default=0.1)
    parser.add_argument("--contrastive_target", choices=["lng", "rgm", "both"], default="lng")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--print_every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
