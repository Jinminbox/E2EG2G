"""Run E2EG2G on DEAP with subject-dependent 10-fold trial CV."""

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import KFold, train_test_split
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


def parse_subjects(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def load_deap_subject(data_path, subject, label_mode, input_time):
    item = np.load(os.path.join(data_path, f"s{subject:02d}.npy"), allow_pickle=True).item()
    data = np.asarray(item["data"], dtype=np.float32)
    labels = np.asarray(item["label"], dtype=np.float32)
    n_trials, n_channels, total_time = data.shape
    n_segments = total_time // input_time
    data = data[:, :, : n_segments * input_time]
    segments = data.reshape(n_trials, n_channels, n_segments, input_time).transpose(0, 2, 1, 3)
    raw = labels[:, 1] if label_mode == "A" else labels[:, 0]
    y = (raw > 5.0).astype(np.int64)
    y = np.repeat(y[:, None], n_segments, axis=1)
    return segments, y


def flatten_segments(x, y):
    return x.reshape(-1, x.shape[2], x.shape[3]), y.reshape(-1)


def normalize_by_train(x_train, x_valid, x_test):
    mean = x_train.mean()
    std = x_train.std() + 1e-6
    return (x_train - mean) / std, (x_valid - mean) / std, (x_test - mean) / std


def segment_augment(x_source, y_source, args):
    seg_len = args.input_time // args.n_seg
    per_class = max(1, args.n_aug * int(args.batch_size / args.n_class))
    aug_x, aug_y = [], []
    for cls in range(args.n_class):
        cls_data = x_source[y_source == cls]
        if len(cls_data) == 0:
            continue
        out = np.zeros((per_class, args.n_channel, args.input_time), dtype=x_source.dtype)
        for row in range(per_class):
            picks = np.random.randint(0, len(cls_data), size=args.n_seg)
            for seg in range(args.n_seg):
                s, e = seg * seg_len, (seg + 1) * seg_len
                out[row, :, s:e] = cls_data[picks[seg], :, s:e]
        aug_x.append(out)
        aug_y.append(np.full(per_class, cls, dtype=np.int64))
    if not aug_x:
        return None, None
    x = np.concatenate(aug_x)
    y = np.concatenate(aug_y)
    idx = np.random.permutation(len(y))
    return torch.from_numpy(x[idx]).unsqueeze(1), torch.from_numpy(y[idx])


def make_loader(x, y, batch_size, shuffle, generator=None):
    ds = TensorDataset(torch.from_numpy(x).float().unsqueeze(1), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


def compute_metrics(true, pred):
    return {
        "acc": accuracy_score(true, pred) * 100.0,
        "f1": f1_score(true, pred, average="binary", zero_division=0) * 100.0,
        "kappa": cohen_kappa_score(true, pred),
    }


def evaluate(model, loader, device):
    model.eval()
    true, pred = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device).float())["logits"]
            pred.append(logits.argmax(1).cpu().numpy())
            true.append(y.numpy())
    return compute_metrics(np.concatenate(true), np.concatenate(pred))


def train_split(args, x_train, y_train, x_valid, y_valid, x_test, y_test, device, seed):
    generator = set_seed(seed)
    x_train, x_valid, x_test = normalize_by_train(x_train, x_valid, x_test)
    train_loader = make_loader(x_train, y_train, args.batch_size, True, generator)
    valid_loader = make_loader(x_valid, y_valid, args.batch_size, False)
    test_loader = make_loader(x_test, y_test, args.batch_size, False)
    model = E2EG2G(
        n_channels=args.n_channel,
        n_classes=args.n_class,
        input_time=args.input_time,
        dropout=args.dropout,
        contrastive_target="lng",
        use_lng_residual=True,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    ce = nn.CrossEntropyLoss()
    scl = SupConLoss(args.supcon_temp)
    best_state, best = None, {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x_aug, y_aug = segment_augment(x_train, y_train, args)
            if x_aug is not None:
                x = torch.cat([x, x_aug], 0)
                y = torch.cat([y, y_aug], 0)
            x, y = x.to(device).float(), y.to(device).long()
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = ce(out["logits"], y)
            for z in out["z"].values():
                loss = loss + args.alpha * scl(z, y)
            loss.backward()
            opt.step()
        losses = []
        model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                losses.append(float(ce(model(x.to(device).float())["logits"], y.to(device)).cpu()))
        val_loss = float(np.mean(losses))
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch}
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"best_epoch": best["epoch"], **evaluate(model, test_loader, device)}


def run_deap(args, protocol):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    labels = ["A", "V"] if args.label_mode == "both" else [args.label_mode]
    for label_mode in labels:
        out_dir = Path(args.log_root) / f"deap_{protocol}_{label_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        subjects = parse_subjects(args.subjects)
        if protocol == "dependent":
            for subject in subjects:
                data, label = load_deap_subject(args.data_path, subject, label_mode, args.input_time)
                trial_idx = np.arange(data.shape[0])
                kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed + subject)
                for fold, (tr_idx, te_idx) in enumerate(kfold.split(trial_idx), start=1):
                    x_trval, y_trval = flatten_segments(data[tr_idx], label[tr_idx])
                    x_test, y_test = flatten_segments(data[te_idx], label[te_idx])
                    x_train, x_valid, y_train, y_valid = train_test_split(
                        x_trval, y_trval, test_size=args.val_rate, random_state=args.seed + subject * 100 + fold, stratify=y_trval
                    )
                    result = train_split(args, x_train, y_train, x_valid, y_valid, x_test, y_test, device, args.seed + subject * 1000 + fold)
                    rows.append({"subject": subject, "fold": fold, **result})
                    print(rows[-1], flush=True)
        else:
            all_subjects = parse_subjects(args.all_subjects)
            cache = {s: load_deap_subject(args.data_path, s, label_mode, args.input_time) for s in all_subjects}
            for holdout in subjects:
                x_test, y_test = flatten_segments(*cache[holdout])
                x_parts, y_parts = [], []
                for subject in all_subjects:
                    if subject != holdout:
                        x, y = flatten_segments(*cache[subject])
                        x_parts.append(x)
                        y_parts.append(y)
                x_trval = np.concatenate(x_parts)
                y_trval = np.concatenate(y_parts)
                x_train, x_valid, y_train, y_valid = train_test_split(
                    x_trval, y_trval, test_size=args.val_rate, random_state=args.seed + holdout, stratify=y_trval
                )
                result = train_split(args, x_train, y_train, x_valid, y_valid, x_test, y_test, device, args.seed + holdout)
                rows.append({"subject": holdout, **result})
                print(rows[-1], flush=True)
        with (out_dir / "results.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path", default="/home/ming/data/DEAP")
    p.add_argument("--log_root", default="logs")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--subjects", default="1-32")
    p.add_argument("--all_subjects", default="1-32")
    p.add_argument("--label_mode", choices=["A", "V", "both"], default="both")
    p.add_argument("--n_splits", type=int, default=10)
    p.add_argument("--val_rate", type=float, default=0.2)
    p.add_argument("--n_channel", type=int, default=32)
    p.add_argument("--n_class", type=int, default=2)
    p.add_argument("--input_time", type=int, default=512)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--supcon_temp", type=float, default=0.1)
    p.add_argument("--n_aug", type=int, default=3)
    p.add_argument("--n_seg", type=int, default=8)
    p.add_argument("--seed", type=int, default=222)
    return p.parse_args()


if __name__ == "__main__":
    run_deap(parse_args(), protocol="dependent")

