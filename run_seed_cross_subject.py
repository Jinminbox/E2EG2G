"""Run E2EG2G on SEED binary cross-subject LOSO protocol."""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
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


def load_subject(data_root, subject, session):
    root = Path(data_root)
    x = np.load(root / f"S{subject}_session{session}.npy")
    y = np.load(root / f"S{subject}_session{session}_label.npy")
    mask = y != 0
    x = x[mask]
    y = np.where(y[mask] == -1, 0, 1)
    return x.astype(np.float32), y.astype(np.int64)


def load_fold(data_root, session, test_subject):
    train_x, train_y = [], []
    test_x, test_y = None, None
    for subject in range(1, 16):
        x, y = load_subject(data_root, subject, session)
        if subject == test_subject:
            test_x, test_y = x, y
        else:
            train_x.append(x)
            train_y.append(y)
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    mean = train_x.mean()
    std = train_x.std() + 1e-6
    return (train_x - mean) / std, train_y, (test_x - mean) / std, test_y


def loader(x, y, batch_size, shuffle, generator=None):
    ds = TensorDataset(torch.from_numpy(x).float().unsqueeze(1), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


def metric_dict(true, pred):
    return {
        "acc": accuracy_score(true, pred) * 100.0,
        "f1": f1_score(true, pred, average="weighted", zero_division=0) * 100.0,
        "bacc": balanced_accuracy_score(true, pred) * 100.0,
        "kappa": cohen_kappa_score(true, pred),
    }


def evaluate(model, data_loader, device):
    model.eval()
    true, pred = [], []
    with torch.no_grad():
        for x, y in data_loader:
            logits = model(x.to(device).float())["logits"]
            pred.append(logits.argmax(1).cpu().numpy())
            true.append(y.numpy())
    return metric_dict(np.concatenate(true), np.concatenate(pred))


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_root", default="/home/ming/work/E2EG2G/data/one_second_trval")
    p.add_argument("--log_root", default="logs")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--session", type=int, default=1)
    p.add_argument("--seed", type=int, default=222)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_rate", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--supcon_temp", type=float, default=0.1)
    return p.parse_args()


def main(args):
    generator = set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.log_root) / f"seed_binary_loso_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in range(1, 16):
        train_x, train_y, test_x, test_y = load_fold(args.data_root, args.session, subject)
        x_train, x_valid, y_train, y_valid = train_test_split(
            train_x, train_y, test_size=args.val_rate, random_state=args.seed + subject, stratify=train_y
        )
        train_loader = loader(x_train, y_train, args.batch_size, True, generator)
        valid_loader = loader(x_valid, y_valid, args.batch_size, False)
        test_loader = loader(test_x, test_y, args.batch_size, False)
        model = E2EG2G(62, 2, 200, temporal_kernel=32, pool1=4, pool2=4, contrastive_target="lng").to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        ce = nn.CrossEntropyLoss()
        scl = SupConLoss(args.supcon_temp)
        best_state, best = None, {"acc": -1, "epoch": 0}
        for epoch in range(1, args.epochs + 1):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device).float(), y.to(device).long()
                opt.zero_grad(set_to_none=True)
                out = model(x)
                loss = ce(out["logits"], y)
                for z in out["z"].values():
                    loss = loss + args.alpha * scl(z, y)
                loss.backward()
                opt.step()
            val = evaluate(model, valid_loader, device)
            if val["acc"] > best["acc"]:
                best = {"epoch": epoch, **val}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        row = {"subject": subject, "best_epoch": best["epoch"], **evaluate(model, test_loader, device)}
        rows.append(row)
        print(row, flush=True)
    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main(parse_args())

