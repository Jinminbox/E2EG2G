"""Run E2EG2G on SEED three-class cross-session protocol.

Expected data: EMOD-preprocessed SEED LMDB with ``train``, ``val`` and ``test``
keys.  This matches the protocol used in the final manuscript:
train=session 1, validation=session 2, test=session 3.
"""

import argparse
import csv
import os
import pickle
import random
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader, Dataset

from model import E2EG2G, SupConLoss


_ENV_CACHE = {}


class SeedLmdbDataset(Dataset):
    def __init__(self, data_dir, split, norm_stats=None):
        if data_dir not in _ENV_CACHE:
            _ENV_CACHE[data_dir] = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        self.env = _ENV_CACHE[data_dir]
        with self.env.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get(b"__keys__"))[split]
        self.norm_stats = norm_stats

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            item = pickle.loads(txn.get(self.keys[index].encode()))
        x = np.asarray(item["sample"], dtype=np.float32).reshape(62, -1)
        if self.norm_stats is not None:
            mean, std = self.norm_stats
            x = (x - mean) / std
        return torch.from_numpy(x).unsqueeze(0), torch.tensor(int(item["label"]), dtype=torch.long)


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


def compute_channel_norm(data_dir):
    env = lmdb.open(data_dir, readonly=True, lock=False, readahead=False, meminit=False)
    total_sum = np.zeros((62,), dtype=np.float64)
    total_sq = np.zeros((62,), dtype=np.float64)
    total_n = 0
    with env.begin(write=False) as txn:
        keys = pickle.loads(txn.get(b"__keys__"))["train"]
        for key in keys:
            sample = pickle.loads(txn.get(key.encode()))["sample"].astype(np.float64).reshape(62, -1)
            total_sum += sample.sum(axis=1)
            total_sq += np.square(sample).sum(axis=1)
            total_n += sample.shape[1]
    mean = (total_sum / total_n).astype(np.float32).reshape(62, 1)
    var = np.maximum(total_sq / total_n - (total_sum / total_n) ** 2, 1e-12)
    std = np.sqrt(var).astype(np.float32).reshape(62, 1)
    return mean, std


def make_loader(data_dir, split, batch_size, shuffle, generator, num_workers, norm_stats):
    return DataLoader(
        SeedLmdbDataset(data_dir, split, norm_stats),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def metric_dict(true, pred):
    return {
        "acc": accuracy_score(true, pred) * 100.0,
        "f1": f1_score(true, pred, average="weighted", zero_division=0) * 100.0,
        "bacc": balanced_accuracy_score(true, pred) * 100.0,
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
    return metric_dict(np.concatenate(true), np.concatenate(pred))


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", default="/home/ming/work/E2EG2G/SEED/EMOD/processed_data/SEED")
    p.add_argument("--log_root", default="logs")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--supcon_temp", type=float, default=0.1)
    p.add_argument("--print_every", type=int, default=10)
    return p.parse_args()


def main(args):
    generator = set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.log_root) / f"seed_cross_session_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_stats = compute_channel_norm(args.data_dir)
    train_loader = make_loader(args.data_dir, "train", args.batch_size, True, generator, args.num_workers, norm_stats)
    val_loader = make_loader(args.data_dir, "val", args.batch_size, False, None, args.num_workers, norm_stats)
    test_loader = make_loader(args.data_dir, "test", args.batch_size, False, None, args.num_workers, norm_stats)
    model = E2EG2G(62, 3, 1000, dropout=args.dropout, contrastive_target="lng").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    ce = nn.CrossEntropyLoss()
    scl = SupConLoss(args.supcon_temp)
    best_state, best = None, {"acc": -1, "epoch": 0}
    history = []
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
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, **val})
        if val["acc"] > best["acc"]:
            best = {"epoch": epoch, **val}
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if epoch % args.print_every == 0:
            print(f"epoch={epoch} val_acc={val['acc']:.2f} best_epoch={best['epoch']}", flush=True)
    model.load_state_dict(best_state)
    result = {"seed": args.seed, "best_epoch": best["epoch"], **evaluate(model, test_loader, device)}
    with (out_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)
    print(result)


if __name__ == "__main__":
    main(parse_args())

