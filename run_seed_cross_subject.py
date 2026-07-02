"""Run E2EG2G on SEED binary emotion recognition with LOSO evaluation.

This runner follows the EmotionMIL-style SEED binary protocol: neutral trials
are removed, positive/negative trials are evaluated as binary classes, and each
trial is treated as a bag of temporal EEG segments.  E2EG2G is used as the
segment encoder; a small MIL attention head aggregates segment features into one
trial-level prediction.
"""

import argparse
import csv
import pickle
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader, Dataset

from model import E2EG2G, SupConLoss


_ENVS = {}


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


def get_env(data_dir):
    data_dir = str(data_dir)
    if data_dir not in _ENVS:
        _ENVS[data_dir] = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
    return _ENVS[data_dir]


def parse_seed_key(key):
    match = re.match(r"^(\d+)_([^-]+)-(\d+)-(\d+)$", key)
    if match is None:
        raise ValueError(f"Unexpected SEED LMDB key format: {key}")
    subject, session, trial, segment = match.groups()
    return int(subject), session, int(trial), int(segment)


def load_seed_trial_bags(data_dir, cache_path=None, rebuild_cache=False):
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not rebuild_cache:
            with cache_path.open("rb") as f:
                return pickle.load(f)

    bags = defaultdict(list)
    labels = {}
    subjects = {}
    with get_env(data_dir).begin(write=False) as txn:
        split_keys = pickle.loads(txn.get(b"__keys__"))
        for keys in split_keys.values():
            for key in keys:
                item = pickle.loads(txn.get(key.encode()))
                label = int(item["label"])
                if label == 1:
                    continue
                subject, session, trial, segment = parse_seed_key(key)
                trial_id = f"S{subject:02d}_{session}_T{trial:02d}"
                bags[trial_id].append((segment, key))
                labels[trial_id] = 0 if label == 0 else 1
                subjects[trial_id] = subject

    index = []
    for trial_id in sorted(bags):
        keys = [key for _, key in sorted(bags[trial_id])]
        index.append(
            {
                "trial_id": trial_id,
                "subject": subjects[trial_id],
                "label": labels[trial_id],
                "keys": keys,
            }
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(index, f)
    return index


def split_train_valid(bags, val_rate, seed):
    labels = np.asarray([bag["label"] for bag in bags])
    idx0 = np.where(labels == 0)[0].copy()
    idx1 = np.where(labels == 1)[0].copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n0 = int(len(idx0) * (1.0 - val_rate))
    n1 = int(len(idx1) * (1.0 - val_rate))
    train_idx = np.concatenate([idx0[:n0], idx1[:n1]])
    valid_idx = np.concatenate([idx0[n0:], idx1[n1:]])
    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    return [bags[i] for i in train_idx], [bags[i] for i in valid_idx]


def compute_ea_matrix(data_dir, bags, n_channels):
    cov = np.zeros((n_channels, n_channels), dtype=np.float64)
    n = 0
    with get_env(data_dir).begin(write=False) as txn:
        for bag in bags:
            for key in bag["keys"]:
                x = np.asarray(pickle.loads(txn.get(key.encode()))["sample"], dtype=np.float64).reshape(n_channels, -1)
                cov += (x @ x.T) / max(x.shape[1], 1)
                n += 1
    cov /= max(n, 1)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-6, None)
    return (vecs @ np.diag(vals ** -0.5) @ vecs.T).astype(np.float32)


def compute_norm(data_dir, bags, n_channels, channel_norm=True, ea_matrix=None):
    with get_env(data_dir).begin(write=False) as txn:
        if channel_norm:
            total = np.zeros(n_channels, dtype=np.float64)
            total_sq = np.zeros(n_channels, dtype=np.float64)
            count = 0
            for bag in bags:
                for key in bag["keys"]:
                    x = np.asarray(pickle.loads(txn.get(key.encode()))["sample"], dtype=np.float64).reshape(n_channels, -1)
                    if ea_matrix is not None:
                        x = ea_matrix @ x
                    total += x.sum(axis=1)
                    total_sq += (x * x).sum(axis=1)
                    count += x.shape[1]
            mean = total / max(count, 1)
            var = np.maximum(total_sq / max(count, 1) - mean * mean, 1e-12)
            return mean.astype(np.float32).reshape(n_channels, 1), np.sqrt(var).astype(np.float32).reshape(n_channels, 1)

        total, total_sq, count = 0.0, 0.0, 0
        for bag in bags:
            for key in bag["keys"]:
                x = np.asarray(pickle.loads(txn.get(key.encode()))["sample"], dtype=np.float64).reshape(n_channels, -1)
                if ea_matrix is not None:
                    x = ea_matrix @ x
                total += x.sum()
                total_sq += (x * x).sum()
                count += x.size
        mean = np.float32(total / max(count, 1))
        var = max(total_sq / max(count, 1) - float(mean) * float(mean), 1e-12)
        return mean, np.float32(var**0.5)


class SeedTrialBagDataset(Dataset):
    def __init__(self, data_dir, bags, mean, std, n_channels, max_instances=0, training=False, ea_matrix=None):
        self.data_dir = data_dir
        self.bags = bags
        self.mean = mean
        self.std = std
        self.n_channels = n_channels
        self.max_instances = max_instances
        self.training = training
        self.ea_matrix = ea_matrix

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        bag = self.bags[index]
        keys = bag["keys"]
        if self.training and self.max_instances > 0 and len(keys) > self.max_instances:
            keys = random.sample(keys, self.max_instances)

        samples = []
        with get_env(self.data_dir).begin(write=False) as txn:
            for key in keys:
                x = np.asarray(pickle.loads(txn.get(key.encode()))["sample"], dtype=np.float32).reshape(self.n_channels, -1)
                if self.ea_matrix is not None:
                    x = self.ea_matrix @ x
                x = (x - self.mean) / self.std
                samples.append(x)
        x = torch.from_numpy(np.stack(samples).astype(np.float32)).unsqueeze(1)
        y = torch.tensor(bag["label"], dtype=torch.long)
        return x, y, bag["trial_id"]


def collate_bags(batch):
    xs, ys, trial_ids, lengths = [], [], [], []
    for x, y, trial_id in batch:
        xs.append(x)
        ys.append(y)
        trial_ids.append(trial_id)
        lengths.append(x.shape[0])
    return torch.cat(xs, dim=0), torch.stack(ys), trial_ids, lengths


def make_loader(data_dir, bags, mean, std, args, shuffle, generator=None, training=False, ea_matrix=None):
    ds = SeedTrialBagDataset(
        data_dir=data_dir,
        bags=bags,
        mean=mean,
        std=std,
        n_channels=args.n_channels,
        max_instances=args.max_instances if training else 0,
        training=training,
        ea_matrix=ea_matrix,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size if training else args.eval_batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate_bags,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred) * 100.0,
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0) * 100.0,
        "bacc": balanced_accuracy_score(y_true, y_pred) * 100.0,
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


class E2EG2GAttentionMIL(nn.Module):
    def __init__(self, encoder, n_classes=2, hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        feature_dim = encoder.classifier[1].in_features
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x, lengths):
        out = self.encoder(x)
        chunks = torch.split(out["feature"], lengths, dim=0)
        pooled = []
        for chunk in chunks:
            weights = torch.softmax(self.attention(chunk).squeeze(-1), dim=0)
            pooled.append((chunk * weights.unsqueeze(-1)).sum(dim=0))
        logits = self.classifier(torch.stack(pooled, dim=0))
        return {"logits": logits, "z": out["z"]}


def make_model(args, device):
    encoder = E2EG2G(
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        input_time=args.input_time,
        latent_nodes=args.latent_nodes,
        lng_f1=args.lng_f1,
        temporal_kernel=args.temporal_kernel,
        pool1=args.pool1,
        pool2=args.pool2,
        dropout=args.dropout,
        use_lng_residual=True,
        contrastive_target=args.contrastive_target,
    )
    return E2EG2GAttentionMIL(encoder, n_classes=args.n_classes, hidden_dim=args.mil_hidden).to(device)


def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y, _, lengths in loader:
            out = model(x.to(device).float(), lengths)
            pred = out["logits"].argmax(dim=1).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y.numpy())
    return metrics(np.concatenate(all_true), np.concatenate(all_pred))


def run_seed_binary_loso(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator = set_seed(args.seed)
    cache_path = Path(args.cache_dir) / "seed_binary_trial_index.pkl" if args.cache_dir else None
    all_bags = load_seed_trial_bags(args.data_dir, cache_path, args.rebuild_cache)
    subjects = sorted({bag["subject"] for bag in all_bags})
    out_dir = Path(args.log_root) / f"seed_binary_loso_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for subject in subjects:
        train_all = [bag for bag in all_bags if bag["subject"] != subject]
        test_bags = [bag for bag in all_bags if bag["subject"] == subject]
        train_bags, valid_bags = split_train_valid(train_all, args.val_rate, args.seed + subject)
        ea_matrix = compute_ea_matrix(args.data_dir, train_bags, args.n_channels) if args.ea else None
        mean, std = compute_norm(args.data_dir, train_bags, args.n_channels, args.channel_norm, ea_matrix)

        train_loader = make_loader(args.data_dir, train_bags, mean, std, args, True, generator, True, ea_matrix)
        valid_loader = make_loader(args.data_dir, valid_bags, mean, std, args, False, None, False, ea_matrix)
        test_loader = make_loader(args.data_dir, test_bags, mean, std, args, False, None, False, ea_matrix)
        model = make_model(args, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        scl = SupConLoss(args.supcon_temp)
        best_state, best = None, {args.save_metric: -1.0, "epoch": 0}

        for epoch in range(1, args.epochs + 1):
            model.train()
            for x, y, _, lengths in train_loader:
                x = x.to(device).float()
                y = y.to(device).long()
                optimizer.zero_grad(set_to_none=True)
                out = model(x, lengths)
                loss = ce(out["logits"], y)
                if args.alpha > 0:
                    repeated_y = torch.repeat_interleave(y, torch.as_tensor(lengths, device=device))
                    for z in out["z"].values():
                        loss = loss + args.alpha * scl(z, repeated_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            val = evaluate(model, valid_loader, device)
            if val[args.save_metric] > best[args.save_metric]:
                best = {"epoch": epoch, **val}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if epoch % args.print_every == 0:
                print(f"seed binary loso subject={subject:02d} epoch={epoch} val_{args.save_metric}={val[args.save_metric]:.2f}", flush=True)

        model.load_state_dict(best_state)
        row = {"subject": subject, "best_epoch": best["epoch"], **evaluate(model, test_loader, device)}
        rows.append(row)
        print(row, flush=True)

    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for key in ("acc", "f1", "bacc", "kappa"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = values.mean()
        summary[f"{key}_std"] = values.std(ddof=1)
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(summary, flush=True)
    print(f"Saved results to {out_dir / 'results.csv'}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", default="/home/ming/work/E2EG2G/SEED/EMOD/processed_data/SEED")
    p.add_argument("--cache_dir", default="cache/seed_binary_loso")
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--log_root", default="logs")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=222)
    p.add_argument("--n_channels", type=int, default=62)
    p.add_argument("--n_classes", type=int, default=2)
    p.add_argument("--input_time", type=int, default=1000)
    p.add_argument("--latent_nodes", type=int, default=32)
    p.add_argument("--lng_f1", type=int, default=16)
    p.add_argument("--temporal_kernel", type=int, default=64)
    p.add_argument("--pool1", type=int, default=8)
    p.add_argument("--pool2", type=int, default=8)
    p.add_argument("--val_rate", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--max_instances", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.0012)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--supcon_temp", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--mil_hidden", type=int, default=128)
    p.add_argument("--save_metric", choices=["acc", "f1", "bacc", "kappa"], default="acc")
    p.add_argument("--contrastive_target", choices=["lng", "rgm", "both"], default="lng")
    p.add_argument("--channel_norm", dest="channel_norm", action="store_true", default=True)
    p.add_argument("--no_channel_norm", dest="channel_norm", action="store_false")
    p.add_argument("--ea", action="store_true")
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--print_every", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    run_seed_binary_loso(parse_args())
