"""Run E2EG2G on BCIC IV-2a with LOSO evaluation and optional EA.

This runner keeps the paper default model architecture and exposes the
Euclidean Alignment (EA) setting used in our post-acceptance release check.
EA is estimated only from the training subjects in each LOSO fold and is then
applied to the train/validation/test data of that fold.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from model import E2EG2G, SupConLoss
from run_bcic2b_loso_ea import (
    apply_ea,
    fit_ea_transform,
    make_loader,
    segment_augment,
    set_seed,
    split_train_valid,
    str2bool,
)


def load_subject(data_path, subject):
    train = scipy.io.loadmat(os.path.join(data_path, f"A0{subject}T.mat"))
    test = scipy.io.loadmat(os.path.join(data_path, f"A0{subject}E.mat"))
    x = np.concatenate([train["data"], test["data"]], axis=0).astype(np.float32)
    y = np.concatenate([train["label"].squeeze(), test["label"].squeeze()], axis=0).astype(np.int64) - 1
    return x, y


def load_loso_fold(data_path, held_subject, n_subjects):
    train_x, train_y = [], []
    test_x, test_y = None, None
    for subject in range(1, n_subjects + 1):
        x, y = load_subject(data_path, subject)
        if subject == held_subject:
            test_x, test_y = x, y
        else:
            train_x.append(x)
            train_y.append(y)
    return np.concatenate(train_x), np.concatenate(train_y), test_x, test_y


def normalize_train_split(x_train, x_valid, x_test):
    mean, std = x_train.mean(), x_train.std() + 1e-6
    return (x_train - mean) / std, (x_valid - mean) / std, (x_test - mean) / std


def compute_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred) * 100.0,
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0,
        "bacc": balanced_accuracy_score(y_true, y_pred) * 100.0,
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


def prepare_fold(args, subject):
    x_train_all, y_train_all, x_test, y_test = load_loso_fold(args.data_path, subject, args.n_subjects)
    if args.ea:
        ea_transform = fit_ea_transform(x_train_all)
        x_train_all = apply_ea(x_train_all, ea_transform)
        x_test = apply_ea(x_test, ea_transform)

    x_train_all = x_train_all[:, None, :, :]
    x_test = x_test[:, None, :, :]
    x_train, y_train, x_valid, y_valid = split_train_valid(
        x_train_all, y_train_all, args.val_rate, args.seed + subject
    )
    x_train, x_valid, x_test = normalize_train_split(x_train, x_valid, x_test)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def validation_score(model, loader, device, ce, metric_name):
    model.eval()
    losses, all_true, all_pred = [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device).float())["logits"]
            y_device = y.to(device)
            losses.append(float(ce(logits, y_device).cpu()))
            all_pred.append(logits.argmax(dim=1).cpu().numpy())
            all_true.append(y.numpy())
    val_loss = float(np.mean(losses))
    if metric_name == "loss":
        return val_loss, val_loss
    val_acc = accuracy_score(np.concatenate(all_true), np.concatenate(all_pred)) * 100.0
    return -val_acc, val_acc


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_name = (
        f"bcic2a_loso_{'ea' if args.ea else 'noea'}_seed{args.seed}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(args.log_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = set_seed(args.seed)
    rows = []

    for subject in range(1, args.n_subjects + 1):
        x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_fold(args, subject)
        train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=args.drop_last_train,
        )
        valid_loader = make_loader(x_valid, y_valid, args.batch_size, False)
        test_loader = make_loader(x_test, y_test, args.batch_size, False)

        model = E2EG2G(
            n_channels=22,
            n_classes=4,
            input_time=args.input_time,
            dropout=args.dropout,
            temporal_kernel=args.temporal_kernel,
            contrastive_target=args.contrastive_target,
            use_lng_residual=True,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        ce = nn.CrossEntropyLoss()
        scl = SupConLoss(args.supcon_temp)
        best_state, best = None, {"score": float("inf"), "epoch": 0, "val": None}

        for epoch in range(1, args.epochs + 1):
            model.train()
            for x, y in train_loader:
                x_aug, y_aug = segment_augment(
                    x_train, y_train, args.n_aug, args.batch_size, 4, args.n_seg, 22, args.input_time
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

            score, val_value = validation_score(model, valid_loader, device, ce, args.selection_metric)
            if score < best["score"]:
                best = {"score": score, "epoch": epoch, "val": val_value}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if epoch % args.print_every == 0:
                print(
                    f"BCIC2a LOSO {'EA' if args.ea else 'no-EA'} subject={subject:02d} "
                    f"epoch={epoch} {args.selection_metric}={val_value:.4f}",
                    flush=True,
                )

        model.load_state_dict(best_state)
        row = {"subject": subject, "best_epoch": best["epoch"], **evaluate(model, test_loader, device)}
        rows.append(row)
        print(row, flush=True)

    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("acc", "f1", "bacc", "kappa")
    }
    summary.update({
        f"{key}_std": float(np.std([row[key] for row in rows], ddof=1))
        for key in ("acc", "f1", "bacc", "kappa")
    })
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(f"Saved results to {out_dir / 'results.csv'}")
    print(summary)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", default="/home/ming/data/BCIC2A/")
    parser.add_argument("--log_root", default="paper_logs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=222)
    parser.add_argument("--n_subjects", type=int, default=9)
    parser.add_argument("--input_time", type=int, default=1000)
    parser.add_argument("--val_rate", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--temporal_kernel", type=int, default=64)
    parser.add_argument("--n_aug", type=int, default=0)
    parser.add_argument("--n_seg", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--supcon_temp", type=float, default=0.1)
    parser.add_argument("--contrastive_target", choices=["lng", "rgm", "both"], default="rgm")
    parser.add_argument("--ea", type=str2bool, default=True)
    parser.add_argument("--selection_metric", choices=["loss", "val_acc"], default="val_acc")
    parser.add_argument("--drop_last_train", type=str2bool, default=True)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--print_every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
