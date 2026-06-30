# E2EG2G for BCIC IV-2a

This repository provides a compact, readable implementation of **E2EG2G** for the BCIC IV-2a benchmark. It includes the shared model definition and two runnable protocols:

- Subject-dependent evaluation using the official training/evaluation split.
- Leave-one-subject-out (LOSO) evaluation.

Other datasets from the paper will be released after their reproduction scripts are fully checked.

## Files

| File | Description |
|---|---|
| `model.py` | E2EG2G model components, including LNG, RGM, GFE backbones, DHPM, and supervised contrastive loss. |
| `run_bcic2a_subject_dependent.py` | BCIC IV-2a subject-dependent runner. |
| `run_bcic2a_loso.py` | BCIC IV-2a LOSO runner. |
| `requirements.txt` | Minimal Python dependencies. |

## Reproduction Results

The following results were obtained in our local paper-compatible check using the current BCIC IV-2a hyperparameter setting. Values are reported as mean ± standard deviation across the nine subjects.

| Dataset | Protocol | Seed | Acc / Std (%) | F1 / Std (%) | Kappa / Std (%) |
|---|---|---:|---:|---:|---:|
| BCIC IV-2a | Subject-dependent | 555 | 85.73 ± 7.89 | 85.54 ± 8.04 | 80.97 ± 10.52 |
| BCIC IV-2a | LOSO | 42 | 63.45 ± 13.41 | 62.96 ± 13.60 | 51.26 ± 17.88 |

These runs are close to the accepted manuscript results for BCIC IV-2a: 85.80% subject-dependent accuracy and 63.43% LOSO accuracy.

## Data Layout

The scripts expect preprocessed `.mat` files with `data` and `label` arrays:

```text
/path/to/BCIC2A/
├── A01T.mat
├── A01E.mat
├── ...
├── A09T.mat
└── A09E.mat
```

`data` should have shape `[trials, channels, time]`, and `label` should contain class labels from 1 to 4.

## Installation

```bash
pip install -r requirements.txt
```

## Example Commands

Subject-dependent evaluation:

```bash
python run_bcic2a_subject_dependent.py \
  --data_path /path/to/BCIC2A \
  --device cuda:0 \
  --seed 555
```

LOSO evaluation:

```bash
python run_bcic2a_loso.py \
  --data_path /path/to/BCIC2A \
  --device cuda:0 \
  --seed 42 \
  --n_aug 0
```

## Notes

`GridFeatureExtractor` is implemented as a wrapper for multiple relation-grid backbones. The default paper setting uses the lightweight ConvNet backbone, while ResNet and ViT-style alternatives are available in `model.py` for controlled backbone replacement experiments.
