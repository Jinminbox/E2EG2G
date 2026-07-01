# E2EG2G Reproducible Runners

Official release runners for the accepted paper:

**E2EG2G: Latent Node Generation and Graph-to-Grid Conversion for Unified EEG Decoding**

The model definition is centralized in `model.py`; each dataset/protocol has a
small runnable script with protocol-specific defaults.

## Files

| File | Protocol |
|---|---|
| `model.py` | LNG, RGM, GFE wrapper, ConvNet/ResNet/ViT GFE backbones, projection heads, SupCon loss, and the E2EG2G model |
| `run_bcic2a_subject_dependent.py` | BCIC IV-2a official subject-dependent split |
| `run_bcic2a_loso.py` | BCIC IV-2a leave-one-subject-out |
| `run_bcic2b_subject_dependent.py` | BCIC IV-2b official subject-dependent split |
| `run_bcic2b_loso.py` | BCIC IV-2b leave-one-subject-out, EA optional |
| `run_bcic2b_loso_ea.py` | BCIC IV-2b LOSO implementation used by `run_bcic2b_loso.py` |
| `run_deap_subject_dependent.py` | DEAP subject-dependent 10-fold trial CV |
| `run_deap_loso.py` | DEAP leave-one-subject-out |
| `run_seed_cross_session.py` | SEED three-class cross-session setting |
| `run_seed_cross_subject.py` | SEED binary cross-subject LOSO setting |

## Default Hyperparameters

The defaults are selected from the final manuscript runs and release
reproduction checks.

| Dataset | Protocol | Main defaults |
|---|---|---|
| BCIC IV-2a | subject-dependent / LOSO | epochs 1000, batch 48, lr 0.004, dropout 0.5, segment augmentation 3 |
| BCIC IV-2b | subject-dependent | epochs 1000, batch 72, lr 0.004, latent nodes 8, LNG F1 8, segment augmentation 5, alpha 0.20, SupCon target LNG |
| BCIC IV-2b | LOSO | epochs 400, batch 72, lr 0.004, latent nodes 8, LNG F1 8, segment augmentation 0, alpha 0.20, SupCon target LNG, EA off by default |
| DEAP | subject-dependent / LOSO | epochs 300, batch 64, lr 0.001, 4 s windows, 10-fold dependent CV |
| SEED | cross-session | epochs 150, batch 128, lr 0.001, channel-wise train normalization, alpha 0.25 |
| SEED | binary cross-subject | epochs 100, batch 64, lr 0.001, one-second binary samples |

## Release Reproduction Results

The following values are from our release sanity/reproduction runs using the
code in this folder.  Means and standard deviations are computed over subjects
for BCIC runs and over seeds for the SEED cross-session run.

| Dataset | Protocol | Default command | Acc / Std | F1 / Std | Kappa / Std |
|---|---|---|---:|---:|---:|
| BCIC IV-2a | subject-dependent | `run_bcic2a_subject_dependent.py` | 85.73 +/- 7.89 | 85.54 +/- 8.04 | 80.97 +/- 10.52 |
| BCIC IV-2a | LOSO | `run_bcic2a_loso.py` | 63.45 +/- 13.41 | 62.96 +/- 13.60 | 51.26 +/- 17.88 |
| BCIC IV-2b | subject-dependent | `run_bcic2b_subject_dependent.py` | 88.04 +/- 9.08 | 87.36 +/- 10.52 | 76.08 +/- 18.17 |
| BCIC IV-2b | LOSO | `run_bcic2b_loso.py` | 76.92 +/- 7.34 | 76.82 +/- 7.54 | 53.85 +/- 14.68 |
| BCIC IV-2b | LOSO, optional EA | `run_bcic2b_loso.py --ea true --seed 888` | 77.47 +/- 6.55 | 77.21 +/- 7.60 | 54.94 +/- 13.09 |

BCIC IV-2b LOSO uses no EA by default because EA was not consistently better
across the release checks.  The optional EA command is retained for readers who
want to reproduce the best single LOSO rerun from our checks.

## Model Backbone Notes

`GridFeatureExtractor` is a wrapper for plug-and-play relation-grid backbones. The paper default is `gfe_backbone="convnet"`, implemented by `ConvNetGFE`. Optional controlled replacements include `gfe_backbone="resnet18"`, `gfe_backbone="resnet50"`, and `gfe_backbone="vit"` / `"tiny_vit"`.

## Installation

```bash
pip install -r requirements.txt
```

## Example Commands

```bash
python run_bcic2a_subject_dependent.py --data_path /path/to/BCIC2A
python run_bcic2a_loso.py --data_path /path/to/BCIC2A
python run_bcic2b_subject_dependent.py --data_path /path/to/BCI2B
python run_bcic2b_loso.py --data_path /path/to/BCI2B
python run_bcic2b_loso.py --data_path /path/to/BCI2B --ea true --seed 888
python run_deap_subject_dependent.py --data_path /path/to/DEAP --label_mode both
python run_deap_loso.py --data_path /path/to/DEAP --label_mode both
python run_seed_cross_session.py --data_dir /path/to/EMOD/processed_data/SEED
python run_seed_cross_subject.py --data_root /path/to/one_second_trval
```

## Expected Data Layout

BCIC IV-2a:

```text
A01T.mat A01E.mat ... A09T.mat A09E.mat
```

BCIC IV-2b:

```text
B01T.mat B01E.mat ... B09T.mat B09E.mat
```

Each `.mat` file is expected to contain `data` and `label`.

DEAP:

```text
s01.npy ... s32.npy
```

Each `.npy` file should be a dictionary with `data` and `label`.

SEED cross-session:

An EMOD-style LMDB containing `__keys__` with `train`, `val`, and `test` splits.

SEED binary cross-subject:

```text
S1_session1.npy
S1_session1_label.npy
...
S15_session1.npy
S15_session1_label.npy
```

Labels are expected as `-1`, `0`, and `1`; neutral samples (`0`) are removed and
the remaining labels are mapped to binary classes.
