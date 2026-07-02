# E2EG2G: Latent Node Generation and Graph-to-Grid Conversion for Unified EEG Decoding

This repository provides release runners for the Information Fusion paper
**E2EG2G: Latent Node Generation and Graph-to-Grid Conversion for Unified EEG Decoding**.
E2EG2G is a compact end-to-end EEG decoding framework that maps heterogeneous
EEG inputs into fixed latent nodes, converts latent-node relations into
image-like relation grids, and decodes them with a lightweight grid feature
extractor plus supervised contrastive regularization.

The model definition is centralized in `model.py`; each runnable script keeps a
protocol-specific data loading, training, and evaluation entry point. The code
currently includes protocols that have passed our release reproduction checks.
DEAP runners will be added after the full reproduction check is completed.

## Files

| File | Protocol |
|---|---|
| `model.py` | LNG, RGM, GFE wrapper, ConvNet/ResNet/ViT GFE backbones, projection heads, SupCon loss, and the E2EG2G model |
| `run_bcic2a_subject_dependent.py` | BCIC IV-2a official subject-dependent split |
| `run_bcic2a_loso.py` | BCIC IV-2a leave-one-subject-out |
| `run_bcic2b_subject_dependent.py` | BCIC IV-2b official subject-dependent split |
| `run_bcic2b_loso.py` | BCIC IV-2b leave-one-subject-out, EA optional |
| `run_bcic2b_loso_ea.py` | BCIC IV-2b LOSO implementation used by `run_bcic2b_loso.py` |
| `run_seed_cross_session.py` | SEED three-class cross-session setting |
| `run_seed_cross_subject.py` | SEED binary cross-subject LOSO with EmotionMIL-style trial bags |

## Default Hyperparameters

The defaults are selected from the final manuscript runs and release
reproduction checks.

| Dataset | Protocol | Main defaults |
|---|---|---|
| BCIC IV-2a | subject-dependent / LOSO | epochs 1000, batch 48, lr 0.004, dropout 0.5, segment augmentation 3 |
| BCIC IV-2b | subject-dependent | epochs 1000, batch 72, lr 0.004, latent nodes 8, LNG F1 8, segment augmentation 5, alpha 0.20, SupCon target LNG |
| BCIC IV-2b | LOSO | epochs 400, batch 72, lr 0.004, latent nodes 8, LNG F1 8, segment augmentation 0, alpha 0.20, SupCon target LNG, EA off by default |
| SEED | three-class cross-session | epochs 150, batch 128, lr 0.001, channel-wise train normalization, alpha 0.25 |
| SEED | binary cross-subject LOSO | epochs 70, batch 64, lr 0.0012, max instances 4, alpha 0.10, channel-wise train normalization |

## Release Reproduction Results

The following values are from our release sanity/reproduction runs using the
code in this folder. BCIC and SEED binary LOSO means/stds are computed over
subjects. SEED cross-session values are computed over five seeds.

| Dataset | Protocol | Default command | Acc / Std | F1 / Std | Kappa / Std |
|---|---|---|---:|---:|---:|
| BCIC IV-2a | subject-dependent | `python run_bcic2a_subject_dependent.py` | 85.73 +/- 7.89 | 85.54 +/- 8.04 | 80.97 +/- 10.52 |
| BCIC IV-2a | LOSO | `python run_bcic2a_loso.py` | 63.45 +/- 13.41 | 62.96 +/- 13.60 | 51.26 +/- 17.88 |
| BCIC IV-2b | subject-dependent | `python run_bcic2b_subject_dependent.py` | 88.04 +/- 9.08 | 87.36 +/- 10.52 | 76.08 +/- 18.17 |
| BCIC IV-2b | LOSO | `python run_bcic2b_loso.py` | 76.92 +/- 7.34 | 76.82 +/- 7.54 | 53.85 +/- 14.68 |
| BCIC IV-2b | LOSO, optional EA | `python run_bcic2b_loso.py --ea true --seed 888` | 77.47 +/- 6.55 | 77.21 +/- 7.60 | 54.94 +/- 13.09 |
| SEED | three-class cross-session | `python run_seed_cross_session.py` | 59.69 +/- 1.13 | 60.15 +/- 1.19 | 39.61 +/- 1.69 |
| SEED | binary cross-subject LOSO | `python run_seed_cross_subject.py` | 81.78 +/- 14.47 | 80.04 +/- 18.45 | 0.636 +/- 0.289 |
| DEAP | subject-dependent / LOSO | pending |  |  |  |

BCIC IV-2b LOSO uses no EA by default because EA was not consistently better
across the release checks. The optional EA command is retained for readers who
want to reproduce the best single LOSO rerun from our checks.

For SEED cross-session, the no-EA setting is retained as the default. A matched
EA check produced lower results in our release runs and is therefore not used as
the default configuration.

DEAP reproduction is comparatively time-consuming to verify. Because we are
currently reserving limited GPU resources for an upcoming conference submission,
we leave the DEAP implementation/results pending for now and will add the
validated runners and numbers in a later update.

## Model Backbone Notes

`GridFeatureExtractor` is a wrapper for plug-and-play relation-grid backbones.
The paper default is `gfe_backbone="convnet"`, implemented by `ConvNetGFE`.
Optional controlled replacements include `gfe_backbone="resnet18"`,
`gfe_backbone="resnet50"`, and `gfe_backbone="vit"` / `"tiny_vit"`.

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
python run_seed_cross_session.py --data_dir /path/to/EMOD/processed_data/SEED
python run_seed_cross_subject.py --data_dir /path/to/EMOD/processed_data/SEED
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

SEED cross-session and SEED binary cross-subject:

An EMOD-style LMDB containing `__keys__`. For binary cross-subject LOSO,
neutral trials are removed and positive/negative trials are mapped to binary
classes. Each trial is treated as a bag of temporal EEG segments.

## Citation

If you use this code, please cite:

```bibtex
@article{jin2026e2eg2g,
  title={E2EG2G: Latent Node Generation and Graph-to-Grid Conversion for Unified EEG Decoding},
  author={Jin, Ming and Yu, Lanlan and Wei, Xin and Li, Kesang},
  journal={Information Fusion},
  pages={104580},
  year={2026},
  publisher={Elsevier}
}
```
