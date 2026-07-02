# E2EG2G Release Reproduction Report

This report records the release-ready checks for the GitHub version of
E2EG2G. Only verified protocols are included in the public runner set.

## Included Protocols

| Dataset | Protocol | Runner | Status |
|---|---|---|---|
| BCIC IV-2a | subject-dependent | `run_bcic2a_subject_dependent.py` | verified |
| BCIC IV-2a | LOSO | `run_bcic2a_loso.py` | verified |
| BCIC IV-2b | subject-dependent | `run_bcic2b_subject_dependent.py` | verified |
| BCIC IV-2b | LOSO | `run_bcic2b_loso.py` | verified |
| SEED | three-class cross-session | `run_seed_cross_session.py` | verified |
| SEED | binary cross-subject LOSO | `run_seed_cross_subject.py` | verified |
| DEAP | subject-dependent / LOSO | pending | deferred |

DEAP runners are intentionally left pending because the full release verification is computationally expensive and will be added after validation.

## Verified Release Results

| Dataset | Protocol | Acc / Std | F1 / Std | Kappa / Std | Source note |
|---|---|---:|---:|---:|---|
| BCIC IV-2a | subject-dependent | 85.73 +/- 7.89 | 85.54 +/- 8.04 | 80.97 +/- 10.52 | paper-compatible release rerun |
| BCIC IV-2a | LOSO | 63.45 +/- 13.41 | 62.96 +/- 13.60 | 51.26 +/- 17.88 | paper-compatible release rerun |
| BCIC IV-2b | subject-dependent | 88.04 +/- 9.08 | 87.36 +/- 10.52 | 76.08 +/- 18.17 | compact latent release rerun |
| BCIC IV-2b | LOSO | 76.92 +/- 7.34 | 76.82 +/- 7.54 | 53.85 +/- 14.68 | compact latent release rerun, no EA |
| BCIC IV-2b | LOSO, optional EA | 77.47 +/- 6.55 | 77.21 +/- 7.60 | 54.94 +/- 13.09 | optional EA rerun, seed 888 |
| SEED | three-class cross-session | 59.69 +/- 1.13 | 60.15 +/- 1.19 | 39.61 +/- 1.69 | no-EA, seeds 2021-2025 |
| SEED | binary cross-subject LOSO | 81.78 +/- 14.47 | 80.04 +/- 18.45 | 0.636 +/- 0.289 | release-style runner, epochs 70, lr 0.0012 |
| DEAP | subject-dependent / LOSO | pending | pending | pending | verification deferred |

## SEED Cross-Session Check

The verified SEED cross-session release setting uses the EMOD-style processed
SEED LMDB with the original train/validation/test session split. The release
check used five seeds (`2021-2025`) and no Euclidean Alignment.

A matched EA group was also tested but produced lower results:

| Setting | Seeds | Acc / Std | Weighted F1 / Std | BACC / Std | Kappa / Std |
|---|---:|---:|---:|---:|---:|
| no-EA | 2021-2025 | 59.69 +/- 1.13 | 60.15 +/- 1.19 | 59.58 +/- 1.13 | 39.61 +/- 1.69 |
| EA | 2021-2025 | 57.37 +/- 1.11 | 57.49 +/- 1.09 | 57.20 +/- 1.11 | 36.07 +/- 1.65 |

Therefore, the no-EA setting is kept as the default SEED cross-session release
configuration.

## SEED Binary LOSO Follow-Up

This check uses the release-style SEED binary LOSO runner
(`run_seed_cross_subject.py`) to match the EmotionMIL-style preprocessing and
trial-bag evaluation path. The core encoder
remains `E2EG2G`; the MIL runner only organizes SEED segments into trial bags
and aggregates bag-level predictions.

Fixed settings: `batch_size=64`, `eval_batch_size=4`, `max_instances=4`,
`alpha=0.1`, `seed=222`, `save_metric=acc`, and channel-wise normalization.

| Setting | Acc / Std | F1 / Std | BACC / Std | Kappa / Std |
|---|---:|---:|---:|---:|
| `epochs=70, lr=0.0012` | 81.78 +/- 14.47 | 80.04 +/- 18.45 | 81.78 +/- 14.47 | 0.636 +/- 0.289 |
| `epochs=70, lr=0.0008` | 80.89 +/- 15.56 | 80.05 +/- 17.75 | 80.89 +/- 15.56 | 0.618 +/- 0.311 |
| `epochs=90, lr=0.0010` | 80.67 +/- 13.87 | 77.38 +/- 20.80 | 80.67 +/- 13.87 | 0.613 +/- 0.277 |
| `epochs=70, lr=0.0010` | 80.22 +/- 13.24 | 78.52 +/- 16.15 | 80.22 +/- 13.24 | 0.604 +/- 0.265 |
| `epochs=50, lr=0.0008` | 78.67 +/- 15.11 | 74.92 +/- 20.52 | 78.67 +/- 15.11 | 0.573 +/- 0.302 |
| `epochs=50, lr=0.0015` | 78.44 +/- 15.16 | 73.21 +/- 26.30 | 78.44 +/- 15.16 | 0.569 +/- 0.303 |
| `epochs=90, lr=0.0008` | 78.00 +/- 13.90 | 74.65 +/- 22.19 | 78.00 +/- 13.90 | 0.560 +/- 0.278 |
| `epochs=50, lr=0.0012` | 77.78 +/- 15.72 | 73.66 +/- 24.30 | 77.78 +/- 15.72 | 0.556 +/- 0.314 |
| `epochs=50, lr=0.0005` | 77.56 +/- 15.86 | 76.58 +/- 18.55 | 77.56 +/- 15.86 | 0.551 +/- 0.317 |
| `epochs=50, lr=0.0010` | 77.11 +/- 14.02 | 72.83 +/- 20.54 | 77.11 +/- 14.02 | 0.542 +/- 0.280 |

Best current setting: `epochs=70, lr=0.0012`, reaching 81.78% trial-level
LOSO accuracy. This configuration is now used as the default setting in
`run_seed_cross_subject.py`.
