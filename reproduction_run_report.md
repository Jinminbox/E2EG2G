# E2EG2G Release Reproduction Report

This report records the release-ready checks for the GitHub version of
E2EG2G. Only verified protocols are included in the public runner set.

## Included Protocols

| Dataset | Protocol | Runner | Status |
|---|---|---|---|
| BCIC IV-2a | subject-dependent | `run_bcic2a_subject_dependent.py` | verified |
| BCIC IV-2a | LOSO | `run_bcic2a_loso.py` | verified |
| BCIC IV-2a | LOSO, optional EA | `run_bcic2a_loso_ea.py` | verified |
| BCIC IV-2b | subject-dependent | `run_bcic2b_subject_dependent.py` | verified |
| BCIC IV-2b | LOSO | `run_bcic2b_loso.py` | verified |
| SEED | three-class cross-session | `run_seed_cross_session.py` | verified |

DEAP and SEED binary cross-subject runners are intentionally excluded from this
release until their release hyperparameters and results are re-verified.

## Verified Release Results

| Dataset | Protocol | Acc / Std | F1 / Std | Kappa / Std | Source note |
|---|---|---:|---:|---:|---|
| BCIC IV-2a | subject-dependent | 85.73 +/- 7.89 | 85.54 +/- 8.04 | 80.97 +/- 10.52 | paper-compatible release rerun |
| BCIC IV-2a | LOSO | 63.45 +/- 13.41 | 62.96 +/- 13.60 | 51.26 +/- 17.88 | paper-compatible release rerun |
| BCIC IV-2a | LOSO, optional EA | 67.28 +/- 0.58 | 66.50 +/- 0.69 | 56.37 +/- 0.78 | optional EA release check |
| BCIC IV-2b | subject-dependent | 88.04 +/- 9.08 | 87.36 +/- 10.52 | 76.08 +/- 18.17 | compact latent release rerun |
| BCIC IV-2b | LOSO | 76.92 +/- 7.34 | 76.82 +/- 7.54 | 53.85 +/- 14.68 | compact latent release rerun, no EA |
| BCIC IV-2b | LOSO, optional EA | 77.47 +/- 6.55 | 77.21 +/- 7.60 | 54.94 +/- 13.09 | optional EA rerun, seed 888 |
| SEED | three-class cross-session | 59.69 +/- 1.13 | 60.15 +/- 1.19 | 39.61 +/- 1.69 | no-EA, seeds 2021-2025 |

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
