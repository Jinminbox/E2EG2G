"""Run E2EG2G on BCIC IV-2b with leave-one-subject-out evaluation.

This wrapper uses the fixed BCIC IV-2b LOSO defaults from
``run_bcic2b_loso_ea.py``.  Euclidean Alignment is available through
``--ea true`` but is disabled by default because it did not consistently
improve the release reproduction runs.
"""

from run_bcic2b_loso_ea import parse_args, run


if __name__ == "__main__":
    run(parse_args())
