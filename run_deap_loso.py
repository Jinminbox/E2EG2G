"""Run E2EG2G on DEAP with leave-one-subject-out evaluation."""

from run_deap_subject_dependent import parse_args, run_deap


if __name__ == "__main__":
    run_deap(parse_args(), protocol="loso")

