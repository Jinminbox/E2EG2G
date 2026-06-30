"""Run E2EG2G on BCIC IV-2a with leave-one-subject-out evaluation."""

from run_bcic2a_subject_dependent import parse_args, run_bcic_dataset


if __name__ == "__main__":
    run_bcic_dataset(
        parse_args(),
        dataset_name="bcic2a",
        prefix="A",
        n_channels=22,
        n_classes=4,
        default_data_path="/home/ming/data/BCIC2A/",
        protocol="loso",
    )

