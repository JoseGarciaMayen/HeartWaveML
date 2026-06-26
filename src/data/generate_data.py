import os

import pandas as pd
import wfdb

from src.preprocessing import extract_features_from_dataframe, feature_extracting, split_data
from src.utils import get_class_mapping, get_record_numbers


def generateData(data=None, record_numbers=None, window_size=187):
    """
    Generates all datasets that are necessary to train the model.
    First, it generates a interim dataset that contains the features for each heartbeat,
    then it generates the filtered and scaled training, validation and testing datasets.
    """
    if record_numbers is None:
        record_numbers = get_record_numbers()

    if os.path.exists("data/interim/mitbih_combined_records.csv"):
        print("Interim dataset already exists. Skipping generation.")
    else:
        rows = []

        for record_number in record_numbers:
            record = wfdb.rdrecord(f"data/raw/mit-bih-arrhythmia-database-1.0.0/{record_number}")
            ann = wfdb.rdann(f"data/raw/mit-bih-arrhythmia-database-1.0.0/{record_number}", "atr")
            class_mapping = get_class_mapping()

            n_ann = len(ann.sample)
            for i in range(n_ann):
                symbol = ann.symbol[i]
                if symbol in class_mapping:
                    center = ann.sample[i]
                    w2 = window_size // 2
                    start = max(0, center - (w2))
                    end = min(len(record.p_signal[:, 0]), center + (w2) + (window_size % 2))

                    if end - start == window_size:
                        beat_samples = record.p_signal[start:end, 0].tolist()
                        beat_samples.append(class_mapping[symbol])
                        beat_samples.append(record_number)
                        beat_samples.append(int(center))
                        rows.append(beat_samples)

        column_names = [f"sample_{i}" for i in range(window_size)] + [
            "class",
            "record",
            "beat_center",
        ]

        df = pd.DataFrame(rows, columns=column_names)

        df.to_csv("data/interim/mitbih_combined_records.csv", index=False)

    if data is None:
        split_data("data/interim/mitbih_combined_records.csv")
        feature_extracting()
        extract_features_from_dataframe()
        split_data("data/interim/mitbih_features.csv")
        split_data("data/interim/mitbih_features_only.csv")
    elif data == "deterministic":
        # Everything that does not need the trained CNN feature extractor:
        split_data("data/interim/mitbih_combined_records.csv")
        extract_features_from_dataframe()
        split_data("data/interim/mitbih_features.csv")
        split_data("data/interim/mitbih_features_only.csv")
    elif data == "cnn":
        # Only the CNN feature extraction step (requires feature_extractor.keras).
        feature_extracting()
    elif data == "no_feat":
        split_data("data/interim/mitbih_combined_records.csv")
    elif data == "cnn_full":
        split_data("data/interim/mitbih_combined_records.csv")
        feature_extracting()
    elif data == "feat":
        extract_features_from_dataframe()
        split_data("data/interim/mitbih_features.csv")
    elif data == "feat_only":
        extract_features_from_dataframe()
        split_data("data/interim/mitbih_features_only.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate HeartWaveML datasets.")
    parser.add_argument(
        "--mode",
        default="deterministic",
        choices=["deterministic", "cnn", "no_feat", "cnn_full", "feat", "feat_only", "all"],
        help=(
            "deterministic: base/feat/feat_only (no CNN). "
            "cnn: only the CNN feature extraction (needs feature_extractor.keras). "
            "all: full pipeline. Others are kept for manual use."
        ),
    )
    args = parser.parse_args()

    generateData(data=None if args.mode == "all" else args.mode)
