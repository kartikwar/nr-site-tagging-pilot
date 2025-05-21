import shutil
import pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path
from config import INPUT_DIR, OUTPUT_DIR, LOG_PATH, GOLD_FILES_DIR, GOLD_METADATA_PATH, EVALUATION_DIR
from main import main


import config
config.INPUT_DIR = config.GOLD_FILES_DIR
config.OUTPUT_DIR = config.EVALUATION_DIR / "output"
OUTPUT_DIR = config.OUTPUT_DIR
config.LOG_PATH = config.EVALUATION_DIR / "evaluation_log.csv"
LOG_PATH = config.EVALUATION_DIR / "evaluation_log.csv"
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def files_preparation():
    # Clear previous logs and input directory
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy gold files to input directory
    for pdf in GOLD_FILES_DIR.glob("*.pdf"):
        shutil.copy(pdf, INPUT_DIR / pdf.name)


def load_evaluation_dataframe():
    pred_df = pd.read_csv(LOG_PATH)
    gold_df = pd.read_csv(GOLD_METADATA_PATH, header=3)

    # Normalize filename column for alignment
    pred_df["Original_Filename"] = pred_df["Original_Filename"].str.strip()
    gold_df["Current BC Mail title"] = gold_df["Current BC Mail title"].str.strip()

    # rename so that column names are same

    gold_df = gold_df.rename(columns={"Duplicate  (Y/N)": "Duplicate"})
    gold_df = gold_df.rename(
        columns={'Site Registry releaseable': 'Site_Registry_Releaseable'})

    # Merge on original filename
    merged_df = pd.merge(
        gold_df,
        pred_df,
        left_on="Current BC Mail title",
        right_on="Original_Filename",
        suffixes=("_gold", "_pred"),
        how="inner"
    )

    # Convert Duplicate to binary labels
    merged_df["Duplicate_gold"] = merged_df["Duplicate_gold"].apply(
        lambda x: "yes" if x.lower() in ["contained", "ocr", "yes"] else "no")
    merged_df["Duplicate_pred"] = merged_df["Duplicate_pred"].apply(
        lambda x: "yes" if x.lower() in ["contained", "likely_duplicate_ocr", "yes"] else "no")

    # Convert Site Registry to lowercase yes/no
    merged_df["Site_Registry_Releaseable_gold"] = merged_df["Site_Registry_Releaseable_gold"].str.lower(
    ).str.strip()
    merged_df["Site_Registry_Releaseable_pred"] = merged_df["Site_Registry_Releaseable_pred"].str.lower(
    ).str.strip()

    return merged_df


def compute_scores():
    f1_duplicate = f1_score(
        merged_df["Duplicate_gold"], merged_df["Duplicate_pred"], pos_label="yes")
    f1_releasable = f1_score(merged_df["Site_Registry_Releaseable_gold"],
                             merged_df["Site_Registry_Releaseable_pred"], pos_label="yes")

    # Step 5: Store scores in array
    f1_scores = [f1_duplicate, f1_releasable]

    # Step 11: Print results
    print(f"F1 Score – Duplicate: {f1_duplicate:.4f}")
    print(f"F1 Score – Site Registry Releasable: {f1_releasable:.4f}")


if __name__ == '__main__':
    # Step 1: Prepare files
    files_preparation()

    # Step 2: Run the pipeline
    main()

    # Step 3: Load evaluation dataframe
    merged_df = load_evaluation_dataframe()

    # Step 4: Compute F1 scores
    compute_scores()
