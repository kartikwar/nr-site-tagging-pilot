import shutil
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from pathlib import Path
from config import INPUT_DIR, OUTPUT_DIR, LOG_PATH, GOLD_FILES_DIR, GOLD_METADATA_PATH, EVALUATION_DIR
from main import main
from rouge_score import rouge_scorer


import config
config.INPUT_DIR = config.GOLD_FILES_DIR
config.OUTPUT_DIR = config.EVALUATION_DIR / "output"
OUTPUT_DIR = config.OUTPUT_DIR
config.LOG_PATH = config.EVALUATION_DIR / "evaluation_log.csv"
LOG_PATH = config.EVALUATION_DIR / "evaluation_log.csv"
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def files_preparation():
    # Just clear the previous evaluation log
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    # Do NOT touch INPUT_DIR or copy any files


def normalize_columns(df, columns):
    for column in columns:
        df[column + "_gold"] = df[column + "_gold"].str.lower(
        ).str.strip()
        df[column + "_pred"] = df[column + "_pred"].str.lower(
        ).str.strip()
    return df


def load_evaluation_dataframe():
    pred_df = pd.read_csv(LOG_PATH)
    gold_df = pd.read_csv(GOLD_METADATA_PATH, header=3, encoding='ISO-8859-1')

    # Normalize filename column for alignment
    pred_df["Original_Filename"] = pred_df["Original_Filename"].str.strip()
    gold_df["Current BC Mail title"] = gold_df["Current BC Mail title"].str.strip()

    # rename so that column names are same for both pred and gold
    rename_dict = {
        "Duplicate  (Y/N)": "Duplicate",
        'Site Registry releaseable': 'Site_Registry_Releaseable',
        'Title/Subject': 'Title',
        'Sender/Author': 'Sender'
    }

    gold_df = gold_df.rename(columns=rename_dict)

    # Merge on original filename
    merged_df = pd.merge(
        gold_df,
        pred_df,
        left_on="Current BC Mail title",
        right_on="Original_Filename",
        suffixes=("_gold", "_pred"),
        how="inner"
    )

    # Normalize Duplicate_gold from 'Y'/'N' to 'yes'/'no'
    merged_df["Duplicate_gold"] = merged_df["Duplicate_gold"].astype(str).str.strip().str.upper().map({
        "Y": "yes",
        "N": "no"
    }).fillna("no")

    # Normalize Duplicate_pred from prediction strings to 'yes'/'no'
    merged_df["Duplicate_pred"] = merged_df["Duplicate_pred"].astype(str).str.strip().str.lower().map(
        lambda x: "yes" if x in ["contained", "likely_duplicate_ocr", "yes"] else "no"
    )

    # Normalize Site_Registry_Releaseable columns
    merged_df["Site_Registry_Releaseable_gold"] = merged_df["Site_Registry_Releaseable_gold"].astype(str).str.strip().str.upper().map({
        "Y": "yes",
        "N": "no",
        "N (DUPLICATE)": "no"
    }).fillna("no")

    merged_df["Site_Registry_Releaseable_pred"] = merged_df["Site_Registry_Releaseable_pred"].astype(str).str.strip().str.lower().map(
        lambda x: "yes" if x in ["yes", "y"] else "no"
    )

    # Normalize other string columns for ROUGE comparison
    merged_df = normalize_columns(
        merged_df, ["Title", "Sender", "Receiver"]
    )

    return merged_df


def get_rouge1_recall(gold, pred):
    if not isinstance(gold, str) or not isinstance(pred, str):
        return 0.0
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    score = scorer.score(gold, pred)['rouge1']
    return score.recall


def compute_rouge_recall(df, gol_col, pred_col, store_as):
    df[store_as] = df.apply(
        lambda row: get_rouge1_recall(row[gol_col], row[pred_col]), axis=1)
    return df


def compute_scores(merged_df):
    # Compute ROUGE recall scores for text fields
    merged_df = compute_rouge_recall(merged_df, 'Title_gold', 'Title_pred', 'Title_recall')
    merged_df = compute_rouge_recall(merged_df, 'Receiver_gold', 'Receiver_pred', 'Receiver_recall')
    merged_df = compute_rouge_recall(merged_df, 'Sender_gold', 'Sender_pred', 'Sender_recall')

    # Compute classification metrics for 'Duplicate'
    f1_duplicate = f1_score(merged_df["Duplicate_gold"], merged_df["Duplicate_pred"], pos_label="yes", zero_division=1)
    precision_duplicate = precision_score(merged_df["Duplicate_gold"], merged_df["Duplicate_pred"], pos_label="yes", zero_division=1)
    recall_duplicate = recall_score(merged_df["Duplicate_gold"], merged_df["Duplicate_pred"], pos_label="yes", zero_division=1)

    # Compute classification metrics for 'Site_Registry_Releaseable'
    f1_releasable = f1_score(merged_df["Site_Registry_Releaseable_gold"], merged_df["Site_Registry_Releaseable_pred"], pos_label="yes", zero_division=1)
    precision_releasable = precision_score(merged_df["Site_Registry_Releaseable_gold"], merged_df["Site_Registry_Releaseable_pred"], pos_label="yes", zero_division=1)
    recall_releasable = recall_score(merged_df["Site_Registry_Releaseable_gold"], merged_df["Site_Registry_Releaseable_pred"], pos_label="yes", zero_division=1)

    # Print results
    print("\n===== DUPLICATE =====")
    print(f"Precision: {precision_duplicate:.4f}")
    print(f"Recall:    {recall_duplicate:.4f}")
    print(f"F1 Score:  {f1_duplicate:.4f}")

    print("\n===== SITE REGISTRY RELEASABLE =====")
    print(f"Precision: {precision_releasable:.4f}")
    print(f"Recall:    {recall_releasable:.4f}")
    print(f"F1 Score:  {f1_releasable:.4f}")

    # Log scores into DataFrame (as constants for every row)
    merged_df["F1_Score_Duplicate"] = f1_duplicate
    merged_df["Precision_Duplicate"] = precision_duplicate
    merged_df["Recall_Duplicate"] = recall_duplicate

    merged_df["F1_Score_Site_Registry_Releaseable"] = f1_releasable
    merged_df["Precision_Site_Registry_Releaseable"] = precision_releasable
    merged_df["Recall_Site_Registry_Releaseable"] = recall_releasable

    # Save for inspection
    merged_df.to_csv(config.EVALUATION_DIR / "evaluation_merged_output.csv", index=False)


if __name__ == '__main__':
    # Step 1: Prepare files
    files_preparation()

    # Step 2: Run the pipeline
    main()

    # Step 3: Load evaluation dataframe
    merged_df = load_evaluation_dataframe()

    # Step 4: Compute F1 scores
    compute_scores(merged_df)
