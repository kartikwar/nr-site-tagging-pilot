import shutil
import pandas as pd
from sklearn.metrics import f1_score
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
    # Clear previous logs and input directory
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy gold files to input directory
    for pdf in GOLD_FILES_DIR.glob("*.pdf"):
        shutil.copy(pdf, INPUT_DIR / pdf.name)


def normalize_columns(df, columns):
    for column in columns:
        df[column + "_gold"] = df[column + "_gold"].str.lower(
        ).str.strip()
        df[column + "_pred"] = df[column + "_pred"].str.lower(
        ).str.strip()
    return df


def load_evaluation_dataframe():
    pred_df = pd.read_csv(LOG_PATH)
    gold_df = pd.read_csv(GOLD_METADATA_PATH, header=3)

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

    # Convert Duplicate to binary labels
    merged_df["Duplicate_gold"] = merged_df["Duplicate_gold"].apply(
        lambda x: "yes" if x.lower() in ["contained", "ocr", "yes"] else "no")
    merged_df["Duplicate_pred"] = merged_df["Duplicate_pred"].apply(
        lambda x: "yes" if x.lower() in ["contained", "likely_duplicate_ocr", "yes"] else "no")

    # strip spaces and lower words
    merged_df = normalize_columns(
        merged_df, ["Site_Registry_Releaseable", "Title", "Sender", "Receiver"])

    return merged_df


def get_rouge1_recall(gold, pred):
    scorer = rouge_scorer.RougeScorer(['rouge1'],
                                      use_stemmer=True)
    score = scorer.score(gold, pred)['rouge1']
    return score.recall


def compute_rouge_recall(df, gol_col, pred_col, store_as):
    df[store_as] = df.apply(
        lambda row: get_rouge1_recall(row[gol_col], row[pred_col]), axis=1)
    return df


def compute_scores(merged_df):
    # compute rouge 1 recall score for title and store it in df
    merged_df = compute_rouge_recall(
        merged_df, 'Title_gold', 'Title_pred', 'Title_recall')

    merged_df = compute_rouge_recall(
        merged_df, 'Receiver_gold', 'Receiver_pred', 'Receiver_recall')

    merged_df = compute_rouge_recall(
        merged_df, 'Sender_gold', 'Sender_pred', 'Sender_recall')

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
    compute_scores(merged_df)
