import shutil
import pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path
from config import INPUT_DIR, OUTPUT_DIR, LOG_PATH, GOLD_FILES_DIR, GOLD_METADATA_PATH, EVALUATION_DIR
from main import main
import config
config.INPUT_DIR = config.GOLD_FILES_DIR  # already correct
config.OUTPUT_DIR = config.EVALUATION_DIR / "output"
config.LOG_PATH = config.EVALUATION_DIR / "evaluation_log.csv"

config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Step 1: Clear previous logs and input directory
if LOG_PATH.exists():
    LOG_PATH.unlink()
if INPUT_DIR.exists():
    shutil.rmtree(INPUT_DIR)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 2: Copy gold files to input directory
for pdf in GOLD_FILES_DIR.glob("*.pdf"):
    shutil.copy(pdf, INPUT_DIR / pdf.name)

# Step 3: Run the pipeline
main()

# Step 4: Load predictions and gold metadata
pred_df = pd.read_csv(LOG_PATH)
gold_df = pd.read_csv(GOLD_METADATA_PATH)

# Step 5: Normalize filename column for alignment
pred_df["Original_Filename"] = pred_df["Original_Filename"].str.strip()
gold_df["filename"] = gold_df["filename"].str.strip()

# Step 6: Merge on original filename
merged_df = pd.merge(
    gold_df,
    pred_df,
    left_on="filename",
    right_on="Original_Filename",
    suffixes=("_gold", "_pred"),
    how="inner"
)

# Step 7: Convert Duplicate to binary labels
merged_df["Duplicate_gold"] = merged_df["Duplicate_gold"].apply(lambda x: "yes" if x.lower() in ["contained", "ocr", "yes"] else "no")
merged_df["Duplicate_pred"] = merged_df["Duplicate_pred"].apply(lambda x: "yes" if x.lower() in ["contained", "likely_duplicate_ocr", "yes"] else "no")

# Step 8: Convert Site Registry to lowercase yes/no
merged_df["Site_Registry_Releaseable_gold"] = merged_df["Site_Registry_Releaseable_gold"].str.lower().str.strip()
merged_df["Site_Registry_Releaseable_pred"] = merged_df["Site_Registry_Releaseable_pred"].str.lower().str.strip()

# Step 9: Compute F1 scores
f1_duplicate = f1_score(merged_df["Duplicate_gold"], merged_df["Duplicate_pred"], pos_label="yes")
f1_releasable = f1_score(merged_df["Site_Registry_Releaseable_gold"], merged_df["Site_Registry_Releaseable_pred"], pos_label="yes")

# Step 10: Store scores in array
f1_scores = [f1_duplicate, f1_releasable]

# Step 11: Print results
print(f"F1 Score – Duplicate: {f1_duplicate:.4f}")
print(f"F1 Score – Site Registry Releasable: {f1_releasable:.4f}")