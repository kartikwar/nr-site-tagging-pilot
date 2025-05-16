# BC Environment Capstone: Document Classification Pipeline

This project automates the classification, renaming, and metadata extraction of scanned site remediation documents (PDFs) provided by the BC Ministry of Environment.

---

For each PDF in the input folder:

1. Extract the **Site ID** from the filename using regex, or fall back to **LLM (Mistral via Ollama)**.
2. Use a **document classifier** to assign a type (e.g., REPORT, PSI, CORR):
   - Default: regex-based keyword matcher
   - Optional: ML-based BERT model (Hugging Face)
3. Generate a standardized filename:  
   `YYYY-MM-DD – SITE_ID – TYPE.pdf`
4. Copy the renamed file into `data/output/{TYPE}/`
5. Log extracted metadata to `logs/metadata_log.csv`
6. Optionally compare results to a gold-standard metadata CSV for validation

---

## Folder Setup

Your folder should look like this:

### **Directory Structure**

```plaintext
CAPSTONE/
├── data/
│   ├── input/                                      ← Raw PDFs go here
│   ├── output/                                     ← Renamed + organized PDFs will go here
│   ├── logs/                                       ← CSV log will go here
│   └── lookups/                                    ← Required lookup/reference files
│       └── site_registry_mapping.xlsx              ← REQUIRED: Document type to release eligibility mapping
│       └── site_ids.csv                            ← REQUIRED: Site ID to Address mapping
├── nr-site-tagging-pilot/
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt
│   ├── README.md
│   ├── CODE_OF_CONDUCT.md
│   ├── CONTRIBUTING.md
│   ├── LICENSE.txt
│   ├── prompts/
│   │   └── metadata_prompt.txt                     ← LLM prompt template
│   │   └── address_reprompt.txt
│   │   └── site_id_reprompt.txt
│   ├── models/ ← Save model here, we will share this with team
│   │   └── document_classification_model ← ensure same name
│   └── utils/                   ← Helper scripts (see internal README)

```
---

REQUIRED LOOKUP FILES

The following file must be placed inside the data/lookups/ directory **before running the pipeline**:

1. site_registry_mapping.xlsx  
   - Purpose: Determines whether a document is publicly releasable based on its type.  
   - Columns:
     • Document_Type  
     • Site_Registry_Releaseable (values must be "yes" or "no")

2. site_ids.csv  
   - Purpose: Contains site id to address mapping.  

This file is mandatory. The pipeline will not run if it is missing.

---

## How to Run

1. **Install dependencies**  
`pip install -r requirements.txt`

2. **Download ollama and install the .exe file** from following <https://ollama.com/download/>

3. **Install Models**

`ollama pull mistral`

4. **Run LLaMA 2 model**  
`ollama run mistral`

5. **Run the pipeline**  
`python main.py`

---

## Output

Each processed PDF will appear in `outputs/Site ID/TYPE/` with a new standardized filename. A CSV log is created at `logs/metadata_log.csv` with:

- Original_Filename            → Name of the original input PDF
- New_Filename                 → Standardized renamed file
- Site_id                      → 4-digit Site ID (from filename or LLM)
- Document_Type                → Classified type of the document
- Site_Registry_Releaseable    → "yes", "no" or "no (duplicate)", if ready for public release
- Title                        → Full document or report title
- Receiver                     → Person or organization the document is addressed to
- Sender                       → Person or organization that authored/submitted the document
- Address                      → Site or project location
- Duplicate                    → "yes" if flagged as a duplicate, otherwise "no"
- Readable                     → "yes" if the document was mostly readable, otherwise "no"
- Output_Path                  → Full path to the renamed file in the output directory

---

## Notes

- Raw PDFs are never modified
- The LLM is only used when metadata is incomplete or ambiguous
- Document types are inferred using keyword matching
- To use the ML classifier, set USE_ML_CLASSIFIER = True in main.py

This is an early prototype to demonstrate pipeline automation and reproducibility.
