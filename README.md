# BC Environment Capstone: Document Classification Pipeline

This project automates the classification, renaming, and metadata extraction of scanned site remediation documents (PDFs) provided by the BC Ministry of Environment.

---

## How It Works

For each PDF in the input folder:

1. Extract the **site ID** from the filename or fall back to **LLM (LLaMA 2 via Ollama)**.
2. Classify the document type (e.g., REPORT, CORR, PSI).
3. Generate a clean filename:  
   `YYYY-MM-DD – SITE_ID – TYPE.pdf`
4. Copy the renamed file into `outputs/TYPE/`.
5. Log metadata to `logs/metadata_log.csv`.

---

## Folder Setup

Your folder should look like this:
### **Directory Structure**
```plaintext
CAPSTONE/
├── data/                         ← place raw PDFs here (outside the repo)
└── nr-site-tagging-pilot/       ← this repo
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── logs/
│   └── metadata_log.csv      ← created automatically
├── outputs/
│   └── REPORT/
│       └── renamed files
└── utils/
└── helper scripts
```

---

## How to Run

1. **Install dependencies**  
`pip install -r requirements.txt`

2. **Run LLaMA 2 model**  
`ollama run llama2`

3. **Run the pipeline**  
`python main.py`

---

## Output

Each processed PDF will appear in `outputs/TYPE/` with a new standardized filename. A CSV log is created at `logs/metadata_log.csv` with:

- Original filename  
- New filename  
- Site ID  
- Document type  
- Output path

---

## Notes

- Raw PDFs are never modified
- LLMs are used only if site ID is missing from filename
- Document types are inferred using keyword matching

This is an early prototype to demonstrate pipeline automation and reproducibility.