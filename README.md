# Local Document RAG System

A Retrieval-Augmented Generation (RAG) pipeline built from scratch in Python using **ChromaDB**, **pypdf**, and **Google Gemini (`gemini-3.8-flash`)**. It ingests a 54-page biotechnology thesis PDF, chunks and embeds the text into a local vector database, retrieves the most relevant passages for a user query, and grounds the LLM's response strictly in the retrieved context with page citations.

## Architecture
1. **Ingestion (`ingest.py`)**: Extracts text page-by-page from PDFs in `data/`, splits it into sentence-aware overlapping chunks (~1,800 characters / ~450 tokens with a 350-character overlap), and stores vector embeddings locally in ChromaDB (`all-MiniLM-L6-v2`).
2. **Retrieval & Generation (`rag.py`)**: Embeds the user's question, fetches the top-4 most similar chunks with source/page metadata, and runs a side-by-side comparison between a plain LLM call (no context) and a RAG-grounded LLM call.

## Dataset Indexed
* **Document:** `1822-b.tech-biotech-batchno-22.pdf` (*Formulation and Evaluation of Herbal Based Patch for Wound Healing Through In-Vitro Scratch Assay*)
* **Stats:** 54 pages total → 50 text-bearing pages extracted → **58 chunks** indexed in ChromaDB.

## Project Structure
```text
rag-system/
├── data/
│   └── 1822-b.tech-biotech-batchno-22.pdf
├── ingest.py
├── rag.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup & Usage
```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Mac/Linux: source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a .env file and add your Gemini API key
# GEMINI_API_KEY=your_api_key_here

# 4. Ingest and index the PDF
python ingest.py

# 5. Query the RAG system (runs With RAG vs. Without RAG comparison)
python rag.py "Where were the fresh tubers of Mimosa pudica collected in this study, where was Senna auriculata obtained from, and how many days were the tubers shade dried?"
```

## Proof: With RAG vs. Without RAG

**Question:**
> *Where were the fresh tubers of Mimosa pudica collected in this study, where was Senna auriculata obtained from, and how many days were the tubers shade dried?*

### 1. Retrieved Context (Top Chunks from ChromaDB)
```text
1. 1822-b.tech-biotech-batchno-22.pdf (Page 23) | Distance: 0.4482
   Preview: Chapter 4 Experimental or Material and Methods Used 4.1 COLLECTION OF PLANT MATERIAL Fresh tubers of Mimosa pudica , which were collected from Shollinganallur, ...

2. 1822-b.tech-biotech-batchno-22.pdf (Page 29) | Distance: 0.5834
   Preview: Fig 4.7 Qualitative analysis of Mimosa pudica Fig 4.8 Qualitative analysis of Senna auriculata Qualitative analysis results...

3. 1822-b.tech-biotech-batchno-22.pdf (Page 25) | Distance: 0.7479
   Preview: Fig 4.3 Work flow of gel formation Materials ● Senna auriculata ● Mimosa pudica Fig 4.4 Sample collection Plant extraction ▪ Materials extracts with solvent (et...

4. 1822-b.tech-biotech-batchno-22.pdf (Page 26) | Distance: 0.7942
   Preview: Fig 4.5 Extraction using soxhlet method ● Mimosa pudica extracts with solvent (ethyl acetate) using soxhlet method. Fig 4.6: Mimosa pudica extraction ● Senna au...
```

### 2. Without RAG (Plain LLM — No Retrieval)
> It appears that you forgot to attach or cite the specific study you are referring to.
>
> Because *Mimosa pudica* typically produces taproots rather than true tubers, and collection details vary by publication, the exact details cannot be confirmed without the source text.
>
> Please provide the **text**, **title**, or **DOI/link** of the study, and I will gladly extract the exact collection sites and drying duration for you.

### 3. With RAG (Grounded in Retrieved Chunks)
> Based on the provided context:
>
> - **Collection of *Mimosa pudica* tubers:** They were collected from Shollinganallur, Chennai, Tamil Nadu.
> - **Source of *Senna auriculata*:** Dried powder of *Senna auriculata* was collected from Amazon, an online market.
> - **Duration of shade drying:** The tubers were shade dried for 4 days.
>
> **Citation:** [Source: 1822-b.tech-biotech-batchno-22.pdf, Page 23]