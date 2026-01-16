# 🏗️ Brand Brain - Developer Documentation

> **Version**: 1.7
> **Status**: Active Development

This document serves as the technical manual for **Brand Brain**. It details the system architecture, data pipelines, retrieval logic, and the "Knowledge Loop" that powers the AI.

---

## 🧩 Architecture Overview

Brand Brain uses a **Hybrid RAG (Retrieval-Augmented Generation)** architecture. We strictly separate **Reasoning** (the LLM) from **Memory** (Database) to prevent hallucinations.

### High-Level Diagram

```mermaid
graph TD
    User[User Query] --> Valid[Safety & Intent Check]
    Valid -- Fail --> Reject[Refusal Response]
    Valid -- Pass --> Embed[Generate Query Vector]
    
    subgraph Memory System
        Embed --> Pinecone[(Pinecone Vector DB)]
        Pinecone -->|Chunk IDs| Postgres[(Postgres DB)]
        Postgres -->|Raw Text + Metadata| Context[Retrieved Context]
    end
    
    Context --> Orchestrator
    
    subgraph "Ephemeral Knowledge (Type C)"
        Orchestrator -.->|Live Search Request| GoogleSearch[Google Search]
        GoogleSearch -.->|Summary (Not Saved)| LiveData[Live Context]
    end
    
    Orchestrator --> LLM[Gemini 2.5 Flash]
    LiveData --> LLM
    LLM --> Final[Final On-Brand Response]
```

---

## 💾 The Data Pipeline

The "Knowledge Pickup" phase transforms raw text into a format the AI can understand and retrieve.

### 1. Ingestion (`core/ingestion.py`)

We do not simply dump text. All data is categorized during ingestion:

* **Extraction**: We map raw input fields (e.g., `mission`, `visualStyle`) to semantic **Asset Types**:
  * `guideline`: Inviolable rules.
  * `copy`: Examples of good writing.
  * `website`: Factual data.
* **Vector Types**: We also tag them for specific retrieval spaces:
  * `brand_voice`: For "How should I sound?" queries.
  * `strategy`: For "Who are our competitors?" queries.

### 2. Chunking & Embedding

* **Splitter**: `RecursiveCharacterTextSplitter` (Chunk Size: 350 chars, Overlap: 50).
* **Model**: `gemini-embedding-001`.
* **Dimensions**: 768.
* **Task Type**: We explicitly set `task_type='RETRIEVAL_DOCUMENT'` during ingestion to optimize the vectors for storage.

### 3. Storage Strategy (Hybrid)

We use the **"Reference Pattern"**:

1. **Pinecone**: Stores the **Vector** + `chunk_id`. (Optimized for search speed).
2. **Postgres**: Stores the `chunk_id` + **Full Text** + **Metadata**. (Optimized for storage and joining).

> **Why?** Vector DBs are expensive and poor at storing large metadata. SQL is cheap and robust. We only use Pinecone to find the *ID*, then fetch the actual content from Postgres.

---

## 🧠 The Knowledge Loop (Runtime)

When `main.py interactive` is run, the following pipeline executes for every message:

### Step 1: Intent & Safety Layer (`core/intent.py`)

Before answering, we check **"Can we answer this?"**

1. **Hard Guardrails**: Regex check for forbidden terms (e.g., "cheap", "clearance").
2. **Intent Classification**:
    * `KNOWLEDGE`: Asking about facts (Allowed).
    * `REASONING`: Asking for advice (Allowed).
    * `CREATIVE`: Asking to generate new content (Blocked/Restricted in v1).
3. **Semantic Drift**: We compare the query vector to the "Brand Centroid". If it's too far away (e.g., discussion about sports cars for a kitchen brand), it's flagged.

### Step 2: Retrieval (`core/retrieval.py`)

1. Generate Query Embedding (`task_type='RETRIEVAL_QUERY'`).
2. Query Pinecone in the specific **Namespace** (e.g., `org_id:brand_id:brand_voice`).
3. **Filter (v1.6+)**: Retrieve candidates from Postgres and filter out any assets marked as `deprecated` or with low `confidence`.

### Step 3: Reasoning (`core/reasoning.py`)

The LLM (`gemini-2.5-flash`) acts as a **Reasoner**, not a Knowledge Base.

* **System Prompt**: "You are [Brand Name]. Use ONLY the provided context."
* **Context**: We paste the trusted text chunks retrieved from Postgres.
* **Explainability**: The system output includes *why* it gave the answer and its confidence score.

---

## 📂 Directory Structure

```
d:/brand-brain/
├── .env                 # Secrets (API Keys)
├── DEV_README.md        # This file
├── README.md            # Public overview
├── main.py              # CLI Entry Point
├── requirements.txt     # Python Dependencies
└── brand_brain/         # Application Package
    ├── __init__.py
    ├── config.py        # Global Constants
    ├── database.py      # DB Connection Factory
    ├── tables.sql       # SQL Schema Definition
    ├── core/
    │   ├── ingestion.py # Asset Extraction & Storage
    │   ├── intent.py    # Classification Logic
    │   ├── memory_gov.py# Memory Management (Approve/Reject)
    │   ├── orchestrator.py # Main Pipeline Controller
    │   ├── reasoning.py # LLM Prompting & Response
    │   ├── retrieval.py # Vector Search Logic
    │   └── validation.py# Test Suites
    └── services/
        ├── gemini.py    # Google AI SDK Wrapper
        └── pinecone_svc.py # Pinecone Wrapper
```

## 🧪 Development Guide

### Setting Up Locally

1. **Environment**: Ensure `.env` is populated.
2. **Database**: You must run the SQL in `brand_brain/tables.sql` on your Neon/Postgres instance before running the app.
3. **Virtual Env**:

    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```

### Running Tests

We have a built-in validation suite in `main.py`:

```bash
# Runs the standard validation suite (Voice, Strategy, Safety)
python main.py validate
```

This will output a report checking if:

* [x] Brand voice is retrieved correctly.
* [x] Off-brand queries are blocked.
* [x] Live context is not persisted.
