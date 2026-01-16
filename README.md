# 🧠 Brand Brain

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini%202.5-orange?logo=google&logoColor=white)
![Pinecone](https://img.shields.io/badge/Vector%20DB-Pinecone-green?logo=pinecone&logoColor=white)
![Postgres](https://img.shields.io/badge/Database-Postgres%20(Neon)-336791?logo=postgresql&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-grey)

**Brand Brain** is a read-only, retrieval-augmented intelligence system designed to serve as the "Source of Truth" for brand identity. It ingests brand guidelines, strategy documents, and website content to answer questions with strict adherence to brand voice, safety, and factual accuracy.

Unlike generic chatbots, Brand Brain is **deterministic regarding facts** and **creative only within safety boundaries**. It uses a hybrid memory system (Vector + SQL) to ensure that every response is grounded in approved assets.

---

## ✨ Key Features

* **🛡️ Strict Brand Safety**: Automatically blocks off-brand requests (e.g., "cheap", "clearance") and detects "semantic drift" to ensure the AI never goes rogue.
* **🧠 Hybrid Memory Architecture**: Combines **Vector Search** (Pinecone) for semantic understanding with **SQL Lookup** (Postgres) for precise content retrieval.
* **🔍 Grounding-Assisted Ingestion**: Proactively searches the web to find and ingest "evergreen" brand philosophy, filtering out temporary noise like sales or news.
* **📝 Interactive Chat Playground**: A CLI-based chat interface that provides detailed **Explainability** for every answer (Intent, Confidence Score, Reasoning).
* **🚫 Ephemeral Live Context**: Fetches live data (e.g., "latest trends") for context but **never saves it**, keeping the core memory pure.

---

## 🛠️ Technology Stack

* **Core**: Python 3.11+
* **LLM**: Google Gemini 2.5 Flash (Reasoning & Tool Use)
* **Embeddings**: Gemini Embedding 001 (768 dimensions)
* **Vector Database**: Pinecone Serverless (AWS)
* **Relational Database**: PostgreSQL (via Neon)
* **Orchestration**: Custom Python RAG Pipeline (No heavy frameworks)

---

## 🚀 Quick Start

### Prerequisites

* Python 3.10+
* PostgreSQL Database (Neon recommended)
* Pinecone API Key
* Google Gemini API Key

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/brand-brain.git
    cd brand-brain
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Configure Environment:**
    Create a `.env` file in the root directory:

    ```ini
    GOOGLE_API_KEY=your_key_here
    PINECONE_API_KEY=your_key_here
    NEON_DB_URL=postgresql://user:pass@host/db
    ```

4. **Initialize Database:**
    Run the SQL script in your Postgres console to create tables:

    ```bash
    # (Manual step: Run contents of brand_brain/tables.sql in your DB tool)
    ```

### Usage

**1. Ingest Brand Data**
Load the initial brand guidelines into memory:

```bash
python main.py ingest
```

**2. Start Interactive Chat**
Talk to Brand Brain in your terminal:

```bash
python main.py interactive
```

**3. Run Validation Tests**
Verify system integrity and safety checks:

```bash
python main.py validate
```

---

## 📂 Project Structure

```
brand-brain/
├── brand_brain/        # Main Source Code
│   ├── core/           # Core Logic (Ingestion, Retrieval, Reasoning)
│   ├── services/       # API Wrappers (Gemini, Pinecone)
│   └── tables.sql      # Database Schema
├── notebook-dev/       # Experimental Notebooks
├── main.py             # CLI Entry Point
└── requirements.txt    # Dependencies
```

---

## 📄 License

This project is licensed under the MIT License.
