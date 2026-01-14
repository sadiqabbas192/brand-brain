# Brand Brain (v1.7) - Developer Documentation

Brand Brain is a **read-only brand intelligence system** designed to answer questions about a brand's identity, strategy, and guidelines while strictly enforcing safety and brand voice. It uses a Retrieval-Augmented Generation (RAG) architecture powered by Google Gemini and Pinecone.

## 📂 Project Structure

The project is organized as a Python package with a clear separation of concerns:

```
d:/brand-brain/
├── .env                # Environment variables (API Keys, DB Credentials)
├── main.py             # CLI Entry Point (Ingest, Validate, Interactive Chat)
├── requirements.txt    # Python dependencies
└── brand_brain/        # Main Package
    ├── config.py       # Configuration & Constants (Allowed Intents, Keywords)
    ├── database.py     # Database Connections (Postgres & Pinecone)
    ├── core/           # Core Business Logic
    │   ├── ingestion.py    # Logic for extracting, chunking, and storing assets
    │   ├── intent.py       # Hybrid Intent Classification (Regex + Gemini)
    │   ├── reasoning.py    # RAG Orchestrator, Safety Checks, Response Generation
    │   ├── retrieval.py    # Context retrieval from Pinecone
    │   ├── memory_gov.py   # Memory Governance (Review/Approve/Reject)
    │   └── validation.py   # Verification scripts (v1.5/v1.6/v1.7 logic)
    └── services/       # External Service Wrappers
        ├── gemini.py       # Google GenAI SDK wrapper
        ├── embedding.py    # Text chunking and embedding generation
        └── storage.py      # CRUD helpers (if applicable)
```

---

## 🏗️ Technical Architecture

### 1. Ingestion Pipeline (`core/ingestion.py`)
This process transforms raw brand JSON into searchable vector memory.

1.  **Extraction**: Raw fields (e.g., `mission`, `brandVoice`) are converted into labeled **Assets**.
2.  **Chunking**: Large assets are split into smaller text chunks.
3.  **Embedding**: Each chunk is embedded using `gemini-embedding-001`.
4.  **Storage**:
    *   **Postgres**: Stores metadata (Assets, Chunks, Embeddings mappings).
    *   **Pinecone**: Stores vectors for semantic search.

### 2. The Chat Pipeline (`core/reasoning.py` & `main.py`)
When a user asks a question, the data flows as follows:

1.  **Intent Classification** (`core/intent.py`):
    *   **Rule-Based**: Checks for keywords like "create", "write" (fast fail for Creative requests).
    *   **LLM Fallback**: Uses Gemini to classify ambiguity into `KNOWLEDGE`, `REASONING`, or `CREATIVE`.
2.  **Safety Check** (`core/reasoning.py`):
    *   Checks for `FORBIDDEN_KEYWORDS`.
    *   **Soft Safety**: If intent is `REASONING` but keywords are found, it returns `PASS_WITH_WARNING` instead of a hard block, allowing the model to explain *why* something is off-brand.
3.  **Retrieval** (`core/retrieval.py`):
    *   Fetches relevant chunks from Pinecone.
    *   Filters by `brand_id` and supported memory types.
4.  **Response Generation** (`core/reasoning.py`):
    *   Constructs a strict system prompt.
    *   If `PASS_WITH_WARNING`, injects instructions to explain the conflict politely.
    *   Generates the final response using `gemini-2.5-flash` with low temperature (0.3).

### 3. Memory Model
*   **Type A (Static)**: Core guidelines ingested from JSON/Docs.
*   **Type B (Grounded)**: AI-extracted insights from the web (e.g., "Find our design principles from the website"), reviewed and stored.
*   **Type C (Ephemeral)**: Live web search for one-off queries. **Never stored.**

---

## 🚀 Setup & Usage

### 1. Environment Variables
Create a `.env` file in the root directory:

```ini
GOOGLE_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
NEON_DB_URL=your_postgres_connection_string
```

### 2. CLI Commands (`main.py`)

**Ingest Data**
Ingests the sample "Westinghouse India" brand data defined in `main.py`.
```bash
python main.py ingest
```

**Interactive Mode**
Starts a CLI chat session with Brand Brain.
```bash
python main.py interactive
```

**Validation**
Runs automated tests to verify ingestion and retrieval logic.
```bash
python main.py validate
# or for specific v1.6 tests
python main.py validate --v1.6
```

---

## 🧩 Key Logic Details

### Hybrid Intent Classifier (`core/intent.py`)
To optimize latency and cost, we use a waterfall approach:
1.  **Regex**: Immediate rejection of obvious creative prompts ("write me a blog...").
2.  **Gemini Flash**: If regex passes, a small LLM call classifies the intent.

### Soft Safety (`PASS_WITH_WARNING`)
We don't want to just say "No" when a user asks, "Can we utilize neon pink for our luxury brand?"
*   Instead of blocking, we flag it as `PASS_WITH_WARNING`.
*   The final prompt receives this warning and instructs the model to **explain** the misalignment (e.g., "Neon pink conflicts with our 'Subtle & Premium' color palette...") rather than executing the request.
