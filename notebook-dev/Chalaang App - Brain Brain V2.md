# Chalaang Brand Brain — Technical Architecture & Development Guide

**Audience:** Backend, Platform, AI/ML Engineers  
**Purpose:** Single source of truth for building **Brand Brain v1**  
**Tone:** Engineering-first, implementation-focused  
**Status:** 🔒 FINALIZED (Phase 1 Locked)

---

## 1. What Chalaang Is (Context)

Chalaang is a **brand intelligence platform** used by agencies and teams to manage brands, projects, and creatives.

The existing Chalaang application:
- Uses **Next.js**
- Stores operational data in **DynamoDB**
- Handles authentication via **AWS Cognito**
- Manages brands, projects, creatives, workflows, and users

The Chalaang app is **already functional** and acts as the **operational system of record**.

---

## 2. What Brand Brain Is (Critical Definition)

**Brand Brain is NOT the Chalaang app.**

Brand Brain is a **separate cognitive system** whose responsibility is:

> To understand, remember, and protect brand identity  
> — not to generate content blindly.

Brand Brain is:
- A **persistent memory system**
- A **semantic reasoning layer**
- A **brand-safety enforcement engine**

Brand Brain is **not**:
- A chatbot
- A UI feature
- A stateless content generator
- A replacement for DynamoDB

---

## 3. System Separation (LOCKED)

```

┌────────────────────┐
│  Chalaang App      │
│  (Next.js)         │
│                    │
│  • Users           │
│  • Brands          │
│  • Projects        │
│  • Creatives       │
│  • DynamoDB        │
└─────────┬──────────┘
│
│ Brand JSON (DynamoDB export)
▼
┌──────────────────────────┐
│  Brand Brain Service     │
│  (New Repo)              │
│                          │
│  • Ingestion             │
│  • Memory                │
│  • Embeddings            │
│  • Retrieval             │
│                          │
│  Postgres + Pinecone     │
└──────────────────────────┘

```

---

## 4. Core Architectural Principle (DO NOT VIOLATE)

> **DynamoDB stores what users enter.**  
> **Postgres stores what the Brand Brain understands.**  
> **Pinecone stores how meaning is recalled.**

Each layer has **one responsibility only**.

---

## 5. Brand Brain v1 Scope (LOCKED)

### ✅ Included

- Brand ingestion from DynamoDB-style JSON
- Semantic asset extraction
- Chunking & text embeddings
- Pinecone vector storage
- Brand-scoped semantic retrieval
- Multi-brand isolation
- Off-brand detection (rule-based)
- Explainable recall

### ❌ Explicitly Excluded (Phase 1)

- Feedback learning loops
- Drift dashboards
- Performance scoring
- Image / video generation
- Agentic workflows
- Multi-LLM routing
- Cross-brand learning

> **Phase 1 proves correctness, not scale.**

---

## 6. Brand Brain Lifecycle (FINAL)

```

User Input
↓
Intent Classification
↓
Brand-Scoped Retrieval
↓
Memory Filtering
↓
Layered Prompt Composition
↓
LLM Response
↓
Off-Brand Validation
↓
Explain / Reject / Accept
↓
(Feedback Stored — later phase)

```

---

## 7. Ingestion Strategy (FINAL)

### 7.1 Source of Truth

- **Input:** Brand JSON from DynamoDB (Chalaang App)
- **Direction:** One-way (DynamoDB → Brand Brain)
- **Never:** Brand Brain → DynamoDB

---

### 7.2 Brand Brain Ingestion Adapter

A stateless ingestion layer that:

1. Reads brand JSON
2. Extracts semantic brand assets
3. Stores structured memory in Postgres
4. Generates embeddings
5. Pushes vectors to Pinecone

No UI, no authentication, no business logic.

---

## 8. Semantic Asset Extraction Rules (LOCKED)

One DynamoDB brand record produces **multiple semantic assets**.

| Source Field | asset_type | vector_type | Purpose |
|-------------|-----------|-------------|---------|
| Mission Statement | guideline | strategy | Brand intent |
| Brand Voice | guideline | brand_voice | Tone & language |
| Tone of Voice | guideline | brand_voice | Linguistic constraints |
| Visual Style (text) | guideline | brand_voice | Descriptive style |
| Audience Data | strategy | strategy | Reasoning context |
| Competitors (strengths / weaknesses) | strategy | strategy | Market positioning |
| Inspirational Brands | strategy | strategy | Ideation context |

⚠️ Social URLs, color hex codes, IDs, and emails are **not embedded**.

---

## 9. Chunking Rules (FROZEN)

- Target size: **200–350 tokens**
- Overlap: **40–60 tokens**
- Sentence-aware chunking
- One language per chunk
- Chunks are **versioned**, never edited in place

---

## 10. Embedding Strategy (v1)

- **Embedding model:** `gemini-embedding-001`
- **Similarity metric:** Cosine similarity
- **One embedding per chunk**
- **UUIDs used everywhere**

### Vector Namespace (CRITICAL)

```

org_id:brand_id:vector_type

```

This guarantees:
- Zero cross-brand leakage
- Strong isolation
- Deterministic retrieval

---

## 11. Storage Responsibilities

### Postgres (Brand Brain Memory)

Stores:
- Organizations
- Brands
- Brand assets
- Brand chunks
- Embedding metadata
- Brand insights (later)
- Feedback (later)

### Pinecone (Semantic Recall Engine)

Stores:
- Vector ID = embedding_id
- Vector values
- Namespace
- Minimal metadata only

---

## 12. Multi-Brand Safety Rules (NON-NEGOTIABLE)

1. No global namespace
2. No fallback memory
3. No generation without brand memory
4. Empty retrieval → safe rejection
5. Brand rules override user instructions

---

## 13. Brand Drift Detection (Logic Only)

Used for validation (dashboards later):

- Embedding similarity threshold
- Insight violation count
- Memory coverage check
- Edit distance (future)
- Rejection frequency

---

## 14. Brand Brain Quality Checklist (v1)

A Brand Brain is considered **working** if:

- Different brands sound unmistakably different
- Off-brand prompts are rejected
- Retrieval explains *why* an output fits
- No cross-brand contamination
- Empty memory fails safely

---

## 15. Development Strategy (FINAL)

### Phase 0 — Validation (Current)

- One **Jupyter Notebook**
- End-to-end ingestion + retrieval
- Manual verification
- No FastAPI, no services

### Phase 1 — Productionization

- Convert notebook logic into services
- Introduce FastAPI
- Background ingestion jobs
- API contracts between repos

---

## 16. Repository Strategy

### Repo 1: `chalaang-app`
- Next.js
- DynamoDB
- Cognito
- UI & workflows

### Repo 2: `chalaang-brand-brain`
- Python
- Postgres (Neon)
- Pinecone
- Gemini embeddings

Repos communicate via **data contracts**, not shared code.

---

## 17. Final Architectural Truth

> **Brand Brain is a memory system, not a generator.**  
> **If memory is wrong, intelligence is useless.**

Everything built later depends on Phase 1 being correct.

---

## 18. Phase 1 — LOCKED ✅

- Brand Brain lifecycle
- Chunking rules
- Retrieval logic
- Prompt layering (conceptual)
- Multi-brand isolation
- MVP vs Production boundary

---

**End of Document**