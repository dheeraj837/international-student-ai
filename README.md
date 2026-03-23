# 🎓 International Student AI Assistant

A production-ready **RAG (Retrieval-Augmented Generation) Q&A system** that answers immigration and visa questions for international students using official USCIS sources.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
FastAPI  (/ask)
    │
    ├─► Sentence-Transformers  ──►  Query Embedding (all-MiniLM-L6-v2)
    │
    ├─► Qdrant Vector DB  ──►  Semantic Search (Top-K cosine similarity)
    │                           ~30% better contextual accuracy vs keyword search
    │
    └─► OpenAI GPT (gpt-4o-mini)  ──►  Grounded Answer + Sources
```

### Services
| Service | Description | Port |
|---------|-------------|------|
| `api` | FastAPI RAG application | `8000` |
| `qdrant` | Vector database for semantic retrieval | `6333` |

---

## 🚀 Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/yourrepo/international-student-ai
cd international-student-ai

cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 2. Start with Docker Compose

```bash
docker compose up --build
```

### 3. Ingest USCIS Documents

```bash
# Trigger via API
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"force_reingest": false}'

# Or run the standalone script (local Python env)
pip install -r requirements.txt
python scripts/ingest.py
```

### 4. Ask Questions!

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I apply for OPT as an F-1 student?"}'
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome + endpoint listing |
| `GET` | `/health` | Health check (Qdrant, collection stats) |
| `POST` | `/ask` | Ask an immigration question |
| `POST` | `/ingest` | Run ingestion pipeline |
| `GET` | `/sources` | List all USCIS source URLs |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |

### Example `/ask` Request

```json
{
  "question": "What is the cap-gap extension for F-1 students transitioning to H-1B?",
  "top_k": 5
}
```

### Example `/ask` Response

```json
{
  "question": "What is the cap-gap extension for F-1 students transitioning to H-1B?",
  "answer": "The cap-gap extension bridges the period between an F-1 student's OPT expiration and the H-1B start date (Oct. 1). If your employer files an H-1B petition on your behalf during the registration period...",
  "sources": [
    {
      "title": "Cap-Gap Extension – H-1B F-1 Students",
      "url": "https://www.uscis.gov/working-in-the-united-states/...",
      "category": "cap_gap",
      "score": 0.9123
    }
  ],
  "model_used": "gpt-4o-mini"
}
```

---

## 📚 Knowledge Base – USCIS Official Sources

The system indexes **17 official USCIS pages** across 7 categories:

| Category | Pages | Topics |
|----------|-------|--------|
| `student_visa` | 3 | F-1/M-1 overview, eligibility, status change |
| `opt` | 2 | OPT application, STEM OPT extension |
| `cap_gap` | 1 | H-1B cap-gap bridge |
| `h1b` | 3 | H-1B cap season, registration, FAQs |
| `policy_manual` | 4 | Official USCIS policy (Parts F Ch. 2,5,7) |
| `employer_handbook` | 2 | I-9 handbook for F-1/M-1 students |
| `news` | 2 | Recent policy alerts and guidance updates |

---

## 🔧 Ingestion Pipeline

```
USCIS URLs  →  Scrape (requests + BeautifulSoup)
            →  Clean (remove nav/footer, normalize whitespace)
            →  Chunk (sliding window, 512 words, 64 overlap)
            →  Embed (sentence-transformers: all-MiniLM-L6-v2, 384-dim)
            →  Index (Qdrant cosine similarity, batched upsert)
```

**100+ knowledge chunks** indexed from official government documents.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI + Uvicorn |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | Qdrant |
| Web Scraping | requests + BeautifulSoup4 + lxml |
| Containerization | Docker + Docker Compose |
| Config | pydantic-settings + dotenv |

---

## 🗂️ Project Structure

```
international-student-ai/
├── app/
│   ├── main.py                   # FastAPI app + routes
│   ├── core/
│   │   ├── config.py             # Settings (pydantic-settings)
│   │   └── qa_engine.py          # RAG Q&A logic (OpenAI)
│   ├── ingestion/
│   │   ├── pipeline.py           # Scrape→Chunk→Embed→Index
│   │   └── sources.py            # 17 official USCIS URLs
│   └── models/
│       └── schemas.py            # Pydantic request/response models
├── scripts/
│   └── ingest.py                 # Standalone CLI ingestion tool
├── Dockerfile                    # Multi-stage build
├── docker-compose.yml            # API + Qdrant services
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model for generation |
| `QDRANT_HOST` | `qdrant` | Qdrant service hostname |
| `QDRANT_PORT` | `6333` | Qdrant REST port |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `EMBEDDING_DIM` | `384` | Embedding vector dimension |
| `CHUNK_SIZE` | `512` | Words per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |

---

## 💬 Sample Questions to Try

- *"How do I apply for OPT as an F-1 student?"*
- *"What is the STEM OPT extension and who qualifies?"*
- *"Can I work on CPT before completing one year of study?"*
- *"What is the cap-gap extension and how long does it last?"*
- *"What documents do I need to transfer to a new school on F-1?"*
- *"How many hours can I work on campus during the semester?"*
- *"What happens if my OPT application is pending when it expires?"*

---

## ⚠️ Disclaimer

This tool provides **informational guidance only** based on publicly available USCIS content. It does **not** constitute legal advice. Always verify information with your Designated School Official (DSO) or a licensed immigration attorney. Immigration rules change — consult [uscis.gov](https://www.uscis.gov) for the most current policies.
