# RAG-Based AIOps Framework for Automated Incident Diagnosis

> **Reference implementation for:**
> *"Retrieval-Augmented Generation for Automated Incident Diagnosis in Cloud-Native DevOps Environments"*
> Abhimanyu Garg, Rajendranath Rengan — Journal of Software: Evolution and Process, Springer

---

## Overview

This repository contains the full Proof-of-Concept (PoC) implementation of the RAG-based AIOps framework described in the paper. It demonstrates:

- **5-layer architecture**: Data Sources → Ingestion → Indexing → Retrieval → LLM Reasoning
- **Knowledge indexing pipeline**: Chunking, embedding generation, FAISS vector indexing
- **Real-time incident diagnosis**: Semantic retrieval + LLM-generated root cause analysis
- **REST API**: FastAPI-based service for integration with incident management tools
- **Evaluation harness**: Reproduce paper metrics (accuracy, MTTD reduction, ablation study)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG-AIOps Framework                         │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Ingestion   │   Indexing   │  Retrieval   │   LLM Reasoning    │
│              │              │              │                    │
│ • Tickets    │ • Chunking   │ • FAISS ANN  │ • GPT-4 / Claude   │
│ • Runbooks   │ • Embeddings │ • Re-ranking │ • Chain-of-thought  │
│ • Post-mort. │ • Metadata   │ • HyDE expan │ • Structured output │
│ • Alerts     │ • FAISS idx  │ • Filtering  │ • Confidence score  │
└──────────────┴──────────────┴──────────────┴────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional, for full stack)
- OpenAI API key (or Azure OpenAI endpoint)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your API keys and settings
```

### 3. Index sample knowledge base

```bash
python scripts/index_knowledge_base.py --data-dir data/samples
```

### 4. Run diagnosis on a sample incident

```bash
python scripts/diagnose_incident.py --incident data/samples/incidents/sample_incident.json
```

### 5. Start the REST API

```bash
uvicorn src.api.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 6. Run evaluation (reproduce paper results)

```bash
python scripts/evaluate.py --dataset data/samples/eval_dataset.json
```

---

## Docker Deployment

```bash
docker-compose up --build
```

Services started:
- `aiops-api` — FastAPI service on port 8000
- `faiss-server` — Vector index server on port 8001  
- `redis` — Query embedding cache on port 6379

---

## Project Structure

```
aiops-rag/
├── src/
│   ├── ingestion/          # Data source connectors
│   │   ├── base.py         # Abstract ingestion interface
│   │   ├── ticket_ingester.py
│   │   ├── runbook_ingester.py
│   │   └── alert_ingester.py
│   ├── indexing/           # Knowledge indexing pipeline
│   │   ├── chunker.py      # Document chunking strategies
│   │   ├── embedder.py     # Embedding generation
│   │   └── vector_store.py # FAISS vector store wrapper
│   ├── retrieval/          # Semantic retrieval engine
│   │   ├── retriever.py    # Main retrieval logic
│   │   ├── reranker.py     # Cross-encoder re-ranking
│   │   └── query_expander.py # HyDE query expansion
│   ├── diagnosis/          # LLM reasoning module
│   │   ├── context_builder.py
│   │   ├── llm_client.py
│   │   └── diagnoser.py
│   └── api/                # FastAPI REST interface
│       ├── main.py
│       ├── models.py
│       └── routes.py
├── config/
│   ├── config.example.yaml
│   └── config.yaml         # (gitignored — contains secrets)
├── data/
│   └── samples/            # Sample incidents, runbooks, tickets
├── scripts/
│   ├── index_knowledge_base.py
│   ├── diagnose_incident.py
│   └── evaluate.py
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_diagnoser.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{garg2025ragaiops,
  title     = {Retrieval-Augmented Generation for Automated Incident Diagnosis
               in Cloud-Native DevOps Environments},
  author    = {Garg, Abhimanyu and Rengan, Rajendranath},
  journal   = {Journal of Software: Evolution and Process},
  publisher = {Springer},
  year      = {2025}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
