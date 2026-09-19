# One-Notebook E-commerce RAG Chatbot



`notebook/complete_ecommerce_rag_chatbot.ipynb`

Run it from top to bottom. It trains language detection, sentiment, intent routing, builds and evaluates the RAG index, integrates Groq, tests the chatbot, and exports one artifact archive.

## Contents

- `notebook/complete_ecommerce_rag_chatbot.ipynb` — the only training/integration notebook.
- `app/` — local FastAPI API and browser chat interface.


## Local deployment

After the notebook creates `complete_rag_artifacts.zip`, extract it as `artifacts/` beside this README. Then:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # Windows: copy .env.example .env
# Edit .env and add GROQ_API_KEY
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`; API documentation is at `/docs`.

## Expected artifacts

```text
artifacts/
├── evaluation_summary.json
├── language/language_pipeline.joblib
├── sentiment/config.json
├── sentiment/model.safetensors
├── sentiment/tokenizer files
├── intent/intent_pipeline.joblib
└── rag/
    ├── faiss.index
    ├── knowledge_base.parquet
    └── metadata.json
```

## Safety

The chatbot does not connect to a real order database. It must not claim actual order/refund status or completed actions. Low-relevance retrieval routes to human escalation. Never commit `.env`, credentials, or generated artifacts.
