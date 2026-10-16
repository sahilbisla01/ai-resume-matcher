# Autonomous Job Description & Resume Matcher Agent

A production-ready, modular AI agent that parses a resume (PDF or Word DOCX), compares it
against a target job description, and returns:

- An **ATS match score (0-100)** with rationale
- **Missing hard/soft skills** critical to the JD
- **Formatting red flags** likely to break ATS parsing
- **X-Y-Z formatted rewrites** of your 2-3 weakest bullet points, tailored to the JD's keywords
- **Tailored Cover Letter Generator** automatically customized to the role
- **Multi-LLM Support:** Free tiers (**Google Gemini 1.5 Flash**, **Groq Llama 3.3**), offline (**Local Ollama**), or paid (**Anthropic Claude**)

Built with **FastAPI** (backend), **Streamlit** (frontend), **spaCy** (deterministic NLP baseline), and **Multi-Provider LLM Agents**.

---

## 1. Architecture

```
ai-resume-matcher/
├── .env.example              # Copy to .env and fill in your API key
├── requirements.txt
├── backend/
│   ├── app.py                 # FastAPI: /health /parse /evaluate /rewrite /match
│   ├── parser.py               # pypdf -> pdfplumber fallback + text cleaning
│   ├── nlp_extractor.py        # spaCy NER + baseline token/keyword overlap score
│   └── agents.py                # Claude Agent 1 (ATS Evaluator) + Agent 2 (Bullet Tailor)
└── frontend/
    └── streamlit_app.py        # Upload UI, score gauge, keyword badges, before/after bullets
```

**Pipeline flow:**

1. `parser.py` extracts and cleans raw text from the uploaded resume PDF.
2. `nlp_extractor.py` runs spaCy to extract skills/degrees/experience and
   computes a fast, deterministic **baseline score** (token overlap +
   keyword match) — this works even if Claude is unavailable, and is also
   passed to Agent 1 as grounding context to reduce score hallucination.
3. `agents.py` calls Claude twice:
   - **Agent 1 (ATS Evaluator):** strict-JSON score, missing skills, red flags.
   - **Agent 2 (Bullet Point Tailor):** rewrites weak bullets in Google's
     X-Y-Z format, using JD keywords.
4. `app.py` exposes it all as async FastAPI endpoints.
5. `streamlit_app.py` is the dashboard: resume/JD side-by-side, a score
   gauge, clickable keyword badges, and a before/after bullet accordion
   with copy-to-clipboard buttons.

---

## 2. Prerequisites

- Python 3.10+
- An Anthropic API key: https://console.anthropic.com/settings/keys

---

## 3. Setup

```bash
# 1. Clone / enter the project
cd ai-resume-matcher

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy English model
python -m spacy download en_core_web_sm

# 5. Configure environment variables
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

> `.env` is read by **both** the backend (via `python-dotenv` in `app.py`)
> and the frontend (via `python-dotenv` in `streamlit_app.py`). Keep one
> `.env` at the project root, or copy it into each subfolder if you prefer
> to run them from there — the code loads whichever `.env` is discoverable
> from the current working directory.

---

## 4. Running the app

Open **two terminals** (backend and frontend run as separate processes).

### Terminal 1 — Backend (FastAPI)

```bash
cd ai-resume-matcher/backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- API docs (Swagger UI): http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Terminal 2 — Frontend (Streamlit)

```bash
cd ai-resume-matcher/frontend
streamlit run streamlit_app.py
```

- Dashboard: http://localhost:8501

The sidebar in the dashboard will show a green **"Backend connected"**
badge once both are running correctly.

---

## 5. API Reference

| Method | Endpoint    | Body (multipart/form-data)        | Description                                   |
|--------|-------------|------------------------------------|------------------------------------------------|
| GET    | `/health`   | —                                   | Liveness + config check                        |
| POST   | `/parse`    | `resume` (PDF file)                 | Extract text + spaCy entities                   |
| POST   | `/evaluate` | `resume` (PDF), `jd_text` (string)  | Agent 1 only — ATS score & gaps                 |
| POST   | `/rewrite`  | `resume` (PDF), `jd_text` (string)  | Agent 2 only — X-Y-Z bullet rewrites            |
| POST   | `/match`    | `resume` (PDF), `jd_text` (string)  | Full pipeline (parse + Agent 1 + Agent 2)       |

Example `curl`:

```bash
curl -X POST http://localhost:8000/match \
  -F "resume=@/path/to/resume.pdf" \
  -F "jd_text=We are looking for a Python engineer with FastAPI, AWS, and NLP experience..."
```

---

## 6. Configuration reference (`.env`)

| Variable               | Default                          | Purpose                                    |
|-------------------------|-----------------------------------|---------------------------------------------|
| `ANTHROPIC_API_KEY`     | *(required)*                      | Your Claude API key                         |
| `ANTHROPIC_MODEL`       | `claude-sonnet-4-5`                | Model used by both agents                   |
| `ANTHROPIC_MAX_TOKENS`  | `2048`                             | Max output tokens per agent call            |
| `BACKEND_HOST`          | `0.0.0.0`                          | FastAPI bind host                           |
| `BACKEND_PORT`          | `8000`                             | FastAPI bind port                           |
| `ALLOWED_ORIGINS`       | `http://localhost:8501,...`       | CORS allow-list                             |
| `BACKEND_URL`           | `http://localhost:8000`           | URL the Streamlit app calls                 |
| `SPACY_MODEL`           | `en_core_web_sm`                   | spaCy model name                            |
| `MAX_UPLOAD_MB`         | `10`                                | Max resume PDF upload size                  |

> **Model note:** `claude-sonnet-4-5` is the current recommended default
> (best balance of intelligence/speed/cost). If your account has access to
> a newer model generation, just change `ANTHROPIC_MODEL` in `.env` — no
> code changes required.

---

## 7. Design notes / production considerations

- **Graceful PDF fallback:** `parser.py` tries `pypdf` first (fast) and
  automatically falls back to `pdfplumber` (more robust layout engine) if
  extraction yields too little text — common with multi-column resumes.
- **Deterministic baseline before LLM:** `nlp_extractor.py` computes a free,
  instant token/keyword overlap score. This is shown in the UI immediately
  and also fed into Agent 1's prompt as a grounding reference to reduce
  hallucinated scores.
- **Strict JSON contracts:** both agents are prompted for JSON-only output
  and parsed defensively (`agents._extract_json`) to tolerate stray
  markdown fences the model occasionally adds.
- **Retries:** Claude API calls use `tenacity` with exponential backoff on
  rate limits/transient errors (3 attempts).
- **CORS:** configurable allow-list via `ALLOWED_ORIGINS`, not `*`, for
  production hygiene.
- **Upload limits & content-type validation** guard the `/parse`,
  `/evaluate`, `/rewrite`, and `/match` endpoints.
- **Separation of concerns:** `parser.py` knows nothing about NLP or LLMs;
  `nlp_extractor.py` knows nothing about PDFs or Claude; `agents.py` knows
  nothing about PDFs or spaCy internals — each module is independently
  testable and swappable.

### Suggested next steps for a real deployment

- Add authentication (API key or OAuth) in front of the FastAPI service.
- Persist results (Postgres/S3) instead of in-memory Streamlit session state.
- Add structured logging + request tracing (e.g. OpenTelemetry).
- Containerize with Docker Compose (one service per backend/frontend).
- Add automated tests (`pytest` + `httpx.AsyncClient` for the API,
  `pytest` fixtures with sample PDFs for the parser).
- Rate-limit `/evaluate`, `/rewrite`, `/match` per user to control LLM spend.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `OSError: [E050] Can't find model 'en_core_web_sm'` | Run `python -m spacy download en_core_web_sm` |
| Sidebar shows "Cannot reach backend" | Make sure `uvicorn` is running on the port in `BACKEND_URL` |
| `502` on `/evaluate` or `/rewrite` | Check `ANTHROPIC_API_KEY` is set and valid in `.env` |
| `422` on `/parse` | The PDF is likely scanned/image-only with no embedded text (no OCR) |
| CORS errors in browser console | Add the frontend's origin to `ALLOWED_ORIGINS` in `.env` |
