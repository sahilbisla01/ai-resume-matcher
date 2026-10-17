<div align="center">

# 🎯 ResumeAI: Autonomous ATS Matcher & Resume Optimizer

**Scan your resume against any Job Description, uncover skill gaps, rewrite bullets with Google's X-Y-Z formula, and export an ATS 90+ Word document in 1-click.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sahilbisla01-ai-resume-matcher-frontendstreamlit-app-7kcmcr.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Groq LPU](https://img.shields.io/badge/Powered%20By-Groq%20LPU%20(Llama%203.1)-F05A28?logo=groq&logoColor=white)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Author: Sahil Bisla](https://img.shields.io/badge/Author-Sahil%20Bisla-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sahilbisla)

### 🚀 [Click Here for Live Demo](https://sahilbisla01-ai-resume-matcher-frontendstreamlit-app-7kcmcr.streamlit.app/)

</div>

---

## 🌟 Overview

Landing technical interviews requires passing through Applicant Tracking Systems (ATS) and demanding human recruiters. **ResumeAI** is an autonomous multi-agent copilot that evaluates resumes with the precision of automated ATS filters, identifies critical hard/soft skill voids, elevates weak bullet points using **Google's X-Y-Z formula**, drafts tailored cover letters, and generates a brand-new **ATS 90+ optimized resume in Word (`.docx`)** ready for application.

---

## ✨ Key Features

- **📊 Comprehensive ATS Compatibility Score (0-100):** Real-time grounding score combining deterministic token overlap, spaCy skill extraction, and deep semantic evaluation.
- **🔍 Gap & Red-Flag Analysis:** Automatically isolates critical missing technical competencies, absent soft skills, and ATS-breaking formatting red flags.
- **✍️ Google X-Y-Z Bullet Point Tailoring:** Rewrites vague resume bullets into high-impact metrics: *"Accomplished [X], as measured by [Y], by doing [Z]"*.
- **✉️ Role-Specific Cover Letter Generator:** Drafts compelling, tailored cover letters highlighting your exact relevant achievements for the target role.
- **🚀 1-Click Full Resume Upgrader (`.docx` & `.txt`):** Generates a complete, re-architected resume preserving your genuine history while infusing missing keywords, available for instant Word download.
- **⚡ 100% Free & Blazing Fast:** Powered by **Groq Cloud LPUs** running open-source models (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`).
- **🎨 Glassmorphism Dark Mode UI:** Modern interface styled with Google Fonts (*Outfit* and *Plus Jakarta Sans*) with responsive, intuitive interactions.

---

## 🧠 System Architecture & Multi-Agent Flow

```
                               ┌───────────────────────────┐
                               │  Uploaded Resume (PDF/DOCX)│
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     Document Parser       │
                               │  (pdfplumber / python-docx)│
                               └─────────────┬─────────────┘
                                             │ Cleaned Text
                                             ▼
                               ┌───────────────────────────┐
                               │   Deterministic NLP Engine │
                               │(spaCy NER + Regex Baseline)│
                               └─────────────┬─────────────┘
                                             │ Grounding Signal
                                             ▼
                 ┌─────────────────────────────────────────────────────────┐
                 │                  GROQ LPU MULTI-AGENT PIPELINE          │
                 │                                                         │
                 │  ┌─────────────────┐           ┌─────────────────────┐  │
                 │  │     Agent 1     │           │       Agent 2       │  │
                 │  │  ATS Evaluator  │           │   X-Y-Z Tailorer    │  │
                 │  └────────┬────────┘           └──────────┬──────────┘  │
                 │           │                               │             │
                 │  ┌────────┴────────┐           ┌──────────┴──────────┐  │
                 │  │     Agent 3     │           │       Agent 4       │  │
                 │  │   Cover Letter  │           │Full Resume Upgrader │  │
                 │  └─────────────────┘           └──────────┬──────────┘  │
                 └───────────────────────────────────────────┼─────────────┘
                                                             │
                                                             ▼
                                                ┌─────────────────────────┐
                                                │ Pure Python DOCX Engine │
                                                │ (Instant Word Download) │
                                                └─────────────────────────┘
```

### Agent Roles:
1. **Agent 1 — ATS Evaluator:** Performs strict JSON gap analysis, produces realistic 0-100 score, missing skills list, and formatting red flags.
2. **Agent 2 — Bullet Point Tailor:** Pinpoints weak achievements and transforms them into Google X-Y-Z structured bullets with quantified impact.
3. **Agent 3 — Cover Letter Architect:** Connects 2-3 key resume accomplishments directly to the core requirements of the job description.
4. **Agent 4 — Full Resume Upgrader:** Re-synthesizes the full resume into an ATS 90+ structure with clean section hierarchy and exports directly to `.docx`.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/) with custom CSS (Glassmorphism, Outfit & Plus Jakarta Sans typography)
- **Backend:** [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/) with in-process cloud fallback
- **LLM Engine:** [Groq Cloud](https://groq.com/) LPUs (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`)
- **Document Parsing:** [pdfplumber](https://github.com/jsvine/pdfplumber), [pypdf](https://github.com/py-pdf/pypdf)
- **Document Export:** Custom Zero-Dependency XML `.docx` generator
- **NLP Baseline:** [spaCy](https://spacy.io/) with pure-Python regex fallback for instant compatibility

---

## 💻 Local Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/sahilbisla01/ai-resume-matcher.git
cd ai-resume-matcher
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a `.env` file in the project root:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_free_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```
*(You can get a 100% free API key from [Groq Console](https://console.groq.com/keys)).*

### 5. Launch the Application
```bash
streamlit run frontend/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser!

---

## ☁️ Deployment on Streamlit Community Cloud

This project is optimized for 1-click deployment on **Streamlit Community Cloud**:

1. Fork or push this repository to your GitHub account.
2. Head to [share.streamlit.io](https://share.streamlit.io) and click **"New app"**.
3. Select your repository: `ai-resume-matcher`
4. Set Main file path: `frontend/streamlit_app.py`
5. In **Advanced Settings**:
   - Select **Python version: 3.11**
   - Add your secrets in **Secrets**:
     ```toml
     GROQ_API_KEY = "your_free_groq_api_key"
     LLM_PROVIDER = "groq"
     GROQ_MODEL = "llama-3.1-8b-instant"
     ```
6. Click **Deploy!**

---

## 👨‍💻 Author

**Sahil Bisla**  
- **GitHub:** [@sahilbisla01](https://github.com/sahilbisla01)  
- **LinkedIn:** [linkedin.com/in/sahilbisla](https://www.linkedin.com/in/sahilbisla)  
- **Email:** [sahilbisla@sveltetechnologies.com](mailto:sahilbisla@sveltetechnologies.com)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Feel free to star ⭐ the repository if you found it useful!
