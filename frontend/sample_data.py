"""
sample_data.py
--------------
Pre-packaged sample Job Description and Resume for instant testing,
demos, and verification of the AI Resume Matcher Agent.
"""

from __future__ import annotations

import os

SAMPLE_JD_TEXT = """Job Title: Senior Full-Stack Python & AI Engineer
Company: CloudScale AI Solutions
Location: Remote / Hybrid
Job Type: Full-Time

About the Role:
CloudScale AI is seeking a talented Senior Full-Stack Python & AI Engineer to build scalable, AI-powered web applications and high-throughput microservices. In this role, you will work cross-functionally with product and machine learning teams to design robust backend architectures, integrate Large Language Models (LLMs), and create responsive web interfaces.

Key Responsibilities:
- Design, build, and deploy production-grade REST APIs and microservices using Python, FastAPI, and PostgreSQL.
- Architect and integrate Generative AI capabilities, including LLM orchestration, RAG (Retrieval Augmented Generation), and vector database integrations.
- Develop modern, responsive frontend interfaces using React, TypeScript, and Tailwind CSS.
- Optimize database queries, indexing, and caching layers with Redis and PostgreSQL to maintain sub-100ms response times.
- Containerize services with Docker and manage deployment pipelines using CI/CD and AWS (ECS, Lambda, S3, RDS).
- Participate in code reviews, technical architecture sessions, and mentor junior engineers in an Agile/Scrum environment.

Requirements & Qualifications:
- 4+ years of professional software engineering experience with Python and web frameworks (FastAPI or Django).
- Strong proficiency in SQL database design and query optimization (PostgreSQL).
- Experience building single-page web applications with React, TypeScript, and modern state management.
- Hands-on experience with containerization using Docker and orchestrating deployments via CI/CD pipelines (GitHub Actions or GitLab CI).
- Cloud platform experience with AWS (ECS, EC2, S3, RDS, CloudWatch).
- Familiarity with caching systems such as Redis.
- Solid understanding of Git, version control workflows, and Agile development practices.

Preferred / Nice-to-Have Skills:
- Experience with Generative AI, Prompt Engineering, OpenAI/Anthropic APIs, and RAG architectures.
- Experience with Kubernetes or Terraform for Infrastructure as Code.
- Familiarity with message brokers such as Kafka or RabbitMQ.
- Bachelor's or Master's degree in Computer Science or a related technical discipline.
"""

SAMPLE_RESUME_TEXT = """ALEX MORGAN
alex.morgan@email.com | (555) 019-2834 | San Francisco, CA | github.com/alexmorgan | linkedin.com/in/alexmorgan

PROFESSIONAL SUMMARY
Senior Software Engineer with 5 years of experience building high-performance backend systems, distributed microservices, and modern web applications. Proven track record in Python, FastAPI, PostgreSQL, Docker, and React. Experienced in leading agile teams, architecting cloud solutions, and improving system throughput and database latency.

CORE COMPETENCIES & TECHNICAL SKILLS
- Languages: Python, JavaScript, TypeScript, SQL, Bash
- Backend Frameworks: FastAPI, Django, Flask, REST APIs, Microservices
- Frontend: React, Redux, HTML5, CSS3, Tailwind CSS
- Databases & Caching: PostgreSQL, MySQL, Redis, MongoDB
- DevOps & Cloud: Docker, Git, GitHub Actions, CI/CD, GCP (Google Cloud Platform), Linux
- Methodologies: Agile, Scrum, Jira, Code Reviews, Unit Testing, System Design

PROFESSIONAL EXPERIENCE

Senior Backend Engineer | DataPulse Technologies
San Francisco, CA | July 2022 – Present
- Architected and deployed 8+ RESTful microservices using Python and FastAPI, handling over 15 million daily API requests with 99.95% uptime.
- Redesigned relational schema and optimized complex SQL queries in PostgreSQL, reducing average query response times by 42%.
- Integrated Redis caching layer for session management and hot data queries, cutting server load by 35%.
- Built automated CI/CD deployment pipelines using GitHub Actions and Docker, reducing deployment cycle times from 45 minutes to 8 minutes.
- Mentored a squad of 4 junior and mid-level developers through daily code reviews, pair programming, and architectural design workshops.

Full-Stack Developer | Nexus Labs
San Jose, CA | August 2020 – June 2022
- Developed customer-facing web applications using React, TypeScript, and Python (Django), supporting 50,000+ monthly active users.
- Designed and maintained relational database models with PostgreSQL, ensuring data integrity and GDPR compliance.
- Implemented real-time dashboard analytics with asynchronous Celery workers and Redis task queues.
- Containerized development and staging environments using Docker, cutting developer onboarding time by 50%.
- Collaborated in a 7-person Agile/Scrum team participating in 2-week sprints, sprint retrospectives, and Jira tracking.

Software Engineer Intern | Vertex Systems
Oakland, CA | June 2019 – May 2020
- Built internal monitoring dashboards using Python (Flask) and SQLite to track API health metrics.
- Wrote automated unit and integration tests with PyTest, increasing test coverage from 60% to 85%.

PROJECTS

AI Document Assistant (Side Project)
- Built an exploratory AI document search prototype using Python, LangChain, and OpenAI API with Chroma vector store.
- Created an interactive UI in React to upload and query technical documentation.

EDUCATION
Bachelor of Science in Computer Science
University of California, Davis | 2016 – 2020
"""

def get_sample_docx_bytes() -> bytes:
    """Generate Word (.docx) bytes for the sample resume."""
    try:
        from docx_export import generate_docx_bytes
        return generate_docx_bytes(SAMPLE_RESUME_TEXT)
    except Exception:
        return b""


def ensure_samples_dir(target_dir: str | None = None) -> str:
    """Ensure sample files exist on disk in the samples/ folder."""
    if target_dir is None:
        target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
    os.makedirs(target_dir, exist_ok=True)

    jd_path = os.path.join(target_dir, "sample_job_description.txt")
    if not os.path.exists(jd_path):
        with open(jd_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_JD_TEXT)

    txt_path = os.path.join(target_dir, "sample_resume.txt")
    if not os.path.exists(txt_path):
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_RESUME_TEXT)

    docx_path = os.path.join(target_dir, "sample_resume.docx")
    if not os.path.exists(docx_path):
        docx_bytes = get_sample_docx_bytes()
        if docx_bytes:
            with open(docx_path, "wb") as f:
                f.write(docx_bytes)

    return target_dir
