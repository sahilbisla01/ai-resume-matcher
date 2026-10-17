@echo off
echo ========================================================
echo Starting AI Resume Matcher (Backend + Frontend)
echo ========================================================

REM Activate virtual environment if present
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo Generating sample resume docx if not present...
python create_sample_docx.py

echo Starting FastAPI Backend on http://localhost:8000...
start "AI Resume Matcher - Backend" cmd /k "cd backend && uvicorn app:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 >nul

echo Starting Streamlit Frontend on http://localhost:8501...
streamlit run frontend\streamlit_app.py

pause
