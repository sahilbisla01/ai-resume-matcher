#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh
# Starts both Backend (FastAPI) and Frontend (Streamlit) together.
# Pressing Ctrl+C will shut down both.
#
# Usage:
#   chmod +x run.sh
#   ./run.sh
# ---------------------------------------------------------------------------

set -e

if [ ! -d "venv" ]; then
    echo "❌ venv not found. Please run ./setup.sh first."
    exit 1
fi

source venv/bin/activate

# When the script exits (Ctrl+C), shut down the backend process as well
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

echo "🚀 Starting backend (FastAPI) on http://localhost:8000 ..."
(cd backend && uvicorn app:app --host 0.0.0.0 --port 8000) &
BACKEND_PID=$!

# Give backend a moment to start up
sleep 3

echo "🚀 Starting frontend (Streamlit) on http://localhost:8501 ..."
cd frontend && streamlit run streamlit_app.py

# If Streamlit stops, shut down the backend as well
cleanup
