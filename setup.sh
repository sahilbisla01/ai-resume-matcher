#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh
# In a single command: creates venv, installs dependencies,
# downloads spaCy model, and prepares the .env file.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ---------------------------------------------------------------------------

set -e  # Exit immediately if any command fails

echo "🔧 Step 1/4: Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.10+ first: https://python.org"
    exit 1
fi
python3 --version

echo ""
echo "🔧 Step 2/4: Creating virtual environment (venv/)..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ venv created."
else
    echo "✅ venv already exists, skipping."
fi

# Activate venv (within this script)
source venv/bin/activate

echo ""
echo "🔧 Step 3/4: Installing dependencies (this will take 1-2 minutes)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
python -m spacy download en_core_web_sm --quiet

echo ""
echo "🔧 Step 4/4: Setting up .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ .env file created by copying from .env.example."
    echo "⚠️  IMPORTANT: Now open .env file and add your ANTHROPIC_API_KEY!"
    echo "    Get your key here: https://console.anthropic.com/settings/keys"
else
    echo "✅ .env file already exists, skipping."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next step: Add your API key to the .env file (if not added yet)."
echo "Then use these commands to start the servers:"
echo ""
echo "  Terminal 1 (backend):"
echo "    source venv/bin/activate"
echo "    cd backend && uvicorn app:app --reload --port 8000"
echo ""
echo "  Terminal 2 (frontend):"
echo "    source venv/bin/activate"
echo "    cd frontend && streamlit run streamlit_app.py"
echo ""
echo "Or simply run ./run.sh (starts both simultaneously)."
