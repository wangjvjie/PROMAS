#!/bin/bash
set -e

echo "🛡️  SecDev Agent — Starting..."

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo "   Set it with: export OPENAI_API_KEY='sk-...'"
    echo ""
fi

# Install deps if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r backend/requirements.txt
fi

# Run
echo ""
echo "🚀 Starting server at http://localhost:8000"
echo ""
python3 -m backend.main
