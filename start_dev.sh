#!/bin/bash

# Jaqpot RAG Development Startup Script

echo "🚀 Starting Jaqpot RAG Development Environment"
echo "============================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please create .env file with your configuration:"
    echo "   cp .env.example .env"
    echo "   Edit .env and add your PINECONE_API_KEY"
    exit 1
fi

# Check if PINECONE_API_KEY is set
if ! grep -q "PINECONE_API_KEY=" .env || grep -q "PINECONE_API_KEY=your_pinecone_api_key_here" .env; then
    echo "❌ PINECONE_API_KEY not configured in .env file"
    echo "   Please edit .env and add your Pinecone API key"
    exit 1
fi

# Check if Ollama is running locally
echo "🦙 Checking Ollama status..."
if ! curl -s http://localhost:11434/api/version > /dev/null; then
    echo "⚠️  Ollama not running locally. You can:"
    echo "   1. Start Ollama locally: ollama serve"
    echo "   2. Or use Docker: docker run -p 11434:11434 ollama/ollama"
    echo "   3. Or continue without Ollama (some features may not work)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Ollama is ready"
    
    # Check if required model is available
    if ! ollama list | grep -q "llama3.1:8b"; then
        echo "📥 Downloading llama3.1:8b model..."
        ollama pull llama3.1:8b
    fi
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."

# Upgrade pip first
pip install --upgrade pip

cd backend
echo "   Installing backend dependencies..."
pip install --upgrade huggingface_hub==0.20.3
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install backend dependencies"
    exit 1
fi

cd ../frontend
echo "   Installing frontend dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install frontend dependencies"
    exit 1
fi

cd ..

# Build knowledge base
echo "📚 Building knowledge base..."
echo "   - Scraping Jaqpot documentation"
echo "   - Creating embeddings"
echo "   - Uploading to Pinecone"

cd backend
python document_processor.py

if [ $? -eq 0 ]; then
    echo "✅ Documentation scraped successfully"
    
    # Initialize RAG pipeline and upload to Pinecone
    echo "🔄 Initializing RAG pipeline and uploading to Pinecone..."
    python -c "
import asyncio
import sys
from rag_pipeline import RAGPipeline

async def main():
    try:
        rag = RAGPipeline()
        await rag.initialize()
        stats = await rag.get_stats()
        print(f'✅ RAG Pipeline initialized successfully!')
        print(f'   Total documents in index: {stats[\"total_documents\"]}')
        print(f'   Index fullness: {stats[\"index_fullness\"]:.2%}')
        return 0
    except Exception as e:
        print(f'❌ Failed to initialize RAG pipeline: {e}')
        return 1

sys.exit(asyncio.run(main()))
"
    
    if [ $? -eq 0 ]; then
        echo "✅ Knowledge base built successfully"
    else
        echo "❌ Failed to build knowledge base"
        exit 1
    fi
else
    echo "❌ Failed to scrape documentation"
    exit 1
fi

cd ..

echo ""
echo "🎉 Setup complete! You can now start the services:"
echo ""
echo "🖥️  Start Backend API:"
echo "   cd backend && python app.py"
echo ""  
echo "🌐 Start Frontend (in another terminal):"
echo "   cd frontend && streamlit run app.py"
echo ""
echo "📊 Access Points:"
echo "   Frontend: http://localhost:8501"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "💡 Tip: Use 'tmux' or separate terminal windows to run both services"
echo "📚 Check README.md for more detailed instructions"
