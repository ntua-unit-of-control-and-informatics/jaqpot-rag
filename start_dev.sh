#!/bin/bash

# Jaqpot RAG Development Startup Script

echo "🚀 Starting Jaqpot RAG Development Environment"
echo "============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Start infrastructure services
echo "📦 Starting infrastructure services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 10

# Check Ollama health
echo "🦙 Checking Ollama status..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/version > /dev/null; then
        echo "✅ Ollama is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Ollama failed to start"
        exit 1
    fi
    sleep 2
done

# Check Pinecone Local health  
echo "📊 Checking Pinecone Local status..."
for i in {1..15}; do
    if curl -s http://localhost:5080/health > /dev/null; then
        echo "✅ Pinecone Local is ready"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "❌ Pinecone Local failed to start"
        exit 1
    fi
    sleep 2
done

echo ""
echo "🎉 Infrastructure ready! Next steps:"
echo ""
echo "1. Install dependencies:"
echo "   cd backend && pip install -r requirements.txt"
echo "   cd ../frontend && pip install -r requirements.txt"
echo ""
echo "2. Build knowledge base:"
echo "   cd backend && python ../scripts/build_index.py"
echo ""
echo "3. Start services:"
echo "   Terminal 1: cd backend && python app.py"
echo "   Terminal 2: cd frontend && streamlit run app.py"
echo ""
echo "4. Access:"
echo "   Frontend: http://localhost:8501"
echo "   Backend API: http://localhost:8000"
echo ""
echo "📚 Check README.md for detailed instructions"