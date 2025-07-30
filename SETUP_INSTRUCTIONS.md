# Setup Instructions for Jaqpot RAG Agent

## Quick Start
1. Move this folder to `~/Projects/ntua/jaqpot-rag-agent`
2. Run Claude from the new directory
3. Execute: `docker-compose up -d`
4. Access the interface at http://localhost:8501

## Detailed Setup

### Prerequisites
- Docker Desktop installed and running
- At least 8GB RAM available for Ollama
- Internet connection for initial setup

### Directory Structure (After Move)
```
~/Projects/ntua/jaqpot-rag-agent/
├── CLAUDE.md                  # This context file
├── PROJECT_PLAN.md           # Implementation roadmap
├── SETUP_INSTRUCTIONS.md     # This file
├── docker-compose.yml        # Services orchestration
├── backend/                  # FastAPI server
├── frontend/                 # Streamlit UI
├── scripts/                  # Data processing
└── data/                     # Documentation storage
```

### Environment Variables
Create `.env` file with:
```bash
# Pinecone Local
PINECONE_API_KEY=dummy-key
PINECONE_ENVIRONMENT=local
PINECONE_HOST=pinecone-local:5080

# Ollama
OLLAMA_HOST=ollama:11434
OLLAMA_MODEL=llama3.1:8b

# API Settings
BACKEND_PORT=8000
FRONTEND_PORT=8501
```

### First Run Commands
```bash
# 1. Start services
docker-compose up -d

# 2. Wait for Ollama to download model (first time ~5GB)
docker-compose logs -f ollama

# 3. Initialize Jaqpot documentation index
docker-compose exec backend python scripts/build_index.py

# 4. Test the system
curl http://localhost:8000/health
```

### Development Workflow
1. **Backend changes**: Auto-reload enabled in FastAPI
2. **Frontend changes**: Streamlit auto-reloads on file changes
3. **Documentation updates**: Re-run `build_index.py`
4. **Model changes**: Update `OLLAMA_MODEL` in `.env`

### Troubleshooting
- **Ollama model download fails**: Check available disk space (need 5GB+)
- **Pinecone connection errors**: Verify container is running
- **Slow responses**: Increase RAM allocation to Docker
- **Empty search results**: Re-run documentation indexing

### Next Steps After Setup
1. Test with sample queries about Jaqpot
2. Review retrieved document chunks for relevance
3. Adjust chunk size and embedding parameters
4. Add more documentation sources if needed

## Moving the Project
```bash
# From current location (/Users/alex.arvanitidis/Projects/rag)
cd ~/Projects
mkdir -p ntua
mv rag ntua/jaqpot-rag-agent
cd ntua/jaqpot-rag-agent

# Then restart Claude from this directory
```

This setup provides a complete local RAG system for Jaqpot documentation with no external API dependencies.