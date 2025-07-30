# Jaqpot RAG Agent 🤖

A Retrieval-Augmented Generation (RAG) system for querying Jaqpot documentation using local AI models.

## 🚀 Quick Start

1. **Start infrastructure services:**
   ```bash
   docker-compose up -d
   ```

2. **Install Python dependencies:**
   ```bash
   # Backend
   cd backend && pip install -r requirements.txt

   # Frontend  
   cd ../frontend && pip install -r requirements.txt
   ```

3. **Build the knowledge base:**
   ```bash
   cd backend && python ../scripts/build_index.py
   ```

4. **Start the services:**
   ```bash
   # Terminal 1: Backend
   cd backend && python app.py

   # Terminal 2: Frontend
   cd frontend && streamlit run app.py
   ```

5. **Access the interface:**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │───▶│   FastAPI       │───▶│  Pinecone Local │
│   Frontend      │    │   Backend       │    │  Vector Store   │
│   (Port 8501)   │    │   (Port 8000)   │    │  (Port 5080)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │     Ollama      │
                       │   Llama 3.1     │
                       │  (Port 11434)   │
                       └─────────────────┘
```

## 🛠️ Components

### Infrastructure (Docker Compose)
- **Pinecone Local**: Vector database for document embeddings
- **Ollama**: Local LLM inference (Llama 3.1 8B)

### Backend (FastAPI)
- **RAG Pipeline**: LangChain-based retrieval and generation
- **Document Processor**: Web scraper for jaqpot.org/docs
- **Vector Store**: Sentence transformer embeddings + Pinecone

### Frontend (Streamlit)
- **Chat Interface**: Question/answer with sources
- **System Status**: Backend health and knowledge base stats
- **Example Questions**: Pre-built Jaqpot queries

## 📚 How It Works

1. **Document Ingestion**: Scrapes https://jaqpot.org/docs
2. **Chunking**: Splits documents into semantic segments
3. **Embedding**: Generates vectors using all-MiniLM-L6-v2
4. **Indexing**: Stores embeddings in Pinecone Local
5. **Query Processing**: 
   - User asks question
   - Query embedding generated
   - Semantic search in vector store
   - Top-k relevant chunks retrieved
   - Context + question sent to Llama 3.1
   - AI-generated response with sources

## 🔧 Development

### Project Structure
```
jaqpot-rag/
├── docker-compose.yml         # Infrastructure services
├── .env                       # Environment variables
├── backend/
│   ├── app.py                # FastAPI server
│   ├── rag_pipeline.py       # RAG logic with LangChain  
│   ├── document_processor.py # Documentation scraper
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Streamlit interface
│   └── requirements.txt
├── scripts/
│   ├── build_index.py       # Build knowledge base
│   └── run_scraper.py       # Standalone scraper
└── data/
    └── jaqpot_docs/         # Scraped documentation
```

### Running Services Individually

**Infrastructure:**
```bash
docker-compose up pinecone-local ollama
```

**Backend only:**
```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend only:**
```bash
cd frontend  
streamlit run app.py --server.port 8501
```

### Rebuilding Knowledge Base
```bash
# Scrape fresh documentation
python scripts/run_scraper.py

# Rebuild index
python scripts/build_index.py
```

## 🔍 Example Queries

- "How do I install jaqpotpy?"
- "What is the Jaqpot API?"
- "Show me how to upload a model"
- "How do I make predictions with jaqpotpy?"
- "What machine learning frameworks does Jaqpot support?"
- "How do I use Docker with Jaqpot models?"

## 📊 Configuration

### Environment Variables (.env)
```bash
# Pinecone Local
PINECONE_API_KEY=dummy-key
PINECONE_ENVIRONMENT=local
PINECONE_HOST=localhost:5080

# Ollama
OLLAMA_HOST=localhost:11434
OLLAMA_MODEL=llama3.1:8b

# API Settings
BACKEND_PORT=8000
FRONTEND_PORT=8501
```

### Customization
- **Chunk size**: Modify `chunk_size` in `document_processor.py`
- **Embedding model**: Change model in `rag_pipeline.py`
- **LLM model**: Update `OLLAMA_MODEL` in `.env`
- **Scraping scope**: Adjust `max_pages` in scraper calls

## 🔧 Troubleshooting

### Common Issues

**Backend not starting:**
- Check if Pinecone Local and Ollama are running
- Verify environment variables in `.env`
- Look at logs: `docker-compose logs`

**Empty search results:**
- Rebuild index: `python scripts/build_index.py`
- Check scraping logs for errors
- Verify Pinecone Local has data

**Slow responses:**
- Increase Docker memory allocation
- Use smaller LLM model (llama3.1:7b)
- Reduce `max_results` parameter

**Ollama model download fails:**
- Ensure 8GB+ free disk space
- Check internet connection
- Try: `docker-compose exec ollama ollama pull llama3.1:8b`

### Health Checks
```bash
# Backend health
curl http://localhost:8000/health

# Pinecone Local
curl http://localhost:5080/health

# Ollama
curl http://localhost:11434/api/version

# Knowledge base stats
curl http://localhost:8000/stats
```

## 🎯 Performance

- **Response time**: ~3-5 seconds per query
- **Memory usage**: ~4-6GB (Ollama + embeddings)
- **Storage**: ~1-2GB (models + index)
- **Throughput**: ~10-20 queries/minute

## 🚀 Next Steps

- [ ] Add conversation memory
- [ ] Support for code examples
- [ ] Integration with Jaqpot API for live data
- [ ] Multi-language support
- [ ] Advanced filtering by documentation sections
- [ ] Automated documentation updates

## 📄 License

MIT License - See LICENSE file for details.

---

Built with ❤️ for the Jaqpot ML Platform community