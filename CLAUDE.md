# Jaqpot RAG Agent - Context for Claude

## Project Overview
This is a Retrieval-Augmented Generation (RAG) system specifically designed for querying Jaqpot documentation. The goal is to create an intelligent assistant that can answer questions about Jaqpot's APIs, features, and usage patterns using the official documentation as a knowledge base.

## What is Jaqpot?
Jaqpot is a platform for deploying and managing machine learning models. Key features:
- Model upload, storage, and management
- Prediction services via Python client (jaqpotpy) and REST API
- Supports various ML frameworks and deployment patterns
- Documentation available at https://jaqpot.org/docs

## Tech Stack for RAG System
- **Vector Database**: Pinecone Local (Docker container for development)
- **LLM**: Ollama with Llama 3.1 (local inference)
- **Backend**: FastAPI with LangChain
- **Frontend**: Streamlit for simple Q&A interface
- **Orchestration**: Docker Compose
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)

## Project Structure
```
jaqpot-rag-agent/
├── docker-compose.yml          # Pinecone Local + Ollama + services
├── backend/
│   ├── app.py                  # FastAPI server
│   ├── rag_pipeline.py         # RAG logic with LangChain
│   ├── document_processor.py   # Jaqpot docs scraping & chunking
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py        # Simple Q&A interface
│   └── requirements.txt
├── data/
│   └── jaqpot_docs/           # Scraped documentation
└── README.md
```

## Key Implementation Details

### Document Processing
- Scrape https://jaqpot.org/docs and subpages
- Chunk documents into semantic segments
- Generate embeddings using sentence-transformers
- Store in Pinecone Local with metadata (page title, URL, section)

### RAG Pipeline
1. User asks question about Jaqpot
2. Query embedding generated
3. Semantic search in Pinecone Local
4. Retrieve top-k relevant document chunks
5. Context + question sent to Ollama Llama 3.1
6. Generate contextual answer

### Example Queries
- "How do I upload a model to Jaqpot?"
- "What is jaqpotpy and how do I install it?"
- "Show me the API endpoints for making predictions"
- "How do I use Docker with Jaqpot models?"

## Development Notes
- Use Pinecone Local for development (no API costs)
- Ollama runs locally on Mac with Llama 3.1 8B model
- Target deployment: Docker Compose for easy local setup
- Focus on Jaqpot-specific terminology and context

## Success Criteria
- Accurate answers to common Jaqpot questions
- Fast response times (<5 seconds)
- Proper citation of source documentation
- Easy setup with single docker-compose command

## Next Steps
1. Set up Docker Compose with Pinecone Local + Ollama
2. Implement documentation scraper for jaqpot.org/docs
3. Build RAG pipeline with LangChain
4. Create simple Streamlit interface
5. Test with various Jaqpot-related queries
6. Deploy and document for easy reproduction