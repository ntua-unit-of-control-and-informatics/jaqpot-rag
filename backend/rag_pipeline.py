import os
import json
import logging
from typing import List, Dict, Any
import asyncio
from dotenv import load_dotenv

import pinecone
from sentence_transformers import SentenceTransformer
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import httpx

from document_processor import JaqpotDocsScraper, chunk_documents

load_dotenv()
logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG Pipeline for Jaqpot documentation"""
    
    def __init__(self):
        self.embedding_model = None
        self.pinecone_index = None
        self.llm = None
        self.qa_chain = None
        self.documents_loaded = False
        
        # Configuration
        self.index_name = "jaqpot-docs"
        self.embedding_dimension = 384  # all-MiniLM-L6-v2 dimension
        self.data_dir = "../data/jaqpot_docs"
        
        # Pinecone configuration
        self.pinecone_host = os.getenv("PINECONE_HOST", "localhost:5080")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "dummy-key")
        
        # Ollama configuration
        self.ollama_host = os.getenv("OLLAMA_HOST", "localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    
    async def initialize(self):
        """Initialize all components of the RAG pipeline"""
        logger.info("Initializing RAG pipeline...")
        
        # Initialize embedding model
        await self._init_embeddings()
        
        # Initialize Pinecone
        await self._init_pinecone()
        
        # Initialize LLM
        await self._init_llm()
        
        # Load documents if not already loaded
        await self._ensure_documents_loaded()
        
        logger.info("RAG pipeline initialization complete")
    
    async def _init_embeddings(self):
        """Initialize sentence transformer model"""
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded")
    
    async def _init_pinecone(self):
        """Initialize Pinecone client and index"""
        logger.info("Initializing Pinecone...")
        
        try:
            # Configure Pinecone for local instance
            pinecone.init(
                api_key=self.pinecone_api_key,
                environment="local",
                host=f"http://{self.pinecone_host}"
            )
            
            # Check if index exists, create if not
            existing_indexes = pinecone.list_indexes()
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine"
                )
            
            self.pinecone_index = pinecone.Index(self.index_name)
            logger.info("Pinecone initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise e
    
    async def _init_llm(self):
        """Initialize Ollama LLM"""
        logger.info("Initializing Ollama LLM...")
        
        try:
            # Test Ollama connection
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{self.ollama_host}/api/version")
                response.raise_for_status()
                logger.info("Ollama connection successful")
            
            # Initialize LLM
            self.llm = Ollama(
                model=self.ollama_model,
                base_url=f"http://{self.ollama_host}"
            )
            
            # Create QA chain
            qa_prompt = PromptTemplate(
                template="""You are a helpful AI assistant that answers questions about Jaqpot, a machine learning platform. 
Use the following context from Jaqpot documentation to answer the user's question.

Context:
{context}

Question: {question}

Instructions:
- Provide accurate, helpful answers based on the context
- If the context doesn't contain enough information, say so clearly
- Include relevant code examples when available
- Mention the source sections when possible
- Be concise but comprehensive

Answer:""",
                input_variables=["context", "question"]
            )
            
            self.qa_chain = LLMChain(llm=self.llm, prompt=qa_prompt)
            logger.info("LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise e
    
    async def _ensure_documents_loaded(self):
        """Ensure documents are scraped and indexed"""
        chunks_file = os.path.join(self.data_dir, "document_chunks.json")
        
        if not os.path.exists(chunks_file):
            logger.info("No document chunks found, scraping Jaqpot docs...")
            await self._scrape_and_index_documents()
        else:
            # Check if index has documents
            stats = self.pinecone_index.describe_index_stats()
            if stats['total_vector_count'] == 0:
                logger.info("Index is empty, loading documents...")
                await self._load_and_index_documents()
            else:
                logger.info(f"Found {stats['total_vector_count']} documents in index")
                self.documents_loaded = True
    
    async def _scrape_and_index_documents(self):
        """Scrape documentation and create index"""
        logger.info("Scraping Jaqpot documentation...")
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Scrape documents
        scraper = JaqpotDocsScraper(data_dir=self.data_dir)
        documents = scraper.scrape_all(max_pages=30)
        scraper.save_documents(documents)
        
        # Create chunks
        chunks = chunk_documents(documents)
        
        # Save chunks
        chunks_file = os.path.join(self.data_dir, "document_chunks.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        # Index chunks
        await self._index_chunks(chunks)
        self.documents_loaded = True
        
        logger.info(f"Scraped and indexed {len(chunks)} document chunks")
    
    async def _load_and_index_documents(self):
        """Load existing documents and create index"""
        chunks_file = os.path.join(self.data_dir, "document_chunks.json")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        await self._index_chunks(chunks)
        self.documents_loaded = True
        
        logger.info(f"Loaded and indexed {len(chunks)} document chunks")
    
    async def _index_chunks(self, chunks: List[Dict]):
        """Index document chunks in Pinecone"""
        logger.info(f"Indexing {len(chunks)} chunks...")
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Generate embeddings for batch
            texts = [chunk["text"] for chunk in batch]
            embeddings = self.embedding_model.encode(texts)
            
            # Prepare vectors for upload
            vectors = []
            for j, chunk in enumerate(batch):
                vectors.append({
                    "id": f"chunk_{chunk['chunk_id']}",
                    "values": embeddings[j].tolist(),
                    "metadata": {
                        "text": chunk["text"][:1000],  # Truncate for metadata
                        "title": chunk["title"],
                        "url": chunk["url"],
                        "section": chunk["section"],
                        **chunk["metadata"]
                    }
                })
            
            # Upload to Pinecone
            self.pinecone_index.upsert(vectors=vectors)
            
            if (i + batch_size) % 500 == 0:
                logger.info(f"Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
        
        logger.info("Indexing complete")
    
    async def query(self, question: str, max_results: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.documents_loaded:
            raise Exception("Documents not loaded")
        
        logger.info(f"Processing query: {question}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([question])[0]
        
        # Search in Pinecone
        search_results = self.pinecone_index.query(
            vector=query_embedding.tolist(),
            top_k=max_results,
            include_metadata=True
        )
        
        # Extract context and sources
        contexts = []
        sources = []
        retrieved_chunks = []
        
        for match in search_results['matches']:
            metadata = match['metadata']
            contexts.append(metadata['text'])
            
            sources.append({
                "title": metadata['title'],
                "url": metadata['url'],
                "section": metadata.get('section', ''),
                "score": match['score']
            })
            
            retrieved_chunks.append({
                "text": metadata['text'],
                "title": metadata['title'],
                "score": match['score'],
                "metadata": metadata
            })
        
        # Generate answer using LLM
        context = "\n\n".join(contexts)
        
        try:
            answer = await self.qa_chain.arun(
                context=context,
                question=question
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = "Sorry, I encountered an error generating the answer. Please try again."
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        if not self.pinecone_index:
            return {"error": "Index not initialized"}
        
        stats = self.pinecone_index.describe_index_stats()
        
        return {
            "total_documents": stats['total_vector_count'],
            "index_fullness": stats.get('index_fullness', 0),
            "documents_loaded": self.documents_loaded,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": self.ollama_model
        }