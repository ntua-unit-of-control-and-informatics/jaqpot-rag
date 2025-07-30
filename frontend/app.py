import streamlit as st
import requests
import time
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Jaqpot RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1e3a8a;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e0f2fe;
        border-left: 4px solid #0277bd;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #7b1fa2;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 3px solid #28a745;
    }
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

def check_backend_health():
    """Check if backend is available"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, str(e)

def query_backend(question: str, max_results: int = 5):
    """Send query to backend"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": question, "max_results": max_results},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying backend: {e}")
        return None

def get_backend_stats():
    """Get backend statistics"""
    try:
        response = requests.get(f"{BACKEND_URL}/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    # Header
    st.markdown("<h1 class='main-header'>🤖 Jaqpot RAG Agent</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Ask questions about Jaqpot ML platform using AI-powered documentation search</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 System Status")
    
    # Check backend health
    health_ok, health_data = check_backend_health()
    
    if health_ok:
        st.sidebar.success("✅ Backend Connected")
        if health_data:
            st.sidebar.json(health_data)
    else:
        st.sidebar.error("❌ Backend Unavailable")
        st.sidebar.markdown(f"**Error:** {health_data}")
        st.markdown("<div class='error-message'>⚠️ Backend service is not available. Please ensure the FastAPI server is running on port 8000.</div>", unsafe_allow_html=True)
        return
    
    # Get and display statistics
    with st.sidebar.expander("📈 Knowledge Base Stats"):
        stats = get_backend_stats()
        if "error" not in stats:
            st.metric("Total Documents", stats.get("total_documents", 0))
            st.metric("Index Fullness", f"{stats.get('index_fullness', 0):.1%}")
            st.text(f"Model: {stats.get('llm_model', 'Unknown')}")
            st.text(f"Embeddings: {stats.get('embedding_model', 'Unknown')}")
        else:
            st.error(f"Stats error: {stats['error']}")
    
    # Sample questions
    st.sidebar.title("💡 Example Questions")
    sample_questions = [
        "How do I install jaqpotpy?",
        "What is the Jaqpot API?",
        "How do I upload a model to Jaqpot?",
        "Show me how to make predictions",
        "What machine learning frameworks does Jaqpot support?",
        "How do I use Docker with Jaqpot?"
    ]
    
    for question in sample_questions:
        if st.sidebar.button(question, key=f"sample_{hash(question)}"):
            st.session_state.sample_question = question
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Main chat interface
    st.markdown("---")
    
    # Query settings
    col1, col2 = st.columns([3, 1])
    with col2:
        max_results = st.selectbox("Max Results", [3, 5, 7, 10], index=1)
        show_sources = st.checkbox("Show Sources", value=True)
        show_chunks = st.checkbox("Show Retrieved Chunks", value=False)
    
    # Chat input
    with col1:
        # Check for sample question
        default_question = ""
        if hasattr(st.session_state, 'sample_question'):
            default_question = st.session_state.sample_question
            del st.session_state.sample_question
        
        question = st.text_input(
            "Ask a question about Jaqpot:",
            value=default_question,
            placeholder="e.g., How do I deploy a model using jaqpotpy?",
            key="question_input"
        )
    
    # Submit button
    if st.button("🔍 Ask Question", type="primary") or question:
        if question.strip():
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Show thinking message
            with st.spinner("🤔 Searching through Jaqpot documentation..."):
                result = query_backend(question, max_results)
            
            if result:
                # Add assistant response to chat
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "chunks": result.get("retrieved_chunks", [])
                })
    
    # Display chat history
    if st.session_state.messages:
        st.markdown("## 💬 Conversation")
        
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🙋‍♂️ You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            
            else:  # assistant
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>{message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources if available and enabled
                if show_sources and "sources" in message and message["sources"]:
                    st.markdown("**📚 Sources:**")
                    for j, source in enumerate(message["sources"]):
                        score_color = "green" if source["score"] > 0.8 else "orange" if source["score"] > 0.6 else "red"
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>{source["title"]}</strong><br>
                            <a href="{source["url"]}" target="_blank">{source["url"]}</a><br>
                            <small>Section: {source.get("section", "N/A")} | 
                            Relevance: <span style="color: {score_color};">{source["score"]:.2f}</span></small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show retrieved chunks if enabled
                if show_chunks and "chunks" in message and message["chunks"]:
                    with st.expander(f"📄 Retrieved Chunks ({len(message['chunks'])})"):
                        for j, chunk in enumerate(message["chunks"]):
                            st.markdown(f"**Chunk {j+1}** (Score: {chunk['score']:.2f})")
                            st.markdown(f"**Title:** {chunk['title']}")
                            st.text_area(
                                f"Content {j+1}:", 
                                chunk["text"], 
                                height=100, 
                                key=f"chunk_{i}_{j}"
                            )
                            st.markdown("---")
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🤖 Powered by Ollama + Pinecone Local + Sentence Transformers<br>
        Built for Jaqpot ML Platform Documentation
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()