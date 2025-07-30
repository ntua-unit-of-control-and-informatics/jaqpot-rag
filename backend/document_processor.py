import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import json
import time
from typing import List, Dict, Set
import logging
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class Document:
    title: str
    content: str
    url: str
    section: str = ""
    metadata: Dict = None

class JaqpotDocsScraper:
    """Scraper for Jaqpot documentation"""
    
    def __init__(self, base_url: str = "https://jaqpot.org/docs", data_dir: str = "../data/jaqpot_docs"):
        self.base_url = base_url
        self.data_dir = data_dir
        self.visited_urls: Set[str] = set()
        self.documents: List[Document] = []
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Headers to avoid blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def is_valid_jaqpot_url(self, url: str) -> bool:
        """Check if URL is a valid Jaqpot docs URL"""
        parsed = urlparse(url)
        return (
            parsed.netloc == "jaqpot.org" and 
            parsed.path.startswith("/docs") and
            url not in self.visited_urls
        )
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> List[Document]:
        """Extract content from a documentation page"""
        documents = []
        
        # Get page title
        title_elem = soup.find('h1') or soup.find('title')
        page_title = title_elem.get_text().strip() if title_elem else "Untitled"
        
        # Remove navigation, header, footer elements
        for elem in soup.find_all(['nav', 'header', 'footer', 'aside']):
            elem.decompose()
        
        # Find main content area
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', class_=re.compile(r'content|main|docs')) or
            soup.find('body')
        )
        
        if not main_content:
            logger.warning(f"No main content found for {url}")
            return documents
        
        # Extract sections
        sections = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        if not sections:
            # If no sections, treat entire content as one document
            content = self.clean_text(main_content.get_text())
            if content.strip():
                documents.append(Document(
                    title=page_title,
                    content=content,
                    url=url,
                    section="main",
                    metadata={"page_title": page_title}
                ))
        else:
            # Extract content by sections
            for i, section in enumerate(sections):
                section_title = section.get_text().strip()
                
                # Get content until next section
                content_parts = []
                current = section.next_sibling
                
                while current and not (current.name and current.name.lower() in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    if hasattr(current, 'get_text'):
                        text = current.get_text().strip()
                        if text:
                            content_parts.append(text)
                    current = current.next_sibling
                
                content = self.clean_text(' '.join(content_parts))
                
                if content.strip():
                    documents.append(Document(
                        title=f"{page_title} - {section_title}",
                        content=content,
                        url=url,
                        section=section_title,
                        metadata={
                            "page_title": page_title,
                            "section_title": section_title,
                            "section_level": section.name
                        }
                    ))
        
        return documents
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep useful punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\&\*\+\=\<\>]', '', text)
        return text.strip()
    
    def get_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Extract all valid documentation links from a page"""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(current_url, href)
            
            if self.is_valid_jaqpot_url(absolute_url):
                links.append(absolute_url)
        
        return links
    
    def scrape_page(self, url: str) -> List[Document]:
        """Scrape a single page"""
        if url in self.visited_urls:
            return []
        
        logger.info(f"Scraping: {url}")
        self.visited_urls.add(url)
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            documents = self.extract_content(soup, url)
            
            # Extract new links for crawling
            new_links = self.get_links(soup, url)
            
            # Add delay to be respectful
            time.sleep(1)
            
            return documents, new_links
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return [], []
    
    def scrape_all(self, max_pages: int = 50) -> List[Document]:
        """Scrape all Jaqpot documentation"""
        logger.info("Starting Jaqpot documentation scraping...")
        
        urls_to_visit = [self.base_url]
        all_documents = []
        pages_scraped = 0
        
        while urls_to_visit and pages_scraped < max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
            
            documents, new_links = self.scrape_page(current_url)
            all_documents.extend(documents)
            
            # Add new links to queue
            for link in new_links:
                if link not in self.visited_urls and link not in urls_to_visit:
                    urls_to_visit.append(link)
            
            pages_scraped += 1
            
            if pages_scraped % 5 == 0:
                logger.info(f"Scraped {pages_scraped} pages, found {len(all_documents)} documents")
        
        logger.info(f"Scraping complete. Total documents: {len(all_documents)}")
        return all_documents
    
    def save_documents(self, documents: List[Document]):
        """Save documents to JSON file"""
        output_file = os.path.join(self.data_dir, "jaqpot_docs.json")
        
        docs_data = []
        for doc in documents:
            docs_data.append({
                "title": doc.title,
                "content": doc.content,
                "url": doc.url,
                "section": doc.section,
                "metadata": doc.metadata or {}
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(documents)} documents to {output_file}")
    
    def load_documents(self) -> List[Document]:
        """Load documents from JSON file"""
        input_file = os.path.join(self.data_dir, "jaqpot_docs.json")
        
        if not os.path.exists(input_file):
            logger.warning(f"No existing documents found at {input_file}")
            return []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        
        documents = []
        for doc_data in docs_data:
            documents.append(Document(
                title=doc_data["title"],
                content=doc_data["content"],
                url=doc_data["url"],
                section=doc_data.get("section", ""),
                metadata=doc_data.get("metadata", {})
            ))
        
        logger.info(f"Loaded {len(documents)} documents from {input_file}")
        return documents

def chunk_documents(documents: List[Document], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Split documents into chunks for better retrieval"""
    chunks = []
    
    for doc in documents:
        text = doc.content
        
        # If document is short enough, keep as one chunk
        if len(text) <= chunk_size:
            chunks.append({
                "text": text,
                "title": doc.title,
                "url": doc.url,
                "section": doc.section,
                "metadata": doc.metadata or {},
                "chunk_id": len(chunks)
            })
            continue
        
        # Split into overlapping chunks
        start = 0
        chunk_num = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence end
            if end < len(text):
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    end = break_point + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "title": f"{doc.title} (Part {chunk_num + 1})",
                    "url": doc.url,
                    "section": doc.section,
                    "metadata": {**(doc.metadata or {}), "chunk_num": chunk_num, "total_chunks": "unknown"},
                    "chunk_id": len(chunks)
                })
                chunk_num += 1
            
            start = end - overlap
    
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scraper = JaqpotDocsScraper()
    documents = scraper.scrape_all(max_pages=30)
    scraper.save_documents(documents)
    
    # Create chunks
    chunks = chunk_documents(documents)
    
    # Save chunks
    chunks_file = os.path.join(scraper.data_dir, "document_chunks.json")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Scraping complete! Found {len(documents)} documents and {len(chunks)} chunks.")