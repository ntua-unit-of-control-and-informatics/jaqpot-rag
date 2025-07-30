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

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

logger = logging.getLogger(__name__)

@dataclass
class Document:
    title: str
    content: str
    url: str
    section: str = ""
    metadata: Dict = None

class JaqpotDocsScraper:
    """Scraper for Jaqpot documentation using Selenium for JavaScript-rendered content"""
    
    def __init__(self, base_url: str = "https://jaqpot.org/docs", data_dir: str = None):
        self.base_url = base_url
        if data_dir is None:
            self.data_dir = "./data/jaqpot_docs" if os.path.exists("./data") else "../data/jaqpot_docs"
        else:
            self.data_dir = data_dir
        self.visited_urls: Set[str] = set()
        self.documents: List[Document] = []
        self.driver = None
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize Selenium WebDriver
        self._init_driver()
    
    def _init_driver(self):
        """Initialize Chrome WebDriver with appropriate options"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("Chrome WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Chrome WebDriver: {e}")
            raise e
    
    def __del__(self):
        """Clean up WebDriver on destruction"""
        if self.driver:
            self.driver.quit()
    
    def is_valid_jaqpot_url(self, url: str) -> bool:
        """Check if URL is a valid Jaqpot docs URL"""
        parsed = urlparse(url)
        return (
            parsed.netloc == "jaqpot.org" and 
            parsed.path.startswith("/docs") and
            url not in self.visited_urls
        )
    
    def get_next_page_url(self, current_url: str) -> str:
        """Extract the 'Next' button URL using Selenium to wait for JavaScript to load"""
        logger.debug(f"Looking for next page from: {current_url}")
        
        try:
            # Load the page
            self.driver.get(current_url)
            
            # Wait for the page to load and for navigation elements to appear
            wait = WebDriverWait(self.driver, 10)
            
            # Try multiple selectors for the next button
            next_selectors = [
                "a.pagination-nav__link.pagination-nav__link--next",
                "a[class*='pagination-nav__link--next']",
                "a[class*='next']",
                ".pagination-nav__link--next"
            ]
            
            for selector in next_selectors:
                try:
                    # Wait for the element to be present and clickable
                    next_element = wait.until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    
                    href = next_element.get_attribute('href')
                    logger.debug(f"Found next link with selector '{selector}': {href}")
                    
                    if href and self.is_valid_jaqpot_url(href):
                        return href
                        
                except TimeoutException:
                    logger.debug(f"Timeout waiting for selector: {selector}")
                    continue
                except Exception as e:
                    logger.debug(f"Error with selector '{selector}': {e}")
                    continue
            
            # If specific selectors don't work, try finding by text content
            try:
                next_elements = self.driver.find_elements(By.XPATH, "//a[contains(text(), 'Next') or contains(@class, 'next')]")
                for element in next_elements:
                    href = element.get_attribute('href')
                    if href and self.is_valid_jaqpot_url(href):
                        logger.debug(f"Found next link by text search: {href}")
                        return href
            except Exception as e:
                logger.debug(f"Error searching by text: {e}")
            
            logger.debug("No next page link found")
            return None
            
        except Exception as e:
            logger.error(f"Error getting next page URL from {current_url}: {e}")
            return None

    def extract_content(self, soup: BeautifulSoup, url: str) -> List[Document]:
        """Extract content from a Docusaurus documentation page"""
        documents = []
        
        # Get page title from h1 or title tag
        title_elem = soup.find('h1') or soup.find('title')
        page_title = title_elem.get_text().strip() if title_elem else "Untitled"
        
        # For Docusaurus, look for the main content area
        main_content = (
            soup.find('article') or
            soup.find('main') or 
            soup.find('div', class_=re.compile(r'docPage|content|main')) or
            soup.find('div', {'role': 'main'})
        )
        
        if not main_content:
            logger.warning(f"No main content found for {url}")
            return documents
        
        # Remove code blocks temporarily to avoid splitting them
        code_blocks = []
        for code in main_content.find_all(['pre', 'code']):
            placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
            code_blocks.append(code.get_text())
            code.replace_with(placeholder)
        
        # Get all text content
        full_content = main_content.get_text()
        
        # Restore code blocks
        for i, code_text in enumerate(code_blocks):
            full_content = full_content.replace(f"__CODE_BLOCK_{i}__", code_text)
        
        # Clean the content
        content = self.clean_text(full_content)
        
        if content.strip():
            documents.append(Document(
                title=page_title,
                content=content,
                url=url,
                section="main",
                metadata={
                    "page_title": page_title,
                    "content_length": len(content)
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
        """Extract next page link from Docusaurus navigation"""
        links = []
        next_url = self.get_next_page_url(soup, current_url)
        if next_url:
            links.append(next_url)
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
    
    def get_all_sidebar_links(self, url: str) -> List[str]:
        """Get all documentation links from sidebar navigation using Selenium"""
        links = set()
        
        try:
            # Load the page with Selenium
            self.driver.get(url)
            
            # Wait for sidebar to load
            wait = WebDriverWait(self.driver, 10)
            
            # Find Docusaurus sidebar
            sidebar_selectors = [
                'nav[class*="sidebar"]',
                'aside[class*="sidebar"]', 
                'div[class*="sidebar"]',
                'nav[class*="menu"]',
                'ul[class*="menu"]',
                '.menu',
                '.sidebar'
            ]
            
            sidebar_element = None
            for selector in sidebar_selectors:
                try:
                    sidebar_element = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    logger.debug(f"Found sidebar with selector: {selector}")
                    break
                except TimeoutException:
                    continue
            
            if sidebar_element:
                # Get all links from sidebar
                link_elements = sidebar_element.find_elements(By.TAG_NAME, "a")
                
                for link_element in link_elements:
                    href = link_element.get_attribute('href')
                    if href:
                        if href.startswith('/docs'):
                            full_url = urljoin('https://jaqpot.org', href)
                            links.add(full_url)
                            logger.debug(f"Found sidebar link: {full_url}")
                        elif href.startswith('https://jaqpot.org/docs'):
                            links.add(href)
                            logger.debug(f"Found sidebar link: {href}")
            
        except Exception as e:
            logger.error(f"Error getting sidebar links: {e}")
        
        return list(links)

    def scrape_all(self, max_pages: int = 50) -> List[Document]:
        """Scrape all Jaqpot documentation following Next button navigation with Selenium"""
        logger.info("Starting Jaqpot documentation scraping using Selenium and Next button navigation...")
        
        all_documents = []
        current_url = self.base_url
        pages_scraped = 0
        
        while current_url and pages_scraped < max_pages:
            if current_url in self.visited_urls:
                logger.info(f"Already visited {current_url}, stopping to avoid loop")
                break
            
            logger.info(f"Scraping page {pages_scraped + 1}: {current_url}")
            
            try:
                # Load page with Selenium and extract content
                self.driver.get(current_url)
                
                # Wait for page to load
                wait = WebDriverWait(self.driver, 10)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                
                # Additional wait for React content to render
                time.sleep(3)
                
                # Get page source and parse with BeautifulSoup
                page_source = self.driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')
                
                # Extract content
                documents = self.extract_content(soup, current_url)
                self.visited_urls.add(current_url)
                
                # Add documents (with duplicate checking)
                for doc in documents:
                    is_duplicate = False
                    for existing_doc in all_documents:
                        # Check for duplicates based on content similarity
                        if (len(doc.content.strip()) > 100 and
                            doc.content.strip() == existing_doc.content.strip()):
                            is_duplicate = True
                            logger.debug(f"Skipping duplicate content from {doc.url}")
                            break
                    
                    if not is_duplicate:
                        all_documents.append(doc)
                        logger.info(f"Added document: {doc.title} ({len(doc.content)} chars)")
                
                pages_scraped += 1
                
                # Find next page using Selenium
                next_url = self.get_next_page_url(current_url)
                
                if next_url and next_url != current_url:
                    current_url = next_url
                    logger.info(f"Found next page: {next_url}")
                else:
                    logger.info("No more next pages found, scraping complete")
                    break
                
                # Add delay to be respectful
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scraping {current_url}: {e}")
                break
        
        logger.info(f"Scraping complete. Total documents: {len(all_documents)} from {pages_scraped} pages")
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
    logging.basicConfig(level=logging.DEBUG)
    
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