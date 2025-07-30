#!/usr/bin/env python3
"""
Standalone script to scrape Jaqpot documentation
"""

import sys
import os
import logging

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from document_processor import JaqpotDocsScraper, chunk_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Scrape Jaqpot documentation"""
    logger.info("Starting documentation scraping...")
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'jaqpot_docs')
    
    try:
        # Initialize scraper
        scraper = JaqpotDocsScraper(data_dir=data_dir)
        
        # Scrape documents
        documents = scraper.scrape_all(max_pages=30)
        scraper.save_documents(documents)
        
        # Create chunks
        chunks = chunk_documents(documents)
        
        # Save chunks
        chunks_file = os.path.join(data_dir, "document_chunks.json")
        import json
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Scraping complete! Found {len(documents)} documents and {len(chunks)} chunks.")
        return 0
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)