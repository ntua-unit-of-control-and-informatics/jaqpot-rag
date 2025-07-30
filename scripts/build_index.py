#!/usr/bin/env python3
"""
Script to build the Jaqpot documentation index
"""

import sys
import os
import asyncio
import logging

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Build the documentation index"""
    logger.info("Starting index build process...")
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline()
        await rag.initialize()
        
        # Get stats
        stats = await rag.get_stats()
        logger.info(f"Index build complete! Stats: {stats}")
        
    except Exception as e:
        logger.error(f"Index build failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)