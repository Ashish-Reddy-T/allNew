# retrieval/search_index.py
import logging
import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Union, Tuple
import sys
import os
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ingestion.metadata_utils import load_and_merge_metadata

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_index_and_metadata(
    index_path: str,
    meta_path: str,
) -> tuple:
    """
    Load a FAISS index and associated metadata.
    """
    base_dir = Path(__file__).resolve().parent.parent
    index_file = base_dir / index_path
    meta_file = base_dir / meta_path

    logger.info(f"Reading FAISS index from {index_file}")
    index = faiss.read_index(str(index_file))

    logger.info(f"Loading metadata from {meta_file}")
    with open(meta_file, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

class Retriever:
    """
    Advanced retrieval system for finding places based on user queries.
    Features:
    - Semantic search
    - Enhanced query understanding
    - Dynamic result ranking
    - LLM integration (optional)
    """
    def __init__(
        self,
        text_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        text_index_path: str = "ingestion/text_index.faiss",
        text_meta_path: str = "ingestion/text_metadata.pkl",
    ):
        logger.info("Initializing Retriever")
        
        # Load text index and metadata
        try:
            self.text_index, self.text_metadata = load_index_and_metadata(text_index_path, text_meta_path)
            logger.info(f"Loaded {len(self.text_metadata)} places from text metadata")
            self.search_available = True
        except Exception as e:
            logger.error(f"Failed to load text index or metadata: {e}")
            self.search_available = False
            
        # Initialize text embedder
        logger.info(f"Loading text embedding model: {text_model_name}")
        self.embedder = SentenceTransformer(text_model_name)
        
        # Store the dimension of embeddings
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Check if index and model dimensions match
        if self.search_available and self.text_index.d != self.embedding_dim:
            logger.warning(f"Dimension mismatch! Index: {self.text_index.d}, Model: {self.embedding_dim}")
            logger.warning("Using model that created the index for searching")
            # This is a critical warning, but we'll try to proceed
        
        logger.info("Retriever initialized successfully")
    
    def get_embedding_model(self):
        """Return the embedding model for external use"""
        return self.embedder

    def search(self, query: str, k: int = 10, use_hyde: bool = False) -> List[Dict]:
        """
        Perform semantic search with optional enhancements.
        
        Args:
            query: User search query
            k: Number of results to return
            use_hyde: Whether to use query expansion techniques
            
        Returns:
            List of matched places with metadata
        """
        if not self.search_available:
            logger.error("Search not available")
            return []
            
        logger.info(f"Searching for: {query}")
        
        # Create query embedding
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb).astype("float32")
        
        # Ensure dimensions match
        if q_emb.shape[1] != self.text_index.d:
            logger.error(f"Query embedding dimension ({q_emb.shape[1]}) does not match index dimension ({self.text_index.d})")
            return []
        
        # Search the index
        logger.info(f"Searching index for top {k} matches")
        distances, indices = self.text_index.search(q_emb, k)
        
        # Format and return results
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.text_metadata):
                result = self.text_metadata[idx].copy()
                
                # Add relevance score
                result["score"] = float(score)
                
                # Generate a match reason
                result["match_reason"] = self._generate_match_reason(query, result)
                
                results.append(result)
        
        logger.info(f"Found {len(results)} matching places")
        return results
    
    def _generate_match_reason(self, query: str, result: Dict) -> str:
        """Generate a reason for why this result matches the query."""
        name = result.get("name", "")
        neighborhood = result.get("neighborhood", "")
        tags = result.get("tags", "")
        description = result.get("short_description", "")
        
        # Create a basic reason based on available metadata
        components = []
        
        if name:
            components.append(f"{name}")
        
        if neighborhood:
            components.append(f"in {neighborhood}")
        
        if tags:
            components.append(f"is a {tags}")
        
        if description:
            components.append(f"known for {description}")
        
        if components:
            return " ".join(components)
        else:
            return "This place matches your query semantically."