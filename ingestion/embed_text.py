# ingestion/embed_text.py
import numpy as np
import faiss
import pickle
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
import os

# Ensure project root is on path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ingestion.metadata_utils import load_and_merge_metadata

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def embed_and_save(
    model_name: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 32,
    index_path: str = None,
    meta_path: str = None,
):
    """
    Create and save text embeddings for all places.
    
    Args:
        model_name: Name of the sentence-transformers model to use
        batch_size: Batch size for encoding
        index_path: Custom path for saving the FAISS index
        meta_path: Custom path for saving the metadata
    """
    # Determine paths
    base_dir = Path(__file__).resolve().parent
    index_path = index_path or base_dir / "text_index.faiss"
    meta_path = meta_path or base_dir / "text_metadata.pkl"

    # Load metadata
    logger.info("Loading metadata...")
    data = load_and_merge_metadata()
    texts = [item["full_text"] for item in data]
    
    # Load model
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Calculate embeddings in batches
    logger.info(f"Embedding {len(texts)} texts...")
    embeddings_list = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True)
        embeddings_list.append(emb)
    
    # Combine all embeddings
    all_embeddings = np.vstack(embeddings_list).astype("float32")
    
    # Create and save FAISS index
    logger.info("Creating FAISS index...")
    index = faiss.IndexFlatIP(all_embeddings.shape[1])
    index.add(all_embeddings)
    
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    logger.info(f"Saving index to {index_path}")
    faiss.write_index(index, str(index_path))
    
    logger.info(f"Saving metadata to {meta_path}")
    with open(meta_path, "wb") as f:
        pickle.dump(data, f)
    
    logger.info(f"Done! Saved index with {len(data)} embeddings")
    return index_path, meta_path

if __name__ == "__main__":
    embed_and_save()