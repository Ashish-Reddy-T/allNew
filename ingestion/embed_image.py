# ingestion/embed_image.py
import torch
import faiss
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from pathlib import Path
import logging
import os
import sys
from PIL import Image
from io import BytesIO
import requests
from typing import List, Dict, Optional, Tuple
from transformers import CLIPProcessor, CLIPModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_image_from_url(url: str) -> Optional[Image.Image]:
    """Load an image from a URL, returning None if it fails."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        logger.warning(f"Failed to load image from {url}: {e}")
        return None

def embed_images(
    model_name: str = "openai/clip-vit-base-patch32",
    media_path: str = "media.csv",
    places_path: str = "places.csv",
    batch_size: int = 16,
    max_images_per_place: int = 5,  # Limit number of images per place
    index_path: str = None,
    meta_path: str = None,
) -> Tuple[str, str]:
    """
    Create and save CLIP embeddings for images.
    
    Args:
        model_name: Name of the CLIP model to use
        media_path: Path to the media CSV
        places_path: Path to the places CSV
        batch_size: Batch size for processing
        max_images_per_place: Maximum number of images to process per place
        index_path: Custom path for saving the FAISS index
        meta_path: Custom path for saving the metadata
        
    Returns:
        Tuple of (index_path, meta_path)
    """
    # Determine paths
    base_dir = Path(__file__).resolve().parent
    index_path = index_path or base_dir / "image_index.faiss"
    meta_path = meta_path or base_dir / "image_metadata.pkl"
    
    # Load media and places data
    logger.info(f"Loading media data from {media_path}")
    media_df = pd.read_csv(Path(__file__).resolve().parent.parent / media_path)
    
    logger.info(f"Loading places data from {places_path}")
    places_df = pd.read_csv(Path(__file__).resolve().parent.parent / places_path)
    
    # Join media with places to get place names and neighborhoods
    media_with_place = media_df.merge(
        places_df[['place_id', 'name', 'neighborhood', 'tags', 'short_description']],
        on='place_id',
        how='left'
    )
    
    # Initialize CLIP model
    logger.info(f"Loading CLIP model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Process images and create embeddings
    all_embeddings = []
    metadata = []
    
    # Group by place_id to limit images per place
    logger.info("Processing images...")
    for place_id, group in tqdm(media_with_place.groupby('place_id')):
        place_name = group['name'].iloc[0]
        place_neighborhood = group['neighborhood'].iloc[0]
        place_tags = group['tags'].iloc[0]
        place_description = group['short_description'].iloc[0]
        
        # Take only a subset of images per place
        for i, (_, row) in enumerate(group.iterrows()):
            if i >= max_images_per_place:
                break
                
            img_url = row['media_url']
            img = load_image_from_url(img_url)
            
            if img is not None:
                # Process and embed the image
                try:
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        image_features = model.get_image_features(**inputs)
                        
                    # Normalize the features
                    image_embedding = image_features.cpu().numpy()
                    image_embedding = image_embedding / np.linalg.norm(image_embedding, axis=1, keepdims=True)
                    
                    # Add to our collections
                    all_embeddings.append(image_embedding)
                    metadata.append({
                        'place_id': place_id,
                        'name': place_name,
                        'neighborhood': place_neighborhood,
                        'tags': place_tags,
                        'short_description': place_description,
                        'image_url': img_url
                    })
                except Exception as e:
                    logger.warning(f"Error processing image {img_url}: {e}")
    
    if not all_embeddings:
        logger.error("No valid images were processed!")
        return index_path, meta_path
        
    # Stack all embeddings
    embeddings_array = np.vstack(all_embeddings).astype('float32')
    
    # Create FAISS index
    logger.info("Creating FAISS index...")
    d = embeddings_array.shape[1]  # Embedding dimension
    index = faiss.IndexFlatIP(d)
    index.add(embeddings_array)
    
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Save index and metadata
    logger.info(f"Saving index with {len(metadata)} images to {index_path}")
    faiss.write_index(index, str(index_path))
    
    logger.info(f"Saving metadata to {meta_path}")
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
        
    logger.info("Image embedding complete")
    return index_path, meta_path

if __name__ == "__main__":
    embed_images()