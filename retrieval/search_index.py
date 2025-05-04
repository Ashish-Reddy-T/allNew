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
from transformers import CLIPProcessor, CLIPModel

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

class MultiModalSearchEngine:
    """
    Enhanced RAG-based search engine that uses both text and image embeddings.
    Features:
    - Text-based semantic search
    - Image-based vibe search
    - Query understanding and enhancement
    - Result explanation
    """
    def __init__(
        self,
        text_model_name: str = "BAAI/bge-large-en-v1.5",
        text_index_path: str = "ingestion/text_index.faiss",
        text_meta_path: str = "ingestion/text_metadata.pkl",
        image_model_name: str = "openai/clip-vit-base-patch32",
        image_index_path: str = "ingestion/image_index.faiss",
        image_meta_path: str = "ingestion/image_metadata.pkl",
    ):
        logger.info("Initializing MultiModalSearchEngine")
        
        # Load text index and metadata
        try:
            self.text_index, self.text_metadata = load_index_and_metadata(text_index_path, text_meta_path)
            logger.info(f"Loaded {len(self.text_metadata)} places from text metadata")
            self.text_search_available = True
        except Exception as e:
            logger.error(f"Failed to load text index or metadata: {e}")
            self.text_search_available = False
            
        # Try to load image index and metadata
        try:
            self.image_index, self.image_metadata = load_index_and_metadata(image_index_path, image_meta_path)
            logger.info(f"Loaded {len(self.image_metadata)} images from image metadata")
            
            # Load CLIP model for image search
            logger.info(f"Loading CLIP model: {image_model_name}")
            self.device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            self.clip_model = CLIPModel.from_pretrained(image_model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(image_model_name)
            self.image_search_available = True
            logger.info("Image search initialized successfully")
        except Exception as e:
            logger.warning(f"Image search not available: {e}")
            self.image_search_available = False
            
        # Initialize text embedder
        logger.info(f"Loading text embedding model: {text_model_name}")
        self.embedder = SentenceTransformer(text_model_name)
        
        # Build a mapping from place_id to images
        self.place_to_images = {}
        if self.image_search_available:
            for item in self.image_metadata:
                place_id = item.get("place_id")
                if place_id:
                    if place_id not in self.place_to_images:
                        self.place_to_images[place_id] = []
                    self.place_to_images[place_id].append({
                        "image_url": item.get("image_url"),
                        "metadata_index": self.image_metadata.index(item)
                    })
            
        logger.info("MultiModalSearchEngine initialized successfully")
        
    def semantic_search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform semantic search using text embeddings.
        """
        if not self.text_search_available:
            logger.error("Text search not available")
            return []
            
        logger.info(f"Performing semantic search for: {query}")
        
        # Enhance the query to better capture the semantics
        enhanced_query = self._enhance_query(query)
        logger.info(f"Enhanced query: {enhanced_query}")
        
        # Create query embedding
        q_emb = self.embedder.encode([enhanced_query], normalize_embeddings=True)
        q_emb = np.array(q_emb).astype("float32")
        
        # Search the index
        logger.info(f"Searching FAISS index for top {k} matches")
        distances, indices = self.text_index.search(q_emb, k)
        
        # Format results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.text_metadata):
                result = self.text_metadata[idx].copy()
                result["score"] = float(score)
                result["match_reason"] = self._generate_reason(query, result)
                
                # Add image URLs if available
                place_id = result.get("place_id")
                if place_id in self.place_to_images:
                    image_info = self.place_to_images[place_id][:3]  # Limit to 3 images
                    result["representative_images"] = [info["image_url"] for info in image_info]
                    result["image_indices"] = [info["metadata_index"] for info in image_info]
                
                results.append(result)
                
        return results
    
    def image_search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform image search using CLIP text-to-image embeddings.
        """
        if not self.image_search_available:
            logger.warning("Image search not available")
            return []
            
        logger.info(f"Performing image search for: {query}")
        
        # Enhance query for visual search
        visual_query = f"A photo of {query}"
        logger.info(f"Visual query: {visual_query}")
        
        # Create the text embedding using CLIP
        inputs = self.clip_processor(text=visual_query, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            
        # Convert to numpy and normalize
        q_emb = text_features.cpu().numpy()
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_emb = q_emb.astype("float32")
        
        # Search the image index
        logger.info(f"Searching image FAISS index for top {k*2} matches")
        distances, indices = self.image_index.search(q_emb, k*2)  # Get more for deduplication
        
        # Group by place_id to avoid duplication
        place_scores = {}
        place_images = {}
        
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.image_metadata):
                item = self.image_metadata[idx]
                place_id = item.get("place_id")
                
                if place_id not in place_scores or score > place_scores[place_id]:
                    place_scores[place_id] = float(score)
                    place_images[place_id] = item.get("image_url")
        
        # Format results
        results = []
        for place_id, score in sorted(place_scores.items(), key=lambda x: x[1], reverse=True)[:k]:
            # Find corresponding place info in text metadata
            place_info = None
            for item in self.text_metadata:
                if item.get("place_id") == place_id:
                    place_info = item
                    break
                    
            if place_info:
                result = {
                    "place_id": place_id,
                    "name": place_info.get("name", "Unknown"),
                    "neighborhood": place_info.get("neighborhood", "Unknown"),
                    "tags": place_info.get("tags", ""),
                    "short_description": place_info.get("short_description", ""),
                    "score": score,
                    "representative_image": place_images.get(place_id),
                    "source": "image_search",
                    "match_reason": f"The visual appearance of this place matches the vibe of '{query}'"
                }
                results.append(result)
                
        logger.info(f"Image search returned {len(results)} results")
        return results
    
    def _enhance_query(self, query: str) -> str:
        """
        Enhance the query to better capture implied meaning.
        """
        # Map of common vibe queries to enhanced semantic queries
        vibe_query_map = {
            "hot guys": "Popular venues and places known for attractive men and good dating scene",
            "date night": "Romantic and intimate places perfect for couples and special evenings",
            "study": "Quiet places with wifi and good atmosphere for working or studying",
            "cowork": "Places suitable for working with laptop, wifi, and comfortable seating",
            "instagram": "Visually striking and photogenic places with aesthetic appeal",
            "sunset": "Places with beautiful views, especially during sunset hours",
            "lively": "Energetic and vibrant venues with active social atmosphere",
            "chill": "Relaxed and laid-back places with comfortable ambiance",
            "dance": "Places for dancing with good music and energetic atmosphere",
            "hipster": "Trendy and alternative spots popular with the hip crowd",
            "fancy": "Upscale and elegant establishments with refined atmosphere",
            "dive": "Authentic, unpretentious, and casual local establishments",
            "hidden gem": "Lesser-known but excellent places that locals love",
            "girls night": "Fun places perfect for groups of friends to socialize and celebrate",
            "tourist": "Popular attractions and must-visit spots for visitors",
            "first date": "Casual yet impressive places with conversation-friendly atmosphere",
            "birthday": "Celebratory venues suitable for special occasions and groups",
            "coffee": "Quality coffee shops with good atmosphere for relaxing or meeting",
            "brunch": "Places known for excellent weekend brunch service",
            "sunny day": "Places to enjoy nice weather, outdoor seating, or natural light",
            "rainy day": "Cozy indoor places perfect for when the weather is bad"
        }
        
        # Check if we have a predefined enhancement
        query_lower = query.lower()
        
        for key, enhanced in vibe_query_map.items():
            if key in query_lower:
                # Keep any location information from the original query
                neighborhoods = ["manhattan", "brooklyn", "queens", "bronx", "williamsburg", 
                              "east village", "west village", "soho", "tribeca", "downtown"]
                
                for neighborhood in neighborhoods:
                    if neighborhood in query_lower:
                        return f"{enhanced} in {neighborhood}"
                        
                return enhanced
                
        # If no predefined enhancement matched, return the original query
        return query
    
    def _generate_reason(self, query: str, result: Dict) -> str:
        """
        Generate an explanation of why this result matches the query.
        """
        name = result.get("name", "")
        tags = result.get("tags", "")
        description = result.get("short_description", "")
        neighborhood = result.get("neighborhood", "")
        
        # Common vibe keywords to check for
        query_lower = query.lower()
        
        if "hot guys" in query_lower or "cute guys" in query_lower:
            if "bar" in tags.lower() or "club" in tags.lower():
                return f"{name} is a {tags} in {neighborhood} known for its social atmosphere and popularity among stylish clientele."
            elif "dance" in tags.lower() or "dance" in description.lower():
                return f"{name} is a dance venue in {neighborhood} that attracts a lively and attractive crowd."
            else:
                return f"{name} in {neighborhood} is a social hotspot that matches your search for places to meet people."
                
        elif "date" in query_lower:
            return f"{name} in {neighborhood} offers a {description} ambiance that's perfect for a romantic evening."
            
        elif "work" in query_lower or "study" in query_lower:
            return f"{name} in {neighborhood} provides a good environment for focusing, likely with WiFi and good seating."
            
        elif "instagram" in query_lower or "photo" in query_lower:
            return f"{name} features visually striking {description} that make for great photos and social media content."
            
        elif "dance" in query_lower:
            return f"{name} is known for music and dancing with its {description} atmosphere."
            
        # Default explanation based on available metadata
        if tags and description:
            return f"{name} in {neighborhood} is a {tags} featuring {description}."
        elif tags:
            return f"{name} in {neighborhood} is categorized as {tags}."
        elif description:
            return f"{name} in {neighborhood} is described as: {description}."
        else:
            return f"{name} in {neighborhood} matches semantic elements of your search query."
        
    def search(self, query: str, k: int = 10, use_images: bool = True) -> Dict:
        """
        Perform a complete search using both text and image modalities.
        
        Args:
            query: The search query
            k: Number of results to return
            use_images: Whether to include image search
            
        Returns:
            Search results with metadata
        """
        # Start with text search
        text_results = self.semantic_search(query, k=k)
        
        # Add image search if available and requested
        image_results = []
        if use_images and self.image_search_available:
            image_results = self.image_search(query, k=k//2)
            
        # Combine results, prioritizing text but including unique image results
        combined = text_results.copy()
        
        # Add unique results from image search
        existing_place_ids = {r["place_id"] for r in combined}
        
        for img_result in image_results:
            if img_result["place_id"] not in existing_place_ids:
                combined.append(img_result)
                existing_place_ids.add(img_result["place_id"])
                
                # Stop if we've reached the desired number of results
                if len(combined) >= k:
                    break
        
        # Sort by score
        combined.sort(key=lambda x: x["score"], reverse=True)
        
        # Prepare the final response
        return {
            "query": query,
            "enhanced_query": self._enhance_query(query),
            "results": combined[:k],
            "total_results": len(combined),
            "text_results_count": len(text_results),
            "image_results_count": len(image_results)
        }

if __name__ == "__main__":
    searcher = MultiModalSearchEngine()
    
    while True:
        query = input("Enter search query (or 'exit' to quit): ")
        if query.lower() in ('exit', 'quit', 'q'):
            break
            
        use_images = input("Include image search? (y/n, default=y): ").lower() != 'n'
        
        results = searcher.search(query, k=10, use_images=use_images)
            
        print("\n--- Search Results ---\n")
        print(f"Query: {results['query']}")
        print(f"Enhanced query: {results['enhanced_query']}")
        print(f"Total results: {results['total_results']} (Text: {results['text_results_count']}, Image: {results['image_results_count']})")
        print()
        
        if not results["results"]:
            print("No results found.")
        else:
            for i, result in enumerate(results["results"], 1):
                print(f"{i}. {result['name']} ({result['neighborhood']})")
                print(f"   Score: {result['score']:.4f}")
                print(f"   Tags: {result['tags']}")
                print(f"   Description: {result['short_description']}")
                print(f"   Why it matches: {result['match_reason']}")
                
                # Show image info
                if "representative_images" in result:
                    print(f"   Images: {len(result['representative_images'])}")
                elif "representative_image" in result:
                    print(f"   Representative image: {result['representative_image']}")
                print()