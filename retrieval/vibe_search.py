# retrieval/vibe_search.py
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ingestion.metadata_utils import load_and_merge_metadata

class VibeSearch:
    """
    Integrated multi-modal vibe search system combining text and image understanding.
    
    This system:
    1. Processes natural language queries with powerful embedding models
    2. Combines text and image search for a complete "vibe" understanding
    3. Uses LangChain with powerful models for query enhancement
    4. Ranks results by relevance across modalities
    """
    
    def __init__(
        self,
        text_model_name: str = "BAAI/bge-large-en-v1.5",  
        text_index_path: str = "ingestion/text_index.faiss",
        text_meta_path: str = "ingestion/text_metadata.pkl",
        image_index_path: str = "ingestion/image_index.faiss",
        image_meta_path: str = "ingestion/image_metadata.pkl",
        use_llm: bool = True,
        llm_type: str = "ollama",
        llm_model: str = "mistral"
    ):
        """
        Initialize the complete vibe search system.
        
        Args:
            text_model_name: Name of the text embedding model
            text_index_path: Path to the text FAISS index
            text_meta_path: Path to the text metadata
            image_index_path: Path to the image FAISS index
            image_meta_path: Path to the image metadata
            use_llm: Whether to use LLM for query enhancement 
            llm_type: Type of LLM to use
            llm_model: Model name to use
        """
        logger.info("Initializing VibeSearch")
        
        # Initialize text search
        self.text_search_available = self._init_text_search(text_model_name, text_index_path, text_meta_path)
        
        # Initialize image search
        self.image_search_available = self._init_image_search(image_index_path, image_meta_path)
            
        # Initialize LLM if enabled
        self.use_llm = use_llm
        if use_llm:
            self._init_llm(llm_type, llm_model)
        
        # Create a mapping from place_id to all data
        self._build_place_mapping()
        
        # Build neighborhood mapping for filtering
        self._build_neighborhood_mapping()
            
        logger.info(f"VibeSearch initialized successfully (Text: {self.text_search_available}, Image: {self.image_search_available}, LLM: {self.use_llm})")
    
    def _init_text_search(self, model_name, index_path, meta_path):
        """Initialize text search components."""
        try:
            # Load the text index and metadata
            base_dir = Path(__file__).resolve().parent.parent
            index_file = base_dir / index_path
            meta_file = base_dir / meta_path
            
            logger.info(f"Loading text search index from {index_file}")
            self.text_index = faiss.read_index(str(index_file))
            
            logger.info(f"Loading text metadata from {meta_file}")
            with open(meta_file, "rb") as f:
                self.text_metadata = pickle.load(f)
            
            # Initialize the embedding model
            logger.info(f"Loading text embedding model: {model_name}")
            self.text_embedder = SentenceTransformer(model_name)
            self.text_embedding_dim = self.text_embedder.get_sentence_embedding_dimension()
            
            # Check for dimension mismatch
            if self.text_index.d != self.text_embedding_dim:
                logger.warning(f"Text dimension mismatch: index={self.text_index.d}, model={self.text_embedding_dim}")
                logger.warning("Text search may not work properly due to dimension mismatch")
            
            # Get the model that created the index if possible
            if hasattr(self.text_embedder, "get_model_name"):
                logger.info(f"Text index was likely created with: {self.text_embedder.get_model_name()}")
            
            logger.info(f"Text search initialized with {len(self.text_metadata)} places")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize text search: {e}")
            return False
    
    def _init_image_search(self, index_path, meta_path):
        """Initialize image search components."""
        try:
            # Load the image index and metadata
            base_dir = Path(__file__).resolve().parent.parent
            index_file = base_dir / index_path
            meta_file = base_dir / meta_path
            
            logger.info(f"Loading image search index from {index_file}")
            self.image_index = faiss.read_index(str(index_file))
            
            logger.info(f"Loading image metadata from {meta_file}")
            with open(meta_file, "rb") as f:
                self.image_metadata = pickle.load(f)
            
            # We'll load CLIP only when needed to save memory
            
            logger.info(f"Image search initialized with {len(self.image_metadata)} images")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize image search: {e}")
            return False
    
    def _init_llm(self, llm_type, model_name):
        """Initialize the LLM for query enhancement."""
        try:
            # Try to import the new LangChain packages
            try:
                if llm_type == "ollama":
                    from langchain_community.llms import Ollama
                    self.llm = Ollama(model=model_name)
                    logger.info(f"Initialized Ollama with model {model_name}")
                else:
                    # Default to HF
                    from langchain_community.llms import HuggingFaceTextGenInference
                    self.llm = HuggingFaceTextGenInference(
                        inference_server_url="http://localhost:8080/",
                        max_new_tokens=512,
                        temperature=0.1,
                        timeout=120,
                    )
                    logger.info(f"Initialized HuggingFace inference endpoint")
                
                from langchain_core.prompts import PromptTemplate
                from langchain_core.output_parsers import StrOutputParser
                from langchain_core.runnables import RunnablePassthrough
            except ImportError:
                # Fall back to old imports
                if llm_type == "ollama":
                    from langchain.llms import Ollama
                    self.llm = Ollama(model=model_name)
                    logger.info(f"Initialized Ollama with model {model_name}")
                else:
                    from langchain.llms import HuggingFaceTextGenInference
                    self.llm = HuggingFaceTextGenInference(
                        inference_server_url="http://localhost:8080/",
                        max_new_tokens=512,
                        temperature=0.1,
                        timeout=120,
                    )
                    logger.info(f"Initialized HuggingFace inference endpoint")
                
                from langchain.prompts import PromptTemplate
                from langchain.schema.output_parser import StrOutputParser
                from langchain.schema.runnable import RunnablePassthrough
            
            # Set up the enhance query chain with context about available data
            enhance_template = """
            You are an AI assistant helping with a search system for places in New York City.

            USER QUERY: {query}

            Your task is to:
            1. Understand the user's intent about what places they want to visit
            2. Create an enhanced search query that includes specific areas, types of places, and ordered itinerary if requested
            3. If they ask for a chronological order or itinerary, explicitly include this in your query

            Your enhanced query should be detailed and include SPECIFIC LOCATIONS and neighborhoods.

            ENHANCED QUERY:
            """
            
            self.enhance_chain = (
                {"query": RunnablePassthrough()} 
                | PromptTemplate.from_template(enhance_template)
                | self.llm
                | StrOutputParser()
            )
            
            # Set up the explanation chain
            explain_template = """
            You are helping explain why a specific venue matches someone's search.
            
            USER QUERY: {query}
            
            PLACE DETAILS:
            Name: {name}
            Location: {location}
            Type: {tags}
            Description: {description}
            
            In 1-2 conversational sentences, explain specifically why this place would be a good match for their search.
            Focus on the specific request in the query. Be friendly and helpful.
            
            EXPLANATION:
            """
            
            self.explain_chain = (
                {"query": RunnablePassthrough(), "name": RunnablePassthrough(), 
                 "location": RunnablePassthrough(), "tags": RunnablePassthrough(),
                 "description": RunnablePassthrough()} 
                | PromptTemplate.from_template(explain_template)
                | self.llm
                | StrOutputParser()
            )
            
            # Add a chain for improving rankings
            ranking_template = """
            You are helping rank search results based on relevance to a query.
            
            USER QUERY: {query}
            
            For each place, assign a relevance score from 0-100 based on how well it matches the query:
            
            {places}
            
            Output ONLY a comma-separated list of scores in the same order as the places.
            Example: 85,72,95,63,41
            
            SCORES:
            """
            
            self.ranking_chain = (
                {"query": RunnablePassthrough(), "places": RunnablePassthrough()} 
                | PromptTemplate.from_template(ranking_template)
                | self.llm
                | StrOutputParser()
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.use_llm = False
            return False
    
    def _build_place_mapping(self):
        """Build a mapping from place_id to all relevant data."""
        self.place_mapping = {}
        
        # Add text metadata
        if hasattr(self, 'text_metadata'):
            for item in self.text_metadata:
                place_id = item.get('place_id')
                if place_id:
                    if place_id not in self.place_mapping:
                        self.place_mapping[place_id] = {'place_info': item, 'images': []}
        
        # Add image metadata
        if hasattr(self, 'image_metadata'):
            for idx, item in enumerate(self.image_metadata):
                place_id = item.get('place_id')
                if place_id:
                    if place_id not in self.place_mapping:
                        # Create entry if it doesn't exist
                        self.place_mapping[place_id] = {
                            'place_info': {
                                'place_id': place_id,
                                'name': item.get('name', ''),
                                'neighborhood': item.get('neighborhood', ''),
                                'tags': item.get('tags', ''),
                                'short_description': item.get('short_description', '')
                            },
                            'images': []
                        }
                    
                    # Add image
                    self.place_mapping[place_id]['images'].append({
                        'image_url': item.get('image_url', ''),
                        'index': idx
                    })
        
        logger.info(f"Built place mapping with {len(self.place_mapping)} places")
    
    def _build_neighborhood_mapping(self):
        """Build mapping of neighborhoods for filtering."""
        self.neighborhoods = set()
        self.neighborhood_to_places = {}
        
        # Extract all neighborhoods
        for place_id, data in self.place_mapping.items():
            # Convert neighborhood to string first to handle floats
            neighborhood_value = data['place_info'].get('neighborhood', '')
            neighborhood = str(neighborhood_value).lower() if neighborhood_value else ''
            
            if neighborhood:
                self.neighborhoods.add(neighborhood)
                
                if neighborhood not in self.neighborhood_to_places:
                    self.neighborhood_to_places[neighborhood] = []
                
                self.neighborhood_to_places[neighborhood].append(place_id)
        
        logger.info(f"Found {len(self.neighborhoods)} unique neighborhoods")
    
    def enhance_query(self, query: str) -> str:
        """Enhance the query with LLM understanding."""
        if not self.use_llm:
            return query
        
        try:
            enhanced = self.enhance_chain.invoke(query)
            logger.info(f"Enhanced query: {enhanced}")
            return enhanced.strip()
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query
    
    def explain_match(self, query: str, place: Dict) -> str:
        """Generate an explanation for why a place matches the query."""
        if not self.use_llm:
            return self._simple_explanation(place)
        
        try:
            explanation = self.explain_chain.invoke({
                "query": query,
                "name": place.get('name', ''),
                "location": place.get('neighborhood', ''),
                "tags": place.get('tags', ''),
                "description": place.get('short_description', '')
            })
            return explanation.strip()
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return self._simple_explanation(place)
    
    def _simple_explanation(self, place: Dict) -> str:
        """Generate a simple explanation without using LLM."""
        name = place.get('name', '')
        neighborhood = place.get('neighborhood', '')
        tags = place.get('tags', '')
        description = place.get('short_description', '')
        
        if description:
            return f"{name} in {neighborhood} features {description}"
        elif tags:
            return f"{name} is a {tags} in {neighborhood}"
        else:
            return f"{name} in {neighborhood} matches your search"
    
    def _extract_neighborhoods(self, query: str) -> List[str]:
        """Extract mentioned neighborhoods from query."""
        query_lower = query.lower()
        
        # Check if any neighborhood is mentioned
        mentioned = []
        for neighborhood in self.neighborhoods:
            if neighborhood in query_lower:
                mentioned.append(neighborhood)
        
        # Also check for common NYC area names
        common_areas = {
            "downtown": ["lower manhattan", "financial district", "tribeca", "soho"],
            "midtown": ["midtown", "times square", "theater district"],
            "uptown": ["upper east side", "upper west side", "harlem"],
            "brooklyn": ["williamsburg", "dumbo", "bushwick", "park slope"],
            "queens": ["astoria", "long island city", "flushing"]
        }
        
        for area, neighborhoods in common_areas.items():
            if area in query_lower:
                for n in neighborhoods:
                    if n in self.neighborhood_to_places:
                        mentioned.append(n)
        
        return mentioned
    
    def text_search(self, query: str, neighborhoods: List[str] = None, k: int = 20) -> List[Dict]:
        """Perform text-based semantic search with optional neighborhood filtering."""
        if not self.text_search_available:
            logger.error("Text search not available")
            return []
        
        try:
            # Create the query embedding
            query_emb = self.text_embedder.encode([query], normalize_embeddings=True)
            query_emb = np.array(query_emb).astype('float32')
            
            # Ensure dimensions match
            if query_emb.shape[1] != self.text_index.d:
                logger.error("Query embedding dimension mismatch")
                return []
            
            # Search the index
            logger.info(f"Performing text search for: {query}")
            
            # If neighborhoods are specified, limit search to those neighborhoods
            if neighborhoods:
                # Get all place indices for these neighborhoods
                place_ids = []
                for n in neighborhoods:
                    if n in self.neighborhood_to_places:
                        place_ids.extend(self.neighborhood_to_places[n])
                
                # If we found relevant places, search only those
                if place_ids:
                    logger.info(f"Filtering by {len(neighborhoods)} neighborhoods with {len(place_ids)} places")
                    
                    # Get indices in the metadata array
                    place_indices = []
                    for i, place in enumerate(self.text_metadata):
                        if place.get('place_id') in place_ids:
                            place_indices.append(i)
                    
                    # Extract vectors for these places
                    vectors = np.vstack([self.text_index.reconstruct(i) for i in place_indices]).astype('float32')
                    
                    # Create a temporary index
                    temp_index = faiss.IndexFlatIP(self.text_index.d)
                    temp_index.add(vectors)
                    
                    # Search in this index
                    distances, indices = temp_index.search(query_emb, min(k, len(vectors)))
                    
                    # Map back to original indices
                    original_indices = [place_indices[i] for i in indices[0]]
                    distances = distances[0]
                else:
                    # No places in these neighborhoods, search normally
                    distances, indices = self.text_index.search(query_emb, k)
                    original_indices = indices[0]
                    distances = distances[0]
            else:
                # Search the full index
                distances, indices = self.text_index.search(query_emb, k)
                original_indices = indices[0]
                distances = distances[0]
            
            # Process results
            results = []
            for idx, score in zip(original_indices, distances):
                if idx < len(self.text_metadata):
                    place_data = self.text_metadata[idx]
                    results.append({
                        'place_id': place_data.get('place_id', ''),
                        'name': place_data.get('name', ''),
                        'neighborhood': place_data.get('neighborhood', ''),
                        'tags': place_data.get('tags', ''),
                        'short_description': place_data.get('short_description', ''),
                        'score': float(score),
                        'source': 'text'
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []
    
    def image_search(self, query: str, k: int = 10) -> List[Dict]:
        """Perform image-based "vibe" search with shorter text input."""
        if not self.image_search_available:
            logger.error("Image search not available")
            return []
        
        try:
            # We need to load the CLIP model
            import torch
            from transformers import CLIPProcessor, CLIPModel
            
            # Initialize CLIP on first use
            if not hasattr(self, 'clip_model'):
                logger.info("Loading CLIP model for image search")
                model_name = "openai/clip-vit-base-patch32"
                self.device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
                self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained(model_name)
                logger.info(f"CLIP model loaded on {self.device}")
            
            # Truncate query if needed
            # CLIP has a token limit - keep it short
            short_query = ' '.join(query.split()[:10])
            visual_query = f"A photo of {short_query}"
            
            # Prepare the query
            inputs = self.clip_processor(text=[visual_query], return_tensors="pt").to(self.device)
            
            # Get embedding
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            
            # Convert to numpy and normalize
            query_emb = text_features.cpu().numpy()
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
            query_emb = query_emb.astype('float32')
            
            # Search the index
            logger.info(f"Performing image search for: {visual_query}")
            distances, indices = self.image_index.search(query_emb, k*2)  # Get more results for deduplication
            
            # Process results - group by place_id to avoid duplication
            place_scores = {}
            for idx, score in zip(indices[0], distances[0]):
                if idx < len(self.image_metadata):
                    item = self.image_metadata[idx]
                    place_id = item.get('place_id')
                    
                    if place_id not in place_scores or score > place_scores[place_id]['score']:
                        place_scores[place_id] = {
                            'score': float(score),
                            'image_url': item.get('image_url', ''),
                            'name': item.get('name', ''),
                            'neighborhood': item.get('neighborhood', ''),
                            'tags': item.get('tags', ''),
                            'short_description': item.get('short_description', '')
                        }
            
            # Convert to list and sort
            results = []
            for place_id, data in sorted(place_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:k]:
                results.append({
                    'place_id': place_id,
                    'name': data['name'],
                    'neighborhood': data['neighborhood'],
                    'tags': data['tags'],
                    'short_description': data['short_description'],
                    'score': data['score'],
                    'image_url': data['image_url'],
                    'source': 'image'
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Error in image search: {e}")
            return []
    
    def rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Use LLM to rerank results based on relevance to query."""
        if not self.use_llm or not results:
            return results
            
        try:
            # Format places for the LLM
            places_text = ""
            for i, result in enumerate(results):
                places_text += f"{i+1}. {result['name']} ({result['neighborhood']}) - {result['tags']} - {result['short_description']}\n"
            
            # Get LLM ranking
            scores_text = self.ranking_chain.invoke({
                "query": query,
                "places": places_text
            })
            
            # Parse scores
            scores = []
            try:
                scores = [float(s.strip()) for s in scores_text.split(',')]
                
                # Ensure we have the right number of scores
                if len(scores) != len(results):
                    logger.warning(f"Ranking returned {len(scores)} scores for {len(results)} results")
                    scores = scores[:len(results)] if len(scores) > len(results) else scores + [0] * (len(results) - len(scores))
            except Exception as e:
                logger.error(f"Error parsing ranking scores: {e}")
                return results  # Return original results if parsing fails
            
            # Apply scores and resort
            for i, score in enumerate(scores):
                results[i]['llm_score'] = score
                # Combine with original score (weighted average)
                results[i]['combined_score'] = 0.3 * results[i]['score'] + 0.7 * (score / 100.0)
            
            # Sort by combined score
            results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return results  # Return original results if reranking fails
    
    def search(
        self, 
        query: str, 
        k: int = 10, 
        use_text: bool = True,
        use_images: bool = True,
        enhance: bool = True,
        explain: bool = True,
        rerank: bool = True
    ) -> Dict:
        """
        Perform a complete vibe search using all available modalities.
        
        Args:
            query: User's search query
            k: Number of results to return
            use_text: Whether to use text search
            use_images: Whether to use image search
            enhance: Whether to enhance the query with LLM
            explain: Whether to explain results with LLM
            rerank: Whether to rerank results with LLM
            
        Returns:
            Dictionary with search results and metadata
        """
        logger.info(f"VibeSearch for: {query}")
        
        # Extract neighborhoods if mentioned
        neighborhoods = self._extract_neighborhoods(query)
        
        # Enhance query if requested
        original_query = query
        if enhance and self.use_llm:
            query = self.enhance_query(query)
        
        # Collect results from all sources
        all_results = []
        
        # Text search
        if use_text and self.text_search_available:
            text_results = self.text_search(query, neighborhoods=neighborhoods, k=k*2)
            for result in text_results:
                all_results.append(result)
        
        # Image search (with shorter query to avoid token limit errors)
        if use_images and self.image_search_available:
            image_results = self.image_search(query[:50], k=k)
            
            # Add unique places from image search
            existing_place_ids = {r['place_id'] for r in all_results}
            for result in image_results:
                if result['place_id'] not in existing_place_ids:
                    all_results.append(result)
                    existing_place_ids.add(result['place_id'])
        
        # Sort by score and take top k
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply LLM reranking if requested
        if rerank and self.use_llm and all_results:
            all_results = self.rerank_results(original_query, all_results)
        
        # Take top k
        final_results = all_results[:k]
        
        # Generate explanations if requested
        if explain and final_results:
            for result in tqdm(final_results, desc="Generating explanations"):
                result['match_reason'] = self.explain_match(original_query, result)
        
        # Add image URLs if available
        for result in final_results:
            place_id = result.get('place_id')
            if place_id in self.place_mapping and self.place_mapping[place_id]['images']:
                images = self.place_mapping[place_id]['images']
                result['image_urls'] = [img['image_url'] for img in images[:3]]  # Up to 3 images
        
        # Build response
        response = {
            "original_query": original_query,
            "processed_query": query if query != original_query else original_query,
            "neighborhoods": neighborhoods,
            "results": final_results,
            "result_count": len(final_results),
            "text_search_used": use_text and self.text_search_available,
            "image_search_used": use_images and self.image_search_available
        }
        
        return response

if __name__ == "__main__":
    import pickle
    import argparse
    parser = argparse.ArgumentParser(description="VibeSearch - multi-modal place finder")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM enhancements")
    parser.add_argument("--text-only", action="store_true", help="Use only text search")
    parser.add_argument("--image-only", action="store_true", help="Use only image search")
    parser.add_argument("--k", type=int, default=10, help="Number of results to return")
    args = parser.parse_args()
    
    # Create the search system
    searcher = VibeSearch(use_llm=not args.no_llm)
    
    # Interactive search loop
    while True:
        query = input("\nEnter search query (or 'exit' to quit): ")
        if query.lower() in ('exit', 'quit', 'q'):
            break
            
        # Handle search mode
        use_text = not args.image_only
        use_images = not args.text_only
        
        # Perform search
        results = searcher.search(
            query, 
            k=args.k, 
            use_text=use_text, 
            use_images=use_images,
            enhance=not args.no_llm,
            explain=not args.no_llm,
            rerank=not args.no_llm
        )
            
        # Display results
        print(f"\n--- Vibe Search Results ---")
        print(f"Query: {results['original_query']}")
        if results['processed_query'] != results['original_query']:
            print(f"Enhanced query: {results['processed_query']}")
        
        if results['neighborhoods']:
            print(f"Detected neighborhoods: {', '.join(results['neighborhoods'])}")
            
        print(f"Found {results['result_count']} results")
        print(f"Search modes: Text={results['text_search_used']}, Image={results['image_search_used']}")
        print()
        
        if not results['results']:
            print("No matching places found.")
        else:
            for i, result in enumerate(results['results'], 1):
                print(f"{i}. {result['name']} ({result['neighborhood']})")
                
                if 'tags' in result and result['tags']:
                    print(f"   Type: {result['tags']}")
                    
                if 'short_description' in result and result['short_description']:
                    print(f"   Description: {result['short_description']}")
                    
                print(f"   Match reason: {result.get('match_reason', 'Good semantic match')}")
                
                if 'image_urls' in result and result['image_urls']:
                    print(f"   Images: {len(result['image_urls'])} available")
                elif 'image_url' in result and result['image_url']:
                    print(f"   Image: {result['image_url']}")
                    
                if 'combined_score' in result:
                    print(f"   Score: {result.get('combined_score', 0):.3f}")
                else:
                    print(f"   Score: {result.get('score', 0):.3f}")
                
                print(f"   Source: {result.get('source', 'combined')}")
                print()