# retrieval/vibe_search.py
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import faiss
import torch
import pickle
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
# sys.path.append(str(Path(__file__).resolve().parent.parent))
# from ingestion.metadata_utils import load_and_merge_metadata # Assuming this is correctly located

class VibeSearch:
    """
    Integrated multi-modal vibe search system combining text and image understanding.
    """
    
    def __init__(
        self,
        text_model_name: str = "BAAI/bge-large-en-v1.5",  
        text_index_path: str = "ingestion/text_index.faiss",
        text_meta_path: str = "ingestion/text_metadata.pkl",
        image_index_path: str = "ingestion/image_index.faiss",
        image_meta_path: str = "ingestion/image_metadata.pkl",
        use_llm: bool = True, # This flag determines if LLM components are LOADED
        llm_type: str = "ollama",
        llm_model: str = "mistral:7b-instruct-v0.3-q4_K_M"
    ):
        logger.info(f"Initializing VibeSearch with use_llm={use_llm}")
        
        self.text_search_available = self._init_text_search(text_model_name, text_index_path, text_meta_path)
        self.image_search_available = self._init_image_search(image_index_path, image_meta_path)
            
        self.use_llm_globally = use_llm # Renamed to avoid confusion with method parameter
        self.llm = None
        self.enhance_chain = None
        self.explain_chain = None
        self.ranking_chain = None

        if self.use_llm_globally:
            self._init_llm(llm_type, llm_model)
        else:
            logger.info("LLM components will not be initialized as use_llm_globally is False.")
        
        self._build_place_mapping()
        self._build_neighborhood_mapping()
            
        logger.info(f"VibeSearch initialized (Text: {self.text_search_available}, Image: {self.image_search_available}, LLM Loaded: {self.use_llm_globally and self.llm is not None})")
    
    def _init_text_search(self, model_name, index_path, meta_path):
        try:
            base_dir = Path(__file__).resolve().parent.parent
            index_file = base_dir / index_path
            meta_file = base_dir / meta_path
            
            logger.info(f"Loading text search index from {index_file}")
            self.text_index = faiss.read_index(str(index_file))
            
            logger.info(f"Loading text metadata from {meta_file}")
            with open(meta_file, "rb") as f:
                self.text_metadata = pickle.load(f)
            
            mps_device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
            logger.info(f"Using device: {mps_device} for text embedder")
            self.text_embedder = SentenceTransformer(model_name, device=mps_device)
            self.text_embedding_dim = self.text_embedder.get_sentence_embedding_dimension()
            
            if self.text_index.d != self.text_embedding_dim:
                logger.warning(f"Text dimension mismatch: index={self.text_index.d}, model={self.text_embedding_dim}")
            
            logger.info(f"Text search initialized with {len(self.text_metadata)} places")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize text search: {e}", exc_info=True)
            return False
    
    def _init_image_search(self, index_path, meta_path):
        try:
            base_dir = Path(__file__).resolve().parent.parent
            index_file = base_dir / index_path
            meta_file = base_dir / meta_path
            
            logger.info(f"Loading image search index from {index_file}")
            self.image_index = faiss.read_index(str(index_file))
            
            logger.info(f"Loading image metadata from {meta_file}")
            with open(meta_file, "rb") as f:
                self.image_metadata = pickle.load(f)
            
            self.clip_model = None # Lazy load CLIP model
            self.clip_processor = None
            self.clip_device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


            logger.info(f"Image search initialized with {len(self.image_metadata)} images. CLIP will load on first use on device: {self.clip_device}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize image search: {e}", exc_info=True)
            return False
    
    def _ensure_clip_loaded(self):
        """Loads CLIP model and processor if not already loaded."""
        if self.clip_model is None or self.clip_processor is None:
            logger.info("Lazy loading CLIP model for image search...")
            from transformers import CLIPProcessor, CLIPModel # Local import
            model_name = "openai/clip-vit-base-patch32"
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.clip_device)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model.eval() # Set to evaluation mode
            logger.info(f"CLIP model loaded on {self.clip_device}")

    def _init_llm(self, llm_type, model_name):
        try:
            # Langchain imports
            try: # New imports
                from langchain_community.llms import Ollama, HuggingFaceTextGenInference
                from langchain_core.prompts import PromptTemplate
                from langchain_core.output_parsers import StrOutputParser
                from langchain_core.runnables import RunnablePassthrough
            except ImportError: # Fallback to old imports
                from langchain.llms import Ollama, HuggingFaceTextGenInference # type: ignore
                from langchain.prompts import PromptTemplate # type: ignore
                from langchain.schema.output_parser import StrOutputParser # type: ignore
                from langchain.schema.runnable import RunnablePassthrough # type: ignore
            
            if llm_type == "ollama":
                self.llm = Ollama(model=model_name)
                logger.info(f"Initialized Ollama with model {model_name}")
            elif llm_type == "huggingface": # Assuming a TGI endpoint
                self.llm = HuggingFaceTextGenInference(
                    inference_server_url="http://localhost:8080/", # Make this configurable
                    max_new_tokens=150, # Reduced for faster explanations/enhancements
                    temperature=0.6,
                    top_p=0.9,
                    timeout=120,
                )
                logger.info("Initialized HuggingFace TGI endpoint")
            else:
                logger.error(f"Unsupported LLM type: {llm_type}. LLM will not be used.")
                self.use_llm_globally = False # Disable LLM usage if type is wrong
                return False

            enhance_template = """
            You are an AI assistant helping with a search system for places in New York City.
            USER QUERY: {query}
            Your task is to:
            1. Understand the user's intent about what places they want to visit.
            2. Create an enhanced search query that includes specific areas, types of places, and ordered itinerary if requested.
            3. If they ask for a chronological order or itinerary, explicitly include this in your query.
            Your enhanced query should be detailed and include SPECIFIC LOCATIONS and neighborhoods if implied.
            Output ONLY the enhanced query.
            ENHANCED QUERY:
            """
            self.enhance_chain = (
                {"query": RunnablePassthrough()} 
                | PromptTemplate.from_template(enhance_template)
                | self.llm
                | StrOutputParser()
            )
            
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
            
            ranking_template = """
            You are helping rank search results based on relevance to a query.
            USER QUERY: {query}
            For each place, assign a relevance score from 0-100 based on how well it matches the query:
            {places}
            Output ONLY a comma-separated list of scores in the same order as the places. Example: 85,72,95,63,41
            SCORES:
            """
            self.ranking_chain = (
                {"query": RunnablePassthrough(), "places": RunnablePassthrough()} 
                | PromptTemplate.from_template(ranking_template)
                | self.llm
                | StrOutputParser()
            )
            logger.info("LLM chains initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LLM components: {e}", exc_info=True)
            self.use_llm_globally = False # Disable LLM if initialization fails
            return False

    def _build_place_mapping(self):
        self.place_mapping = {}
        if hasattr(self, 'text_metadata') and self.text_metadata:
            for item in self.text_metadata:
                place_id = item.get('place_id')
                if place_id:
                    if place_id not in self.place_mapping:
                        self.place_mapping[place_id] = {'place_info': item, 'images': []}
        
        if hasattr(self, 'image_metadata') and self.image_metadata:
            for idx, item in enumerate(self.image_metadata):
                place_id = item.get('place_id')
                if place_id:
                    if place_id not in self.place_mapping:
                        self.place_mapping[place_id] = {
                            'place_info': {k: item.get(k, '') for k in ['place_id', 'name', 'neighborhood', 'tags', 'short_description']},
                            'images': []
                        }
                    self.place_mapping[place_id]['images'].append({'image_url': item.get('image_url', ''), 'index': idx})
        logger.info(f"Built place mapping with {len(self.place_mapping)} places")

    def _build_neighborhood_mapping(self):
        self.neighborhoods = set()
        self.neighborhood_to_places = {}
        for place_id, data in self.place_mapping.items():
            neighborhood_value = data['place_info'].get('neighborhood', '')
            neighborhood = str(neighborhood_value).strip().lower() if neighborhood_value else ''
            if neighborhood:
                self.neighborhoods.add(neighborhood)
                self.neighborhood_to_places.setdefault(neighborhood, []).append(place_id)
        logger.info(f"Found {len(self.neighborhoods)} unique neighborhoods")

    def enhance_query(self, query: str) -> str:
        if not self.use_llm_globally or not self.enhance_chain:
            logger.info("Skipping query enhancement (LLM not enabled/initialized).")
            return query
        try:
            enhanced = self.enhance_chain.invoke(query)
            logger.info(f"Enhanced query: {enhanced}")
            return enhanced.strip()
        except Exception as e:
            logger.error(f"Error enhancing query: {e}", exc_info=True)
            return query

    def explain_match(self, query: str, place: Dict) -> str:
        if not self.use_llm_globally or not self.explain_chain:
            # logger.info("Skipping LLM explanation (LLM not enabled/initialized). Using simple explanation.")
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
            logger.error(f"Error generating LLM explanation: {e}", exc_info=True)
            return self._simple_explanation(place)

    def _simple_explanation(self, place: Dict) -> str:
        name = place.get('name', 'This place')
        neighborhood = place.get('neighborhood', 'an area')
        description = place.get('short_description', '')
        tags = place.get('tags', '')
        
        if description:
            return f"{name} in {neighborhood} (features: {description})."
        elif tags:
            return f"{name} in {neighborhood} (tags: {tags})."
        return f"{name} in {neighborhood} is a potential match."

    def _extract_neighborhoods(self, query: str) -> List[str]:
        query_lower = query.lower()
        mentioned = [n for n in self.neighborhoods if n and n in query_lower] # Ensure n is not empty
        
        common_areas = {
            "downtown": ["lower manhattan", "financial district", "tribeca", "soho"],
            "midtown": ["midtown", "times square", "theater district"],
            "uptown": ["upper east side", "upper west side", "harlem"],
        }
        for area, neighborhoods_in_area in common_areas.items():
            if area in query_lower:
                for n_area in neighborhoods_in_area:
                    if n_area in self.neighborhood_to_places and n_area not in mentioned:
                        mentioned.append(n_area)
        return list(set(mentioned)) # Return unique neighborhoods

    def text_search(self, query: str, neighborhoods: List[str] = None, k: int = 20) -> List[Dict]:
        if not self.text_search_available:
            logger.error("Text search not available, returning empty list.")
            return []
        try:
            query_emb = self.text_embedder.encode([query], normalize_embeddings=True)
            query_emb = np.array(query_emb).astype('float32')
            
            if query_emb.shape[1] != self.text_index.d:
                logger.error(f"Query embedding dim ({query_emb.shape[1]}) != index dim ({self.text_index.d}).")
                return []
            
            logger.info(f"Performing text search for: '{query}' with k={k}")
            # Neighborhood filtering logic can be complex, simplified here for brevity
            # For a real app, consider if filtering should happen before or after initial search
            
            distances, indices = self.text_index.search(query_emb, k * 2) # Fetch more for diversity
            
            results = []
            seen_place_ids = set()
            for idx, score in zip(indices[0], distances[0]):
                if idx < len(self.text_metadata):
                    place_data = self.text_metadata[idx]
                    place_id = place_data.get('place_id')
                    if place_id in seen_place_ids:
                        continue # Avoid duplicates from text search itself
                    seen_place_ids.add(place_id)

                    # Filter by neighborhood if specified
                    if neighborhoods:
                        place_neighborhood = str(place_data.get('neighborhood', '')).lower()
                        if not any(n in place_neighborhood for n in neighborhoods):
                            continue # Skip if not in desired neighborhoods

                    results.append({
                        'place_id': place_id,
                        'name': place_data.get('name', ''),
                        'neighborhood': place_data.get('neighborhood', ''),
                        'tags': place_data.get('tags', ''),
                        'short_description': place_data.get('short_description', ''),
                        'score': float(score),
                        'source': 'text'
                    })
                    if len(results) >= k:
                        break 
            return results
        except Exception as e:
            logger.error(f"Error in text search: {e}", exc_info=True)
            return []

    def image_search(self, query: str, k: int = 10) -> List[Dict]:
        if not self.image_search_available:
            logger.error("Image search not available, returning empty list.")
            return []
        try:
            self._ensure_clip_loaded() # Make sure CLIP model is loaded

            short_query = ' '.join(query.split()[:15]) # Slightly longer for better context
            visual_query = f"A photo of: {short_query}"
            
            inputs = self.clip_processor(text=[visual_query], return_tensors="pt", truncation=True, padding=True).to(self.clip_device)
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            
            query_emb = text_features.cpu().numpy()
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
            query_emb = query_emb.astype('float32')
            
            logger.info(f"Performing image search for: '{visual_query}' with k={k}")
            distances, indices = self.image_index.search(query_emb, k * 2) 
            
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
            
            results = []
            for place_id, data in sorted(place_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:k]:
                results.append({**data, 'place_id': place_id, 'source': 'image'})
            return results
        except Exception as e:
            logger.error(f"Error in image search: {e}", exc_info=True)
            return []

    def rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        if not self.use_llm_globally or not results or not self.ranking_chain:
            logger.info("Skipping LLM reranking (LLM not enabled/initialized or no results).")
            return results
        try:
            places_text = ""
            for i, result in enumerate(results):
                places_text += f"{i+1}. {result.get('name','N/A')} ({result.get('neighborhood','N/A')}) - Tags: {result.get('tags','N/A')} - Desc: {result.get('short_description','N/A')}\n"
            
            scores_text = self.ranking_chain.invoke({"query": query, "places": places_text})
            
            parsed_scores = []
            try:
                raw_scores = [s.strip() for s in scores_text.split(',')]
                for s in raw_scores:
                    if s: # Ensure not empty string
                        parsed_scores.append(float(s))

                if len(parsed_scores) != len(results):
                    logger.warning(f"LLM Reranking score count mismatch: got {len(parsed_scores)}, expected {len(results)}. Scores: '{scores_text}'")
                    # Pad with 0 or use original scores if mismatch is too large
                    # For simplicity, we'll use original scores if count is very off
                    if abs(len(parsed_scores) - len(results)) > len(results) / 2:
                         return results
                    parsed_scores = parsed_scores[:len(results)] + [0.0] * (len(results) - len(parsed_scores))

            except ValueError as ve:
                logger.error(f"Error parsing LLM ranking scores: '{scores_text}'. Error: {ve}", exc_info=True)
                return results 
            
            for i, score in enumerate(parsed_scores):
                if i < len(results): # Ensure we don't go out of bounds
                    results[i]['llm_score'] = score
                    results[i]['combined_score'] = (0.4 * results[i]['score']) + (0.6 * (score / 100.0))
            
            results.sort(key=lambda x: x.get('combined_score', x['score']), reverse=True)
            logger.info("Results reranked by LLM.")
            return results
        except Exception as e:
            logger.error(f"Error reranking results: {e}", exc_info=True)
            return results

    def search(
        self, 
        query: str, 
        k: int = 10, 
        use_text: bool = True,
        use_images: bool = True,
        enhance: bool = True, # User's preference from frontend for enhancing
        explain: bool = True, # User's preference from frontend for explaining
        rerank: bool = True,  # User's preference from frontend for reranking
        quick_search: bool = False # If True, bypasses LLM operations for this request
    ) -> Dict:
        logger.info(f"VibeSearch: Query='{query}', QuickSearch={quick_search}, EnhancePref={enhance}, ExplainPref={explain}, RerankPref={rerank}")
        
        # Determine if LLM features should be practically used for THIS request
        # LLM must be globally enabled (self.use_llm_globally) AND quick_search must be False.
        allow_llm_operations_for_this_request = self.use_llm_globally and not quick_search

        original_query = query
        processed_query = query 

        if enhance and allow_llm_operations_for_this_request:
            logger.info("Attempting query enhancement...")
            processed_query = self.enhance_query(original_query)
        else:
            logger.info("Skipping query enhancement (QuickSearch enabled, LLM disabled globally, or Enhance preference is False).")
        
        neighborhoods = self._extract_neighborhoods(processed_query) # Use processed query for neighborhood extraction
        
        all_results = []
        seen_place_ids = set()

        if use_text and self.text_search_available:
            text_results = self.text_search(processed_query, neighborhoods=neighborhoods, k=k*2) # Fetch more for merging
            for res in text_results:
                if res['place_id'] not in seen_place_ids:
                    all_results.append(res)
                    seen_place_ids.add(res['place_id'])
        
        if use_images and self.image_search_available:
            # Use processed_query for image search as well, but keep it concise for CLIP
            image_query_text = ' '.join(processed_query.split()[:15])
            image_results = self.image_search(image_query_text, k=k*2) # Fetch more for merging
            for res in image_results:
                if res['place_id'] not in seen_place_ids:
                    all_results.append(res) # Add if not already present from text search
                    seen_place_ids.add(res['place_id'])
                else: # If already present, update score if image score is higher (or average, etc.)
                    for existing_res in all_results:
                        if existing_res['place_id'] == res['place_id']:
                            existing_res['score'] = max(existing_res['score'], res['score']) # Example: take max score
                            if 'image_url' not in existing_res and 'image_url' in res : # Add image if text result didn't have one
                                existing_res['image_url'] = res['image_url']
                            break


        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        if rerank and allow_llm_operations_for_this_request and all_results:
            logger.info("Attempting LLM reranking...")
            all_results = self.rerank_results(original_query, all_results[:k*2]) # Rerank a larger pool before final cut
        else:
            logger.info("Skipping LLM reranking (QuickSearch enabled, LLM disabled globally, Rerank preference is False, or no results).")
        
        final_results = all_results[:k]
        
        if explain and final_results:
            logger.info("Attempting explanations...")
            for result in tqdm(final_results, desc="Generating explanations", disable=not allow_llm_operations_for_this_request):
                if allow_llm_operations_for_this_request:
                    result['match_reason'] = self.explain_match(original_query, result)
                else:
                    result['match_reason'] = self._simple_explanation(result) 
        else:
             logger.info("Skipping explanations (QuickSearch enabled, LLM disabled globally, Explain preference is False, or no results).")
             # Add simple explanations if explain was false but quick_search was also false (meaning LLM could have run but user unchecked explain)
             if not allow_llm_operations_for_this_request and not quick_search and explain and final_results:
                 for result in final_results:
                     result['match_reason'] = self._simple_explanation(result)


        for result in final_results:
            place_id = result.get('place_id')
            if place_id in self.place_mapping and self.place_mapping[place_id]['images']:
                images_data = self.place_mapping[place_id]['images']
                result['image_urls'] = [img['image_url'] for img in images_data[:3]] 
        
        response = {
            "original_query": original_query,
            "processed_query": processed_query,
            "neighborhoods": neighborhoods,
            "results": final_results,
            "result_count": len(final_results),
            "text_search_used": use_text and self.text_search_available,
            "image_search_used": use_images and self.image_search_available
        }
        return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VibeSearch - multi-modal place finder")
    parser.add_argument("--no-llm-init", action="store_true", help="Do not initialize LLM components at startup.")
    parser.add_argument("--text-only", action="store_true", help="Use only text search")
    parser.add_argument("--image-only", action="store_true", help="Use only image search")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()
    
    # Initialize VibeSearch based on --no-llm-init
    # This affects if LLM components are LOADED.
    search_system_use_llm = not args.no_llm_init 
    searcher = VibeSearch(use_llm=search_system_use_llm)
    
    while True:
        raw_query = input(f"\nEnter search query (LLM loaded: {search_system_use_llm}) (or 'exit' to quit): ")
        if raw_query.lower() in ('exit', 'quit', 'q'):
            break
        
        # Simulate frontend options for CLI testing
        # For CLI, let's assume quick_search is false unless we add another arg for it.
        # The enhance, explain, rerank flags here are for testing the VibeSearch.search method's parameters.
        cli_quick_search = False # Default for CLI, could be another arg
        cli_enhance = True
        cli_explain = True
        cli_rerank = True

        if cli_quick_search:
            cli_enhance = False
            cli_explain = False
            cli_rerank = False
            print("CLI: Quick Search is ON (LLM operations will be skipped for this search).")


        use_text_cli = not args.image_only
        use_images_cli = not args.text_only
        
        results = searcher.search(
            raw_query, 
            k=args.k, 
            use_text=use_text_cli, 
            use_images=use_images_cli,
            enhance=cli_enhance, # User's preference for this search
            explain=cli_explain, # User's preference for this search
            rerank=cli_rerank,   # User's preference for this search
            quick_search=cli_quick_search # Overrides LLM usage for this search
        )
            
        print(f"\n--- Vibe Search CLI Results ---")
        print(f"Original Query: {results['original_query']}")
        if results['processed_query'] != results['original_query']:
            print(f"Processed Query: {results['processed_query']}")
        if results['neighborhoods']:
            print(f"Detected neighborhoods: {', '.join(results['neighborhoods'])}")
        print(f"Found {results['result_count']} results.")
        print(f"Search modes used: Text={results['text_search_used']}, Image={results['image_search_used']}")
        print(f"Quick Search active for this query: {cli_quick_search}")
        print(f"LLM operations (enhance, explain, rerank) attempted if not quick_search and preferences were True.")
        print()
        
        if not results['results']:
            print("No matching places found.")
        else:
            for i, r_item in enumerate(results['results'], 1):
                print(f"{i}. {r_item.get('name', 'N/A')} ({r_item.get('neighborhood', 'N/A')})")
                if 'tags' in r_item and r_item['tags']: print(f"   Type: {r_item['tags']}")
                if 'short_description' in r_item and r_item['short_description']: print(f"   Desc: {r_item['short_description']}")
                print(f"   Reason: {r_item.get('match_reason', 'N/A')}")
                if 'image_urls' in r_item and r_item['image_urls']: print(f"   Images: {len(r_item['image_urls'])} available")
                elif 'image_url' in r_item and r_item['image_url']: print(f"   Image: {r_item['image_url']}")
                score_key = 'combined_score' if 'combined_score' in r_item else 'score'
                print(f"   Score: {r_item.get(score_key, 0):.3f} (Source: {r_item.get('source', 'N/A')})")
                print()