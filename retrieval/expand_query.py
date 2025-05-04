# retrieval/expand_query.py
import logging
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Tuple, Optional, List
import re
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class QueryExpander:
    """
    Improved query expansion using a language model:
    1. Extracts filters like neighborhood, cuisine, etc.
    2. Generates expanded hypothetical descriptions (HyDE)
    3. Can handle context like weather, time, etc.
    """
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: Optional[str] = None,
    ):
        # Determine device
        if device:
            self.device = device
        else:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        logger.info(f"Using device: {self.device}")

        # Load tokenizer and model
        logger.info(f"Loading model {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device in ("cuda", "mps") else torch.float32,
            device_map="auto" if self.device != "cpu" else None,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")

    def _generate_text(
        self, 
        prompt: str, 
        max_new_tokens: int = 200, 
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the prompt from the output
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
            
        return text

    def extract_filters(self, query: str) -> Dict[str, str]:
        """
        Extract structured filters from query (neighborhood, place type, etc.)
        """
        filter_prompt = (
            "Given this search query, extract any filters like neighborhood, place type, or cuisine. "
            "Format your response as a Python dictionary. Only include filters that are explicitly mentioned. "
            f"Query: '{query}'\n\n"
            "Extracted filters:"
        )
        
        try:
            result = self._generate_text(filter_prompt, max_new_tokens=100, temperature=0.1)
            logger.info(f"Filter extraction result: {result}")
            
            # Try to extract a dictionary using regex
            dict_pattern = r'{[\s\S]*}'
            dict_match = re.search(dict_pattern, result)
            
            if dict_match:
                dict_str = dict_match.group(0)
                try:
                    # Try to parse as JSON
                    filter_dict = json.loads(dict_str)
                    return filter_dict
                except json.JSONDecodeError:
                    # Fallback to a simpler approach
                    filter_dict = {}
                    if "neighborhood" in result.lower():
                        match = re.search(r"'neighborhood':\s*'([^']*)'", result)
                        if match:
                            filter_dict["neighborhood"] = match.group(1).lower()
                    
                    if "place_type" in result.lower():
                        match = re.search(r"'place_type':\s*'([^']*)'", result)
                        if match:
                            filter_dict["place_type"] = match.group(1).lower()
                            
                    return filter_dict
            
            # If no dictionary pattern found, process line by line
            filter_dict = {}
            lines = result.split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().strip("'\"").lower()
                    value = value.strip().strip("'\"").lower()
                    if key and value:
                        filter_dict[key] = value
            
            return filter_dict
            
        except Exception as e:
            logger.warning(f"Failed to extract filters: {e}")
            return {}

    def generate_hyde(
        self,
        query: str,
        weather: Optional[str] = None,
        time_context: Optional[str] = None,
    ) -> str:
        """
        Generate a hypothetical document paragraph that captures the user's intent.
        """
        # Add context if provided
        context = ""
        if weather:
            context += f" The weather is {weather}."
        if time_context:
            context += f" It's {time_context}."

        prompt = (
            "You are a helpful assistant.\n"
            f"Based on the user's search query: '{query}'{context}, "
            "write a detailed paragraph describing the perfect place that would match this query. "
            "Include details about atmosphere, clientele, activities, and vibe. "
            "Write about 80-100 words."
        )
        
        logger.info(f"Generating HyDE for query: {query}")
        hyde_doc = self._generate_text(prompt, max_new_tokens=150)
        logger.info(f"Generated HyDE: {hyde_doc[:100]}...")
        
        return hyde_doc

    def process_query(self, query: str) -> Tuple[Dict[str, str], str]:
        """
        Complete query processing: extract filters and generate HyDE document
        Returns: (filters, expanded_query)
        """
        # Extract structured filters
        filters = self.extract_filters(query)
        logger.info(f"Extracted filters: {filters}")
        
        # Generate HyDE document
        hyde_doc = self.generate_hyde(query)
        
        return filters, hyde_doc

if __name__ == "__main__":
    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    expander = QueryExpander()
    
    while True:
        q = input("Enter query for expansion (or 'exit' to quit): ")
        if q.lower() in ('exit', 'quit', 'q'):
            break
            
        filters, hyde_doc = expander.process_query(q)
        
        print("\n--- Extracted Filters ---")
        for k, v in filters.items():
            print(f"{k}: {v}")
            
        print("\n--- Generated HyDE Document ---")
        print(hyde_doc)
        print("\n" + "-"*50)