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
    Dynamic query expansion using a language model:
    1. Extracts structured components (filters, intents, etc.)
    2. Identifies the underlying user need beyond literal words
    3. Generates expanded hypothetical descriptions (HyDE)
    4. Considers implicit context and intent
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

    def extract_structured_info(self, query: str) -> Dict[str, str]:
        """
        Extract structured information from query (intent, filters, search type)
        """
        structure_prompt = (
            "You are a query understanding expert. Analyze this search query and extract: \n"
            "1. Main intent - the core purpose (finding food, entertainment, dating spot, etc.)\n"
            "2. Filters - specific constraints (neighborhood, cuisine, price, etc.)\n"
            "3. Vibe words - descriptive terms about atmosphere/feeling\n"
            "4. Activities - what the user wants to do\n"
            "5. Time context - any time of day/week mentions\n\n"
            f"Query: '{query}'\n\n"
            "Format your response as a Python dictionary with these keys. Include only what's explicitly or strongly implied:"
        )
        
        try:
            result = self._generate_text(structure_prompt, max_new_tokens=150, temperature=0.1)
            logger.info(f"Structure extraction result: {result}")
            
            # Try to extract a dictionary using regex
            dict_pattern = r'{[\s\S]*}'
            dict_match = re.search(dict_pattern, result)
            
            if dict_match:
                dict_str = dict_match.group(0)
                # Clean up the dictionary string
                dict_str = dict_str.replace("'", '"')  # Replace single quotes with double quotes
                dict_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*):(\s*)', r'\1"\2":\3', dict_str)  # Add quotes to keys
                
                try:
                    # Try to parse as JSON
                    return json.loads(dict_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON: {e}")
                    # Fallback to a simpler approach
                    info_dict = {}
                    for key in ["main_intent", "filters", "vibe_words", "activities", "time_context"]:
                        pattern = rf'"{key}":\s*"([^"]*)"'
                        match = re.search(pattern, dict_str)
                        if match:
                            info_dict[key] = match.group(1).lower()
                    return info_dict
            
            # If no dictionary pattern found, extract line by line
            info_dict = {}
            lines = result.split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().strip("'\"").lower().replace(" ", "_")
                    value = value.strip().strip("'\"").lower()
                    if key and value:
                        info_dict[key] = value
            
            return info_dict
            
        except Exception as e:
            logger.warning(f"Failed to extract structured info: {e}")
            return {}

    def interpret_intent(self, query: str) -> str:
        """
        Go beyond literal words to understand what the user really wants.
        """
        intent_prompt = (
            "You are an expert in understanding human search intentions. "
            "For this query, explain what the user is REALLY looking for beyond the literal words. "
            "Consider both explicit and implicit needs. "
            f"Query: '{query}'\n\n"
            "What is the user actually looking for? Be specific but concise (30-50 words):"
        )
        
        try:
            intent = self._generate_text(intent_prompt, max_new_tokens=100, temperature=0.3)
            logger.info(f"Intent interpretation: {intent[:100]}...")
            return intent
        except Exception as e:
            logger.warning(f"Failed to interpret intent: {e}")
            return query

    def generate_hyde(
        self,
        query: str,
        structured_info: Dict[str, str],
        weather: Optional[str] = None,
        time_context: Optional[str] = None,
    ) -> str:
        """
        Generate a hypothetical document paragraph that captures the user's intent.
        Now uses structured information to create more targeted hypothetical documents.
        """
        # Extract relevant information
        main_intent = structured_info.get("main_intent", "")
        vibe_words = structured_info.get("vibe_words", "")
        activities = structured_info.get("activities", "")
        
        # Add context if provided
        context = ""
        if weather:
            context += f" The weather is {weather}."
        if time_context:
            context += f" It's {time_context}."
        elif structured_info.get("time_context"):
            context += f" It's {structured_info.get('time_context')}."

        prompt = (
            "You are a NYC expert describing the perfect place for someone.\n\n"
            f"Their request: '{query}'\n"
            f"What they're looking for: {main_intent}\n"
            f"Vibe they want: {vibe_words}\n"
            f"Activities: {activities}\n"
            f"Context: {context}\n\n"
            "Write a detailed paragraph (80-120 words) describing the perfect place that would "
            "match their needs. Include specific details about the atmosphere, typical clientele, "
            "activities available, and the overall experience. Your description should capture "
            "the essence of what they're searching for."
        )
        
        logger.info(f"Generating HyDE for query: {query}")
        hyde_doc = self._generate_text(prompt, max_new_tokens=200, temperature=0.7)
        logger.info(f"Generated HyDE: {hyde_doc[:100]}...")
        
        return hyde_doc
    
    def generate_diverse_queries(self, query: str, structured_info: Dict[str, str], num_queries: int = 3) -> List[str]:
        """
        Generate diverse alternative search queries to capture different aspects of the intent.
        """
        prompt = (
            "You are a search expert helping expand a query into multiple perspectives.\n\n"
            f"Original query: '{query}'\n"
            f"User intention: {structured_info.get('main_intent', '')}\n\n"
            f"Generate {num_queries} alternative search queries that capture different aspects of "
            f"what the user is looking for. Make each one focus on a different facet or interpretation "
            f"of the user's intent. Each query should be between 3-10 words and be different from each other."
            f"\n\nAlternative queries (one per line):"
        )
        
        try:
            result = self._generate_text(prompt, max_new_tokens=150, temperature=0.8)
            # Split by newlines and clean
            alternative_queries = [q.strip() for q in result.split('\n') if q.strip()]
            # Take only the requested number, removing any numbering
            cleaned_queries = []
            for q in alternative_queries[:num_queries]:
                # Remove numbering like "1. " or "1) " if present
                cleaned = re.sub(r'^\d+[\.\)\-]\s*', '', q)
                cleaned_queries.append(cleaned)
                
            logger.info(f"Generated alternative queries: {cleaned_queries}")
            return cleaned_queries
        except Exception as e:
            logger.warning(f"Failed to generate alternative queries: {e}")
            return [query]  # Return original if failed

    def process_query(
        self, 
        query: str,
        weather: Optional[str] = None,
        time_context: Optional[str] = None,
        generate_alternatives: bool = False
    ) -> Dict:
        """
        Complete query processing pipeline.
        Returns a dictionary with all query understanding components.
        """
        # Extract structured information
        structured_info = self.extract_structured_info(query)
        logger.info(f"Extracted structured info: {structured_info}")
        
        # Interpret deeper intent
        intent = self.interpret_intent(query)
        logger.info(f"Interpreted intent: {intent}")
        
        # Generate HyDE document
        hyde_doc = self.generate_hyde(query, structured_info, weather, time_context)
        
        # Optionally generate alternative queries
        alt_queries = []
        if generate_alternatives:
            alt_queries = self.generate_diverse_queries(query, structured_info)
        
        # Combine everything into a comprehensive query understanding
        query_understanding = {
            "original_query": query,
            "structured_info": structured_info,
            "interpreted_intent": intent,
            "hyde_document": hyde_doc,
            "alternative_queries": alt_queries
        }
        
        return query_understanding

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
            
        # Option for generating alternatives
        gen_alt = input("Generate alternative queries? (y/n, default=n): ").lower() == 'y'
        
        # Optional context
        weather = input("Weather context (optional, press Enter to skip): ").strip() or None
        time = input("Time context (optional, press Enter to skip): ").strip() or None
        
        results = expander.process_query(q, weather, time, gen_alt)
        
        print("\n--- Query Understanding Results ---")
        print(f"Original query: {results['original_query']}")
        
        print("\n--- Structured Information ---")
        for k, v in results['structured_info'].items():
            print(f"{k}: {v}")
            
        print("\n--- Interpreted Intent ---")
        print(results['interpreted_intent'])
        
        print("\n--- Generated HyDE Document ---")
        print(results['hyde_document'])
        
        if results['alternative_queries']:
            print("\n--- Alternative Queries ---")
            for i, alt in enumerate(results['alternative_queries'], 1):
                print(f"{i}. {alt}")
        
        print("\n" + "-"*60)