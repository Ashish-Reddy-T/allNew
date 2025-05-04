# generation/generate_summary.py
import sys, os
# allow importing from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from retrieval.search_index import Retriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Justifier:
    """
    Generates natural-language explanations for why each place matches the user's query.
    Uses TinyLlama for concise, on-topic justifications.
    """
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: torch.device = None
    ):
        # Determine device
        if device:
            self.device = device
        else:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        logger.info(f"Justifier using device: {self.device}")

        # Load tokenizer & model
        logger.info(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type in ('cuda','mps') else torch.float32,
            device_map='auto' if self.device.type!='cpu' else None,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("Justifier model loaded successfully")

    def generate_explanation(
        self,
        query: str,
        place: dict,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Given the original user query and a place metadata dict, produce a concise 1-2 sentence explanation.
        """
        name = place.get('name', 'This place')
        neighborhood = place.get('neighborhood', 'an area')
        desc = place.get('short_description', '')

        prompt = (
            f"You are an expert assistant recommending venues in New York City. "
            f"The user wants: '{query}'. "
            f"Consider the place: {name} located in {neighborhood}, described as: '{desc}'. "
            f"Write a concise (1-2 sentence) natural-language explanation why this place matches the user's request."
        )
        logger.info(f"Generating explanation for {name}")

        inputs = self.tokenizer(prompt, return_tensors='pt')
        if self.device.type != 'cpu':
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        logger.info("Explanation generation complete")
        return text


if __name__ == '__main__':
    # Clear caches
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Initialize retriever and justifier
    retriever = Retriever()
    query = input("Enter your search query: ")
    use_hyde = input("Use HyDE expansion? (y/n): ").strip().lower().startswith('y')
    results = retriever.search(query, k=5, use_hyde=use_hyde)

    justifier = Justifier()
    print("\nSearch Results with Explanations:\n")
    for i, place in enumerate(results, 1):
        explanation = justifier.generate_explanation(query, place)
        neighborhood = place.get('neighborhood', 'Unknown')
        print(f"{i}. {place['name']} ({neighborhood})")
        print(f"   -> {explanation}\n")
