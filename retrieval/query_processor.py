# retrieval/query_processor.py
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

class QueryProcessor:
    """
    Advanced query processing using LangChain with various LLM backends.
    
    This class provides:
    1. Query understanding and expansion
    2. Result explanation generation
    3. Support for multiple LLM backends
    """
    
    def __init__(self, llm_type: str = "ollama", model_name: str = "mistral"):
        """
        Initialize the query processor with the specified LLM.
        
        Args:
            llm_type: Type of LLM to use ('ollama', 'huggingface', etc.)
            model_name: Name of the model to use
        """
        logger.info(f"Initializing QueryProcessor with {llm_type}:{model_name}")
        
        try:
            # Import LangChain components
            try:
                from langchain_community.llms import Ollama, HuggingFaceTextGenInference
                from langchain_core.prompts import PromptTemplate
                from langchain_core.output_parsers import StrOutputParser
                from langchain_core.runnables import RunnablePassthrough
            except ImportError:
                # Fall back to older imports
                from langchain.llms import Ollama, HuggingFaceTextGenInference
                from langchain.prompts import PromptTemplate
                from langchain.schema.output_parser import StrOutputParser
                from langchain.schema.runnable import RunnablePassthrough
            
            # Initialize LLM based on type
            if llm_type == "ollama":
                self.llm = Ollama(model=model_name)
                logger.info(f"Initialized Ollama with model {model_name}")
            elif llm_type == "huggingface":
                self.llm = HuggingFaceTextGenInference(
                    inference_server_url="http://localhost:8080/",
                    max_new_tokens=512,
                    temperature=0.1,
                    timeout=120,
                )
                logger.info(f"Initialized HuggingFace inference endpoint")
            else:
                raise ValueError(f"Unsupported LLM type: {llm_type}")
            
            # Set up prompt templates
            self._setup_prompts()
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize QueryProcessor: {e}")
            self.initialized = False
    
    def _setup_prompts(self):
        """Set up the prompt templates for various tasks."""
        # Query enhancement prompt
        enhance_template = """
        You are a search expert helping to understand user queries about places and locations.
        
        User query: {query}
        
        Your task is to:
        1. Understand what the user is looking for
        2. Identify any implicit needs beyond the literal query
        3. Create an enhanced search query that's more likely to find relevant places
        
        Enhanced query:
        """
        
        self.enhance_chain = (
            {"query": RunnablePassthrough()} 
            | PromptTemplate.from_template(enhance_template)
            | self.llm
            | StrOutputParser()
        )
        
        # Result explanation prompt
        explain_template = """
        You are an expert recommender explaining why a place matches a user's query.
        
        User query: {query}
        Place name: {name}
        Place location: {location}
        Place tags: {tags}
        Place description: {description}
        
        Write a brief 1-2 sentence explanation of why this place is a good match for the query.
        Focus on aspects that align with the user's expressed or implied needs.
        Be concise and natural - this is a recommendation explanation.
        
        Explanation:
        """
        
        self.explain_chain = (
            {"query": RunnablePassthrough(), "name": RunnablePassthrough(), 
             "location": RunnablePassthrough(), "tags": RunnablePassthrough(),
             "description": RunnablePassthrough()} 
            | PromptTemplate.from_template(explain_template)
            | self.llm
            | StrOutputParser()
        )
    
    def enhance_query(self, query: str) -> str:
        """
        Use LLM to enhance the query with better understanding of intent.
        
        Args:
            query: The original user query
            
        Returns:
            Enhanced query
        """
        if not self.initialized:
            logger.warning("QueryProcessor not initialized, returning original query")
            return query
        
        try:
            enhanced = self.enhance_chain.invoke(query)
            logger.info(f"Enhanced query: {enhanced}")
            return enhanced.strip()
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query
    
    def explain_result(self, query: str, result: Dict) -> str:
        """
        Generate an explanation for why a search result matches the query.
        
        Args:
            query: The original user query
            result: The search result (with metadata)
            
        Returns:
            Explanation text
        """
        if not self.initialized:
            logger.warning("QueryProcessor not initialized, returning simple explanation")
            return self._simple_explanation(query, result)
        
        try:
            explanation = self.explain_chain.invoke({
                "query": query,
                "name": result.get("name", ""),
                "location": result.get("neighborhood", ""),
                "tags": result.get("tags", ""),
                "description": result.get("short_description", "")
            })
            return explanation.strip()
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return self._simple_explanation(query, result)
    
    def _simple_explanation(self, query: str, result: Dict) -> str:
        """Generate a simple explanation without using LLM."""
        name = result.get("name", "")
        neighborhood = result.get("neighborhood", "")
        tags = result.get("tags", "")
        description = result.get("short_description", "")
        
        if description:
            return f"{name} in {neighborhood} - {description}"
        elif tags:
            return f"{name} is a {tags} in {neighborhood}"
        else:
            return f"{name} in {neighborhood} matches your query"