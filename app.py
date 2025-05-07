from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse # Removed JSONResponse as it's not directly used here
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
import os
import logging
from pathlib import Path

# Import your retrieval functionality
from retrieval.vibe_search import VibeSearch

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Models for API requests/responses
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    use_text: bool = True
    use_images: bool = True
    enhance: bool = True
    explain: bool = True
    rerank: bool = True
    quick_search: bool = False

class SearchResponse(BaseModel):
    results: List[Dict]
    original_query: str
    processed_query: Optional[str] = None
    neighborhoods: Optional[List[str]] = None
    result_count: int
    processing_time: float
    text_search_used: bool
    image_search_used: bool

# Create FastAPI app
app = FastAPI(title="Vibe Search™", description="Find places in NYC that match your vibe")

# Set up static files and templates
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Initialize the search system
# The VibeSearch instance is created once when the app starts.
# Its internal 'use_llm' flag (defaulting to True) determines if LLM components are loaded.
# The 'quick_search' flag from the request will determine if these loaded components are USED.
vibe_search = VibeSearch()
logger.info("Vibe Search system initialized (LLM components loaded if VibeSearch default use_llm=True)")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main search interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    API endpoint for place searches
    """
    start_time = time.time()
    logger.info(f"Search request: Query='{req.query}', QuickSearch={req.quick_search}, Enhance={req.enhance}, Explain={req.explain}")
    
    try:
        # Perform the search using your RAG system
        # Pass the quick_search flag from the request to the VibeSearch method
        results_data = vibe_search.search(
            query=req.query,
            k=req.limit,
            use_text=req.use_text,
            use_images=req.use_images,
            enhance=req.enhance,
            explain=req.explain,
            rerank=req.rerank,
            quick_search=req.quick_search # Pass the flag here
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Print results to terminal (optional, for debugging)
        # Consider removing or reducing verbosity for production
        print("\n=== SEARCH RESULTS (FastAPI) ===")
        print(f"Original Query: {results_data['original_query']}")
        if results_data['processed_query'] != results_data['original_query']:
            print(f"Processed Query: {results_data['processed_query']}")
        print(f"Quick Search Mode: {req.quick_search}")
        print(f"Found {results_data['result_count']} results in {processing_time:.2f}s")
        # for i, place in enumerate(results_data['results'], 1):
        #     print(f"\n{i}. {place.get('name', 'Unknown')}")
        #     if 'match_reason' in place:
        #         print(f"   Reason: {place['match_reason']}")
        print("==============================\n")
        
        # Return formatted response
        return SearchResponse(
            results=results_data["results"],
            original_query=results_data["original_query"],
            processed_query=results_data["processed_query"],
            neighborhoods=results_data.get("neighborhoods", []),
            result_count=results_data["result_count"],
            processing_time=processing_time,
            text_search_used=results_data["text_search_used"],
            image_search_used=results_data["image_search_used"]
        )
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "time": time.strftime("%Y-%m-%d %H:%M:%S")
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    # Ensure reload is False or handled carefully in production
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)


# no options selected, rerank = true; 
# quick search = enabled, rerank = false;
# image search = enabled, rerank = true;  --> thereby takes time
# enhance search = enabled, rerank = true; --> thereby takes time
# explain search = enabled, rerank = true;  --> thereby takes even more time