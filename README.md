# Vibe Search™ - Find NYC Places by Semantic Vibes

Vibe Search is a multimodal semantic search engine for discovering places in New York City(Current Database) that match your specific vibe. This application combines text and image understanding with large language models to create a powerful and an intuitive search experience.

---

## Overview

Have you ever struggled to find the perfect spot that matches exactly what you're looking for? Well, Vibe Search here lets you describe places in natural language and finds matches based on semantic understanding rather than just keywords. Be it "cozy cafes with a literary vibe" or "lively bars with outdoor seating in Williamsburg," Vibe Search understands your intent and finds the perfect match.

---

## Features

- **Natural Language Understanding**: Search using everyday language and conversational queries
- **Multimodal Semantic Search**: Combines both text and image understanding
- **LLM-Enhanced Search**: Uses language models to improve search quality
  - Query enhancement to better understand user intent
  - Personalized explanations for why each place matches
  - Intelligent re-ranking of results
- **Quick Search Mode**: Option to bypass LLM features for faster results
- **Neighborhood Detection**: Automatically identifies and filters by NYC neighborhoods
- **User-Friendly Interface**: Clean, responsive design with search options and visual results

---

## System Architecture

### Backend Components

- **FastAPI Application**: Main web server and API endpoints
- **VibeSearch Engine**: Core search implementation with multimodal capabilities
- **Text Embeddings**: Using `BAAI/bge-large-en-v1.5` sentence transformer model
- **Image Embeddings**: Using OpenAI's `CLIP` model for visual understanding
- **LLM Integration**: Optional `Ollama` or `HuggingFace` integration for enhanced results
- **FAISS Indices**: Fast similarity search for both text and image embeddings

### Frontend Components

- **Bootstrap UI**: Responsive interface with modern design
- **Dynamic Results**: Asynchronous loading and display of search results
- **Search Options**: Toggles for different search modes and features

---

## Installation & Setup

1. Clone the repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Prepare your data:
   - Ensure you have `places.csv`, `reviews.csv`, and `media.csv` in the project root
   - Run the embedding scripts to create search indices:
     ```
     python ingestion/embed_text.py
     python ingestion/embed_image.py
     ```
4. Start the server:
   ```
   python app.py
   ```
5. Access the application at `http://localhost:8000`

---

## Data Requirements

The application expects three CSV files:
- `places.csv`: Basic information about places (name, neighborhood, tags, etc.)
- `reviews.csv`: User reviews for places
- `media.csv`: Links to images for places

---

## Search Options

- **Quick Search**: Faster results without LLM enhancements
- **Image Search**: Include image-based semantic matching
- **Enhance Query**: Use LLM to improve query understanding
- **Explain Results**: Generate personalized explanations for matches

---

## Usage Examples

Try searching for:
- "cafes to cowork from"
- "lively bars in williamsburg"
- "where to spend a sunny day"
- "romantic date night spot"
- "hipster coffee shops"
- "unique dinner experience"
- "where to meet people when feeling lonely"
- "dance-y bars with good music"

---

## Technical Details

### Embedding Process

- **Text**: Creates embeddings from structured text combining place name, location, type, description, and reviews
- **Images**: Processes place images using `CLIP` to create visual embeddings
- Both are indexed in `FAISS` for efficient similarity search

### Search Process

1. **Query Enhancement** (optional): LLM improves the search query
2. **Neighborhood Detection**: Identifies mentioned neighborhoods
3. **Multimodal Search**: Performs both text and image search
4. **Result Merging**: Combines results from both search methods
5. **LLM Re-ranking** (optional): Improves result ordering
6. **Explanation Generation** (optional): Creates personalized match explanations

---

## Performance Considerations

- Use Quick Search mode for faster results (disables LLM features)
- Image search and LLM operations increase response time but improve quality
- `FAISS` indices provide efficient similarity search even with large datasets
- The system lazy-loads models when needed to optimize resource usage

---

## LLM Integration

The system supports two LLM backends:
- **Ollama**: Local LLM inference (default: "mistral:7b-instruct-q4_K_M" quantised model)
- **HuggingFace**: TGI endpoint for external inference

---

## Acknowledgments

- `CLIP` model by OpenAI
- `BGE` embeddings by BAAI
- `FastAPI` for the web framework
- `Bootstrap` for the frontend
- `FAISS` by Facebook Research

---

Built with ❤️ for _NYC explorers_ and ___CORNER___