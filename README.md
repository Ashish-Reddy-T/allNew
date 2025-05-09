# Vibe Search™ - Find NYC Places by Vibes

Vibe Search is a multimodal semantic search engine for discovering places in New York City(Current Database) that match your specific vibe. This application combines text and image understanding with large language models to create a powerful and an intuitive search experience.

[Vibe Search Demo](https://youtu.be/pe6IovJ92xs)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Data Requirements](#data-requirements)
  - [Installation Steps](#installation-steps)
- [Tree Structure](#tree-structure)
- [Running Locally](#running-locally)
- [Search Options](#search-options)
- [Usage Examples](#usage-examples)
- [Technical Details](#technical-details)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

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

### Prerequisites

1. Python 3.8+ installed
2. Pip package manager
3. Git (for cloning the repository)
4. At least 8GB of RAM (16GB recommended for better performance)
5. GPU recommended but not required (CPU-only mode will be slower)
6. For LLM features: Ollama installed locally or access to a HuggingFace Text Generation Inference endpoint

### Data Requirements

**Note: The data files are not included in this repository.**

You will need to prepare three CSV files with the following structure:

1. **places.csv**:
   - Required columns: `place_id`, `name`, `neighborhood`, `tags`, `short_description`
   - Each row represents a place in NYC

2. **reviews.csv**:
   - Required columns: `place_id`, `review_text`
   - Contains user reviews for places

3. **media.csv**:
   - Required columns: `place_id`, `media_url`
   - Contains image URLs for each place

Place these CSV files in the project root directory.

---

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Ashish-Reddy-T/allNew.git
   cd allNew
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Ollama (for LLM features - optional but recommended):
   - Visit [Ollama's website](https://ollama.ai/) and follow the installation instructions
   - Pull the Mistral model:
     ```bash
     ollama pull mistral:7b-instruct-v0.3-q4_K_M
     ```

5. Process the data and create embeddings:
   ```bash
   # Generate text embeddings
   python ingestion/embed_text.py
   
   # Generate image embeddings
   python ingestion/embed_image.py
   ```
   This will create FAISS indices and metadata files in the `ingestion` directory. If it is taking too long, you may access the `*.faiss` and `*.pkl` files from [__here__](https://drive.google.com/drive/folders/1cmV2ic-mgBPC9mnvYprVhxJ69-mvNk8c?usp=sharing). 

   Make sure these files in the folder follow the [Tree Structure](#tree-structure).

---

## Tree Structure

```
.
├── app.py
├── ingestion
│   ├── embed_image.py
│   ├── embed_text.py
│   ├── metadata_utils.py
│   ├── image_index.faiss
│   ├── image_metadata.pkl
│   ├── text_index.faiss
│   └── text_metadata.pkl
├── media.csv              # Should be Placed         
├── places.csv             # Should be Placed
├── reviews.csv            # Should be Placed
├── requirements.txt
├── retrieval
│   └── vibe_search.py
├── static
│   ├── css
│   │   └── style.css
│   └── js
│       └── search.js
└── templates
    └── index.html
```

---

## Running Locally

1. Start the FastAPI server:
   ```bash
   python app.py
   ```

2. Access the application in your browser:
   ```
   http://localhost:8000
   ```

3. Optional: Run with different configurations:
   ```bash
   # Run without LLM components (faster startup, but no enhanced features)
   python -m retrieval.vibe_search --no-llm-init
   
   # Run with text search only
   python -m retrieval.vibe_search --text-only
   
   # Run with image search only
   python -m retrieval.vibe_search --image-only
   ```
---

## Search Options

The web interface provides several toggles to customize your search experience:

- **Quick Search**: Faster results without LLM enhancements (disables all other options)
- **Image Search**: Include image-based semantic matching (may increase search time)
- **Enhance Query with LLM**: Use LLM to improve query understanding (may increase search time)
- **Explain Results**: Generate personalized explanations for matches (may increase search time)

The application handles these dependencies automatically - enabling Quick Search will disable other options that require LLM processing.

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

The application uses two types of embeddings:

1. **Text Embeddings**:
   - Each place is represented as a structured text combining name, location, type, description, and reviews
   - The `BAAI/bge-large-en-v1.5` model creates vector embeddings
   - Implementation in `ingestion/embed_text.py`

2. **Image Embeddings**:
   - CLIP model processes images from place listings
   - Converts images into vector embeddings that capture visual semantics
   - Implementation in `ingestion/embed_image.py`

Both are indexed in FAISS for efficient similarity search.

### Search Process

The search flow in `vibe_search.py` follows these steps:

1. **Query Enhancement** (optional): LLM improves the search query to better capture intent
2. **Neighborhood Detection**: Identifies mentioned neighborhoods to filter results
3. **Text Search**: Performs semantic search on text embeddings
4. **Image Search** (optional): Performs visual search using CLIP
5. **Result Merging**: Combines results from both search methods
6. **LLM Re-ranking** (optional): Improves result ordering using LLM-assigned relevance scores
7. **Explanation Generation** (optional): Creates personalized match explanations

### Frontend-Backend Integration

- FastAPI provides the REST API endpoints
- Frontend JS (`static/js/search.js`) handles UI interactions and API calls
- Bootstrap provides responsive UI components
- Search options determine which backend features are used

---

## Performance Considerations

- **Quick Search Mode**: Disables LLM operations for much faster results when speed is critical
- **Lazy Loading**: The system only loads heavy models (like CLIP) when needed
- **Batch Processing**: Embedding generation is done in batches for efficiency
- **Device Optimization**: Automatically uses MPS (Apple Silicon), CUDA (NVIDIA), or falls back to CPU
- **FAISS Similarity Search**: Provides efficient vector search even with large datasets

---

## Troubleshooting

### Common Issues:

1. **Missing Dependencies**:
   - If you encounter import errors, ensure all requirements are installed:
     ```bash
     pip install -r requirements.txt
     ```

2. **GPU/CUDA Issues**:
   - If you face CUDA errors, the application will automatically fall back to CPU

3. **Embedding Generation Fails**:
   - Ensure you have sufficient disk space and RAM
   - Try reducing batch size in embedding scripts

4. **LLM Features Not Working**:
   - Check if Ollama is running properly:
     ```bash
     ollama list
     ```
   - Verify model availability:
     ```bash
     ollama pull mistral:7b-instruct-v0.3-q4_K_M
     ```

5. **Data Loading Issues**:
   - Ensure CSV files are in the correct format and location
   - Check file permissions

---

## Acknowledgments

- `CLIP` model by OpenAI
- `BGE` embeddings by BAAI
- `FastAPI` for the web framework
- `Bootstrap` for the frontend
- `FAISS` by Facebook Research
- `Ollama` for local LLM capabilities

---

Built with ❤️ for ___Corner___