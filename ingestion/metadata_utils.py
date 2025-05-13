# ingestion/metadata_utils.py
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_and_merge_metadata(
    places_path: str = "places.csv",
    reviews_path: str = "reviews.csv",
    media_path: str = "media.csv",
) -> List[Dict]:
    """
    Load and merge places, reviews, and media into unified metadata structure.
    Returns list of dicts with fields: place_id, name, full_text, image_urls, etc.
    
    This improved version properly handles missing values and creates better text representation.
    """
    base_dir = Path(__file__).resolve().parent.parent
    places_path = base_dir / places_path
    reviews_path = base_dir / reviews_path
    media_path = base_dir / media_path

    # Load all files
    logger.info(f"Loading places from {places_path}")
    places_df = pd.read_csv(places_path)

    # Add parsing for the 'tags' column
    def parse_tags_string(tags_str: Optional[str]) -> Optional[List[str]]:
        if pd.isna(tags_str) or not isinstance(tags_str, str) or not tags_str.strip():
            return None
        # Check if it looks like the set-as-string format "{tag1,tag2}"
        if tags_str.startswith('{') and tags_str.endswith('}'):
            # Remove curly braces and split by comma, then strip whitespace
            return [tag.strip() for tag in tags_str[1:-1].split(',') if tag.strip()]
        # If it's already a simple comma-separated string without braces (e.g., "tag1, tag2")
        elif ',' in tags_str:
            return [tag.strip() for tag in tags_str.split(',') if tag.strip()]
        # If it's a single tag without commas or braces
        else:
            return [tags_str.strip()] if tags_str.strip() else None

    if 'tags' in places_df.columns:
        logger.info("Parsing 'tags' column into lists...")
        places_df['tags'] = places_df['tags'].apply(parse_tags_string)
    else:
        logger.warning("'tags' column not found in places.csv. Skipping tags parsing.")

    
    logger.info(f"Loading reviews from {reviews_path}")
    reviews_df = pd.read_csv(reviews_path)
    
    logger.info(f"Loading media from {media_path}")
    media_df = pd.read_csv(media_path)

    # Process reviews: group by place_id and combine texts
    logger.info("Processing reviews...")
    reviews_grouped = (
        reviews_df.groupby("place_id")["review_text"]
        .apply(lambda x: " ".join(x.dropna().astype(str)))
        .reset_index()
    )

    # Process media: group all images per place into lists
    logger.info("Processing media...")
    media_grouped = (
        media_df.groupby("place_id")["media_url"]
        .apply(list)
        .reset_index()
        .rename(columns={"media_url": "image_urls"})
    )

    # Merge all data
    logger.info("Merging datasets...")
    merged = places_df.merge(reviews_grouped, on="place_id", how="left")
    merged = merged.merge(media_grouped, on="place_id", how="left")

    # Create a structured text field for better embedding
    logger.info("Creating structured text field...")
    merged["full_text"] = merged.apply(
        lambda row: (
            f"Name: {row['name'] if pd.notna(row['name']) else 'Unknown'}. "
            f"Location: {row['neighborhood'] if pd.notna(row['neighborhood']) else 'Unknown'}. "
            # FIXED: Using direct check for list type and emptiness
            f"Type: {', '.join(row['tags']) if row['tags'] and isinstance(row['tags'], list) else 'Unknown'}. "
            f"Description: {row['short_description'] if pd.notna(row['short_description']) else ''}. "
            f"Reviews: {row['review_text'] if pd.notna(row['review_text']) else ''}"
        ),
        axis=1
    )

    # Convert to list of dictionaries
    logger.info("Converting to dictionary format...")
    result = merged.to_dict(orient="records")
    
    # FIX: Handle missing image_urls after conversion to dict
    logger.info("Handling missing image_urls...")
    for item in result:
        # Only initialize image_urls if it doesn't exist or isn't already a list
        if 'image_urls' not in item or not isinstance(item['image_urls'], list):
            item['image_urls'] = []
    
    logger.info(f"Processed {len(result)} places with metadata")
    return result

if __name__ == "__main__":
    # Test the function
    data = load_and_merge_metadata()
    print(f"Loaded {len(data)} enriched place records.")
    
    # Print a sample
    if data:
        for i in range(min(3, len(data))):
            sample = data[i]
            print(f"\n--- Sample Record {i+1} ---")
            for key, value in sample.items():
                if key == "tags":
                    print(f"Tags: {value} (Type: {type(value)})")
                elif key == "image_urls":
                    print(f"Image URLs: {value[:2]}{'...' if len(value) > 2 else ''} (Type: {type(value)}, Length: {len(value)})")
                elif key not in ["full_text", "review_text"]:
                    print(f"{key}: {value}")