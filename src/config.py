"""
All the tunable settings for the pipeline live here, in one place.

Why bother with a config file instead of hardcoding values in each script?
Because when you want to try a bigger chunk size, or switch embedding
models, you change ONE number here instead of hunting through five files.
"""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
# Path(__file__).resolve() gives us the absolute path of THIS file, and
# .parent twice walks up from src/config.py -> src/ -> project root.
# That way the paths below work no matter where you run the scripts from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
EXTRACTED_IMAGES_DIR = PROJECT_ROOT / "data" / "extracted" / "images"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

# --- Chunking ------------------------------------------------------------
# These control how we split page text into smaller pieces before
# embedding. Smaller chunks = more precise retrieval but less context per
# chunk. 800/100 is a reasonable starting point for dense paragraph text;
# tune it if your PDFs are very different (e.g. slide decks need smaller
# chunks, dense legal text can use bigger ones).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Embeddings ----------------------------------------------------------
# We use CLIP for BOTH text and images, on purpose. CLIP was trained to
# put text and images into the SAME vector space, so a text query like
# "a bar chart showing revenue growth" can directly retrieve a matching
# image without any extra translation step. That's what makes this
# "multimodal" rather than two separate search systems bolted together.
CLIP_MODEL_NAME = "clip-ViT-B-32"

# --- Retrieval -------------------------------------------------------------
TOP_K_TEXT = 4
TOP_K_IMAGES = 2

# --- Generation ------------------------------------------------------------
# We talk to an OpenAI-compatible chat completions endpoint. This also
# works with local servers (e.g. Ollama, LM Studio) that mimic the same
# API shape -- just change OPENAI_BASE_URL and OPENAI_API_KEY below.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")

COLLECTION_NAME = "multimodal_rag"
