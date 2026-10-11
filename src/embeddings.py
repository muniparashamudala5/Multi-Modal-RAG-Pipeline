"""
Step 2: turn text and images into vectors (lists of numbers) that live in
the same "meaning space," so we can compare a text question directly
against an image and get a sensible similarity score.

We use a single CLIP model for this. CLIP was trained on millions of
(image, caption) pairs to push matching image/text pairs close together
in vector space and push non-matching pairs apart. That's exactly the
property a multimodal search system needs.
"""

import sys
from functools import lru_cache

from PIL import Image

sys.path.insert(0, ".")
from src.config import CLIP_MODEL_NAME  # noqa: E402


@lru_cache(maxsize=1)
def _get_model():
    """
    Loads the CLIP model once and reuses it for every call.

    @lru_cache(maxsize=1) is a simple way to memoize this -- the first
    call loads the (fairly large) model into memory, and every call after
    that just returns the already-loaded object instead of reloading it.
    Without this, embedding 500 chunks would load the model 500 times.
    """
    # Imported lazily (inside the function) so that just importing this
    # module doesn't require sentence-transformers/torch to be installed --
    # useful if you're only poking at other parts of the pipeline.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(CLIP_MODEL_NAME)


def embed_text(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of text strings into CLIP's shared vector space."""
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()


def embed_images(image_paths: list[str]) -> list[list[float]]:
    """Embeds a batch of image files into the SAME vector space as embed_text."""
    model = _get_model()
    images = [Image.open(p).convert("RGB") for p in image_paths]
    vectors = model.encode(images, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embeds a single search query (what the user typed).

    This is just embed_text() for one string, but it's worth having as
    its own function -- it's the one place retriever.py calls into, and
    keeping the "one query" case separate from the "batch of documents"
    case makes both call sites read more clearly.
    """
    return embed_text([query])[0]
