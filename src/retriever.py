"""
Step 4: given a question, find the most relevant text chunks AND the most
relevant images already sitting in the vector store.

We query for text and images SEPARATELY (two calls to Chroma, filtered by
the "type" metadata field) rather than one mixed call. The reason: if we
asked for "top 6 overall," a question with a strong textual answer could
crowd out every image, even when a relevant chart exists. Reserving fixed
slots for each type (e.g. top 4 text + top 2 images) guarantees the final
answer can always point at supporting visuals when they exist.
"""

import sys
from dataclasses import dataclass

sys.path.insert(0, ".")
from src.config import TOP_K_IMAGES, TOP_K_TEXT  # noqa: E402
from src.embeddings import embed_query  # noqa: E402
from src.vectorstore import load_collection  # noqa: E402


@dataclass
class RetrievedItem:
    type: str          # "text" or "image"
    source: str
    page: int
    content: str         # the chunk text, or the image file path
    distance: float        # lower = more similar (Chroma returns distances, not scores)


def retrieve(query: str, top_k_text: int = TOP_K_TEXT, top_k_images: int = TOP_K_IMAGES) -> list[RetrievedItem]:
    collection = load_collection()
    query_vector = embed_query(query)

    results: list[RetrievedItem] = []

    # --- Text hits -----------------------------------------------------
    text_results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k_text,
        where={"type": "text"},  # only search within text-type documents
    )
    # Chroma's query() returns everything wrapped in an extra outer list
    # (because you can batch multiple queries at once) -- we only sent
    # one query, so we always index into [0] to unwrap it.
    for metadata, distance in zip(text_results["metadatas"][0], text_results["distances"][0]):
        results.append(
            RetrievedItem(
                type="text",
                source=metadata["source"],
                page=metadata["page"],
                content=metadata["text"],
                distance=distance,
            )
        )

    # --- Image hits ------------------------------------------------------
    image_results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k_images,
        where={"type": "image"},
    )
    for metadata, distance in zip(image_results["metadatas"][0], image_results["distances"][0]):
        results.append(
            RetrievedItem(
                type="image",
                source=metadata["source"],
                page=metadata["page"],
                content=metadata["image_path"],
                distance=distance,
            )
        )

    return results
