"""
Step 3: store the embedded documents somewhere we can search later.

We use Chroma because it's a lightweight, file-based vector database --
no separate server to run, it just writes to a folder on disk
(vectorstore/) and persists between runs. For a bigger production system
you might reach for something like Pinecone or Qdrant, but Chroma is the
right amount of "simple" for a portfolio project.
"""

import sys

sys.path.insert(0, ".")
from src.config import COLLECTION_NAME, VECTORSTORE_DIR  # noqa: E402
from src.embeddings import embed_images, embed_text  # noqa: E402
from src.ingest import Document  # noqa: E402


def _get_client():
    # Imported lazily so this module can be imported without chromadb
    # installed, same reasoning as the lazy import in embeddings.py.
    import chromadb

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTORSTORE_DIR))


def build_index(documents: list[Document]) -> None:
    """
    Embeds every document (text and image) and writes them into a Chroma
    collection on disk.

    Chroma collections store three parallel lists that all line up by
    index: ids, embeddings, and metadatas. We split the incoming
    documents into text/image groups first because text and images go
    through different embedding functions (even though both land in the
    same CLIP vector space).
    """
    client = _get_client()
    # get_or_create_collection: if this collection already exists from a
    # previous run, we add to it rather than erroring out. If you want a
    # clean rebuild, delete the vectorstore/ folder first.
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    text_docs = [d for d in documents if d.type == "text"]
    image_docs = [d for d in documents if d.type == "image"]

    if text_docs:
        print(f"Embedding {len(text_docs)} text chunks ...")
        text_vectors = embed_text([d.content for d in text_docs])
        collection.add(
            ids=[d.doc_id for d in text_docs],
            embeddings=text_vectors,
            # We don't store the raw text as the embedded "content" field
            # in Chroma -- we store it in metadata instead, alongside page
            # number and source, so one lookup gives us everything we need
            # to build a citation.
            metadatas=[
                {"source": d.source, "page": d.page, "type": d.type, "text": d.content}
                for d in text_docs
            ],
        )

    if image_docs:
        print(f"Embedding {len(image_docs)} images ...")
        image_vectors = embed_images([d.content for d in image_docs])
        collection.add(
            ids=[d.doc_id for d in image_docs],
            embeddings=image_vectors,
            metadatas=[
                {"source": d.source, "page": d.page, "type": d.type, "image_path": d.content}
                for d in image_docs
            ],
        )

    print(f"Index now has {collection.count()} items total.")


def load_collection():
    """Opens the existing persisted collection (doesn't rebuild it)."""
    client = _get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)
