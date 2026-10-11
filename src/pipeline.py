"""
Ties every stage together into two simple entry points:

    build()       -- run once (or whenever you add new PDFs): extract -> embed -> index
    ask(question)  -- run per question: retrieve -> generate

Keeping this orchestration in its own file means app/api.py (the web
layer) and any future CLI or notebook can both just call build()/ask()
without needing to know how retrieval or generation work internally.
"""

import sys

sys.path.insert(0, ".")
from src.config import PDF_DIR  # noqa: E402
from src.generator import generate_answer  # noqa: E402
from src.ingest import extract_all  # noqa: E402
from src.retriever import retrieve  # noqa: E402
from src.vectorstore import build_index  # noqa: E402


def build() -> None:
    """Indexing pipeline: run this whenever you add/change PDFs in data/pdfs/."""
    print(f"Scanning {PDF_DIR} for PDFs ...")
    documents = extract_all(PDF_DIR)
    if not documents:
        print("Nothing to index -- add a PDF to data/pdfs/ and re-run.")
        return
    build_index(documents)
    print("Done. Run ask('your question') or hit POST /ask to query it.")


def ask(question: str) -> dict:
    """Query pipeline: retrieve relevant context, then generate a grounded answer."""
    context_items = retrieve(question)
    if not context_items:
        return {
            "answer": "I couldn't find anything relevant in the indexed PDFs. "
            "Have you run `python -m src.pipeline` to build the index yet?",
            "sources": [],
        }
    return generate_answer(question, context_items)


if __name__ == "__main__":
    # Lets you run `python -m src.pipeline` to (re)build the index, or
    # `python -m src.pipeline "your question"` to ask something directly
    # from the command line without starting the API.
    if len(sys.argv) > 1:
        result = ask(" ".join(sys.argv[1:]))
        print("\nAnswer:\n" + result["answer"])
        print("\nSources:", result["sources"])
    else:
        build()
