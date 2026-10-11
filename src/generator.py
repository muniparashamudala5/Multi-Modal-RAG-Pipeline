"""
Step 5: take the retrieved text/image context and actually answer the
user's question with an LLM -- grounded in that context, with page
citations, instead of the model just making something up.

The core trick of RAG is in the prompt: we explicitly tell the model
"only answer from what's given below, and cite the page." That one
instruction is most of what separates a trustworthy RAG answer from a
model freely hallucinating.
"""

import sys

import requests

sys.path.insert(0, ".")
from src.config import GENERATION_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL  # noqa: E402
from src.retriever import RetrievedItem  # noqa: E402

SYSTEM_PROMPT = """You are a careful research assistant. You answer questions \
ONLY using the context provided below (text excerpts and image descriptions \
pulled from a PDF). If the context doesn't contain the answer, say so plainly \
instead of guessing. Every claim you make should be followed by a page \
citation in the form [Page N], referencing the context item it came from."""


def _build_context_block(items: list[RetrievedItem]) -> str:
    """
    Turns the retrieved items into a numbered block of text the LLM can
    read and cite back to. Text items get their content pasted in
    directly; image items get a placeholder description, since we're not
    sending the raw image bytes to every model (see the README for the
    one-line swap to a vision-capable model that CAN look at the actual
    pixels).
    """
    lines = []
    for item in items:
        if item.type == "text":
            lines.append(f"[Page {item.page}, text] {item.content}")
        else:
            lines.append(
                f"[Page {item.page}, image] (An image/figure on this page -- "
                f"file: {item.content}. Treat its presence as supporting "
                f"evidence for topics discussed on this page.)"
            )
    return "\n\n".join(lines)


def generate_answer(question: str, context_items: list[RetrievedItem]) -> dict:
    """
    Calls an OpenAI-compatible chat completions endpoint with the
    retrieved context, and returns the answer plus a clean list of
    sources (so the caller doesn't have to re-derive citations from the
    free-text answer).
    """
    context_block = _build_context_block(context_items)

    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer the question using only the context above, with [Page N] "
        f"citations."
    )

    if not OPENAI_API_KEY:
        # Fail loudly and helpfully rather than silently -- a missing API
        # key is the single most common reason this step breaks.
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file or export it "
            "in your shell before calling generate_answer(). See .env.example."
        )

    response = requests.post(
        f"{OPENAI_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": GENERATION_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,  # low temperature: we want grounded, consistent answers, not creative ones
        },
        timeout=60,
    )
    response.raise_for_status()
    answer_text = response.json()["choices"][0]["message"]["content"]

    # Build a de-duplicated, page-sorted source list to return alongside
    # the answer -- handy for a UI to render as "Sources: Page 2, Page 5".
    seen_pages = set()
    sources = []
    for item in context_items:
        key = (item.source, item.page, item.type)
        if key not in seen_pages:
            seen_pages.add(key)
            sources.append({"source": item.source, "page": item.page, "type": item.type})
    sources.sort(key=lambda s: s["page"])

    return {"answer": answer_text, "sources": sources}
