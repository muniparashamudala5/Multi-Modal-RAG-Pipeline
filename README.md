# Multimodal RAG Pipeline (PDF Text + Images)

Ask questions about PDFs that contain both text and images, and get back
answers grounded in the document, with page citations -- built with
PyMuPDF, CLIP embeddings, Chroma, and FastAPI.

## How it works, in one paragraph

A PDF is opened page by page. Each page's text is split into overlapping
chunks, and each embedded image is pulled out and saved to disk. Both the
text chunks and the images are embedded with **CLIP**, which was trained
to put matching text and images close together in the same vector space
-- so a question like "show me the revenue chart" can retrieve an actual
chart image, not just a paragraph that happens to mention revenue. Those
vectors go into a **Chroma** index. At query time, the question is
embedded the same way, the most similar text chunks and images are
retrieved, and both are handed to an LLM with an instruction to answer
*only* from that context and cite the page number for every claim.

## Project structure

```
multimodal-rag/
├── data/
│   ├── pdfs/                 # drop your PDFs here
│   └── extracted/images/     # images pulled out of PDFs land here automatically
├── vectorstore/               # persisted Chroma index (generated)
├── src/
│   ├── config.py              # every tunable setting, in one place
│   ├── ingest.py               # PDF -> text chunks + images
│   ├── embeddings.py            # CLIP embedder for text AND images
│   ├── vectorstore.py            # build / load the Chroma index
│   ├── retriever.py               # question -> top text + image hits
│   ├── generator.py                # builds the grounded prompt, calls the LLM
│   └── pipeline.py                  # build() and ask() -- the two entry points
├── app/
│   └── api.py                        # FastAPI: POST /ingest, POST /ask
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

Every file above has inline comments explaining *why*, not just *what* --
read through `src/ingest.py` and `src/retriever.py` first if you want to
understand the two trickiest parts (chunking with overlap, and why text
and images are retrieved in separate calls).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenAI (or OpenAI-compatible)
API key:

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

> **Using a local model instead of OpenAI?** Point `OPENAI_BASE_URL` in
> `.env` at a local server like Ollama or LM Studio that exposes an
> OpenAI-compatible `/chat/completions` endpoint -- `generator.py`
> doesn't need to change.

## Add a PDF and build the index

Drop any PDF into `data/pdfs/`, then run:

```bash
python -m src.pipeline
```

This extracts every page's text and images, embeds them with CLIP, and
writes the index to `vectorstore/`. Re-run this any time you add more
PDFs -- it adds to the existing index rather than wiping it.

## Ask a question

From the command line:

```bash
python -m src.pipeline "What does the chart on page 3 show?"
```

Or start the API and use `/docs`:

```bash
uvicorn app.api:app --reload
```

Open http://127.0.0.1:8000/docs, use `POST /ingest` to upload a PDF
directly through the browser (this does extraction + indexing in one
step), then `POST /ask` with `{"question": "..."}`.

Example response:
```json
{
  "answer": "The churn rate drops sharply for two-year contracts compared to month-to-month [Page 4].",
  "sources": [
    {"source": "report", "page": 4, "type": "text"},
    {"source": "report", "page": 4, "type": "image"}
  ]
}
```

## Tests

```bash
pytest tests/ -v
```

The chunking logic (`_chunk_text` in `src/ingest.py`) is tested directly
with plain strings, so those tests run without PyMuPDF, CLIP, or any
model weights installed -- useful for quick sanity checks in CI before
the heavier dependencies are even pulled in.

## Docker

```bash
docker build -t multimodal-rag .
docker run -p 8000:8000 --env-file .env multimodal-rag
```

Note the `--env-file .env` -- the container needs your `OPENAI_API_KEY`
at runtime; it's intentionally not baked into the image.

## Deploy

Same pattern as any Dockerized FastAPI app:
- **Render** -- New -> Web Service -> connect the repo -> it detects the
  Dockerfile. Add `OPENAI_API_KEY` under the service's Environment tab.
- **Hugging Face Spaces** -- create a Space with the "Docker" SDK, push
  this repo, and add `OPENAI_API_KEY` as a repo secret.

One thing to plan for: the vector index (`vectorstore/`) and any indexed
PDFs live on local disk, which most free hosting tiers don't persist
across restarts. For a portfolio demo that's usually fine (re-ingest a
sample PDF on startup); for anything longer-lived, swap Chroma's local
`PersistentClient` for a hosted vector DB.

## A note on what's "real" here

The code in this project is complete and was syntax-checked file by
file. The chunking logic was unit-tested directly. The parts that need
PyMuPDF, CLIP/sentence-transformers, Chroma, and a live OpenAI API key
weren't run end-to-end in the environment this was built in (no internet
access there to install those packages or call an API) -- so when you
first run `python -m src.pipeline` on a real PDF, treat it as the first
real test of the ingestion + embedding path, and expect to debug small
things (a path issue, a version mismatch) the way you would with any
new project.

## Ideas to extend this

- Swap the text-only context sent to the LLM for an actual vision-capable
  call (e.g. GPT-4o, Claude) that looks at the retrieved image bytes
  directly, instead of a text placeholder describing "an image exists
  here."
- Add a simple Streamlit UI that shows the answer next to the cited page
  images.
- Track which chunks/images get cited most often, as a signal for which
  parts of a long PDF people actually care about.
