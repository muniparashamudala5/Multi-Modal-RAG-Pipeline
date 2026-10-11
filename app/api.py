"""
The web layer. Thin on purpose -- it just validates requests, saves
uploaded files to the right folder, and calls into src/pipeline.py for
all the actual logic. Keeping the API this thin means you could swap
FastAPI for a CLI or a Streamlit app without touching src/ at all.
"""

import shutil
import sys

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

sys.path.insert(0, ".")
from src.config import PDF_DIR  # noqa: E402
from src.ingest import extract_from_pdf  # noqa: E402
from src.pipeline import ask as pipeline_ask  # noqa: E402
from src.vectorstore import build_index  # noqa: E402

app = FastAPI(
    title="Multimodal RAG API",
    description="Ask questions about PDFs containing both text and images, with page-cited answers.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Multimodal RAG API is running. Visit /docs for the interactive API."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile):
    """
    Accepts a PDF upload, saves it to data/pdfs/, extracts its text and
    images, and adds them to the vector index -- all in one request, so
    the frontend doesn't need to make two separate calls.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = PDF_DIR / file.filename
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # UploadFile gives us a file-like object; shutil.copyfileobj streams it
    # to disk in chunks instead of loading the whole PDF into memory at
    # once, which matters once PDFs get into the tens-of-MB range.
    with open(save_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    documents = extract_from_pdf(save_path)
    build_index(documents)

    text_count = sum(1 for d in documents if d.type == "text")
    image_count = sum(1 for d in documents if d.type == "image")
    return {
        "filename": file.filename,
        "text_chunks_indexed": text_count,
        "images_indexed": image_count,
    }


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: AskRequest):
    """Answers a question using whatever PDFs have been ingested so far."""
    result = pipeline_ask(request.question)
    return result
