"""
Step 1 of the pipeline: turn a raw PDF into a list of "documents" we can
embed and search over.

A PDF page usually has two kinds of content worth indexing:
  - the TEXT on the page (which we split into smaller overlapping chunks)
  - any IMAGES on the page (charts, diagrams, photos, screenshots)

We keep track of which page each piece of text or image came from, because
that's what lets us answer "what page is this from?" later on -- citations
are the whole point of a trustworthy RAG system.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, ".")
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EXTRACTED_IMAGES_DIR  # noqa: E402


@dataclass
class Document:
    """
    A single retrievable unit -- either a chunk of text or one image.
    Keeping text and images in the same shape makes the rest of the
    pipeline (embedding, storing, retrieving) not have to special-case
    either type.
    """

    doc_id: str          # unique id, e.g. "mypdf_p3_text_0" or "mypdf_p3_img_1"
    source: str           # the PDF filename this came from
    page: int              # 1-indexed page number (how people usually count pages)
    type: Literal["text", "image"]
    content: str            # the text itself, OR the file path to the image
    

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits a long string into overlapping chunks.

    Why overlap at all? If we split with zero overlap, a sentence that
    straddles a chunk boundary gets cut in half, and neither half makes
    much sense on its own. Overlapping by ~100 characters means important
    context rarely gets fully severed.

    This is a simple character-based splitter. LangChain's
    RecursiveCharacterTextSplitter does something smarter (it tries to
    break on paragraph/sentence boundaries first), and you can swap this
    function out for that if you want cleaner chunk edges -- see the
    README for the one-line swap.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move forward by (chunk_size - overlap), not chunk_size, so the
        # next chunk re-includes the last `overlap` characters of this one.
        start += chunk_size - overlap
    return chunks


def extract_from_pdf(pdf_path: Path) -> list[Document]:
    """
    Opens one PDF and pulls out every text chunk and every embedded image,
    tagging each with the page it came from.
    """
    # Imported lazily so this module is importable (and _chunk_text is
    # unit-testable) even in an environment where PyMuPDF isn't installed.
    import fitz  # this is the import name for PyMuPDF, confusingly

    documents: list[Document] = []
    source_name = pdf_path.stem  # filename without the .pdf extension

    # Each PDF gets its own image subfolder, so images from different
    # PDFs never collide or overwrite each other.
    image_out_dir = EXTRACTED_IMAGES_DIR / source_name
    image_out_dir.mkdir(parents=True, exist_ok=True)

    pdf = fitz.open(pdf_path)

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        page_number = page_index + 1  # humans count pages from 1, not 0

        # --- Text on this page -------------------------------------------
        page_text = page.get_text()
        for chunk_index, chunk in enumerate(_chunk_text(page_text)):
            documents.append(
                Document(
                    doc_id=f"{source_name}_p{page_number}_text_{chunk_index}",
                    source=source_name,
                    page=page_number,
                    type="text",
                    content=chunk,
                )
            )

        # --- Images on this page --------------------------------------------
        # get_images() returns a list of image references on the page;
        # xref is the internal PDF object id we use to pull out the actual
        # image bytes with extract_image().
        for image_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            extension = base_image["ext"]  # e.g. "png", "jpeg"

            image_path = image_out_dir / f"p{page_number}_img{image_index}.{extension}"
            image_path.write_bytes(image_bytes)

            documents.append(
                Document(
                    doc_id=f"{source_name}_p{page_number}_img_{image_index}",
                    source=source_name,
                    page=page_number,
                    type="image",
                    content=str(image_path),
                )
            )

    pdf.close()
    return documents


def extract_all(pdf_dir: Path) -> list[Document]:
    """Runs extract_from_pdf() over every PDF in a folder."""
    all_documents: list[Document] = []
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        print(f"[warn] no PDFs found in {pdf_dir} -- add some and re-run")
        return all_documents

    for pdf_path in pdf_paths:
        print(f"Extracting {pdf_path.name} ...")
        docs = extract_from_pdf(pdf_path)
        text_count = sum(1 for d in docs if d.type == "text")
        image_count = sum(1 for d in docs if d.type == "image")
        print(f"  -> {text_count} text chunks, {image_count} images")
        all_documents.extend(docs)

    return all_documents


if __name__ == "__main__":
    from src.config import PDF_DIR

    docs = extract_all(PDF_DIR)
    print(f"\nTotal: {len(docs)} documents extracted from {PDF_DIR}")
