FROM python:3.11-slim

# PyMuPDF and Pillow need a couple of system libraries to build/run on a
# slim base image -- without these the pip install step fails partway
# through compiling native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY src/ src/

# These stay empty in the image -- you mount or populate them at runtime
# rather than baking PDFs/embeddings into the image itself.
RUN mkdir -p data/pdfs data/extracted/images vectorstore

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
