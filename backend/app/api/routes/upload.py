from fastapi import APIRouter, UploadFile, File
import os
from pypdf import PdfReader
from app.services.embedding_service import create_embeddings

router = APIRouter()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    text = ""

    reader = PdfReader(file_path)

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

    chunk_count = create_embeddings(text)

    return {
        "filename": file.filename,
        "chunks_created": chunk_count,
        "message": "PDF processed successfully"
    }