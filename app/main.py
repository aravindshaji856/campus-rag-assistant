from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import answer

import shutil
from pathlib import Path
from fastapi import UploadFile, File
from app.rag import ingest_pdf

Path("data").mkdir(exist_ok=True)




app = FastAPI(title="Campus RAG Assistant")


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        return answer(q.question)
    except Exception:
        raise HTTPException(status_code=503, detail="Language model unavailable, try again shortly")

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    dest = Path("data") / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    count = ingest_pdf(dest)
    return {"filename": file.filename, "chunks_added": count}        