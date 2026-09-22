from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import answer

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