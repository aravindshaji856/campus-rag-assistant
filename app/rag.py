import os
import re
import time
import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from pathlib import Path
import fitz

CHUNK_WORDS = 300
OVERLAP = 50


def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP):
    words = text.split()
    step = size - overlap
    return [" ".join(words[s:s + size]) for s in range(0, len(words), step) if words[s:s + size]]


def ingest_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    ids, docs, metas = [], [], []
    stem = Path(pdf_path).stem
    for page_num, page in enumerate(doc, start=1):
        for j, chunk in enumerate(chunk_text(page.get_text())):
            ids.append(f"{stem}-p{page_num}-c{j}")
            docs.append(chunk)
            metas.append({"source": Path(pdf_path).name, "page": page_num})
    if docs:
        embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()
        collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    return len(docs)

load_dotenv()
MODEL = "gemini-3.6-flash"
NOT_FOUND = "I couldn't find this in the uploaded notes."

llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
collection = chromadb.PersistentClient(path="chroma_db").get_or_create_collection("notes")


def retrieve(question, k=4):
    query = "Represent this sentence for searching relevant passages: " + question
    emb = embedder.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=emb, n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[Source {i + 1}: {m['source']}, page {m['page']}]\n{doc}"
        for i, (doc, m) in enumerate(chunks)
    )
    return f"""You are a study assistant. Answer the question using ONLY the context below.
Cite the sources you used like [Source 1]. If the answer is not in the context,
reply exactly: "{NOT_FOUND}"

Context:
{context}

Question: {question}"""


def generate(prompt, retries=3):
    for attempt in range(retries):
        try:
            return llm.models.generate_content(model=MODEL, contents=prompt).text
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def answer(question):
    chunks = retrieve(question)
    text = generate(build_prompt(question, chunks))
    if NOT_FOUND in text:
        return {"answer": NOT_FOUND, "sources": []}
    cited = sorted({int(n) for n in re.findall(r"\[Source (\d+)\]", text)})
    sources = [
        {"source": chunks[i - 1][1]["source"], "page": chunks[i - 1][1]["page"]}
        for i in cited
        if 1 <= i <= len(chunks)
    ]
    return {"answer": text, "sources": sources}