from pathlib import Path
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_WORDS = 300
OVERLAP = 50


def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    return [(i + 1, page.get_text()) for i, page in enumerate(doc)]


def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP):
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
    return chunks


def main():
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection("notes")

    for pdf in Path("data").glob("*.pdf"):
        ids, docs, metas = [], [], []
        for page_num, text in extract_pages(pdf):
            for j, chunk in enumerate(chunk_text(text)):
                ids.append(f"{pdf.stem}-p{page_num}-c{j}")
                docs.append(chunk)
                metas.append({"source": pdf.name, "page": page_num})

        if not docs:
            print(f"No text found in {pdf.name} (scanned PDF?)")
            continue

        embeddings = model.encode(docs, normalize_embeddings=True).tolist()
        collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
        print(f"Ingested {pdf.name}: {len(docs)} chunks")


if __name__ == "__main__":
    main()