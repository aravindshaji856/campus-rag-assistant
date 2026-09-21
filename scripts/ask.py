import os
import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()
MODEL = "gemini-3.6-flash"

llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
collection = chromadb.PersistentClient(path="chroma_db").get_collection("notes")


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
reply exactly: "I couldn't find this in the uploaded notes."

Context:
{context}

Question: {question}"""


def answer(question):
    chunks = retrieve(question)
    response = llm.models.generate_content(
        model=MODEL, contents=build_prompt(question, chunks)
    )
    return response.text, chunks


if __name__ == "__main__":
    q = input("Ask a question: ")
    text, chunks = answer(q)
    print("\n" + text)
    print("\nSources:")
    for i, (_, m) in enumerate(chunks):
        print(f"[{i + 1}] {m['source']} - page {m['page']}")