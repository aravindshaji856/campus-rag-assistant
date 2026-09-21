import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
collection = chromadb.PersistentClient(path="chroma_db").get_collection("notes")

question = input("Ask a question: ")
query = "Represent this sentence for searching relevant passages: " + question
q_emb = model.encode([query], normalize_embeddings=True).tolist()

results = collection.query(query_embeddings=q_emb, n_results=3)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"\n[{meta['source']} - page {meta['page']}]")
    print(doc[:400], "...")