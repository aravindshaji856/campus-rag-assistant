import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="Campus RAG Assistant")
st.title("📚 Campus RAG Assistant")

with st.sidebar:
    st.header("Upload notes")
    pdf = st.file_uploader("Choose a PDF", type="pdf")
    if pdf and st.button("Upload & ingest"):
        with st.spinner("Processing..."):
            res = requests.post(API + "/upload", files={"file": (pdf.name, pdf.getvalue(), "application/pdf")})
        if res.ok:
            st.success(f"Added {res.json()['chunks_added']} chunks from {pdf.name}")
        else:
            st.error(res.json().get("detail", "Upload failed"))

question = st.text_input("Ask a question about your notes")
if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        res = requests.post(API + "/ask", json={"question": question})
    if res.ok:
        data = res.json()
        st.write(data["answer"])
        if data["sources"]:
            st.caption("Sources: " + ", ".join(f"{s['source']} (p.{s['page']})" for s in data["sources"]))
    else:
        st.error(res.json().get("detail", "Something went wrong"))