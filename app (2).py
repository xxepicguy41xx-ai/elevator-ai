# elevator_app.py

import os
import json
import torch
import fitz  # PyMuPDF
import chromadb
import streamlit as st
from InstructorEmbedding import INSTRUCTOR
from chromadb.config import Settings
from groq import Groq

# === Configuration ===
PDF_FOLDER = "Elevator_Service_Manuals"  # 📁 Must be in the same GitHub repo
CACHE_PATH = "chunk_cache.json"
CHROMA_PATH = "chromadb_store"

# === Load Groq API key ===
GROQ_API_KEY = st.secrets["groq_api_key"]
client = Groq(api_key=GROQ_API_KEY)

# === Load model and ChromaDB ===
model = INSTRUCTOR("hkunlp/instructor-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("elevator_docs")

# === Helper: Embed and store chunks ===
def chunk_and_embed_all():
    all_chunks = []
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            processed_cache = json.load(f)
    else:
        processed_cache = {}

    for file in sorted(os.listdir(PDF_FOLDER)):
        if not file.endswith(".pdf"): continue
        file_path = os.path.join(PDF_FOLDER, file)
        mod_time = os.path.getmtime(file_path)
        if file in processed_cache and processed_cache[file]["mtime"] == mod_time:
            continue

        try:
            doc = fitz.open(file_path)
            file_chunks = []
            for i, page in enumerate(doc):
                text = page.get_text("text")
                if len(text.strip()) < 50: continue
                chunks = [text[j:j+800] for j in range(0, len(text), 800)]
                for chunk in chunks:
                    file_chunks.append({
                        "content": chunk,
                        "metadata": {
                            "source_file": file,
                            "page": i + 1
                        }
                    })
            all_chunks.extend(file_chunks)
            processed_cache[file] = {"mtime": mod_time, "chunks": len(file_chunks)}
        except Exception as e:
            st.warning(f"Failed to process {file}: {e}")

    with open(CACHE_PATH, "w") as f:
        json.dump(processed_cache, f)

    for i in range(0, len(all_chunks), 32):
        batch = all_chunks[i:i+32]
        texts = [x["content"] for x in batch]
        metas = [x["metadata"] for x in batch]
        ids = [f"id_{i+j}" for j in range(len(batch))]
        with torch.no_grad():
            embeds = model.encode(
                [["Represent the elevator maintenance issue for retrieval:", t] for t in texts],
                convert_to_numpy=True,
                device=device
            )
        collection.add(documents=texts, embeddings=embeds, metadatas=metas, ids=ids)

# === Helper: Ask the assistant ===
def ask_question(query: str, top_k: int = 5):
    embed = model.encode(
        [["Represent the elevator maintenance issue for retrieval:", query]],
        convert_to_numpy=True
    )[0]

    results = collection.query(
        query_embeddings=[embed],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    context_blocks = []
    for doc, meta in zip(docs, metas):
        context_blocks.append(f"**{meta['source_file']} (page {meta['page']})**\n\n{doc}")

    full_context = "\n\n---\n\n".join(context_blocks)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are an expert elevator assistant. Only use the provided context."},
            {"role": "user", "content": f"Context:\n{full_context}\n\nQuestion:\n{query}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# === Streamlit UI ===
st.set_page_config(page_title="Elevator Manual Assistant")
st.title("🛠️ Elevator Manual Assistant")

if st.button("🔁 Index Manuals (only run if changed)"):
    with st.spinner("Reading and embedding manuals..."):
        chunk_and_embed_all()
    st.success("Manuals indexed!")

query = st.text_input("Ask a question about elevator maintenance:")
if query:
    with st.spinner("Thinking..."):
        answer = ask_question(query)
    st.markdown("### ✅ Answer")
    st.markdown(answer)
