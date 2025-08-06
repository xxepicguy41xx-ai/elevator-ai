
import streamlit as st
import chromadb
from chromadb.config import Settings
from InstructorEmbedding import INSTRUCTOR
import torch
import os
from groq import Groq

# Load embedding model
model = INSTRUCTOR("hkunlp/instructor-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Connect to local ChromaDB (manuals only)
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="elevator_db"  # path where your ChromaDB was saved
))
collection = chroma_client.get_or_create_collection(name="elevator_docs")

# Connect to Groq
groq_client = Groq(api_key="YOUR_GROQ_API_KEY")

# Ask function
def ask_question_local(user_input: str):
    query_emb = model.encode(
        [["Represent the elevator maintenance issue for retrieval:", user_input]],
        convert_to_numpy=True,
        device=device
    )[0]

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=5
    )

    # Combine top 5 sources
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    sources = ""
    for doc, meta in zip(docs, metadatas):
        file = meta.get("source_file", "Unknown")
        page = meta.get("page", "?")
        drive_id = meta.get("drive_id", "")
        link = f"https://drive.google.com/file/d/{drive_id}/view?usp=sharing" if drive_id else ""
        sources += f"
• Page {page} of [{file}]({link})"

    context = "\n\n".join(docs)
    prompt = f"Answer the following based only on the context below. If not in the context, say 'I don't know.'\n\nContext:\n{context}\n\nQuestion: {user_input}"

    # Run Groq LLM
    chat_completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = chat_completion.choices[0].message.content.strip()
    return answer, sources

# Streamlit UI
st.title("🚧 Elevator Manual Assistant")
query = st.text_input("Ask a maintenance question:")

if query:
    with st.spinner("Thinking..."):
        answer, sources = ask_question_local(query)
    st.markdown(f"### 💬 Answer:
{answer}")
    st.markdown("### 📚 Sources:")
    st.markdown(sources)
