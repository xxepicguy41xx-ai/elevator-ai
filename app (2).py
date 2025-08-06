import streamlit as st
from InstructorEmbedding import INSTRUCTOR
import chromadb
import torch

# --- Load Chroma vector store from uploaded directory ---
chroma_client = chromadb.PersistentClient(path="chromadb_store")
collection = chroma_client.get_or_create_collection("elevator_docs")

# --- Load embedding model (Instructor base) ---
model = INSTRUCTOR("hkunlp/instructor-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- Groq OpenAI-compatible API client ---
from openai import OpenAI
import os

# Set your Groq API key securely in Streamlit secrets
client = OpenAI(api_key=st.secrets["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")

# --- App UI ---
st.title("🚀 Elevator AI Assistant")
st.write("Ask me questions based on the elevator manuals. I’ll answer using your uploaded documents only.")

user_input = st.text_input("🔍 Enter your question:", placeholder="e.g., What should I check if the elevator stops at the wrong floor?")

if user_input:
    with st.spinner("🔎 Searching the manuals..."):
        # Embed query
        query_embed = model.encode(
            [["Represent the elevator maintenance issue for retrieval:", user_input]],
            convert_to_numpy=True,
            device=device
        )[0]

        # Search top 5 relevant chunks
        results = collection.query(
            query_embeddings=[query_embed],
            n_results=5,
            include=["documents", "metadatas"]
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        # Format context with clickable Drive links
        context_blocks = []
        for doc, meta in zip(docs, metas):
            link = f"https://drive.google.com/file/d/{meta['drive_id']}/view"
            context_blocks.append(f"**📄 [{meta['source_file']} – page {meta['page']}]({link})**\n\n{doc}")

        context = "\n\n---\n\n".join(context_blocks)

        # Send to Groq (GPT-OSS-120b)
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are an expert elevator assistant. Only answer using the context provided."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_input}"}
            ],
            temperature=0.3
        )

        st.markdown("## 💡 Answer")
        st.write(response.choices[0].message.content.strip())

        with st.expander("📚 Sources"):
            st.markdown(context, unsafe_allow_html=True)
