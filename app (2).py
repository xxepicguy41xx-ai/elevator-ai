import streamlit as st
from InstructorEmbedding import INSTRUCTOR
import chromadb
from chromadb.config import Settings
import torch

# Load ChromaDB
chroma_client = chromadb.PersistentClient(path="ChromaDB_Backup")
collection = chroma_client.get_or_create_collection(name="elevator_manuals")

# Load embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = INSTRUCTOR("hkunlp/instructor-base")
model.to(device)

# Query helper
def query_docs(user_input):
    embedded_query = model.encode(
        [["Represent the elevator maintenance issue for retrieval:", user_input]],
        convert_to_numpy=True,
        device=device
    )[0]

    results = collection.query(query_embeddings=[embedded_query], n_results=5)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    st.write("### Top Results")
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        st.markdown(f"""
        **Result {i+1}**
        - **Page:** {meta['page']}
        - **Source File:** {meta['source_file']}
        - **Link:** [Open in Drive]({meta['link']})
        - **Answer Snippet:** {doc[:500]}...
        """)
    
# UI
st.title("Elevator Manual Assistant")
query = st.text_input("Ask a question about elevator maintenance:")

if query:
    query_docs(query)
