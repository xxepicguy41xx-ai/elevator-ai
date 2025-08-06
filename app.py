import streamlit as st
from langchain.chat_models import ChatGroq
from langchain.schema import HumanMessage

# Load Groq API key from Streamlit Secrets
groq_api_key = st.secrets["groq"]["api_key"]

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-20b"
)

st.title("🔧 Elevator AI Assistant")

question = st.text_input("Ask a question about elevator repair")

if question:
    with st.spinner("Thinking..."):
        response = llm([HumanMessage(content=question)])
        st.write(response.content)
