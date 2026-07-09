import os
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.schema import Document
import PyPDF2

# Load .env for local development
load_dotenv()

st.title("🧠 GENAI Chatbot with File Upload")

# Try environment variable first (.env or system env), fall back to Streamlit secrets
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (FileNotFoundError, KeyError):
        api_key = None

if not api_key:
    st.error("Groq API key not found. Please configure GROQ_API_KEY.")
    st.stop()

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
docs = []

if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            docs.append(Document(page_content=text))
    st.success(f"📄 {len(pdf_reader.pages)} pages loaded!")

if docs:
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Groq LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    user_input = st.text_input("Ask a question about the document:")
    if user_input:
        result = qa_chain.invoke({"query": user_input})
        st.write("🤖 Chatbot:", result["result"])