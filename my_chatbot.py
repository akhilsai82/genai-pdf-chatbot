import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.schema import Document
from transformers import pipeline
import PyPDF2

st.title("🧠 GENAI Chatbot with File Upload")

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

    # Local LLM (small model example)
    llm_pipeline = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=200
    )

    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    user_input = st.text_input("Ask a question about the document:")

    if user_input:
        response = qa_chain.run(user_input)
        st.write("🤖 Chatbot:", response)