import streamlit as st
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import PyPDF2
import io

st.title("📄 Offline Document Q&A (Mistral + FAISS)")

# Load LLM
# llm = Ollama(model="mistral")
llm = Ollama(model="gemma:2b")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
index = None
chunk_map = []

uploaded_file = st.file_uploader("Upload PDF or Excel file", type=["pdf", "xlsx", "xls"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == 'pdf':
        reader = PyPDF2.PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file_type in ['xlsx', 'xls']:
        df = pd.read_excel(uploaded_file)
        text = df.to_csv(index=False)
        print(text)
    else:
        st.error("Unsupported file type")
        st.stop()

    docs = splitter.split_documents([Document(page_content=text)])
    texts = [doc.page_content for doc in docs]
    chunk_map = texts

    embeddings = embedder.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    st.success(f"Processed {len(texts)} chunks.")

question = st.text_input("Ask a question about the file")

if question and index is not None:
    query_embedding = embedder.encode([question], convert_to_numpy=True)
    D, I = index.search(query_embedding, 3)
    top_chunks = [chunk_map[i] for i in I[0]]
    
    context = "\n---\n".join(top_chunks)
    # context = context[:2000] 
    prompt = f"Document context:\n{context}\n\nAnswer this question:\n{question}"
    response = llm(prompt)
    st.markdown("### 🤖 Answer:")
    st.write(response)
