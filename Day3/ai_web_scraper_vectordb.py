import requests
from bs4 import BeautifulSoup
import streamlit as st
import faiss
import numpy as np

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter


llm = OllamaLLM(model="llama3.2:1b")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_DIM = 384
index = faiss.IndexFlatL2(VECTOR_DIM)
vector_store = {}

def scrape_website(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return "❌ Failed to fetch website"

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)

        return text[:5000]

    except Exception as e:
        return f"❌ Error: {str(e)}"

def store_in_faiss(text, url):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    vectors = embeddings.embed_documents(chunks)
    vectors = np.array(vectors, dtype=np.float32)

    start_id = len(vector_store)
    index.add(vectors)

    for i, chunk in enumerate(chunks):
        vector_store[start_id + i] = (url, chunk)

    return "✅ Data stored successfully"

def retrieve_and_answer(query):
    if index.ntotal == 0:
        return "❌ No data available"

    query_vector = embeddings.embed_query(query)
    query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

    D, I = index.search(query_vector, k=3)

    context = ""
    for idx in I[0]:
        if idx in vector_store:
            context += vector_store[idx][1] + "\n\n"

    if not context.strip():
        return "❌ No relevant data found"

    prompt = f"""
Context:
{context}

Question: {query}
Answer:
"""
    return llm.invoke(prompt)

st.title("🧠 AI Web Scraper with FAISS")

url = st.text_input("Enter Website URL")

if url:
    content = scrape_website(url)
    if "❌" in content:
        st.error(content)
    else:
        st.success(store_in_faiss(content, url))

query = st.text_input("Ask a question")

if query:
    st.subheader("AI Answer")
    st.write(retrieve_and_answer(query))