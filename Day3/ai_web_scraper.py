import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:1b") 

def scrape_website(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"} 
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return "❌ Failed to fetch website"

        soup = BeautifulSoup(response.text, "html.parser") 
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)

        return text[:2000] 
    except Exception as e:
        return f"❌ Error: {e}"

def summarize_content(content):
    return llm.invoke(
        f"📝 Summarize the following content:\n\n{content[:1000]}"
    )

st.title("🤖 AI-Powered Web Scraper")

url = st.text_input("🌍 Enter Website URL")

if url:
    content = scrape_website(url)

    if "Error" in content or "Failed" in content:
        st.error(content)
    else:
        summary = summarize_content(content)
        st.subheader("✨ Website Summary")
        st.write(summary)
