### Basic AI Agent with WEB UI

import streamlit as st
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from httpx import ConnectError

# Configuration: Get Ollama base URL from environment variable or use default
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Initialize LLM with error handling
@st.cache_resource
def get_llm():
    try:
        llm_instance = OllamaLLM(
            model="llama3.2:1b",
            base_url=OLLAMA_BASE_URL
        )
        # Test connection with a simple call (optional, can be removed if too slow)
        return llm_instance
    except Exception as e:
        # Don't show error here as it's cached - will be handled in UI
        return None

llm = get_llm()

# Initialize Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()  # Stores user-AI conversation history

# Define AI Chat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
)

# Function to run AI chat with memory
def run_chain(question):
    if llm is None:
        return "Error: Ollama LLM is not available. Please ensure Ollama is running locally or configure a remote endpoint."
    
    try:
        # Retrieve past chat history
        chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])
        
        # Generate the AI response
        response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
        
        # Store new user input and AI response in memory
        st.session_state.chat_history.add_user_message(question)
        st.session_state.chat_history.add_ai_message(response)
        
        return response
    except ConnectError as e:
        error_msg = (
            "❌ **Connection Error**: Cannot connect to Ollama.\n\n"
            "**For Local Deployment:**\n"
            "1. Make sure Ollama is installed: https://ollama.ai/\n"
            "2. Start Ollama service\n"
            "3. Pull the model: `ollama pull llama3.2:1b`\n\n"
            "**For Cloud Deployment:**\n"
            "Ollama requires a local installation. For Streamlit Cloud, consider:\n"
            "- Using a cloud LLM API (OpenAI, Anthropic, etc.)\n"
            "- Setting up a remote Ollama server and configuring OLLAMA_BASE_URL\n"
            "- Deploying on a platform that supports Ollama (like Railway, Render with custom setup)"
        )
        return error_msg
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit UI
st.title("AI Chatbot with Memory")
st.write("Ask me anything!")

# Show connection status in sidebar
with st.sidebar:
    st.header("Configuration")
    if llm is None:
        st.error("⚠️ Ollama not connected")
    else:
        st.success("✅ Ollama connected")
    st.caption(f"Endpoint: {OLLAMA_BASE_URL}")
    st.caption("Model: llama3.2:1b")
    
    st.info(
        "**Note:** This app requires Ollama running locally. "
        "For cloud deployment, configure a remote Ollama endpoint via OLLAMA_BASE_URL environment variable."
    )

user_input = st.text_input("Your Question:")

if user_input:
    with st.spinner("Thinking..."):
        response = run_chain(user_input)
    st.write(f"**You:** {user_input}")
    st.write(f"**AI:** {response}")

# Show full chat history
if st.session_state.chat_history.messages:
    st.subheader("Chat History")
    for msg in st.session_state.chat_history.messages:
        st.write(f"**{msg.type.capitalize()}**: {msg.content}")
