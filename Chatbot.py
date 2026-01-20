### Basic AI Agent with WEB UI

import streamlit as st
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from httpx import ConnectError

# Try to import OpenAI (optional, for cloud deployment)
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configuration: Get LLM provider and settings from environment variables
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "openai"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Initialize LLM with error handling
@st.cache_resource
def get_llm():
    if LLM_PROVIDER.lower() == "openai":
        if not OPENAI_AVAILABLE:
            return None, "OpenAI package not installed. Install with: pip install langchain-openai"
        if not OPENAI_API_KEY:
            return None, "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
        try:
            llm_instance = ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0.7
            )
            return llm_instance, "openai"
        except Exception as e:
            return None, f"Failed to initialize OpenAI: {str(e)}"
    else:
        # Default to Ollama
        try:
            llm_instance = OllamaLLM(
                model="llama3.2:1b",
                base_url=OLLAMA_BASE_URL
            )
            return llm_instance, "ollama"
        except Exception as e:
            return None, f"Failed to initialize Ollama: {str(e)}"

llm, provider = get_llm()

# Initialize Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Define AI Chat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
)

# Function to run AI chat with memory
def run_chain(question):
    if llm is None:
        return f"Error: LLM is not available. {provider}"
    
    try:
        # Retrieve past chat history
        chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])
        
        # Format prompt
        formatted_prompt = prompt.format(chat_history=chat_history_text, question=question)
        
        # Generate the AI response
        if provider == "openai":
            # OpenAI uses ChatOpenAI which returns a message object
            response_obj = llm.invoke(formatted_prompt)
            response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
        else:
            # Ollama returns string directly
            response = llm.invoke(formatted_prompt)
        
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
            "Set LLM_PROVIDER=openai and OPENAI_API_KEY in your environment variables."
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
        st.error(f"⚠️ LLM not connected")
        st.caption(f"Error: {provider}")
    else:
        st.success(f"✅ {provider.upper()} connected")
    
    if provider == "ollama":
        st.caption(f"Endpoint: {OLLAMA_BASE_URL}")
        st.caption("Model: llama3.2:1b")
    else:
        st.caption(f"Model: {OPENAI_MODEL}")
    
    st.info(
        "**Provider:** Set LLM_PROVIDER environment variable to 'ollama' or 'openai'.\n\n"
        "**For Cloud:** Use OpenAI by setting:\n"
        "- LLM_PROVIDER=openai\n"
        "- OPENAI_API_KEY=your-api-key"
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
