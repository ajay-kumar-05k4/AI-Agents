import streamlit as st
import speech_recognition as sr
import pyttsx3
import threading

from langchain_ollama import OllamaLLM
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate

# ---------------------------------
# Streamlit Page Config
# ---------------------------------
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎙",
    layout="centered"
)

# ---------------------------------
# Load AI Model (Ollama)
# ---------------------------------
llm = OllamaLLM(model="llama3.2:1b")  # or "llama3"

# ---------------------------------
# Initialize Chat Memory
# ---------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

if "listening" not in st.session_state:
    st.session_state.listening = False

# ---------------------------------
# Text-to-Speech (Thread Safe)
# ---------------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 160)

def speak(text):
    def run():
        engine.stop()
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=run, daemon=True).start()

# ---------------------------------
# Speech Recognition
# ---------------------------------
recognizer = sr.Recognizer()

def listen():
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)

    try:
        query = recognizer.recognize_google(audio)
        st.success(f"🗣 You said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        st.error("❌ Could not understand audio")
        return ""
    except sr.RequestError:
        st.error("❌ Speech Recognition service unavailable")
        return ""

# ---------------------------------
# Prompt Template
# ---------------------------------
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
Previous conversation:
{chat_history}

User: {question}
AI:
"""
)

# ---------------------------------
# AI Processing Function
# ---------------------------------
def run_chain(question):
    chat_history_text = "\n".join(
        [f"{msg.type.capitalize()}: {msg.content}"
         for msg in st.session_state.chat_history.messages]
    )

    response = llm.invoke(
        prompt.format(
            chat_history=chat_history_text,
            question=question
        )
    )

    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)

    return response

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.title("🎙 AI Voice Assistant")
st.caption("Ollama + LangChain + Speech Recognition + TTS")

st.write("Click the button and speak to the AI")

if st.button("🎧 Start Listening", disabled=st.session_state.listening):
    st.session_state.listening = True

    user_query = listen()

    if user_query:
        ai_response = run_chain(user_query)

        st.markdown(f"**You:** {user_query}")
        st.markdown(f"**AI:** {ai_response}")

        speak(ai_response)

    st.session_state.listening = False

# ---------------------------------
# Chat History Display
# ---------------------------------
st.subheader("💬 Chat History")

for msg in st.session_state.chat_history.messages:
    role = "🧑 User" if msg.type == "human" else "🤖 AI"
    st.markdown(f"**{role}:** {msg.content}")
