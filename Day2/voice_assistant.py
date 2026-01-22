import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import playsound
import os

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# =========================
# Load AI Model
# =========================
llm = OllamaLLM(model="llama3.2:1b")  # or "llama3"

# =========================
# Initialize Memory
# =========================
chat_history = ChatMessageHistory()

# =========================
# Text-to-Speech
# =========================
try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
except Exception as e:
    print(f"Error initializing TTS engine: {e}")
    exit(1)

def speak(text):
    print(f"Speaking: {text}")
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"pyttsx3 failed: {e}, using gTTS")
        tts = gTTS(text=text, lang='en')
        tts.save("temp.mp3")
        playsound.playsound("temp.mp3")
        os.remove("temp.mp3")

# =========================
# Speech Recognition
# =========================
recognizer = sr.Recognizer()

def listen():
    try:
        with sr.Microphone() as source:
            print("\n🎤 Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        query = recognizer.recognize_google(audio)
        print(f"🗣 You said: {query}")
        return query.lower()

    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return ""

    except sr.RequestError:
        print("❌ Speech Recognition service unavailable")
        return ""

    except Exception as e:
        print(f"❌ Error with microphone: {e}")
        return ""

# =========================
# Prompt Template
# =========================
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
Previous Conversation:
{chat_history}

User: {question}
AI:
"""
)

# =========================
# AI Processing
# =========================
def run_chain(question):
    try:
        chat_history_text = "\n".join(
            [f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages]
        )

        response = llm.invoke(
            prompt.format(
                chat_history=chat_history_text,
                question=question
            )
        )

        response = str(response)  # Ensure response is a string

        chat_history.add_user_message(question)
        chat_history.add_ai_message(response)

        return response
    except Exception as e:
        print(f"❌ Error in AI processing: {e}")
        return "Sorry, I encountered an error."

# =========================
# Main Loop
# =========================
speak("Hello! I am your AI voice assistant. How can I help you today?")

while True:
    query = listen()

    if not query:
        continue

    if "exit" in query or "stop" in query:
        speak("Goodbye! Have a great day.")
        break

    response = run_chain(query)
    print(f"🤖 AI: {response}")
    if isinstance(response, str):
        speak(response)
    else:
        speak("Sorry, I got an invalid response.")
