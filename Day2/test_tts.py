import pyttsx3

try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.say("Testing text to speech")
    engine.runAndWait()
    print("TTS test completed.")
except Exception as e:
    print(f"TTS error: {e}")
