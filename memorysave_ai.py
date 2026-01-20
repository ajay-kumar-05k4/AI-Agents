from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI Model
llm = OllamaLLM(model="llama3.2:1b")  # Changed to smaller model that fits in memory

# Initialize Memory
chat_history = ChatMessageHistory()  # Stores user-AI conversation history

# Define AI Chat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
)

# Function to run AI chat with memory
def run_chain(question):
    # Retrieve chat history manually
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])
    
    # Format prompt with chat history and question
    formatted_prompt = prompt.format(chat_history=chat_history_text, question=question)
    
    # Get AI response
    response = llm.invoke(formatted_prompt)
    
    # Store new user input and AI response in memory
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)
    
    return response

# Interactive CLI Chatbot
print("\nAI Chatbot with Memory")
print("Type 'exit' to stop.")

while True:
    user_input = input("\nYou: ")
    
    if user_input.lower() == "exit":
        print("\nGoodbye!")
        break
    
    ai_response = run_chain(user_input)
    print(f"\nAI: {ai_response}")
