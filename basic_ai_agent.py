from langchain_ollama import OllamaLLM


def run_demo():
    # Load a smaller AI model from Ollama that fits in memory
    # Make sure you have this model pulled: `ollama pull llama3.2:1b`
    llm = OllamaLLM(model="llama3.2:1b")

    print("\nWelcome to your AI Agent! Ask me anything.")

    while True:
        question = input("\nYour Question (or type 'exit' to stop): ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        response = llm.invoke(question)
        print("\nAI Response:", response)


if __name__ == "__main__":
    run_demo()