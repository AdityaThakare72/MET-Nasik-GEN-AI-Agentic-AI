from langchain_ollama import ChatOllama

# Initialize the model using the exact name from your 'ollama ls' output
model = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0,  # Keeping it deterministic for coding tasks
)

# Simple string invocation
response = model.invoke("Write a Python function to find the nth Fibonacci number.")

print("\n--- Local Model Response ---")
print(response.content)