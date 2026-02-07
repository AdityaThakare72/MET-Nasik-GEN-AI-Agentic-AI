from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# --- Load Gemini API key ---
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not gemini_key:
    print("Error: Gemini API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.")
    exit()

# --- Gemini Chat Model Setup ---
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=gemini_key,
    temperature=0.7,
    max_output_tokens=500,
)

# --- Chat History Initialization ---
chat_history = [
    SystemMessage(content="You are a helpful AI assistant.")
]

print("Type 'exit' to end the chat.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))

    # Pass entire chat history to Gemini
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)

print("\n--- Chat Ended ---")
print("Final Chat History:")
for message in chat_history:
    if isinstance(message, SystemMessage):
        print(f"System: {message.content}")
    elif isinstance(message, HumanMessage):
        print(f"You: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")
