from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


"""  
Why not just put everything in one string?

    Instruction Persistence: Models are trained to give higher priority to the system role. If you put instructions in the user message, the model might "forget" them if the user gives a very long input (this is called "context drift").

    Standardization: Every provider (OpenAI, Anthropic, Google) has a slightly different API. By using LangChain's SystemMessage, you write code once, and LangChain translates it to the correct format for the specific model you're using.

    Safety: It helps prevent "Prompt Injection." By keeping instructions in the system role, it's harder for a user to trick the AI into saying, "Ignore all previous instructions."
"""
# AIMessage -> model's response, HumanMessage -> user input, SystemMessage -> system instructions

load_dotenv()

# --- Load Gemini API key ---
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not gemini_key:
    print("Error: Gemini API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.")
    exit()

# --- Gemini model setup ---
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=gemini_key,
    temperature=0.7,
    max_output_tokens=500,
)

# --- Initial chat messages ---
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about LangChain"),
]

# --- Invoke Gemini with messages ---
result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(f"AI: {result.content}")

print("\n--- Full Message History ---")
for message in messages:
    if isinstance(message, SystemMessage):
        print(f"System: {message.content}")
    elif isinstance(message, HumanMessage):
        print(f"Human: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")
