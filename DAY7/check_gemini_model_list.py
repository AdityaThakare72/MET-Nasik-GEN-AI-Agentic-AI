# file: list_gemini_models_langchain.py

import os
from dotenv import load_dotenv
from google import genai  # google-genai SDK

# Load API key from .env
load_dotenv()



# Create Gemini client (direct Gemini API, not Vertex)
client = genai.Client()

print("Models that support generateContent:\n")

models_supporting_generate = []
for m in client.models.list():  # Lists all models exposed by Gemini API[web:7]
    if "generateContent" in getattr(m, "supported_actions", []):
        models_supporting_generate.append(m.name)

for name in sorted(models_supporting_generate):
    print(name)
