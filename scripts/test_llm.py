import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Initialize the Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

try:
    chat = client.chats.create(model="gemini-3.6-flash")
    response = chat.send_message("Explain RAG in one sentence.")
    
    print("\n--- Output ---")
    print(response.text)

except Exception as e:
    print("\n--- Connection Error ---")
    print(e)