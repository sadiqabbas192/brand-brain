
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents="Hello world"
        )
        # Using the object attribute access for the wrapper
        print(f"Dimension: {len(result.embeddings[0].values)}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("No API Key found")
