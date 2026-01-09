
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    print("\n--- Test 1: title in specific config ---")
    try:
        # Attempting to pass title and output_dimensionality in config
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents="Validation text",
            config={
                'output_dimensionality': 768,
                'task_type': 'RETRIEVAL_DOCUMENT',
                'title': 'Test Title'
            }
        )
        print(f"✅ Success! Dimension: {len(result.embeddings[0].values)}")
    except Exception as e:
        print(f"❌ Failed Test 1: {e}")

    print("\n--- Test 2: title as kwarg (Expected Failure) ---")
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents="Validation text",
            config={'output_dimensionality': 768},
            title="Test Title"
        )
        print(f"✅ Success! Dimension: {len(result.embeddings[0].values)}")
    except Exception as e:
        print(f"❌ Failed Test 2 (As expected): {e}")

else:
    print("No API Key found")
