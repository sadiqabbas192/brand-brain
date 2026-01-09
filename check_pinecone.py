
import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    print("❌ Error: PINECONE_API_KEY not found in environment variables.")
else:
    # Print masked key for verification
    masked_key = f"{PINECONE_API_KEY[:4]}...{PINECONE_API_KEY[-4:]}" if len(PINECONE_API_KEY) > 8 else "****"
    print(f"🔑 Found API Key: {masked_key}")
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        indexes = pc.list_indexes()
        print(f"✅ Connection Successful! Indexes found: {[i.name for i in indexes]}")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
