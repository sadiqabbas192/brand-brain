import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

NEON_DB_URL = os.getenv("NEON_DB_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY4")

if not all([NEON_DB_URL, PINECONE_API_KEY, GEMINI_API_KEY]):
    print("Warning: Missing one or more required environment variables (NEON_DB_URL, PINECONE_API_KEY, GEMINI_API_KEY4).")

DEBUG_MODE = False

def set_debug_mode(enabled: bool):
    global DEBUG_MODE
    DEBUG_MODE = enabled

def print_debug(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)

# Constants
ALLOWED_INTENTS = {"explain_brand", "validate_copy", "justify_decision", "minimal_rewrite", "reasoning"}
FORBIDDEN_KEYWORDS = {"cheap", "free", "lowest price", "clearance", "sale", "loud", "discount", "discounting"}
