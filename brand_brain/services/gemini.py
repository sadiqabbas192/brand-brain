from google import genai
from ..config import GEMINI_API_KEY

_client = None

def get_gemini_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY4 not set in environment variables")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client
