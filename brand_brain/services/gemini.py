from google import genai
import os
from itertools import cycle
from ..config import GEMINI_API_KEY

_gemini_key_cycle = None

def init_gemini_keys():
    global _gemini_key_cycle
    # In a real scenario, we might have multiple keys in env vars
    # For now, we reuse the existing single key or check for others if they exist
    keys = [
        ("GEMINI_API_KEY1", os.getenv("GEMINI_API_KEY1")),
        ("GEMINI_API_KEY2", os.getenv("GEMINI_API_KEY2")),
        ("GEMINI_API_KEY3", os.getenv("GEMINI_API_KEY3")),
        ("GEMINI_API_KEY4", os.getenv("GEMINI_API_KEY4")),
        ("GEMINI_API_KEY5", os.getenv("GEMINI_API_KEY5")),
        ("GEMINI_API_KEY6", os.getenv("GEMINI_API_KEY6")),
        ("GEMINI_API_KEY7", os.getenv("GEMINI_API_KEY7")),
        ("GEMINI_API_KEY8", os.getenv("GEMINI_API_KEY8")),
        ("GEMINI_API_KEY9", os.getenv("GEMINI_API_KEY9")),
        ("GEMINI_API_KEY10", os.getenv("GEMINI_API_KEY10")),
        ("GEMINI_API_KEY11", os.getenv("GEMINI_API_KEY11"))
    ]
    # Filter valid keys
    valid_keys = [k for k in keys if k[1]]
    
    if not valid_keys:
        if GEMINI_API_KEY:
             valid_keys = [("GEMINI_API_KEY", GEMINI_API_KEY)]
        else:
             raise ValueError("No Gemini API keys found.")
    
    _gemini_key_cycle = cycle(valid_keys)

AVAILABLE_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-3-flash-preview"
]

_gemini_model_cycle = cycle(AVAILABLE_MODELS)

def get_gemini_model() -> str:
    global _gemini_model_cycle
    return next(_gemini_model_cycle)

def get_gemini_client():
    global _gemini_key_cycle
    if _gemini_key_cycle is None:
        init_gemini_keys()
    
    key_name, api_key = next(_gemini_key_cycle)
    return genai.Client(api_key=api_key), key_name

def safe_generate_content(model: str = None, contents=None, config=None, max_retries=4):
    """
    Safely call Gemini with automatic API key rotation on failure.
    If model is None, it rotates through the available models.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            current_model = model if model else get_gemini_model()
            
            client, key_name = get_gemini_client()
            response = client.models.generate_content(
                model=current_model,
                contents=contents,
                config=config
            )
            
            # Attach usage info for debugging/display
            response._usage_info = {
                "model_name": current_model,
                "api_key_name": key_name
            }
            return response
            
        except Exception as e:
            last_error = e
            # print(f"⚠️ Gemini call failed (attempt {attempt+1}/{max_retries}): {e}")
    
    raise RuntimeError(f"❌ All Gemini API keys exhausted. Last error: {last_error}")
