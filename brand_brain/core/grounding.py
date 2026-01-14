from typing import List, Dict
from ..services.gemini import safe_generate_content
from google.genai import types
from ..config import print_debug

def should_trigger_grounding(query: str, retrieved_chunks: List[Dict], avg_similarity: float) -> bool:
    """
    Decides whether to trigger live grounding based on retrieval confidence and query content.
    """
    q = query.lower()

    # Rule 1 - Low Retrieval Confidence
    if avg_similarity < 0.65 or len(retrieved_chunks) < 2:
        return True

    # Rule 2 - External Entity Detection
    external_markers = [
        "compare", "vs", "versus", "difference",
        "alternative", "benchmark", "competitor",
        "market", "industry"
    ]
    if any(m in q for m in external_markers):
        return True

    # Rule 3 - Comparison / Benchmark Intent (Covered by Rule 2 markers largely, but checking specifically for intent implies context)
    # The prompt combined Rule 2 and 3 slightly in description but separated them in list. 
    # The code snippet provided by user only had one list for external_markers that included Rule 3 words.
    # I will stick to the user provided python design.
    
    # Rule 4 - Freshness Signals
    freshness_markers = [
        "latest", "current", "recent",
        "today", "now", "trend", "from the web", "from the internet"
    ]
    if any(f in q for f in freshness_markers):
        return True

    return False

def fetch_live_context(query: str, brand_name: str = "the brand") -> str:
    """
    Fetches ephemeral live context using Gemini 2.5 Flash with Google Search tool.
    This context is NEVER stored.
    """
    
    # Using the prompt structure from the requirements, but adding Brand Context
    prompt = f"""
    Research and gather factual information to answer the following query for the brand "{brand_name}".
    
    Query: '{query}'

    Instructions:
    - Search specifically for "{brand_name}".
    - If specific regional information (e.g., India) is missing, look for global/parent brand standards (e.g., Westinghouse Global) that might apply.
    - Provide concrete details (like hex codes, specs, dates) if available.
    - Do NOT market or invent facts.
    """

    try:
        response = safe_generate_content(
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        print_debug(f"   🔎 Grounding Raw Response Valid: {bool(response.text)}")
        return response.text if response.text else ""
    except Exception as e:
        print_debug(f"   ⚠️ Live Fetch Error: {e}")
        return ""
