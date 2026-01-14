import re
from enum import Enum
from ..services.gemini import safe_generate_content

class IntentType(Enum):
    KNOWLEDGE = "knowledge"    # Allow
    REASONING = "reasoning"    # Allow
    CREATIVE = "creative"      # Block

# Regex for Obvious Creative Intents (Rule-Based First)
CREATIVE_PATTERNS = [
    r"create", r"write", r"generate", r"design", r"make me a", r"draft", 
    r"slogan", r"logo", r"ad copy", r"campaign"
]

def classify_intent_hybrid(query: str) -> IntentType:
    # 1. Rule-Based Check
    query_lower = query.lower()
    for pattern in CREATIVE_PATTERNS:
        if re.search(pattern, query_lower):
            # print(f"   🛡️ Rule-Based Intent Detection: CREATIVE (Blocked pattern: '{pattern}')")
            return IntentType.CREATIVE
            
    # 2. Gemini Fallback for Ambiguity
    prompt = f"""
    Classify the following query into one of 3 categories:
    1. KNOWLEDGE (Questions about facts, brand voice, mission, identity)
    2. REASONING (Questions asking for validation, 'is this on-brand?', 'why?')
    3. CREATIVE (Requests to create, write, generate, design new assets)
    
    QUERY: {query}
    
    RETURN ONLY ONE WORD: KNOWLEDGE, REASONING, or CREATIVE.
    """
    
    try:
        response = safe_generate_content(
            contents=prompt
        )
        result = response.text.strip().upper()
        if "CREATIVE" in result: return IntentType.CREATIVE
        if "REASONING" in result: return IntentType.REASONING
        return IntentType.KNOWLEDGE
    except Exception as e:
        print(f"   ⚠️ Intent Classification Failed: {e}. Defaulting to KNOWLEDGE.")
        return IntentType.KNOWLEDGE
