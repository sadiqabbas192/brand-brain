import uuid
import numpy as np
from typing import Dict, List, Any
from pinecone import Pinecone
from ..config import ALLOWED_INTENTS, FORBIDDEN_KEYWORDS
from ..services.embedding import generate_embedding
from ..services.gemini import get_gemini_client
from ..database import get_pinecone_client

def calculate_brand_centroid(brand_id: str, org_id: str, top_n=5) -> List[float]:
    """
    Calculates deterministic centroid from top-N 'brand_voice' chunks.
    In a real system, this is pre-computed. Here we fetch via a 'neutral' query.
    """
    pc = get_pinecone_client()
    idx = pc.Index("brand-brain-index")
    namespace = f"{org_id}:{brand_id}:brand_voice"
    
    # Deterministic query to fetch representative chunks
    # We use a static string that represents the ideal voice to find core chunks
    query_vec = generate_embedding("brand voice tone philosophy identity")
    
    results = idx.query(
        vector=query_vec,
        top_k=top_n,
        namespace=namespace,
        include_values=True
    )
    
    vectors = []
    if results['matches']:
         for m in results['matches']:
             if m.get('values'):
                 vectors.append(m['values'])
             
    if not vectors:
        return []
    
    return np.mean(vectors, axis=0).tolist()

def check_brand_safety(user_query: str, brand_id_str: str, intent: Any) -> Dict:
    # 1. Intent Check
    # [v1.7 Update] Allow IntentType enums or legacy strings
    intent_val = intent.value if hasattr(intent, 'value') else intent
    
    # 2. Keyword Check
    query_lower = user_query.lower()
    violated_keywords = [kw for kw in FORBIDDEN_KEYWORDS if kw in query_lower]
    
    if violated_keywords:
        # [v1.7 SOFT SAFETY]
        # If intent is REASONING (or legacy 'justify_decision'), we Warn instead of Fail
        reasoning_intents = ["reasoning", "justify_decision"]
        
        # Check if intent_val matches any reasoning intent
        is_reasoning = intent_val in reasoning_intents
        
        # Also check against possible Enum values if imported (omitted here for simplicity, relying on string)
        
        if is_reasoning:
             return {
                 "status": "PASS_WITH_WARNING", 
                 "warning_type": "brand_positioning_conflict",
                 "reason": f"This idea conflicts with the brand’s premium positioning (Keywords: {violated_keywords})."
             }
        else:
             # Creative or Knowledge requests with forbidden words still fail
             return {"status": "FAIL", "reason": f"Forbidden keywords detected: {violated_keywords}"}
    
    # 3. Semantic Drift Check
    # Setup IDs
    brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_id_str))
    org_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'default_org'))
    
    centroid = calculate_brand_centroid(brand_uuid, org_id)
    if not centroid:
        return {"status": "PASS", "reason": "No brand memory to validate against (Cold Start)"}
        
    query_vec = generate_embedding(user_query)
    similarity = np.dot(query_vec, centroid) / (np.linalg.norm(query_vec) * np.linalg.norm(centroid))
    
    if similarity < 0.4: 
         return {"status": "FAIL", "reason": f"Semantic drift detected (Score: {similarity:.2f}). Query not aligned with Brand Voice."}

    return {"status": "PASS", "reason": "All checks passed"}

def generate_brand_response(query: str, context: List[Dict], safety_status: Dict, temp_grounding: str = None) -> str:
    if safety_status['status'] == "FAIL":
        return f"🚫 BRAND SAFETY BLOCK: {safety_status['reason']}"
        
    context_str = "\n".join([f"- {c['content']}" for c in context])
    if temp_grounding:
        context_str += f"\n[EXTERNAL EVIDENCE]: {temp_grounding}"
        
    prompt = f"""
    You are Brand Brain. Your job is to Explain, Justify, or Minimally Rewrite.
    Use the provided BRAND MEMORY as the source of truth.
    If external evidence is provided, use it for context but subordinate it to Brand Memory.
    
    QUERY: {query}
    
    BRAND MEMORY:
    {context_str}
    
    INSTRUCTIONS:
    - Do not invent facts.
    - Adhere to the tone found in memory.
    """
    
    client = get_gemini_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-09-2025",
        contents=prompt
    )
    return response.text

def ephemeral_live_fetch(query: str) -> str:
    """
    Fetches live data for Type C memory.
    Guaranteed NO persistence.
    """
    print(f"   🌐 Triggering Ephemeral Live Fetch for: '{query}'")
    
    prompt = f"Search Google for: {query}. Summarize the answer in 2 sentences."
    
    client = get_gemini_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-09-2025",
        contents=prompt,
        config=types.GenerateContentConfig(
             tools=[types.Tool(google_search=types.GoogleSearchRetrieval)]
        )
    )
    return response.text
