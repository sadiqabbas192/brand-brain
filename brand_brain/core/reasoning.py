import uuid
import numpy as np
from typing import Dict, List, Any
from pinecone import Pinecone
from ..config import ALLOWED_INTENTS, FORBIDDEN_KEYWORDS
from ..services.embedding import generate_embedding
from ..services.gemini import get_gemini_client, safe_generate_content
from ..database import get_pinecone_client
from .intent import IntentType

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

SYSTEM_PROMPT = """
You are Brand Brain.
You explain and compare based on provided context.
You do NOT invent facts.
You do NOT create brand assets.
You do NOT store knowledge.

When live grounding is used:
• Explicitly explain differences
• Anchor conclusions back to brand identity (voice, values, positioning, audience)

Your role is to explain, summarize, and reason about a brand using only the context provided to you.

You must strictly follow these rules:
• You do **not** invent facts
• You do **not** create new brand assets
• You do **not** generate creative content
• You do **not** speculate beyond provided or grounded information

You may:
• Explain brand identity, voice, values, and positioning
• Answer factual questions about the brand
• Justify whether ideas or messaging align with the brand
• Politely refuse creative or unsafe requests

If a request asks you to create campaigns, copy, slogans, or visuals:
• Respond with a polite refusal
• Explain that you can evaluate or explain brand guidelines instead

If information is uncertain:
• State the uncertainty clearly
• Do not guess

Your tone must be:
• Clear
• Calm
• Professional
• Brand-aligned

You exist to **protect and explain the brand**, not to create on its behalf.
"""

def generate_explained_response(query: str, context: List[Dict], safety_status: Dict, intent: IntentType, live_context: str = None) -> Dict:
    # Build System Context
    context_str = "\n".join([f"- {c['content']} (Confidence: {c.get('confidence', 'inferred')})" for c in context])
    
    # [v1.7 SOFT SAFETY INJECTION]
    safety_instruction = ""
    if safety_status.get('status') == 'PASS_WITH_WARNING':
        safety_instruction = f"""
        ⚠️ IMPORTANT: The user's idea conflicts with brand positioning: {safety_status['reason']}
        
        YOUR TASK:
        1. Explain WHY this idea conflicts with the brand (using the Context below).
        2. Suggest a principle-based alternative that aligns with the brand.
        3. Maintain a helpful, educational tone. Do NOT scold.
        4. DO NOT generate the requested content/slogan/copy. Just explain the misalignment.
        """
    
    # [v1.8 GROUNDING INJECTION]
    grounding_instruction = ""
    if live_context:
        grounding_instruction = f"""
        🌐 LIVE GROUNDED CONTEXT (Supplement with this ONLY after covering Brand Memory):
        {live_context}

        INSTRUCTION:
        1. **PRIORITY**: ALWAYS start by answering based on 'CONTEXT (Brand Memory)' below.
        2. Then, supplement with the 'LIVE GROUNDED CONTEXT' to add market details or comparisons.
        3. If comparing, clearly highlight differences between your brand (Memory) and external findings (Live).
        4. Anchor any external market info back to the brand's identity.
        """

    full_prompt = f"""
    {SYSTEM_PROMPT}
    
    CONTEXT (Brand Memory):
    {context_str}
    
    {grounding_instruction}
    
    {safety_instruction}
    
    USER QUERY: {query}
    
    Explain your answer based *only* on the context above.
    """
    
    try:
        from google.genai import types
        response = safe_generate_content(
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.3 # Low temp for strict adherence
            )
        )
        answer_text = response.text
    except Exception as e:
        answer_text = f"Error generating response: {e}"

    # Construct Explainability Object
    result = {
        "answer": answer_text,
        "confidence_level": "live" if live_context else ("high" if context else "medium"), 
        "brand_elements_used": list(set([c.get('source_field', 'General') for c in context])) if isinstance(context, list) else [],
        "memory_sources": list(set([c.get('confidence', 'inferred') for c in context])),
        "live_context_used": bool(live_context),
        "safety_status": safety_status['status']
    }
    
    # [v1.9] Attach Usage Info
    if hasattr(response, '_usage_info'):
        result["usage_info"] = response._usage_info
        
    return result
