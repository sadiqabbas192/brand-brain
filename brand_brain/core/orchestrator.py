from typing import Dict, Any
from ..config import DEBUG_MODE
from .intent import classify_intent_hybrid, IntentType
from .retrieval import retrieve_context
from .reasoning import check_brand_safety, generate_explained_response
from .grounding import should_trigger_grounding, fetch_live_context

def print_debug(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)

def chat_session(user_query: str, brand_id: str = "wh_india_001") -> Dict[str, Any]:
    print_debug(f"\n💬 User: {user_query}")
    
    # 1. Intent Classification (Hybrid)
    intent = classify_intent_hybrid(user_query)
    print_debug(f"   🧠 Intent: {intent.value}")
    
    # 2. Creative Block (Pre-computation)
    if intent == IntentType.CREATIVE:
        # Double check: if it was rule-based, we already blocked. 
        # But if Gemini classified it as creative and we didn't catch usage of forbidden words yet...
        # We'll block here. 
        print_debug("   🚫 Creative Request Blocked.")
        return {
            "answer": "I can explain brand guidelines and evaluate ideas, but I don’t generate creative assets yet.",
            "safety_status": "BLOCKED_CREATIVE",
            "intent": intent.value,
        }

    # 3. Retrieval
    # Using v1.6 retrieval (prioritizes approved)
    # If reasoning, we might want 'brand_voice' or 'strategy'. Notebook logic used 'brand_voice' for reasoning.
    vector_type = "brand_voice" if intent == IntentType.REASONING else "strategy"
    context = retrieve_context(brand_id, user_query, vector_type=vector_type)
    
    # 3b. Grounding Decision Engine [v1.8]
    live_context = None
    
    # Calculate Average Similarity
    if context:
        avg_score = sum([c.get('score', 0) for c in context]) / len(context)
    else:
        avg_score = 0.0
        
    if should_trigger_grounding(user_query, context, avg_score):
        print_debug(f"   🌐 [v1.8] Grounding Triggered (Avg Score: {avg_score:.2f})")
        
        # Resolve Brand Name for Grounding Context
        brand_name = "Westinghouse India" if brand_id == "wh_india_001" else brand_id
        
        live_context = fetch_live_context(user_query, brand_name)
        if live_context:
             print_debug(f"   ✅ Live Context Fetched ({len(live_context)} chars)")
    
    # 4. Safety Check (Off-Brand Rules) - BEFORE Reasoner
    # [v1.7] Pass actual intent (IntentType) to safety check
    safety = check_brand_safety(user_query, brand_id, intent)
    
    if safety['status'] == 'FAIL':
        print_debug(f"   🛡️ Safety Block: {safety['reason']}")
        return {
            "answer": f"I cannot answer that. {safety['reason']}",
            "safety_status": "BLOCKED_SAFETY",
            "intent": intent.value
        }
    elif safety['status'] == 'PASS_WITH_WARNING':
        print_debug(f"   ⚠️ Soft Safety Warning: {safety['reason']}")
        # Proceed to Reasoner, passing the warning
        
    # 5. Brand Reasoner [v1.8: Passes live_context]
    response_obj = generate_explained_response(user_query, context, safety, intent, live_context)

    response_obj["intent"] = intent.value

    # Printing debug info just like in notebook
    # print_debug(f"   🤖 Brand Brain: {response_obj['answer'][:100]}...")
    return response_obj
