
import json
import os

NOTEBOOK_PATH = "d:/brand-brain/brand_brain_v1.7.ipynb"

def create_code_cell(source_code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_code.splitlines(keepends=True)
    }

def create_markdown_cell(source_text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_text.splitlines(keepends=True)
    }

v1_7_cells = [
    create_markdown_cell("""## **Section F: Brand Brain v1.7 - Read-Only Chat & Explainability**

This section implements the **Read-Only Chat Playground** with **Explainability**.
It focuses on safe, natural interaction without persistent memory mutation.

**Core Features:**
1.  **Hybrid Intent Classification** (Rules + Gemini)
2.  **Brand Reasoner** (Gemini-2.5-Flash with Strict System Prompt)
3.  **Chat Pipeline** (Intent -> Retrieval -> Safety -> Reasoner)
4.  **Zero-Mutation Guarantee**"""),

    create_code_cell("""# 1. Hybrid Intent Classification

import re
from enum import Enum
import json

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
            print(f"   🛡️ Rule-Based Intent Detection: CREATIVE (Blocked pattern: '{pattern}')")
            return IntentType.CREATIVE
            
    # 2. Gemini Fallback for Ambiguity
    prompt = f\"\"\"
    Classify the following query into one of 3 categories:
    1. KNOWLEDGE (Questions about facts, brand voice, mission, identity)
    2. REASONING (Questions asking for validation, 'is this on-brand?', 'why?')
    3. CREATIVE (Requests to create, write, generate, design new assets)
    
    QUERY: {query}
    
    RETURN ONLY ONE WORD: KNOWLEDGE, REASONING, or CREATIVE.
    \"\"\"
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        result = response.text.strip().upper()
        if "CREATIVE" in result: return IntentType.CREATIVE
        if "REASONING" in result: return IntentType.REASONING
        return IntentType.KNOWLEDGE
    except Exception as e:
        print(f"   ⚠️ Intent Classification Failed: {e}. Defaulting to KNOWLEDGE.")
        return IntentType.KNOWLEDGE

print("✅ Hybrid Intent Classifier Ready")"""),

    create_code_cell("""# 2. Brand Reasoner & Explainability

SYSTEM_PROMPT = \"\"\"
You are Brand Brain, a read-only brand intelligence system.

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
\"\"\"

def generate_explained_response(query: str, context: List[Dict], safety_status: Dict, intent: IntentType) -> Dict:
    # Build System Context
    context_str = "\\n".join([f"- {c['content']} (Confidence: {c.get('confidence', 'inferred')})" for c in context])
    
    full_prompt = f\"\"\"
    {SYSTEM_PROMPT}
    
    CONTEXT (Brand Memory):
    {context_str}
    
    USER QUERY: {query}
    
    Explain your answer based *only* on the context above.
    \"\"\"
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.3 # Low temp for strict adherence
            )
        )
        answer_text = response.text
    except Exception as e:
        answer_text = f"Error generating response: {e}"

    # Construct Explainability Object
    return {
        "answer": answer_text,
        "confidence_level": "high" if context else "medium", # Simplified
        "brand_elements_used": list(set([c.get('source_field', 'General') for c in context])) if isinstance(context, list) else [], # Placeholder logic
        "memory_sources": list(set([c.get('confidence', 'inferred') for c in context])),
        "live_context_used": False, # v1.7 defaults
        "safety_status": safety_status['status']
    }

print("✅ Brand Reasoner Ready")"""),

    create_code_cell("""# 3. Chat Pipeline (Strict Ordering)

def chat_session(user_query: str, brand_id: str = "wh_india_001"):
    print(f"\\n💬 User: {user_query}")
    
    # 1. Intent Classification (Hybrid)
    intent = classify_intent_hybrid(user_query)
    print(f"   🧠 Intent: {intent.value}")
    
    # 2. Creative Block (Pre-computation)
    if intent == IntentType.CREATIVE:
        print("   🚫 Creative Request Blocked.")
        return {
            "answer": "I can explain brand guidelines and evaluate ideas, but I don’t generate creative assets yet.",
            "safety_status": "BLOCKED_CREATIVE"
        }

    # 3. Retrieval
    # Using v1.6 retrieval (prioritizes approved)
    context = retrieve_context(brand_id, user_query, vector_type="brand_voice" if intent == IntentType.REASONING else "strategy")
    
    # 4. Safety Check (Off-Brand Rules) - BEFORE Reasoner
    safety = check_brand_safety(user_query, brand_id, "explain_brand") # Using generic intent for safety check
    
    if safety['status'] == 'FAIL':
        print(f"   🛡️ Safety Block: {safety['reason']}")
        return {
            "answer": f"I cannot answer that. {safety['reason']}",
            "safety_status": "BLOCKED_SAFETY"
        }
        
    # 5. Brand Reasoner
    response_obj = generate_explained_response(user_query, context, safety, intent)
    
    print(f"   🤖 Brand Brain: {response_obj['answer'][:100]}...")
    return response_obj

print("✅ Chat Pipeline Ready")"""),

    create_code_cell("""# 4. v1.7 Validation Harness & DB Snapshot

def get_db_counts():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM brand_assets")
    assets = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM brand_chunks")
    chunks = cur.fetchone()[0]
    cur.close()
    conn.close()
    return assets, chunks

def run_v1_7_validation():
    print("\\n\\n--- 🧪 v1.7 Validation Suite ---")
    
    # 1. Snapshot DB
    assets_before, chunks_before = get_db_counts()
    print(f"📊 DB Before: Assets={assets_before}, Chunks={chunks_before}")
        
    # 2. Test Cases
    queries = [
        ("What are our brand colors?", "KNOWLEDGE"),
        ("Is 'buy now, cheap price' on-brand?", "REASONING"), # Should trigger reasoning or safety
        ("Write a slogan for a summer sale", "CREATIVE") # Should be blocked
    ]
    
    for q, expected in queries:
        print(f"\\n--- Testing: '{q}' (Expected: {expected}) ---")
        res = chat_session(q)
        print(f"   📄 Result: {json.dumps(res, indent=2)}")

    # 3. Verify Mutation
    assets_after, chunks_after = get_db_counts()
    print(f"\\n📊 DB After: Assets={assets_after}, Chunks={chunks_after}")
    
    if assets_before == assets_after and chunks_before == chunks_after:
        print("✅ SUCCESS: Zero DB Mutation Confirmed.")
    else:
        print("❌ FAILURE: DB Mutation Detected!")

run_v1_7_validation()""")
]

# Append to Notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'].extend(v1_7_cells)

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print(f"Successfully appended {len(v1_7_cells)} cells to {NOTEBOOK_PATH}")
