# 1. Setup & Configuration
import os
import json
import uuid
import time
from typing import List, Dict, Any, Optional
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from pinecone import Pinecone, ServerlessSpec
from google import genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv(override=True) # Ensure we reload if .env changed

NEON_DB_URL = os.getenv("NEON_DB_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY4")

if not all([NEON_DB_URL, PINECONE_API_KEY, GEMINI_API_KEY]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# Initialize Clients
client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Database Connection Helper
def get_db_connection():
    return psycopg2.connect(NEON_DB_URL)

print("✅ Configuration Loaded & Clients Initialized")
# CELL SEPARATOR 
# 2. Database Pre-checks (No Table Creation)
def check_connection():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'brand_assets'")
        if cur.fetchone()[0] == 0:
            print("❌ ERROR: Tables not found! Please run tables.sql in Neon console.")
        else:
            print("✅ Connected to Neon DB. Tables exist.")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
    finally:
        cur.close()
        conn.close()

check_connection()
# CELL SEPARATOR 
# 3. Input Brand Data

# Parsed from Westinghouse India.txt
westinghouse_json = {
    "brandId": "wh_india_001",
    "name": "Westinghouse India",
    "industry": "FMEG",
    "mission": "To enrich everyday living with reliable, thoughtfully engineered appliances that combine global heritage, modern innovation, and timeless design—delivering confidence, comfort, and consistency to Indian homes.",
    "brandVoice": "Confident & Reassuring. Premium yet Approachable. Clear & Functional. Trust-First. Design-Conscious.",
    "visualStyle": "Design-forward minimalism. Product as hero. Lifestyle-led context. Retro-modern blend. Premium finishes. Colors: Orange, Red, White, Green, Blue, Black.",
    "audience": "All genders, 25–45 years (core). Upper-middle to affluent households. Interests: Premium home & kitchen appliances, Modern kitchen aesthetics, Smart living. Focus: Tier 1 metros (Mumbai, Delhi NCR...) and affluent Tier 2.",
    "competitors": "Morphy Richards (Strong British Heritage, Wide Portfolio). Weaknesses: Inconsistent Visual Identity, Limited Design Differentiation.",
    "inspiration": "Morphy Richards",
    "website": "https://www.westinghousehomeware.in/"
}

brands_to_ingest = [westinghouse_json]
# CELL SEPARATOR 
# 4. Semantic Asset Extraction Logic

def extract_assets(brand_data: Dict) -> List[Dict]:
    assets = []
    brand_id = brand_data.get("brandId")
    
    # Extraction Rules Mapping
    # Source Field -> (Asset Type [copy/guideline/website], Vector Type [brand_voice/strategy/performance])
    mapping = {
        "mission": ("guideline", "strategy"),
        "brandVoice": ("guideline", "brand_voice"),
        "visualStyle": ("guideline", "brand_voice"),
        "audience": ("guideline", "strategy"),
        "competitors": ("guideline", "strategy"),
        "inspiration": ("guideline", "strategy"),
        "website": ("website", "strategy")
    }
    
    for field, (asset_type, vector_type) in mapping.items():
        content = brand_data.get(field)
        if content:
            assets.append({
                "asset_id": str(uuid.uuid4()),
                "brand_id": brand_id,
                "asset_type": asset_type,
                "vector_type": vector_type,
                "source_field": field,
                "content": content
            })
            
    return assets

print("✅ Asset Extraction Logic Defined")
# CELL SEPARATOR 
# 5. Chunking & Embedding Logic

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def chunk_text(text: str) -> List[str]:
    return text_splitter.split_text(text)

def generate_embedding(text: str) -> List[float]:
    # Using gemini-embedding-001 with truncation to 768 dimensions
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config={
                'output_dimensionality': 768,
                'task_type': 'RETRIEVAL_DOCUMENT',
                'title': 'Brand Asset'
            }
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"Embedding Error: {e}")
        return []

print("✅ Chunking & Embedding Functions Defined (New SDK - 768 dims)")
# CELL SEPARATOR 
# 6. Ingestion Pipeline (Production Schema)

def ingest_brand(brand_data: Dict):
    brand_id_str = brand_data['brandId']
    brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_id_str))
    
    brand_name = brand_data.get('name', 'Unknown')
    org_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'default_org')) # Placeholder Org

    print(f"\n🧠 Ingesting Brand: {brand_name} (UUID: {brand_uuid}) ...")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # 1. Ensure Organization Exists
        cur.execute(
            "INSERT INTO organizations (org_id, name) VALUES (%s, %s) ON CONFLICT (org_id) DO NOTHING",
            (org_id, "Test Org")
        )

        # 2. Ensure Brand Exists
        cur.execute(
            "INSERT INTO brands (brand_id, org_id, name, industry) VALUES (%s, %s, %s, %s) ON CONFLICT (brand_id) DO NOTHING",
            (brand_uuid, org_id, brand_name, brand_data.get('industry', 'Unknown'))
        )

        # 3. Extract Assets
        assets = extract_assets(brand_data)
        print(f"   -> Extracted {len(assets)} semantic assets")

        # Prepare Pinecone
        index_name = "brand-brain-index"
        
        # DEBUG: Check what key is actually being used
        masked = PINECONE_API_KEY[:5] + "..." if PINECONE_API_KEY else "None"
        print(f"   [DEBUG] Checking Pinecone Index with Key: {masked}")
        
        # Create index if not exists
        if index_name not in pc.list_indexes().names():
             pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        idx = pc.Index(index_name)

        total_chunks = 0
        
        for asset in assets:
            # Insert Asset Metadata
            cur.execute(
                "INSERT INTO brand_assets (asset_id, brand_id, asset_type, raw_text, source) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (asset_id) DO NOTHING",
                (asset['asset_id'], brand_uuid, asset['asset_type'], asset['content'], asset['source_field'])
            )
            
            chunks = chunk_text(asset['content'])
            
            for i, chunk_text_content in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                embedding_id = str(uuid.uuid4())
                vector = generate_embedding(chunk_text_content)
                
                if not vector:
                    print(f"Skipping chunk due to embedding failure")
                    continue

                cur.execute(
                    "INSERT INTO brand_chunks (chunk_id, asset_id, brand_id, vector_type, content, token_count) VALUES (%s, %s, %s, %s, %s, %s)",
                    (chunk_id, asset['asset_id'], brand_uuid, asset['vector_type'], chunk_text_content, len(chunk_text_content.split()))
                )
                
                namespace = f"{org_id}:{brand_uuid}:{asset['vector_type']}"
                cur.execute(
                    "INSERT INTO embeddings (embedding_id, chunk_id, brand_id, vector_type, namespace, model) VALUES (%s, %s, %s, %s, %s, %s)",
                    (embedding_id, chunk_id, brand_uuid, asset['vector_type'], namespace, "gemini-embedding-001")
                )

                idx.upsert(
                    vectors=[(chunk_id, vector, {"source": asset['source_field']})],
                    namespace=namespace
                )
                total_chunks += 1
        
        conn.commit()
        print(f"✅ Successfully ingested {total_chunks} chunks for {brand_name}.")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Ingestion Failed: {e}")
    finally:
        cur.close()
        conn.close()

# Run Ingestion
for brand in brands_to_ingest:
    ingest_brand(brand)
# CELL SEPARATOR 
# 7. Retrieval & Validation Logic

def retrieve_context(brand_name_str: str, query: str, vector_type: str = "brand_voice", top_k: int = 3):
    if brand_name_str == "wh_india_001":
        brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_name_str))
        org_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'default_org'))
    else:
        brand_uuid = brand_name_str
        org_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'default_org'))

    print(f"\n🔎 Querying Brand {brand_name_str} (UUID: {brand_uuid}) [{vector_type}]: '{query}'")
    
    # New SDK for Query Embedding
    try:
        query_embedding_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config={
                'output_dimensionality': 768,
                'task_type': 'RETRIEVAL_QUERY'
            }
        )
        query_embedding = query_embedding_result.embeddings[0].values
    except Exception as e:
        print(f"Embedding Error during retrieval: {e}")
        return []
    
    namespace = f"{org_id}:{brand_uuid}:{vector_type}"
    index_name = "brand-brain-index"
    idx = pc.Index(index_name)
    
    results = idx.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    
    if not results['matches']:
        print("   ⚠️ No matches found in namespace:", namespace)
        return []
        
    conn = get_db_connection()
    cur = conn.cursor()
    
    retrieved_docs = []
    chunk_ids = [m['id'] for m in results['matches']]
    
    if chunk_ids:
        placeholders = ', '.join(['%s'] * len(chunk_ids))
        query_sql = f"SELECT content, vector_type FROM brand_chunks WHERE chunk_id IN ({placeholders})"
        cur.execute(query_sql, tuple(chunk_ids))
        rows = cur.fetchall()
        
        for i, row in enumerate(rows):
            score = results['matches'][i]['score']
            print(f"   [{i+1}] Score: {score:.4f} | Content: {row[0][:100]}...")
            retrieved_docs.append({"content": row[0], "score": score})
            
    cur.close()
    conn.close()
    return retrieved_docs

# 8. Run Validation Tests
def run_validation():
    # Test 1: Westinghouse Brand Voice
    print("\n--- TEST 1: Westinghouse Brand Voice ---")
    retrieve_context("wh_india_001", "Describe our design philosophy.", vector_type="brand_voice")
    
    # Test 2: Westinghouse Competitor Context
    print("\n--- TEST 2: Westinghouse Strategy ---")
    retrieve_context("wh_india_001", "Who are we fighting against?", vector_type="strategy")
    
    # Test 3: Off-Brand check
    print("\n--- TEST 3: Isolation / Irrelevant Query ---")
    retrieve_context("wh_india_001", "How to be cheap and loud?", vector_type="brand_voice")

# run_validation()
# CELL SEPARATOR 
# v1.5 Imports
import numpy as np
from google.genai import types
import statistics

print("✅ v1.5 Libraries Loaded")
# CELL SEPARATOR 
# SECTION A: Grounding-Assisted Ingestion (Type B Memory)

def grounding_assisted_ingest(brand_data: Dict, target_vector_type="strategy"):
    """
    Uses Gemini Google Search tools to extract ONLY evergreen brand philosophy.
    Enforces strict prompt filters.
    Stores as Type B memory with version tags.
    """
    brand_name = brand_data['name']
    website = brand_data.get('website', '')
    
    print(f"\n🌍 Starting Grounding-Assisted Ingestion for {brand_name}...")

    # 1. Define Prompt with HARD FILTERS
    prompt = f"""
    You are a Brand Identity Expert.
    SEARCH for "{brand_name} brand philosophy design principles manifesto".
    Also check the provided website: {website}

    EXTRACT ONLY evergreen, high-level brand identity content.
    
    ❌ STRICTLY IGNORE:
    - pricing, offers, discounts
    - launches, new arrivals
    - comparisons, awards
    - timelines, history dates
    - "latest", "new", "recent", "2024", "2025"

    ✅ EXTRACT ONLY:
    - philosophy & mission
    - design principles
    - core values
    - identity statements
    
    RETURN JSON in this format:
    {{
      "brand_philosophy": "string",
      "design_principles": "string",
      "positioning": "string"
    }}
    """

    try:
        # 2. Call Gemini with Search Tool
        # [FIX] Removed response_mime_type to allow Tools to work correctly
        response = client.models.generate_content(
            model="gemini-2.5-flash", # Using Flash for speed/tools
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearchRetrieval)]
            )
        )
        
        # 3. Robust JSON Parsing (Manual)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        extracted_data = json.loads(text.strip())
        print(f"   ✅ Extracted Grounded Data: {list(extracted_data.keys())}")

        # 3. Format as Assets (Type B)
        # Merging into a single text block for embedding is usually better for 'strategy'
        combined_text = f"Philosophy: {extracted_data.get('brand_philosophy', '')}\n"
        combined_text += f"Design Principles: {extracted_data.get('design_principles', '')}\n"
        combined_text += f"Positioning: {extracted_data.get('positioning', '')}"

        grounded_asset = {
            "asset_id": str(uuid.uuid4()),
            "brand_id": brand_data['brandId'],
            "asset_type": "guideline", # Fixed: Must be one of 'copy', 'guideline', 'website'
            "vector_type": target_vector_type,
            "source_field": "grounding_assisted_ingestion",
            "content": combined_text
        }

        # 4. Store & Embed (Reusing ingestion logic pattern)
        ingest_single_asset(grounded_asset, brand_data) # Helper to be defined below

    except Exception as e:
        print(f"❌ Grounding Ingestion Failed: {e}")


def ingest_single_asset(asset: Dict, brand_data: Dict):
    """Helper to ingest a single constructed asset."""
    brand_id_str = brand_data['brandId']
    brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_id_str))
    org_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'default_org'))
    
    conn = get_db_connection()
    cur = conn.cursor()
    idx = pc.Index("brand-brain-index")

    try:
        # Store Asset in Postgres
        # Note: metadata like source_version is stored in source or handled via separate columns in prod
        # Here we pack it into 'source' string or similar for v1 demo
        source_tag = f"{asset['source_field']} | v1.5 | confidence:inferred"
        
        cur.execute(
            "INSERT INTO brand_assets (asset_id, brand_id, asset_type, raw_text, source) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (asset_id) DO NOTHING",
            (asset['asset_id'], brand_uuid, asset['asset_type'], asset['content'], source_tag)
        )

        chunks = chunk_text(asset['content'])
        for chunk_text_content in chunks:
            chunk_id = str(uuid.uuid4())
            embedding_id = str(uuid.uuid4())
            vector = generate_embedding(chunk_text_content)
            
            if not vector: continue

            cur.execute(
                "INSERT INTO brand_chunks (chunk_id, asset_id, brand_id, vector_type, content, token_count) VALUES (%s, %s, %s, %s, %s, %s)",
                (chunk_id, asset['asset_id'], brand_uuid, asset['vector_type'], chunk_text_content, len(chunk_text_content.split()))
            )
            
            namespace = f"{org_id}:{brand_uuid}:{asset['vector_type']}"
            
            cur.execute(
                 "INSERT INTO embeddings (embedding_id, chunk_id, brand_id, vector_type, namespace, model) VALUES (%s, %s, %s, %s, %s, %s)",
                (embedding_id, chunk_id, brand_uuid, asset['vector_type'], namespace, "gemini-embedding-001")
            )
            
            idx.upsert(
                vectors=[(chunk_id, vector, {"source": source_tag})],
                namespace=namespace
            )
        
        conn.commit()
        print(f"   ✅ Successfully stored Type B memory for {brand_data['name']}")
    finally:
        cur.close()
        conn.close()

print("✅ Section A: Grounding-Assisted Ingestion Implementation Ready")
# CELL SEPARATOR 
# SECTION B: Off-Brand Rule Engine (Deterministic)

# [v1.7 Update] Allow generic intents or IntentType
from typing import Any

ALLOWED_INTENTS = {"explain_brand", "validate_copy", "justify_decision", "minimal_rewrite"}
FORBIDDEN_KEYWORDS = {"cheap", "free", "lowest price", "clearance", "sale", "loud", "discount", "discounting"} 

def calculate_brand_centroid(brand_id: str, org_id: str, top_n=5) -> List[float]:
    """
    Calculates deterministic centroid from top-N 'brand_voice' chunks.
    In a real system, this is pre-computed. Here we fetch via a 'neutral' query.
    """
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
        if intent_val in ["reasoning", "justify_decision"] or (hasattr(IntentType, 'REASONING') and intent_val == IntentType.REASONING.value):
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

print("✅ Section B: Off-Brand Rule Engine Ready (Soft Safety Enabled)")

# CELL SEPARATOR 
# SECTION C: Brand Reasoner

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
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

print("✅ Section C: Brand Reasoner Ready")
# CELL SEPARATOR 
# SECTION D: Ephemeral Live Fetch (Type C Memory)

def ephemeral_live_fetch(query: str) -> str:
    """
    Fetches live data for Type C memory.
    Guaranteed NO persistence.
    """
    print(f"   🌐 Triggering Ephemeral Live Fetch for: '{query}'")
    
    prompt = f"Search Google for: {query}. Summarize the answer in 2 sentences."
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearchRetrieval)]
        )
    )
    
    # Extract text from response (ignoring grounding metadata for the summary text)
    return response.text

print("✅ Section D: Ephemeral Live Fetch Ready")
# CELL SEPARATOR 
# SECTION E: v1.5 Validation Harness

def run_v1_5_validation():
    brand_id = "wh_india_001"
    
    # 1. Simulate Type B Ingestion
    print("\n--- TEST 1: Grounding-Assisted Ingestion ---")
    grounding_assisted_ingest(westinghouse_json)
    
    # VERIFICATION OF TYPE B ASSETS
    print("\n--- VERIFYING TYPE B ASSETS IN DB ---")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT count(*) FROM brand_assets WHERE source LIKE '%grounding_assisted%'")
        count = cur.fetchone()[0]
        print(f"✅ Found {count} Type B assets in Postgres.")
        if count == 0:
             print("❌ ERROR: Type B ingestion failed (no rows affected).")
    finally:
        cur.close()
        conn.close()
        
    # 2. Rejection Logic (OMITTED AS REQUESTED)
    # print("\n--- TEST 2: Off-Brand Rejection ---")

    # 3. End-to-End Success + Type C
    print("\n--- TEST 3: Live Query with Ephemeral Context ---")
    good_query = "What are the latest appliance trends suitable for our brand?"
    safety_3 = check_brand_safety(good_query, brand_id, "justify_decision")
    
    if safety_3['status'] == "PASS":
        # Retrieve Memory (Type A/B)
        context = retrieve_context(brand_id, good_query, "strategy")
        
        # Trigger Live Fetch (Type C)
        type_c_data = ephemeral_live_fetch(good_query)
        
        # Reason
        response = generate_brand_response(good_query, context, safety_3, type_c_data)
        print(f"\n🤖 Final Response:\n{response}")

        # PROOF OF NO PERSISTENCE
        print("\n🔒 Verifying Type C Non-Persistence...")
        conn = get_db_connection()
        cur = conn.cursor()
        # Search for 'trends' which comes from live fetch
        # but ensure we don't count type B or A if they happened to have it.
        # We specifically check for *recent* assets that are NOT type B/A?
        # Simple check: search for 'trends' in text, but EXCLUDE source='grounding_assisted_ingestion'
        cur.execute("SELECT count(*) FROM brand_assets WHERE raw_text ILIKE '%trends%' AND source NOT LIKE '%grounding_assisted%'")
        count = cur.fetchone()[0]
        if count == 0:
             print("✅ SUCCESS: Live trend data NOT found in Postgres (ignoring intentional Type A/B).")
        else:
             print("⚠️ NOTE: 'trends' keyword found. Verify it is not from ephemeral fetch.")
        cur.close()
        conn.close()

# run_v1_5_validation()
# CELL SEPARATOR 

# CELL SEPARATOR 
# [v1.6 UPGRADE] Redefining Ingestion to support Confidence Scoring

def ingest_single_asset(asset: Dict, brand_data: Dict):
    """[v1.6] Helper to ingest a single constructed asset with confidence defaults."""
    brand_id_str = brand_data['brandId']
    brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_id_str))
    org_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'default_org'))
    
    conn = get_db_connection()
    cur = conn.cursor()
    idx = pc.Index("brand-brain-index")

    try:
        # Store Asset in Postgres
        source_tag = f"{asset['source_field']} | v1.6 | confidence:inferred"
        
        # [MODIFIED v1.6] Explicitly inserting confidence='inferred'
        # Note: We rely on the schema update (ADD COLUMN confidence) having been run.
        cur.execute(
            "INSERT INTO brand_assets (asset_id, brand_id, asset_type, raw_text, source, confidence) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (asset_id) DO NOTHING",
            (asset['asset_id'], brand_uuid, asset['asset_type'], asset['content'], source_tag, 'inferred')
        )

        chunks = chunk_text(asset['content'])
        for chunk_text_content in chunks:
            chunk_id = str(uuid.uuid4())
            embedding_id = str(uuid.uuid4())
            vector = generate_embedding(chunk_text_content)
            
            if not vector: continue
            
            vt = asset.get('vector_type', 'strategy')

            cur.execute(
                "INSERT INTO brand_chunks (chunk_id, asset_id, brand_id, vector_type, content, token_count) VALUES (%s, %s, %s, %s, %s, %s)",
                (chunk_id, asset['asset_id'], brand_uuid, vt, chunk_text_content, len(chunk_text_content.split()))
            )
            
            namespace = f"{org_id}:{brand_uuid}:{vt}"
            
            cur.execute(
                 "INSERT INTO embeddings (embedding_id, chunk_id, brand_id, vector_type, namespace, model) VALUES (%s, %s, %s, %s, %s, %s)",
                (embedding_id, chunk_id, brand_uuid, vt, namespace, "gemini-embedding-001")
            )
            
            idx.upsert(
                vectors=[(chunk_id, vector, {"source": source_tag})],
                namespace=namespace
            )
        
        conn.commit()
        print(f"   ✅ [v1.6] Successfully stored Type B memory for {brand_data['name']} (Confidence: Inferred)")
    except Exception as e:
        conn.rollback()
        print(f"❌ Ingestion Error: {e}")
    finally:
        cur.close()
        conn.close()
print("✅ [v1.6] Ingestion Logic Updated")
# CELL SEPARATOR 
# [v1.6 UPGRADE] Redefining Retrieval to Prioritize Confidence & Filter Deprecated

def retrieve_context(brand_name_str: str, query: str, vector_type: str = "brand_voice", top_k: int = 3):
    if brand_name_str == "wh_india_001":
        brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_name_str))
        org_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'default_org'))
    else:
        brand_uuid = brand_name_str
        org_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'default_org'))

    print(f"\n🔎 [v1.6] Querying Brand {brand_name_str} (UUID: {brand_uuid}) [{vector_type}]: '{query}'")
    
    try:
        query_embedding_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config={
                'output_dimensionality': 768,
                'task_type': 'RETRIEVAL_QUERY'
            }
        )
        query_embedding = query_embedding_result.embeddings[0].values
    except Exception as e:
        print(f"Embedding Error during retrieval: {e}")
        return []
    
    namespace = f"{org_id}:{brand_uuid}:{vector_type}"
    index_name = "brand-brain-index"
    idx = pc.Index(index_name)
    
    # Fetch more candidates to allow for filtering of deprecated items
    results = idx.query(
        vector=query_embedding,
        top_k=top_k * 3,
        namespace=namespace,
        include_metadata=True
    )
    
    if not results['matches']:
        print("   ⚠️ No matches found in namespace:", namespace)
        return []
        
    conn = get_db_connection()
    cur = conn.cursor()
    
    retrieved_docs = []
    chunk_ids = [m['id'] for m in results['matches']]
    
    if chunk_ids:
        placeholders = ', '.join(['%s'] * len(chunk_ids))
        # [MODIFIED v1.6] Join with brand_assets to fetch 'confidence'
        query_sql = f"""
            SELECT c.chunk_id, c.content, c.vector_type, a.confidence 
            FROM brand_chunks c
            JOIN brand_assets a ON c.asset_id = a.asset_id
            WHERE c.chunk_id IN ({placeholders})
        """
        cur.execute(query_sql, tuple(chunk_ids))
        rows = cur.fetchall()
        
        # Lookup map
        db_map = {row[0]: {"content": row[1], "confidence": row[3]} for row in rows}
        
        valid_candidates = []
        
        for match in results['matches']:
            c_id = match['id']
            score = match['score']
            if c_id not in db_map: 
                continue
                
            data = db_map[c_id]
            confidence = data['confidence'] or 'inferred' # Handle legacy rows where confidence might be NULL
            
            # [RULE] Exclude Deprecated
            if confidence == 'deprecated':
                continue
                
            # [RULE] Prioritize: approved (3) > reviewed (2) > inferred (1)
            priority_score = 1
            if confidence == 'approved': priority_score = 3
            elif confidence == 'reviewed': priority_score = 2
            
            valid_candidates.append({
                "content": data['content'],
                "score": score,
                "confidence": confidence,
                "priority": priority_score
            })
            
        # Sort by Priority (desc), then Similarity Score (desc)
        valid_candidates.sort(key=lambda x: (x['priority'], x['score']), reverse=True)
        
        # Return top_k
        final_results = valid_candidates[:top_k]
        
        for i, res in enumerate(final_results):
            print(f"   [{i+1}] [{res['confidence'].upper()}] Score: {res['score']:.4f} | Content: {res['content'][:100]}...")
            retrieved_docs.append(res)
            
    cur.close()
    conn.close()
    return retrieved_docs
print("✅ [v1.6] Retrieval Logic Updated")
# CELL SEPARATOR 
# SECTION C: Memory Review Functions (v1.6)

def list_inferred_assets(brand_id_str: str):
    print(f"\n📋 Listing Inferred Assets for {brand_id_str}...")
    brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_id_str))
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT asset_id, asset_type, raw_text, source, confidence FROM brand_assets WHERE brand_id = %s AND confidence = 'inferred'", (brand_uuid,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    for r in rows:
        print(f"   [ID: {r['asset_id']}] {r['raw_text'][:50]}... (Source: {r['source']})")
    return rows

def approve_asset(asset_id: str, reviewer: str, notes: str):
    print(f"\n✅ Approving Asset {asset_id}...")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT confidence FROM brand_assets WHERE asset_id = %s", (asset_id,))
        res = cur.fetchone()
        if not res: return
        prev_conf = res[0]
        cur.execute("UPDATE brand_assets SET confidence = 'approved', reviewed_by = %s, review_notes = %s, reviewed_at = NOW() WHERE asset_id = %s", (reviewer, notes, asset_id))
        cur.execute("INSERT INTO memory_reviews (review_id, asset_id, action, previous_confidence, new_confidence, reviewer, notes) VALUES (%s, %s, %s, %s, %s, %s, %s)", (str(uuid.uuid4()), asset_id, 'approve', prev_conf, 'approved', reviewer, notes))
        conn.commit()
        print("   -> Asset Approved.")
    except Exception as e:
        conn.rollback(); print(f"❌ {e}")
    finally:
        cur.close(); conn.close()

def reject_asset(asset_id: str, reviewer: str, notes: str):
    print(f"\n⛔ Rejecting Asset {asset_id}...")
    conn = get_db_connection(); cur = conn.cursor()
    try:
        cur.execute("SELECT confidence FROM brand_assets WHERE asset_id = %s", (asset_id,))
        res = cur.fetchone()
        if not res: return
        prev_conf = res[0]
        cur.execute("UPDATE brand_assets SET confidence = 'deprecated', reviewed_by = %s, review_notes = %s, reviewed_at = NOW() WHERE asset_id = %s", (reviewer, notes, asset_id))
        cur.execute("INSERT INTO memory_reviews (review_id, asset_id, action, previous_confidence, new_confidence, reviewer, notes) VALUES (%s, %s, %s, %s, %s, %s, %s)", (str(uuid.uuid4()), asset_id, 'deprecate', prev_conf, 'deprecated', reviewer, notes))
        conn.commit()
        print("   -> Asset Deprecated.")
    except Exception as e:
        conn.rollback(); print(f"❌ {e}")
    finally:
        cur.close(); conn.close()

def edit_and_promote_asset(asset_id: str, new_text: str, reviewer: str):
    print(f"\n📝 Editing & Promoting Asset {asset_id}...")
    conn = get_db_connection(); cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("SELECT * FROM brand_assets WHERE asset_id = %s", (asset_id,))
        original = cur.fetchone()
        if not original: return
        # Deprecate Old
        cur.execute("UPDATE brand_assets SET confidence = 'deprecated', reviewed_by = %s, review_notes = 'Replaced by edit', reviewed_at = NOW() WHERE asset_id = %s", (reviewer, asset_id))
        # Insert New Approved
        new_asset_id = str(uuid.uuid4())
        cur.execute("INSERT INTO brand_assets (asset_id, brand_id, asset_type, raw_text, source, confidence, reviewed_by, reviewed_at, review_notes) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), 'Created via Edit')", (new_asset_id, original['brand_id'], original['asset_type'], new_text, original['source'], 'approved', reviewer))
        conn.commit()
        print(f"   -> Old Asset Deprecated. New Asset {new_asset_id} Created.")
        # Chunk & Embed New Asset
        chunks = chunk_text(new_text)
        idx = pc.Index("brand-brain-index")
        org_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'default_org'))
        # Need vector_type
        cur.execute("SELECT vector_type FROM brand_chunks WHERE asset_id = %s LIMIT 1", (asset_id,))
        vt = cur.fetchone()['vector_type']
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            vec = generate_embedding(chunk)
            cur.execute("INSERT INTO brand_chunks (chunk_id, asset_id, brand_id, vector_type, content, token_count) VALUES (%s, %s, %s, %s, %s, %s)", (chunk_id, new_asset_id, original['brand_id'], vt, chunk, len(chunk.split())))
            cur.execute("INSERT INTO embeddings (embedding_id, chunk_id, brand_id, vector_type, namespace, model) VALUES (%s, %s, %s, %s, %s, %s)", (str(uuid.uuid4()), chunk_id, original['brand_id'], vt, f"{org_id}:{original['brand_id']}:{vt}", "gemini-embedding-001"))
            idx.upsert(vectors=[(chunk_id, vec, {"source": original['source']})], namespace=f"{org_id}:{original['brand_id']}:{vt}")
        conn.commit()
        print("   -> New/Edited Asset Embeddings Generated.")
    except Exception as e:
        conn.rollback(); print(f"❌ Edit Failed: {e}")
    finally:
        cur.close(); conn.close()
print("✅ [v1.6] Section C: Functions Ready")
# CELL SEPARATOR 
# SECTION D: Validation Tests (v1.6)

def run_v1_6_validation():
    brand_id_str = "wh_india_001"
    brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_id_str))

    approved_asset_id = None
    approved_asset_text = None

    print("\n\n--- 🧪 TEST 1: Ingestion starts as 'inferred' ---")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT count(*) FROM brand_assets WHERE brand_id = %s AND confidence = 'inferred'",
        (brand_uuid,)
    )
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    print(f"✅ Found {count} 'inferred' assets")

    # -------------------------------------------------

    print("\n--- 🧪 TEST 2: Retrieval Priority (Approved > Inferred) [brand_voice] ---")

    # 1️⃣ List inferred assets
    inferred = list_inferred_assets(brand_id_str)

    # 2️⃣ Select an inferred BRAND_VOICE asset explicitly
    brand_voice_target = None
    for asset in inferred:
        if asset["source"] in ["brandVoice", "visualStyle"]:
            brand_voice_target = asset
            break

    if not brand_voice_target:
        print("❌ No inferred brand_voice asset found to test.")
    else:
        approved_asset_id = brand_voice_target["asset_id"]
        approved_asset_text = brand_voice_target["raw_text"]

        print(
            f"   🎯 Targeting Brand Voice Asset: "
            f"{approved_asset_text[:40]}..."
        )

        # 3️⃣ Approve the brand_voice asset
        approve_asset(
            approved_asset_id,
            reviewer="Admin",
            notes="Approved brand voice memory"
        )

        # 4️⃣ Query SAME semantic space + namespace
        res = retrieve_context(
            brand_id_str,
            "design philosophy",
            "brand_voice"
        )

        # 5️⃣ Assert approved asset is prioritized
        if res and res[0]["confidence"] == "approved":
            print("✅ SUCCESS: Approved brand_voice asset correctly prioritized.")
        else:
            print(
                f"❌ FAILURE: Expected approved asset first, got "
                f"{res[0]['confidence'] if res else 'None'}"
            )

    # -------------------------------------------------

    print("\n--- 🧪 TEST 3: Deprecation (Exclude from Retrieval) ---")

    if approved_asset_id:
        reject_asset(
            approved_asset_id,
            reviewer="Admin",
            notes="Deprecating for validation test"
        )

        res = retrieve_context(
            brand_id_str,
            "design philosophy",
            "brand_voice"
        )

        found = any(
            d["content"] == approved_asset_text
            for d in res
        )

        if not found:
            print("✅ Deprecated asset successfully EXCLUDED.")
        else:
            print("❌ FAILURE: Deprecated asset still retrieved!")

    else:
        print("⚠️ Skipping TEST 3: No approved asset available.")

    # -------------------------------------------------

    print("\n--- 🧪 TEST 4: Audit Trail ---")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT action, previous_confidence, new_confidence "
        "FROM memory_reviews ORDER BY created_at DESC LIMIT 5"
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    for r in rows:
        print(f"   - Action: {r[0]} | Old: {r[1]} -> New: {r[2]}")

# Run validation
# run_v1_6_validation()# SECTION D: Validation Tests (v1.6)

def run_v1_6_validation():
    brand_id_str = "wh_india_001"
    brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_id_str))

    approved_asset_id = None
    approved_asset_text = None

    print("\n\n--- 🧪 TEST 1: Ingestion starts as 'inferred' ---")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT count(*) FROM brand_assets WHERE brand_id = %s AND confidence = 'inferred'",
        (brand_uuid,)
    )
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    print(f"✅ Found {count} 'inferred' assets")

    # -------------------------------------------------

    print("\n--- 🧪 TEST 2: Retrieval Priority (Approved > Inferred) [brand_voice] ---")

    # 🔎 Find an inferred BRAND_VOICE asset via DB (robust)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT a.asset_id, a.raw_text
        FROM brand_assets a
        JOIN brand_chunks c ON a.asset_id = c.asset_id
        WHERE a.brand_id = %s
          AND a.confidence = 'inferred'
          AND c.vector_type = 'brand_voice'
        LIMIT 1
    """, (brand_uuid,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        print("⚠️ No inferred brand_voice asset available. TEST 2 skipped (expected if already curated).")
    else:
        approved_asset_id, approved_asset_text = row

        print(f"   🎯 Targeting Brand Voice Asset: {approved_asset_text[:40]}...")

        approve_asset(
            approved_asset_id,
            reviewer="Admin",
            notes="Approved brand voice memory (v1.6 validation)"
        )

        res = retrieve_context(
            brand_id_str,
            "design philosophy",
            "brand_voice"
        )

        if res and res[0]["confidence"] == "approved":
            print("✅ SUCCESS: Approved brand_voice asset correctly prioritized.")
        else:
            print(
                f"❌ FAILURE: Expected approved asset first, got "
                f"{res[0]['confidence'] if res else 'None'}"
            )

    # -------------------------------------------------

    print("\n--- 🧪 TEST 3: Deprecation (Exclude from Retrieval) ---")

    if approved_asset_id:
        reject_asset(
            approved_asset_id,
            reviewer="Admin",
            notes="Deprecating for validation test"
        )

        res = retrieve_context(
            brand_id_str,
            "design philosophy",
            "brand_voice"
        )

        found = any(
            d["content"] == approved_asset_text
            for d in res
        )

        if not found:
            print("✅ Deprecated asset successfully EXCLUDED.")
        else:
            print("❌ FAILURE: Deprecated asset still retrieved!")
    else:
        print("⚠️ Skipping TEST 3: No approved asset available.")

    # -------------------------------------------------

    print("\n--- 🧪 TEST 4: Audit Trail ---")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT action, previous_confidence, new_confidence
        FROM memory_reviews
        ORDER BY created_at DESC
        LIMIT 5
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    for r in rows:
        print(f"   - Action: {r[0]} | Old: {r[1]} -> New: {r[2]}")

# Run validation
# run_v1_6_validation()


# CELL SEPARATOR 

# CELL SEPARATOR 

# CELL SEPARATOR 
import os
from itertools import cycle
from google import genai
from google.genai import types
from IPython.display import display, Markdown

# Load environment variables
load_dotenv(override=True) # Ensure we reload if .env changed

# Load API keys from environment
GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY1"),
    os.getenv("GEMINI_API_KEY2"),
    os.getenv("GEMINI_API_KEY3"),
    os.getenv("GEMINI_API_KEY4"),
    os.getenv("GEMINI_API_KEY5"),
    os.getenv("GEMINI_API_KEY6"),
    os.getenv("GEMINI_API")

]

# Filter out missing keys
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

if not GEMINI_KEYS:
    raise RuntimeError("❌ No Gemini API keys found in environment variables.")

# Create a rotating iterator
_gemini_key_cycle = cycle(GEMINI_KEYS)

def get_gemini_client():
    """
    Returns a Gemini client using the next available API key.
    """
    api_key = next(_gemini_key_cycle)
    return genai.Client(api_key=api_key)

# Safe Gemini Call Wrapper (Auto-Retry)
def safe_generate_content(model: str, contents, config=None, max_retries=4):
    """
    Safely call Gemini with automatic API key rotation on failure.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            client = get_gemini_client()
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except Exception as e:
            last_error = e
            print(f"⚠️ Gemini call failed (attempt {attempt+1}/{max_retries}): {e}")
    
    raise RuntimeError(f"❌ All Gemini API keys exhausted. Last error: {last_error}")

# CELL SEPARATOR 
# 1. Hybrid Intent Classification

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

print("✅ Hybrid Intent Classifier Ready")
# CELL SEPARATOR 
# 2. Brand Reasoner & Explainability

SYSTEM_PROMPT = """
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
"""

def generate_explained_response(query: str, context: List[Dict], safety_status: Dict, intent: IntentType) -> Dict:
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
    
    full_prompt = f"""
    {SYSTEM_PROMPT}
    
    CONTEXT (Brand Memory):
    {context_str}
    
    {safety_instruction}
    
    USER QUERY: {query}
    
    Explain your answer based *only* on the context above.
    """
    
    try:
        response = safe_generate_content(
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
        "confidence_level": "high" if context else "medium", 
        "brand_elements_used": list(set([c.get('source_field', 'General') for c in context])) if isinstance(context, list) else [],
        "memory_sources": list(set([c.get('confidence', 'inferred') for c in context])),
        "live_context_used": False,
        "safety_status": safety_status['status']
    }

print("✅ Brand Reasoner Ready (Soft Safety Enabled)")

# CELL SEPARATOR 
# 3. Chat Pipeline (Strict Ordering)

def chat_session(user_query: str, brand_id: str = "wh_india_001"):
    print(f"\n💬 User: {user_query}")
    
    # 1. Intent Classification (Hybrid)
    intent = classify_intent_hybrid(user_query)
    print(f"   🧠 Intent: {intent.value}")
    
    # 2. Creative Block (Pre-computation)
    if intent == IntentType.CREATIVE:
        # Double check: if it was rule-based, we already blocked. 
        # But if Gemini classified it as creative and we didn't catch usage of forbidden words yet...
        # We'll block here. 
        print("   🚫 Creative Request Blocked.")
        return {
            "answer": "I can explain brand guidelines and evaluate ideas, but I don’t generate creative assets yet.",
            "safety_status": "BLOCKED_CREATIVE"
        }

    # 3. Retrieval
    # Using v1.6 retrieval (prioritizes approved)
    context = retrieve_context(brand_id, user_query, vector_type="brand_voice" if intent == IntentType.REASONING else "strategy")
    
    # 4. Safety Check (Off-Brand Rules) - BEFORE Reasoner
    # [v1.7] Pass actual intent (IntentType) to safety check
    safety = check_brand_safety(user_query, brand_id, intent)
    
    if safety['status'] == 'FAIL':
        print(f"   🛡️ Safety Block: {safety['reason']}")
        return {
            "answer": f"I cannot answer that. {safety['reason']}",
            "safety_status": "BLOCKED_SAFETY"
        }
    elif safety['status'] == 'PASS_WITH_WARNING':
        print(f"   ⚠️ Soft Safety Warning: {safety['reason']}")
        # Proceed to Reasoner, passing the warning
        
    # 5. Brand Reasoner
    response_obj = generate_explained_response(user_query, context, safety, intent)
    
    print(f"   🤖 Brand Brain: {response_obj['answer'][:100]}...")
    return response_obj

print("✅ Chat Pipeline Ready (Soft Safety Enabled)")

# CELL SEPARATOR 
# 4. v1.7 Validation Harness & DB Snapshot

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
    print("\n\n--- 🧪 v1.7 Validation Suite (Soft Safety) ---")
    
    # 1. Snapshot DB
    assets_before, chunks_before = get_db_counts()
    print(f"📊 DB Before: Assets={assets_before}, Chunks={chunks_before}")
        
    # 2. Test Cases
    queries = [
        ("What are our brand colors?", "KNOWLEDGE"),
        ("Is aggressive discounting on-brand?", "REASONING"), # Should trigger Soft Safety (PASS_WITH_WARNING)
        ("Write a Diwali campaign", "CREATIVE") # Should be blocked
    ]
    
    for q, expected in queries:
        print(f"\n--- Testing: '{q}' (Expected Intent/Flow: {expected}) ---")
        res = chat_session(q)
        print(f"   📄 Result: {json.dumps(res, indent=2)}")

    # 3. Verify Mutation
    assets_after, chunks_after = get_db_counts()
    print(f"\n📊 DB After: Assets={assets_after}, Chunks={chunks_after}")
    
    if assets_before == assets_after and chunks_before == chunks_after:
        print("✅ SUCCESS: Zero DB Mutation Confirmed.")
    else:
        print("❌ FAILURE: DB Mutation Detected!")

# run_v1_7_validation()

# CELL SEPARATOR 
def render_brand_brain_response(response: dict):
    """
    Human-friendly rendering of Brand Brain output.
    """

    display(Markdown("### 🤖 Brand Brain"))
    display(Markdown(response.get("answer", "_No response generated._")))

    confidence = response.get("confidence_level", "unknown")
    confidence_badge = {
        "high": "🟢 **High confidence** (Approved brand memory)",
        "medium": "🟡 **Medium confidence** (Inferred brand memory)",
        "live": "🔵 **Live context used**"
    }.get(confidence, "⚪ Confidence unknown")

    display(Markdown(f"**Confidence:** {confidence_badge}"))

    display(Markdown("---"))
    display(Markdown("#### 🔍 Explainability"))

    display(Markdown(f"- **Brand elements used:** {', '.join(response.get('brand_elements_used', [])) or 'N/A'}"))
    display(Markdown(f"- **Memory sources:** {', '.join(response.get('memory_sources', [])) or 'N/A'}"))
    
    safety_status = response.get('safety_status', 'UNKNOWN')
    if safety_status == 'PASS_WITH_WARNING':
        safety_display = "⚠️ **Soft Safety Warning** (Brand Conflict Explained)"
    else:
        safety_display = f"`{safety_status}`"
    
    display(Markdown(f"- **Safety status:** {safety_display}"))


def ask_brand_brain():
    """
    Interactive Brand Brain chat (read-only).
    """
    print("\n💬 Ask Brand Brain (type 'exit' to stop)\n")

    while True:
        user_query = input("You: ").strip()

        if user_query.lower() in ["exit", "quit"]:
            print("👋 Exiting Brand Brain chat.")
            break

        if not user_query:
            print("⚠️ Please enter a question.")
            continue

        try:
            response = chat_session(user_query)
            render_brand_brain_response(response)
        except Exception as e:
            print(f"❌ Error: {e}")

        print("\n" + "=" * 60 + "\n")

# CELL SEPARATOR 
# ask_brand_brain() # Interactive call disabled for automation
# What is our mission and vision?
# What colors and fonts define our brand?
# Is aggressive discounting on-brand?
# How should we sound on LinkedIn?
# Write a Diwali campaign
# CELL SEPARATOR 

# Append this to brand_brain_runner.py or overwrite the end

# DISABLE ALL OTHER RUNS
# run_v1_7_validation()

if __name__ == "__main__":
    print("\n\n--- 🧪 QUOTA & SAFETY VERIFICATION ---")
    
    # Test 1: Knowledge (Triggers classify_intent -> safe_generate_content)
    try:
        q1 = "What is our mission?"
        print(f"\nQuery 1: {q1}")
        res1 = chat_session(q1)
        print("RESULT 1:")
        print(json.dumps(res1, indent=2))
    except Exception as e:
        print(f"❌ Test 1 Failed: {e}")

    # Test 2: Reasoning (Triggers classify -> check_safety -> generate_explained (safe call))
    try:
        q2 = "Is aggressive discounting on-brand?"
        print(f"\nQuery 2: {q2}")
        res2 = chat_session(q2)
        print("RESULT 2:")
        print(json.dumps(res2, indent=2))
    except Exception as e:
        print(f"❌ Test 2 Failed: {e}")
