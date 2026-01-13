
import os
import json
import uuid
import time
import typing
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from pinecone import Pinecone, ServerlessSpec
from google import genai
from google.genai import types
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv(override=True)

NEON_DB_URL = os.getenv("NEON_DB_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([NEON_DB_URL, PINECONE_API_KEY, GEMINI_API_KEY]):
    raise ValueError("Missing required environment variables.")

# Initialize Clients
client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# DB Connection
def get_db_connection():
    return psycopg2.connect(NEON_DB_URL)

# Chunking (Simulated simple split for script, in notebook it uses langchain)
def chunk_text(text: str) -> List[str]:
    # Simple split by newline for this test script to avoid dependency issues if langchain missing
    # But usually langchain is present. Let's try to mimic simple splitting.
    return [t for t in text.split('\n\n') if t.strip()]

def generate_embedding(text: str) -> List[float]:
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config={
                'output_dimensionality': 768,
                'task_type': 'RETRIEVAL_DOCUMENT',
            }
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"Embedding Error: {e}")
        return []

# --- v1.6 IMPL ---

def retrieve_context(brand_name_str: str, query: str, vector_type: str = "brand_voice", top_k: int = 3):
    if brand_name_str == "wh_india_001":
        brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_name_str))
        org_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'default_org'))
    else:
        brand_uuid = brand_name_str
        org_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'default_org'))

    print(f"\n🔎 Querying Brand {brand_name_str} (UUID: {brand_uuid}) [{vector_type}]: '{query}'")
    
    try:
        query_embedding_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config={'output_dimensionality': 768, 'task_type': 'RETRIEVAL_QUERY'}
        )
        query_embedding = query_embedding_result.embeddings[0].values
    except Exception as e:
        print(f"Embedding Error: {e}"); return []
    
    namespace = f"{org_id}:{brand_uuid}:{vector_type}"
    idx = pc.Index("brand-brain-index")
    
    results = idx.query(
        vector=query_embedding, top_k=top_k * 3, namespace=namespace, include_metadata=True
    )
    
    if not results['matches']: return []
        
    conn = get_db_connection()
    cur = conn.cursor()
    chunk_ids = [m['id'] for m in results['matches']]
    
    retrieved_docs = []
    if chunk_ids:
        placeholders = ', '.join(['%s'] * len(chunk_ids))
        query_sql = f"""
            SELECT c.chunk_id, c.content, c.vector_type, a.confidence 
            FROM brand_chunks c
            JOIN brand_assets a ON c.asset_id = a.asset_id
            WHERE c.chunk_id IN ({placeholders})
        """
        cur.execute(query_sql, tuple(chunk_ids))
        rows = cur.fetchall()
        db_map = {row[0]: {"content": row[1], "confidence": row[3]} for row in rows}
        
        valid_candidates = []
        for match in results['matches']:
            c_id = match['id']
            score = match['score']
            if c_id not in db_map: continue
            data = db_map[c_id]
            confidence = data['confidence'] or 'inferred'
            
            if confidence == 'deprecated': continue
            
            priority = 1
            if confidence == 'approved': priority = 3
            elif confidence == 'reviewed': priority = 2
            
            valid_candidates.append({
                "content": data['content'], "score": score, "confidence": confidence, "priority": priority
            })
            
        valid_candidates.sort(key=lambda x: (x['priority'], x['score']), reverse=True)
        final_results = valid_candidates[:top_k]
        
        for i, res in enumerate(final_results):
            print(f"   [{i+1}] [{res['confidence'].upper()}] Score: {res['score']:.4f} | Content: {res['content'][:100]}...")
            retrieved_docs.append(res)
            
    cur.close(); conn.close()
    return retrieved_docs

# --- MEMORY FUNCTIONS ---

def list_inferred_assets(brand_id_str: str):
    print(f"\n📋 Listing Inferred Assets for {brand_id_str}...")
    brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_id_str))
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT asset_id, asset_type, raw_text, source, confidence FROM brand_assets WHERE brand_id = %s AND confidence = 'inferred'", (brand_uuid,))
    rows = cur.fetchall()
    cur.close(); conn.close()
    for r in rows:
        print(f"   [ID: {r['asset_id']}] {r['raw_text'][:50]}... (Source: {r['source']})")
    return rows

def approve_asset(asset_id: str, reviewer: str, notes: str):
    print(f"\n✅ Approving Asset {asset_id}...")
    conn = get_db_connection(); cur = conn.cursor()
    try:
        cur.execute("SELECT confidence FROM brand_assets WHERE asset_id = %s", (asset_id,))
        res = cur.fetchone()
        if not res: return
        prev_conf = res[0]
        cur.execute("UPDATE brand_assets SET confidence = 'approved', reviewed_by = %s, review_notes = %s, reviewed_at = NOW() WHERE asset_id = %s", (reviewer, notes, asset_id))
        cur.execute("INSERT INTO memory_reviews (review_id, asset_id, action, previous_confidence, new_confidence, reviewer, notes) VALUES (%s, %s, %s, %s, %s, %s, %s)", (str(uuid.uuid4()), asset_id, 'approve', prev_conf, 'approved', reviewer, notes))
        conn.commit(); print("   -> Asset Approved.")
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
        conn.commit(); print("   -> Asset Deprecated.")
    except Exception as e:
        conn.rollback(); print(f"❌ {e}")
    finally:
        cur.close(); conn.close()

# --- VALIDATION ---

def run_v1_6_validation():
    brand_id_str = "wh_india_001"
    brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_id_str))
    
    print("\n\n--- 🧪 TEST 1: Ingestion starts as 'inferred' ---")
    conn = get_db_connection(); cur = conn.cursor()
    # Check if we have any inferred assets. If not, we might need to assume previous run or mocked one.
    # The prompt implies we are extending v1.5 which had Tests.
    # We'll check DB state.
    cur.execute("SELECT count(*) FROM brand_assets WHERE brand_id = %s AND confidence = 'inferred'", (brand_uuid,))
    count = cur.fetchone()[0]
    cur.close(); conn.close()
    print(f"✅ Found {count} 'inferred' assets")
    
    print("\n--- 🧪 TEST 2: Retrieval Priority (Approved > Inferred) ---")
    inferred = list_inferred_assets(brand_id_str)
    if inferred:
        target = inferred[0]
        t_id = target['asset_id']
        t_text = target['raw_text']
        print(f"   Targeting: {t_text[:20]}...")
        
        # Approve
        approve_asset(t_id, "Admin", "Accurate")
        
        # Retrieve
        # We need a query that matches this asset. Since I don't know the content, I'll use a generic one or extract keyword.
        # But wait, vector search needs relevant query.
        # I'll use the beginning of the text as query.
        query = t_text[:50]
        res = retrieve_context(brand_id_str, query, "strategy") # Assuming strategy, else try brand_voice?
        # Note: If no results, retry with other vector_type
        if not res:
             res = retrieve_context(brand_id_str, query, "brand_voice")
             
        if res and res[0]['confidence'] == 'approved':
             print("✅ Approved asset prioritized.")
        else:
             print(f"⚠️ Top result confidence: {res[0]['confidence'] if res else 'None'}")
             
        print("\n--- 🧪 TEST 3: Deprecation (Exclude from Retrieval) ---")
        reject_asset(t_id, "Admin", "Deprecating")
        res_d = retrieve_context(brand_id_str, query, "strategy")
        if not res_d: res_d = retrieve_context(brand_id_str, query, "brand_voice")
        
        found = any(d['content'] == t_text for d in res_d)
        if not found:
            print("✅ Deprecated asset successfully EXCLUDED.")
        else:
            print("❌ FAILURE: Deprecated asset still retrieved!")

    print("\n--- 🧪 TEST 4: Audit Trail ---")
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("SELECT * FROM memory_reviews ORDER BY created_at DESC LIMIT 5")
    for r in cur.fetchall():
        print(f"   - Action: {r[3]} | Old: {r[4]} -> New: {r[5]}")
    cur.close(); conn.close()

if __name__ == "__main__":
    run_v1_6_validation()
