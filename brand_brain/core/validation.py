import uuid
from ..database import get_db_connection
from .retrieval import retrieve_context
from .memory_gov import list_inferred_assets, approve_asset, reject_asset

def run_validation():
    print("\n--- TEST 1: Westinghouse Brand Voice ---")
    retrieve_context("wh_india_001", "Describe our design philosophy.", vector_type="brand_voice")
    
    # Test 2: Westinghouse Competitor Context
    print("\n--- TEST 2: Westinghouse Strategy ---")
    retrieve_context("wh_india_001", "Who are we fighting against?", vector_type="strategy")
    
    # Test 3: Off-Brand check
    print("\n--- TEST 3: Isolation / Irrelevant Query ---")
    retrieve_context("wh_india_001", "How to be cheap and loud?", vector_type="brand_voice")

def run_v1_6_validation():
    brand_id_str = "wh_india_001"
    brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_id_str))
    
    print("\n\n--- 🧪 TEST 1: Ingestion starts as 'inferred' ---")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM brand_assets WHERE brand_id = %s AND confidence = 'inferred'", (brand_uuid,))
    res = cur.fetchone()
    count = res[0] if res else 0
    cur.close(); conn.close()
    print(f"✅ Found {count} 'inferred' assets")
    
    print("\n--- 🧪 TEST 2: Retrieval Priority (Approved > Inferred) ---")
    inferred = list_inferred_assets(brand_id_str)
    if inferred:
        target = inferred[0]
        print(f"   Targeting: {target['raw_text'][:20]}...")
        approve_asset(target['asset_id'], "Admin", "Accurate")
        res = retrieve_context(brand_id_str, "brand philosophy", "strategy")
        if res and res[0]['confidence'] == 'approved':
             print("✅ Approved asset prioritized.")
        else:
             print(f"⚠️ Top result confidence: {res[0]['confidence'] if res else 'None'}")

    print("\n--- 🧪 TEST 3: Deprecation (Exclude from Retrieval) ---")
    # Refresh inferred list as we approved one
    # inferred = list_inferred_assets(brand_id_str) 
    # Just list again to find another or reuse if we want to test deprecation on the same one?
    # The notebook reused `target` which was approved. Wait, if it's approved, how do we reject it?
    # The notebook code: `if inferred: reject_asset(target...`.
    # `approve_asset` updates it. `reject_asset` updates it again. So we are deprecating the APPROVED asset.
    if inferred:
        reject_asset(target['asset_id'], "Admin", "Deprecating")
        res = retrieve_context(brand_id_str, "brand philosophy", "strategy")
        found = any(d['content'] == target['raw_text'] for d in res)
        if not found:
            print("✅ Deprecated asset successfully EXCLUDED.")
        else:
            print("❌ FAILURE: Deprecated asset still retrieved!")

    print("\n--- 🧪 TEST 4: Audit Trail ---")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM memory_reviews ORDER BY created_at DESC LIMIT 5")
    for r in cur.fetchall():
        # r structure depends on table definition. Notebook used index access.
        # Assuming: 0:review_id, 1:asset_id, 2:action, 3:prev_conf, 4:new_conf...
        # Notebook: `r[3]` for action, `r[4]` for old, `r[5]` for new.
        # Need to verify column order. 
        # INSERT INTO memory_reviews (review_id, asset_id, action, previous_confidence, new_confidence, reviewer, notes)
        # If select * follows creation order, it might be roughly correct.
        print(f"   - {r}") 
    cur.close(); conn.close()
