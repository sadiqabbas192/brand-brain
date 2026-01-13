import uuid
from psycopg2.extras import RealDictCursor
from ..database import get_db_connection, get_pinecone_client
from ..services.embedding import generate_embedding, chunk_text

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
        pc = get_pinecone_client()
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
