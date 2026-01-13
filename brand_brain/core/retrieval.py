import uuid
from typing import List, Dict, Optional
from ..database import get_db_connection, get_pinecone_client
from ..services.embedding import generate_query_embedding

def retrieve_context(brand_name_str: str, query: str, vector_type: str = "brand_voice", top_k: int = 3) -> List[Dict]:
    if brand_name_str == "wh_india_001":
        brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_name_str))
        org_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'default_org'))
    else:
        brand_uuid = brand_name_str
        org_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'default_org'))

    print(f"\n🔎 Querying Brand {brand_name_str} (UUID: {brand_uuid}) [{vector_type}]: '{query}'")
    
    query_embedding = generate_query_embedding(query)
    if not query_embedding:
        return []
    
    namespace = f"{org_id}:{brand_uuid}:{vector_type}"
    pc = get_pinecone_client()
    index_name = "brand-brain-index"
    idx = pc.Index(index_name)
    
    # Retrieve more candidates to allow for filtering
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
        # [MODIFIED v1.6] Join with brand_assets to get confidence
        query_sql = f"""
            SELECT c.chunk_id, c.content, c.vector_type, a.confidence 
            FROM brand_chunks c
            JOIN brand_assets a ON c.asset_id = a.asset_id
            WHERE c.chunk_id IN ({placeholders})
        """
        cur.execute(query_sql, tuple(chunk_ids))
        rows = cur.fetchall()
        
        # Create lookup
        db_map = {row[0]: {"content": row[1], "confidence": row[3]} for row in rows}
        
        valid_candidates = []
        
        for match in results['matches']:
            c_id = match['id']
            score = match['score']
            if c_id not in db_map: 
                continue
                
            data = db_map[c_id]
            confidence = data['confidence'] or 'inferred' 
            
            # [RULE] Exclude Deprecated
            if confidence == 'deprecated':
                continue
                
            # [RULE] Prioritize: approved > reviewed > inferred
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
        
        # Take top_k
        final_results = valid_candidates[:top_k]
        
        for i, res in enumerate(final_results):
            print(f"   [{i+1}] [{res['confidence'].upper()}] Score: {res['score']:.4f} | Content: {res['content'][:100]}...")
            retrieved_docs.append(res)
            
    cur.close()
    conn.close()
    return retrieved_docs
