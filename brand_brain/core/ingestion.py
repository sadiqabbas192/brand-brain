import uuid
import json
import logging
from typing import Dict, List, Any
from pinecone import ServerlessSpec
from google.genai import types

from ..database import get_db_connection, get_pinecone_client
from ..services.embedding import chunk_text, generate_embedding
from ..services.gemini import get_gemini_client

def extract_assets(brand_data: Dict) -> List[Dict]:
    assets = []
    brand_id = brand_data.get("brandId")
    
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

def ingest_brand(brand_data: Dict):
    brand_id_str = brand_data['brandId']
    brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_id_str))
    
    brand_name = brand_data.get('name', 'Unknown')
    org_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'default_org')) # Placeholder Org

    print(f"\n🧠 Ingesting Brand: {brand_name} (UUID: {brand_uuid}) ...")
    
    conn = get_db_connection()
    cur = conn.cursor()
    pc = get_pinecone_client()
    
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

                # Note: Assuming table schema allows token_count
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
        raise e
    finally:
        cur.close()
        conn.close()

def ingest_single_asset(asset: Dict, brand_data: Dict):
    """Helper to ingest a single constructed asset."""
    brand_id_str = brand_data['brandId']
    brand_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, brand_id_str))
    org_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'default_org'))
    
    conn = get_db_connection()
    cur = conn.cursor()
    pc = get_pinecone_client()
    idx = pc.Index("brand-brain-index")

    try:
        # Store Asset in Postgres w/ confidence
        source_tag = f"{asset['source_field']} | v1.5 | confidence:inferred"
        
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
        print(f"   ✅ Successfully stored Type B memory for {brand_data['name']}")
    finally:
        cur.close()
        conn.close()

def grounding_assisted_ingest(brand_data: Dict, target_vector_type="strategy"):
    brand_name = brand_data['name']
    website = brand_data.get('website', '')
    client = get_gemini_client()
    
    print(f"\n🌍 Starting Grounding-Assisted Ingestion for {brand_name}...")

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
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearchRetrieval)]
            )
        )
        # Robust JSON Parsing
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        extracted_data = json.loads(text.strip())
        print(f"   ✅ Extracted Grounded Data: {list(extracted_data.keys())}")

        combined_text = f"Philosophy: {extracted_data.get('brand_philosophy', '')}\n"
        combined_text += f"Design Principles: {extracted_data.get('design_principles', '')}\n"
        combined_text += f"Positioning: {extracted_data.get('positioning', '')}"

        grounded_asset = {
            "asset_id": str(uuid.uuid4()),
            "brand_id": brand_data['brandId'],
            "asset_type": "guideline",
            "vector_type": target_vector_type,
            "source_field": "grounding_assisted_ingestion",
            "content": combined_text
        }

        ingest_single_asset(grounded_asset, brand_data)

    except Exception as e:
        print(f"❌ Grounding Ingestion Failed: {e}")
