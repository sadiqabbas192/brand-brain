import psycopg2
from psycopg2.extras import RealDictCursor, Json
from pinecone import Pinecone, ServerlessSpec
from .config import NEON_DB_URL, PINECONE_API_KEY

def get_db_connection():
    """Establishes and returns a connection to the Neon Postgres database."""
    return psycopg2.connect(NEON_DB_URL)

def get_pinecone_client():
    """Initializes and returns the Pinecone client."""
    return Pinecone(api_key=PINECONE_API_KEY)

def check_connection():
    """Checks the database connection and existence of required tables."""
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
