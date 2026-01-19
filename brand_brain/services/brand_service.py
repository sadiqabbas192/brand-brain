from typing import List, Dict
from ..database import get_db_connection

def get_all_brands() -> List[Dict[str, str]]:
    """
    Retrieves all brands from the database options.
    Returns:
        List of dictionaries containing 'brand_id' and 'name'.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT brand_id, name FROM brands ORDER BY name ASC")
        rows = cur.fetchall()
        brands = [{"brand_id": row[0], "name": row[1]} for row in rows]
        return brands
    except Exception as e:
        print(f"❌ Error fetching brands: {e}")
        return []
    finally:
        cur.close()
        conn.close()
