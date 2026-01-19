import argparse
import json
import os
import sys
from brand_brain.core.ingestion import ingest_brand

def main():
    parser = argparse.ArgumentParser(description="Ingest brand data from a JSON file into Brand Brain.")
    parser.add_argument("file_path", help="Path to the JSON file containing brand data")
    
    args = parser.parse_args()
    
    file_path = args.file_path
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at '{file_path}'")
        sys.exit(1)
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            brand_data = json.load(f)
            
        print(f"📄 Loaded data for brand: {brand_data.get('name', 'Unknown')}")
        ingest_brand(brand_data)
        
    except json.JSONDecodeError as e:
        print(f"❌ Error: Failed to parse JSON file. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
