import sys
import time
from brand_brain.core.orchestrator import chat_session
from brand_brain.database import get_db_connection

def get_row_counts():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM brand_assets")
    assets_count = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM brand_chunks")
    chunks_count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return assets_count, chunks_count

def run_test(name, query, expected_grounding):
    print(f"\n🧪 Test: {name}")
    print(f"   Query: '{query}'")
    
    response = chat_session(query)
    
    grounding_used = response.get("live_context_used", False)
    status_icon = "✅" if grounding_used == expected_grounding else "❌"
    
    print(f"   Expected Grounding: {expected_grounding}")
    print(f"   Actual Grounding:   {grounding_used}")
    print(f"   Result: {status_icon}")
    
    if grounding_used != expected_grounding:
        print(f"   FAILURE: Unexpected grounding behavior for '{query}'")
        return False
    return True

def main():
    print("🚀 Starting Brand Brain v1.8 Verification")
    print("⏳ Waiting 60s for API quota cooldown...")
    time.sleep(60)
    
    # 1. Initial State
    initial_assets, initial_chunks = get_row_counts()
    print(f"\n📊 Initial DB State: {initial_assets} assets, {initial_chunks} chunks")
    
    tests = [
        ("Internal Memory", "What is our brand mission?", False),
        ("External Comparison", "Compare us with Morphy Richards", True),
        ("Freshness Query", "What are the latest kitchen trends?", True),
        ("Competitor vs", "How do we differ vs Philips?", True),
        ("Reasoning without grounding", "Does our voice allow slang?", False)
    ]
    
    failures = 0
    for name, query, expected in tests:
        if not run_test(name, query, expected):
            failures += 1
        print("   ⏳ Waiting 30s to respect API quota...")
        time.sleep(30)
            
    # 2. Final State
    final_assets, final_chunks = get_row_counts()
    print(f"\n📊 Final DB State: {final_assets} assets, {final_chunks} chunks")
    
    if final_assets != initial_assets or final_chunks != initial_chunks:
        print("❌ FAILURE: Memory Mutation Detected!")
        print(f"   Assets diff: {final_assets - initial_assets}")
        print(f"   Chunks diff: {final_chunks - initial_chunks}")
        failures += 1
    else:
        print("✅ SUCCESS: Zero Memory Mutation Confirmed.")
        
    if failures == 0:
        print("\n🏆 ALL TESTS PASSED for v1.8")
    else:
        print(f"\n💥 {failures} TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
