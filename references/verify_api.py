import requests
import json
from brand_brain.core.orchestrator import chat_session

BASE_URL = "http://127.0.0.1:8001"

def test_health():
    print("Testing /health ...", end=" ")
    try:
        r = requests.get(f"{BASE_URL}/health")
        if r.status_code == 200 and r.json() == {"status": "ok"}:
            print("PASS")
        else:
            print(f"FAIL: {r.status_code} {r.text}")
    except Exception as e:
        print(f"FAIL: {e}")

def test_version():
    print("Testing /version ...", end=" ")
    try:
        r = requests.get(f"{BASE_URL}/version")
        if r.status_code == 200 and r.json()["version"] == "1.8.0":
            print("PASS")
        else:
            print(f"FAIL: {r.status_code} {r.text}")
    except Exception as e:
        print(f"FAIL: {e}")

def test_brands():
    print("Testing /brands ...", end=" ")
    try:
        r = requests.get(f"{BASE_URL}/brands")
        if r.status_code == 200:
            data = r.json()
            # Expect {"brands": [{"id": ..., "name": ...}]}
            if "brands" in data and len(data["brands"]) > 0:
                b = data["brands"][0]
                if b["id"] == "wh_india_001" and b["name"] == "Westinghouse India":
                     print("PASS")
                else:
                     print(f"FAIL: Content Mismatch {data}")
            else:
                 print(f"FAIL: Invalid Structure {data}")
        else:
             print(f"FAIL: {r.status_code} {r.text}")
    except Exception as e:
        print(f"FAIL: {e}")


def test_ask_parity_blocked():
    print("\nTesting /ask Parity (BLOCKED Case - Deterministic)...")
    question = "Write a funny poem about the brand."
    brand_id = "wh_india_001"
    
    # 1. Core Result
    print("   Invoking Core...", end=" ")
    try:
        core_result = chat_session(question, brand_id)
        print("Done")
    except Exception as e:
        print(f"FAIL (Core): {e}")
        return

    # 2. API Result
    print("   Invoking API...", end=" ")
    try:
        payload = {"brand_id": brand_id, "question": question}
        r = requests.post(f"{BASE_URL}/ask", json=payload)
        if r.status_code != 200:
             print(f"FAIL (API Error): {r.status_code} {r.text}")
             return
        api_result = r.json()
        print("Done")
    except Exception as e:
        print(f"FAIL (API Connection): {e}")
        return

    # 3. Compare
    print("   Comparing...")
    mismatch = False
    
    if core_result.get("intent") != api_result.get("intent"):
        print(f"     MISMATCH: Intent Core='{core_result.get('intent')}' API='{api_result.get('intent')}'")
        mismatch = True
    
    if core_result.get("answer") != api_result.get("answer"):
         print(f"     MISMATCH: Answer Core='{core_result.get('answer')}' API='{api_result.get('answer')}'")
         mismatch = True

    if not mismatch:
        print("   ✅ Parity Confirmed (Blocked Case).")
    else:
        print("   ❌ Parity Failed (Blocked Case).")

def test_ask_parity_success():
    print("\nTesting /ask Parity (SUCCESS Case - Structural)...")
    question = "What is our brand philosophy?"
    brand_id = "wh_india_001"

    # 1. Core
    print("   Invoking Core...", end=" ")
    core_result = chat_session(question, brand_id)
    print("Done")

    # 2. API
    print("   Invoking API...", end=" ")
    try:
        r = requests.post(f"{BASE_URL}/ask", json={"brand_id": brand_id, "question": question})
        if r.status_code != 200:
             print(f"FAIL: API Status {r.status_code}")
             return
        api_result = r.json()
        print("Done")
    except Exception as e:
        print(f"FAIL: {e}")
        return

    # 3. Compare
    mismatch = False
    
    if core_result["intent"] != api_result["intent"]:
        print(f"     MISMATCH: Intent Core='{core_result['intent']}' API='{api_result['intent']}'")
        mismatch = True
        
    if core_result["answer"] != api_result["answer"]:
        print("     ℹ️  Answer content differs (Expected due to LLM variance).")
    else:
        print("     ✅ Answer content identical.")

    if not mismatch:
         print("   ✅ Structural Parity Confirmed.")
    else:
         print("   ❌ Structural Parity Failed.")

if __name__ == "__main__":
    test_health()
    test_version()
    test_brands()
    test_ask_parity_blocked()
    test_ask_parity_success()
