import requests
import time
import sys

BASE_URL = "http://localhost:8001"
API_KEY = "digitalf5"

def test_url(method, path, headers=None, params=None, json_body=None, expected_status=200, desc=""):
    url = f"{BASE_URL}{path}"
    print(f"Testing: {method} {url} | {desc}")
    try:
        if method == "GET":
            resp = requests.get(url, headers=headers, params=params)
        else:
            resp = requests.post(url, headers=headers, params=params, json=json_body)
            
        if resp.status_code == expected_status:
            print(f"✅ PASS: Got {resp.status_code}")
        else:
            print(f"❌ FAIL: Expected {expected_status}, Got {resp.status_code}")
            print(f"   Response: {resp.text}")
            return False
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

print("--- 1. Public Routes ---")
test_url("GET", "/health", expected_status=200, desc="Health Check (No Auth)")
test_url("GET", "/version", expected_status=200, desc="Version Check (No Auth)")

print("\n--- 2. Protected Routes (Auth Failure) ---")
test_url("GET", "/brands", expected_status=401, desc="Brands (No Key)")
test_url("POST", "/ask", json_body={"question": "hi", "brand_id": "wh"}, expected_status=401, desc="Ask (No Key)")
test_url("GET", "/brands", headers={"x-api-key": "wrong"}, expected_status=401, desc="Brands (Wrong Key)")

print("\n--- 3. Protected Routes (Auth Success) ---")
test_url("GET", "/brands", headers={"x-api-key": API_KEY}, expected_status=200, desc="Brands (Header Key)")
test_url("GET", "/brands", params={"api_key": API_KEY}, expected_status=200, desc="Brands (Query Key)")
