
# Append this to brand_brain_runner.py or overwrite the end

# DISABLE ALL OTHER RUNS
# run_v1_7_validation()

if __name__ == "__main__":
    print("\n\n--- 🧪 QUOTA & SAFETY VERIFICATION ---")
    
    # Test 1: Knowledge (Triggers classify_intent -> safe_generate_content)
    try:
        q1 = "What is our mission?"
        print(f"\nQuery 1: {q1}")
        res1 = chat_session(q1)
        print("RESULT 1:")
        print(json.dumps(res1, indent=2))
    except Exception as e:
        print(f"❌ Test 1 Failed: {e}")

    # Test 2: Reasoning (Triggers classify -> check_safety -> generate_explained (safe call))
    try:
        q2 = "Is aggressive discounting on-brand?"
        print(f"\nQuery 2: {q2}")
        res2 = chat_session(q2)
        print("RESULT 2:")
        print(json.dumps(res2, indent=2))
    except Exception as e:
        print(f"❌ Test 2 Failed: {e}")
