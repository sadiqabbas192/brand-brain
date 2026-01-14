import argparse
import sys
import threading
import time
from brand_brain.core.ingestion import ingest_brand
from brand_brain.core.validation import run_validation, run_v1_6_validation

# Sample Data
westinghouse_json = {
    "brandId": "wh_india_001",
    "name": "Westinghouse India",
    "industry": "FMEG",
    "mission": "To enrich everyday living with reliable, thoughtfully engineered appliances that combine global heritage, modern innovation, and timeless design—delivering confidence, comfort, and consistency to Indian homes.",
    "brandVoice": "Confident & Reassuring. Premium yet Approachable. Clear & Functional. Trust-First. Design-Conscious.",
    "visualStyle": "Design-forward minimalism. Product as hero. Lifestyle-led context. Retro-modern blend. Premium finishes. Colors: Orange, Red, White, Green, Blue, Black.",
    "audience": "All genders, 25–45 years (core). Upper-middle to affluent households. Interests: Premium home & kitchen appliances, Modern kitchen aesthetics, Smart living. Focus: Tier 1 metros (Mumbai, Delhi NCR...) and affluent Tier 2.",
    "competitors": "Morphy Richards (Strong British Heritage, Wide Portfolio). Weaknesses: Inconsistent Visual Identity, Limited Design Differentiation.",
    "inspiration": "Morphy Richards",
    "website": "https://www.westinghousehomeware.in/"
}

class LoadingAnimation:
    """
    Context manager for a 5-dot loading animation.
    Cycles 1 to 5 dots every second.
    """
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._animate)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * 20 + "\r") # Clear line
        sys.stdout.flush()

    def _animate(self):
        dots = 1
        while not self._stop_event.is_set():
            # Print dots, left aligned in a field of 5 spaces to keep position stable if needed,
            # but user effectively just wants dots growing.
            # \r overwrites the line.
            sys.stdout.write(f"\r{'.' * dots}") 
            sys.stdout.flush()
            time.sleep(1)
            dots = (dots % 5) + 1

def main():
    parser = argparse.ArgumentParser(description="Brand Brain CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest Command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest sample brand data")

    # Validate Command
    validate_parser = subparsers.add_parser("validate", help="Run validation tests")
    validate_parser.add_argument("--v1.6", action="store_true", help="Run v1.6 validation tests")

    # Interactive Command
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive chat mode")
    interactive_parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Set Debug Mode Globally
    if getattr(args, 'debug', False):
        from brand_brain.config import set_debug_mode
        set_debug_mode(True)
        print("🔧 Debug Mode Enabled")

    if args.command == "ingest":
        print("Starting ingestion...")
        ingest_brand(westinghouse_json)
        
    elif args.command == "validate":
        if getattr(args, 'v1.6', False): 
            run_v1_6_validation()
        else:
            run_validation()
            
    elif args.command == "interactive":
        ask_brand_brain()
        
    else:
        parser.print_help()

def render_brand_brain_response_clean(response: dict, intent: str):
    """
    Non-tech friendly Brand Brain response renderer (CLI Version).
    """    
    # Divider
    print("\n" + "-" * 60)
    
    # 1. Intent
    print(f"Intent : {intent}")
    
    # 2. Response
    print(f"Response : {response.get('answer', '_No response generated._')}")
    
    # 3. Confidence
    confidence = response.get("confidence_level", "unknown")
    print(f"Confidence : {confidence}")
    
    # 4. Why this answer
    explain_lines = []
    
    if response.get("brand_elements_used"):
        explain_lines.append("Based on brand identity, voice, and positioning.")

    if response.get("memory_sources"):
        explain_lines.append("Uses existing brand knowledge (no external learning).")

    if str(response.get("safety_status", "")).startswith("PASS_WITH_WARNING"):
        explain_lines.append("This topic was evaluated carefully to protect brand positioning.")
        
    if response.get("live_context_used"):
        explain_lines.append("Incorporates live information from the web.")

    why_text = " ".join(explain_lines)
    if not why_text:
        why_text = "Answered using available brand guidelines."
        
    print(f"why this answer : {why_text}")
    
    # 5. Usage Info [v1.9]
    usage_info = response.get("usage_info", {})
    if usage_info:
        print(f"Model Used : {usage_info.get('model_name', 'Unknown')}")
        print(f"API Key Used : {usage_info.get('api_key_name', 'Unknown')}")
    
    print("-" * 60 + "\n")

def ask_brand_brain():
    """
    Interactive Brand Brain chat loop.
    """
    from brand_brain.core.orchestrator import chat_session
    
    print("\n👋 Welcome to Brand Brain")
    print("Type 'exit' to stop.\n")

    while True:
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting Brand Brain chat.")
            break

        if user_query.lower() in ["exit", "quit"]:
            print("👋 Exiting Brand Brain chat.")
            break

        if not user_query:
            print("⚠️ Please enter a question.")
            continue

        try:
            # Check DEBUG_MODE to decide if we show animation
            # Even if we don't check, if debug mode prints logs, they will mess up the animation line.
            # But the user requirements imply animation is for the clean UX.
            # We can run it unconditionally, accepting that debug mode will look messy (which is expected for debug).
            
            with LoadingAnimation():
                response = chat_session(user_query)
            
            # Extract intent safely
            intent = response.get("intent", "knowledge")

            # Render clean UX
            render_brand_brain_response_clean(response, intent)
            
        except Exception as e:
            print(f"❌ Error during chat session: {e}")

if __name__ == "__main__":
    main()
