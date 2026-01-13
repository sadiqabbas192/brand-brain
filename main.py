import argparse
import sys
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

def main():
    parser = argparse.ArgumentParser(description="Brand Brain CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest Command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest sample brand data")

    # Validate Command
    validate_parser = subparsers.add_parser("validate", help="Run validation tests")
    validate_parser.add_argument("--v1.6", action="store_true", help="Run v1.6 validation tests")

    args = parser.parse_args()

    if args.command == "ingest":
        print("Starting ingestion...")
        ingest_brand(westinghouse_json)
        
    elif args.command == "validate":
        if getattr(args, 'v1.6', False): 
            run_v1_6_validation()
        else:
            run_validation()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
