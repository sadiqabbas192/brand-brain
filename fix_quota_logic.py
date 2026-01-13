import json
import os

NOTEBOOK_PATH = r"d:\brand-brain\brand_brain_v1.7.ipynb"

def fix_quota_logic():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    
    # 1. Identify Cells
    rotation_cell_index = -1
    intent_cell_index = -1
    
    rotation_cell = None
    
    for i, cell in enumerate(cells):
        source = "".join(cell['source'])
        if "def safe_generate_content" in source and "cycle(GEMINI_KEYS)" in source:
            rotation_cell_index = i
            rotation_cell = cell
        if "class IntentType(Enum):" in source and "def classify_intent_hybrid" in source:
            intent_cell_index = i

    if rotation_cell_index == -1 or intent_cell_index == -1:
        print("❌ Could not find target cells. Aborting.")
        return

    print(f"Found Rotation Cell at {rotation_cell_index}")
    print(f"Found Intent Cell at {intent_cell_index}")

    # 2. Move Rotation Cell
    # Remove from old location
    cells.pop(rotation_cell_index)
    
    # Insert before Intent Cell (adjust index if needed)
    # If rotation was after intent (which it is), the intent index is still valid as it was 'before' the popped one
    # But let's be safe: find intent index again or just use the known ID logic if indices shifted
    
    # Re-find intent index to be sure
    for i, cell in enumerate(cells):
        source = "".join(cell['source'])
        if "class IntentType(Enum):" in source and "def classify_intent_hybrid" in source:
            intent_cell_index = i
            break
            
    # Insert rotation cell before intent cell
    cells.insert(intent_cell_index, rotation_cell)
    print(f"Moved Rotation Cell to index {intent_cell_index}")

    # 3. Update Code to use safe_generate_content
    
    # Update Intent Classifier
    intent_cell = cells[intent_cell_index + 1] # It's now after the inserted cell
    source_intent = "".join(intent_cell['source'])
    if "client.models.generate_content" in source_intent:
        new_source_intent = source_intent.replace(
            "response = client.models.generate_content(",
            "response = safe_generate_content("
        )
        intent_cell['source'] = new_source_intent.splitlines(keepends=True)
        print("✅ Updated Intent Classifier to use safe_generate_content")

    # Update Reasoner
    # Find reasoner cell
    for cell in cells:
        source = "".join(cell['source'])
        if "def generate_explained_response" in source:
            if "client.models.generate_content" in source:
                new_source_reasoner = source.replace(
                    "response = client.models.generate_content(",
                    "response = safe_generate_content("
                )
                cell['source'] = new_source_reasoner.splitlines(keepends=True)
                print("✅ Updated Brand Reasoner to use safe_generate_content")
            break

    # 4. Save
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    print("✅ Notebook fixed successfully.")

if __name__ == "__main__":
    fix_quota_logic()
