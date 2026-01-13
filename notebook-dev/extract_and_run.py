import json
import os

NOTEBOOK_PATH = r"d:\brand-brain\brand_brain_v1.7.ipynb"
RUNNER_PATH = r"d:\brand-brain\brand_brain_runner.py"

def extract_and_run():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            # Comment out the interactive ask_brand_brain call if it exists at the very end to avoid blocking
            if "ask_brand_brain()" in source and "def ask_brand_brain" not in source:
                 source = source.replace("ask_brand_brain()", "# ask_brand_brain() # Interactive call disabled for automation")
            
            code_cells.append(source)
            
    full_code = "\n# CELL SEPARATOR \n".join(code_cells)
    
    with open(RUNNER_PATH, 'w', encoding='utf-8') as f:
        f.write(full_code)
        
    print(f"✅ Extracted code to {RUNNER_PATH}")

if __name__ == "__main__":
    extract_and_run()
