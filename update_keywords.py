import json
import os

NOTEBOOK_PATH = r"d:\brand-brain\brand_brain_v1.7.ipynb"

def update_keywords():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        source = "".join(cell['source'])
        if "FORBIDDEN_KEYWORDS =" in source:
            # Replace the line
            new_source = []
            for line in cell['source']:
                if "FORBIDDEN_KEYWORDS =" in line:
                    new_source.append('FORBIDDEN_KEYWORDS = {"cheap", "free", "lowest price", "clearance", "sale", "loud", "discount", "discounting"} \n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print("✅ Updated FORBIDDEN_KEYWORDS")
            break

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)

if __name__ == "__main__":
    update_keywords()
