
import json
import sys

def convert_notebook(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            code_cells.append(source)
            code_cells.append("\n\n")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(code_cells)
    
    print(f"Converted {notebook_path} to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python notebook_to_script.py <input_ipynb> <output_py>")
    else:
        convert_notebook(sys.argv[1], sys.argv[2])
