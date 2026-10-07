import json

notebook_path = '/Users/purevjavgerel/Desktop/analysis/research/analysis/visualize_reports.ipynb'

with open(notebook_path, 'r') as f:
    data = json.load(f)

for cell in data['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        is_confusion_matrix = any('Confusion Matrix' in line for line in source)
        if is_confusion_matrix:
            # We want to insert plt.xticks(rotation=45, ha='right') before plt.tight_layout()
            # Also insert plt.yticks(rotation=0)
            
            new_source = []
            for line in source:
                if 'plt.tight_layout()' in line:
                    new_source.append("plt.xticks(rotation=45, ha='right')\n")
                    new_source.append("plt.yticks(rotation=0)\n")
                new_source.append(line)
            cell['source'] = new_source

with open(notebook_path, 'w') as f:
    json.dump(data, f, indent=2)
