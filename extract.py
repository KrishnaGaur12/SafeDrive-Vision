import json
import codecs

with open('Distrated Driver detection.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with codecs.open('extracted_code.py', 'w', encoding='utf-8') as out:
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, str):
                out.write(source + '\n\n')
            else:
                out.write(''.join(source) + '\n\n')
