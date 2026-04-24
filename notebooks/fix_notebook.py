import json

filepath = r'c:\Users\jorge\OneDrive\Escritorio\Master CEU\TFM v2\notebooks\NJorge3_Entrenamiento_Modelo_Hibrido.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Buscamos la línea a reemplazar
        new_source = []
        for line in cell['source']:
            if "y_log = np.log1p(df['precio'])" in line:
                new_source.append("df['log_precio'] = np.log1p(df['precio'])\n")
                new_source.append("y_log = df['log_precio']\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook actualizado correctamente.")
