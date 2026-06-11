import pandas as pd
import os
from pathlib import Path

# Estado del CSV
df = pd.read_csv('data/raw/propiedades_raw.csv')
print(f'Total propiedades en CSV: {len(df)}')
img_url_cols = [c for c in df.columns if 'img' in c]
print(f'Columnas de imagen (URLs): {img_url_cols}')
print()

# Estado de imagenes locales
img_dir = Path('data/raw/images')
if img_dir.exists():
    images = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.webp']:
        images.extend(img_dir.glob(ext))
    print(f'Imagenes fisicas descargadas: {len(images)}')
    if images:
        sizes = [f.stat().st_size for f in images]
        avg_kb = sum(sizes)/len(sizes)/1024
        total_mb = sum(sizes)/1024/1024
        print(f'  Tamano medio por imagen: {avg_kb:.1f} KB')
        print(f'  Tamano total ocupado: {total_mb:.1f} MB')
        # Proyeccion para 5000 pisos con 3 imagenes cada uno
        total_imgs_5000 = 5000 * 3
        proj_gb = (avg_kb * total_imgs_5000) / 1024 / 1024
        print(f'  Proyeccion para 5000 pisos (3 imgs c/u): {proj_gb:.1f} GB')
else:
    print('Carpeta de imagenes NO existe o esta vacia')

# Columnas local_img en CSV
print()
local_img_cols = [c for c in df.columns if 'local' in c]
print(f'Columnas local_img en CSV: {local_img_cols}')
if local_img_cols:
    for col in local_img_cols:
        no_null = df[col].notna().sum()
        print(f'  {col}: {no_null} registros con ruta guardada')

# Calculos de tiempo
print()
print('--- CALCULO DE TIEMPO ---')
props_actuales = len(df)
objetivo = 5000
props_faltantes = objetivo - props_actuales
props_por_tanda = 40
tandas_necesarias = -(-props_faltantes // props_por_tanda)  # ceil division

print(f'Propiedades actuales: {props_actuales}')
print(f'Objetivo: {objetivo}')
print(f'Faltan: {props_faltantes}')
print(f'Tandas necesarias: {tandas_necesarias}')
print()
print(f'  Escenario A (2 tandas/dia): {tandas_necesarias//2} dias = {tandas_necesarias//2//7} semanas {tandas_necesarias//2%7} dias')
print(f'  Escenario B (3 tandas/dia): {tandas_necesarias//3} dias = {tandas_necesarias//3//7} semanas {tandas_necesarias//3%7} dias')
print(f'  Escenario C (5 tandas/dia): {tandas_necesarias//5} dias = {tandas_necesarias//5//7} semanas {tandas_necesarias//5%7} dias')
print(f'  Tiempo por tanda estimado: ~8-12 min')

# Estado de Inteligencia Artificial (VLM)
print()
print('--- ESTADO DE ETIQUETADO CON IA (GEMINI) ---')
csv_etiquetado = Path('data/processed/propiedades_etiquetadas.csv')
if csv_etiquetado.exists():
    df_ai = pd.read_csv(csv_etiquetado)
    total_etiquetadas = df_ai['estado_conservacion'].notna().sum()
    print(f'Propiedades analizadas por la IA: {total_etiquetadas}')
    
    if total_etiquetadas > 0:
        print('Distribucion de clases (Estado):')
        conteo = df_ai['estado_conservacion'].value_counts()
        for clase, cantidad in conteo.items():
            porcentaje = (cantidad / total_etiquetadas) * 100
            print(f'  - {clase}: {cantidad} ({porcentaje:.1f}%)')
else:
    print('Todavia no existe el archivo propiedades_etiquetadas.csv')
