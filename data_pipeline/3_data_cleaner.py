import pandas as pd
import numpy as np
import re
import os
from pathlib import Path

# Configuracion de rutas relativas
BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "data" / "raw" / "propiedades_raw.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "propiedades_limpias.csv"

def clean_price(price_str):
    if pd.isna(price_str) or str(price_str).lower() in ['consultar', 'precio a consultar']:
        return np.nan
    # Quedarnos con la primera linea (el precio actual, ignorando el precio anterior tachado)
    precio_principal = str(price_str).split('\n')[0]
    # Extraer solo los numeros
    nums = re.sub(r'[^0-9]', '', precio_principal)
    return int(nums) if nums else np.nan

def clean_planta(planta_str):
    if pd.isna(planta_str):
        return np.nan
    planta_str = str(planta_str).lower().strip()
    if 'bajo' in planta_str or 'bj' in planta_str:
        return 0.0
    if 'entresuelo' in planta_str:
        return 0.5
    if 'sotano' in planta_str or 'sótano' in planta_str:
        return -1.0
    
    # Buscar primer numero
    match = re.search(r'\d+', planta_str)
    if match:
        return float(match.group())
    return np.nan

def extract_tipo(titulo):
    if pd.isna(titulo): 
        return "Desconocido"
        
    titulo_str = str(titulo).strip()
    
    # Corregir bugs de encoding detectados en el HTML
    if titulo_str.startswith('tico') or titulo_str.startswith('Ãtico') or titulo_str.startswith('Atico'):
        return 'Ático'
    if titulo_str.startswith('Dplex') or titulo_str.startswith('DÃºplex') or titulo_str.startswith('Duplex'):
        return 'Dúplex'
        
    # Extraer la primera palabra antes de "en venta"
    match = re.match(r"^([A-Za-zÁÉÍÓÚáéíóúÑñ]+)\s+en\s+venta", titulo_str, re.IGNORECASE)
    if match:
        return match.group(1).title()
        
    return "Piso"

def clean_text(text):
    if pd.isna(text): 
        return ""
    # Quitar saltos de linea y espacios multiples
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()

def check_feature(text, keywords):
    if pd.isna(text):
        return 0
    text_lower = str(text).lower()
    for kw in keywords:
        if re.search(r'\b' + kw + r'\b', text_lower):
            return 1
    return 0

def run_cleaning_pipeline():
    print("="*60)
    print("INICIANDO PROCESO DE LIMPIEZA DE DATOS (ETL)")
    print("="*60)
    
    if not RAW_CSV.exists():
        print(f"ERROR: No se encuentra el archivo crudo en {RAW_CSV}")
        return
        
    print("Cargando datos crudos...")
    df = pd.read_csv(RAW_CSV)
    print(f"Propiedades a limpiar: {len(df)}")
    
    print("\nAplicando reglas de limpieza...")
    
    # 1. Limpieza de Precio
    df['precio_limpio'] = df['precio'].apply(clean_price)
    
    # 2. Limpieza de Planta
    df['planta_limpia'] = df['planta'].apply(clean_planta)
    
    # 3. Limpieza de Metros, Habitaciones y Banos
    # Como el scraper ya los extrae como numeros a veces, forzamos a float por seguridad
    df['m2'] = pd.to_numeric(df['m2'], errors="coerce")
    df['habitaciones'] = pd.to_numeric(df['habitaciones'], errors="coerce")
    df['banos'] = pd.to_numeric(df['banos'], errors="coerce")
    
    # 4. Extraccion del Tipo de Inmueble
    df['tipo_inmueble'] = df['titulo'].apply(extract_tipo)
    
    # 4.5. Imputacion inteligente de Planta para Chalets y Casas
    mask_chalet = df['tipo_inmueble'].isin(['Chalet', 'Casa', 'Chalet Adosado', 'Chalet Pareado', 'Casa Adosada'])
    df.loc[mask_chalet & df['planta_limpia'].isna(), 'planta_limpia'] = 0.0
    
    # 5. Feature Engineering: Precio por Metro Cuadrado
    df['precio_m2'] = (df['precio_limpio'] / df['m2']).round(2)
    
    # 6. Limpieza Textual (Descripcion y Barrio)
    df['descripcion_limpia'] = df['descripcion'].apply(clean_text)
    df['barrio_limpio'] = df['barrio'].apply(clean_text).str.title()
    
    # 7. Feature Engineering: Extraccion de Amenidades desde Descripcion
    df['tiene_ascensor'] = df['descripcion_limpia'].apply(lambda x: check_feature(x, ['ascensor', 'elevador']))
    df['tiene_garaje'] = df['descripcion_limpia'].apply(lambda x: check_feature(x, ['garaje', 'parking', 'aparcamiento', 'cochera', 'plaza de garaje']))
    df['tiene_terraza'] = df['descripcion_limpia'].apply(lambda x: check_feature(x, ['terraza', 'balcon', 'balcón', 'azotea', 'patio']))
    df['tiene_piscina'] = df['descripcion_limpia'].apply(lambda x: check_feature(x, ['piscina', 'alberca']))
    df['tiene_trastero'] = df['descripcion_limpia'].apply(lambda x: check_feature(x, ['trastero', 'guardamuebles', 'buhardilla']))
    
    # 8. Filtros de Calidad Estrictos
    df = df.dropna(subset=['precio_limpio', 'm2'])
    
    # Eliminar valores nulos en habitaciones y baños por seguridad
    df = df.dropna(subset=['habitaciones', 'banos'])
    
    # Eliminar outliers (falsos alquileres o m2 imposibles)
    df = df[(df['precio_limpio'] >= 50000) & (df['m2'] >= 15)]
    print(f"Propiedades despues de aplicar filtros estrictos y borrar nulos: {len(df)}")
    
    # 9. Seleccionar y ordenar las columnas que realmente queremos en el CSV final
    # Descartamos las columnas crudas de texto sucio para hacer el archivo mas ligero
    columnas_finales = [
        'id_inmueble', 'url_origen', 
        'tipo_inmueble', 'barrio_limpio', 'planta_limpia', 
        'precio_limpio', 'm2', 'precio_m2', 
        'habitaciones', 'banos',
        'tiene_ascensor', 'tiene_garaje', 'tiene_terraza', 'tiene_piscina', 'tiene_trastero',
        'descripcion_limpia',
        'img_url_1', 'img_url_2', 'img_url_3', 'img_url_4', 'img_url_5',
        'local_img_1', 'local_img_2', 'local_img_3', 'local_img_4', 'local_img_5'
    ]
    
    # Quedarnos solo con las columnas que existen en nuestro subset definido
    columnas_disponibles = [col for col in columnas_finales if col in df.columns]
    df_clean = df[columnas_disponibles]
    
    # Guardar a disco
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_clean.to_csv(CLEAN_CSV, index=False)
    
    print("\n" + "="*60)
    print("LIMPIEZA COMPLETADA CON EXITO")
    print(f"Archivo guardado en: {CLEAN_CSV}")
    print("="*60)
    return df_clean

if __name__ == "__main__":
    import os
    run_cleaning_pipeline()
