import os
import requests
import pandas as pd
from urllib.parse import urlparse
from config import RAW_DIR, IMAGES_DIR
from importlib import import_module
scraper = import_module("1_scraper")
human_delay = scraper.human_delay

def download_image(url, save_path):
    """
    Descarga una imagen desde una URL y la guarda en la ruta especificada.
    Maneja timeouts y errores HTTP.
    """
    if not url:
        return False
        
    try:
        # Si el archivo ya existe y tiene un peso mínimo, asumimos que está bien descargado
        if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
            return True

        # Añadir un User-Agent básico para peticiones requests
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            print(f"Error HTTP {response.status_code} al descargar: {url}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Excepción al descargar la imagen {url}: {e}")
        return False

def process_and_download_images(df):
    """
    Recibe el DataFrame crudo, descarga las imágenes de cada propiedad
    y actualiza el DataFrame con las rutas locales garantizando la trazabilidad.
    """
    local_image_paths_1 = []
    local_image_paths_2 = []
    local_image_paths_3 = []
    local_image_paths_4 = []
    local_image_paths_5 = []
    
    # Asegurar que el directorio existe
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    print(f"Iniciando descarga de imágenes para {len(df)} propiedades...")
    
    for index, row in df.iterrows():
        id_inmueble = row['id_inmueble']
        
        # Procesar Imagen 1
        path_1 = None
        if pd.notna(row.get('img_url_1')):
            ext = os.path.splitext(urlparse(row['img_url_1']).path)[1]
            if not ext: ext = ".jpg" # Por defecto
            
            save_path = IMAGES_DIR / f"{id_inmueble}_1{ext}"
            if download_image(row['img_url_1'], save_path):
                path_1 = str(save_path.relative_to(RAW_DIR.parent))
            human_delay() # Pausa mínima entre descargas
        local_image_paths_1.append(path_1)
        
        # Procesar Imagen 2
        path_2 = None
        if pd.notna(row.get('img_url_2')):
            ext = os.path.splitext(urlparse(row['img_url_2']).path)[1]
            if not ext: ext = ".jpg"
            
            save_path = IMAGES_DIR / f"{id_inmueble}_2{ext}"
            if download_image(row['img_url_2'], save_path):
                path_2 = str(save_path.relative_to(RAW_DIR.parent))
            human_delay()
        local_image_paths_2.append(path_2)
        
        # Procesar Imagen 3
        path_3 = None
        if pd.notna(row.get('img_url_3')):
            ext = os.path.splitext(urlparse(row['img_url_3']).path)[1]
            if not ext: ext = ".jpg"
            
            save_path = IMAGES_DIR / f"{id_inmueble}_3{ext}"
            if download_image(row['img_url_3'], save_path):
                path_3 = str(save_path.relative_to(RAW_DIR.parent))
            human_delay()
        local_image_paths_3.append(path_3)
        
        # Procesar Imagen 4
        path_4 = None
        if pd.notna(row.get('img_url_4')):
            ext = os.path.splitext(urlparse(row['img_url_4']).path)[1]
            if not ext: ext = ".jpg"
            
            save_path = IMAGES_DIR / f"{id_inmueble}_4{ext}"
            if download_image(row['img_url_4'], save_path):
                path_4 = str(save_path.relative_to(RAW_DIR.parent))
            human_delay()
        local_image_paths_4.append(path_4)

        # Procesar Imagen 5
        path_5 = None
        if pd.notna(row.get('img_url_5')):
            ext = os.path.splitext(urlparse(row['img_url_5']).path)[1]
            if not ext: ext = ".jpg"
            
            save_path = IMAGES_DIR / f"{id_inmueble}_5{ext}"
            if download_image(row['img_url_5'], save_path):
                path_5 = str(save_path.relative_to(RAW_DIR.parent))
            human_delay()
        local_image_paths_5.append(path_5)
        
    # Añadir las rutas locales al DataFrame
    df['local_img_1'] = local_image_paths_1
    df['local_img_2'] = local_image_paths_2
    df['local_img_3'] = local_image_paths_3
    df['local_img_4'] = local_image_paths_4
    df['local_img_5'] = local_image_paths_5

    # Guardar en CSV: SIEMPRE añadir al existente (append), nunca sobreescribir
    csv_path = RAW_DIR / "propiedades_raw.csv"
    if csv_path.exists():
        df_existente = pd.read_csv(csv_path)
        df_combinado = pd.concat([df_existente, df], ignore_index=True)
        # Eliminar duplicados por url_origen (por si acaso)
        df_combinado = df_combinado.drop_duplicates(subset=['url_origen'], keep='first')
        df_combinado.to_csv(csv_path, index=False)
        print(f"CSV actualizado: {len(df_combinado)} propiedades en total")
        print(f"Guardado en: {csv_path}")
        return df_combinado
    else:
        df.to_csv(csv_path, index=False)
        print(f"CSV creado con {len(df)} propiedades en: {csv_path}")
        return df

if __name__ == "__main__":
    # Prueba rapida
    pass
