import os
import json
import pandas as pd
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME, PROMPT_VISION, RAW_DIR, PROCESSED_DIR, DATA_DIR

def setup_gemini():
    """Configura la API de Google Gemini."""
    if not GEMINI_API_KEY:
        print("ADVERTENCIA: GEMINI_API_KEY no está configurada. El análisis visual fallará.")
        return False
    genai.configure(api_key=GEMINI_API_KEY)
    return True

def analyze_image_with_gemini(image_path):
    """
    Sube temporalmente la imagen a la API de Gemini (o la envía directa) 
    y le pide que analice las variables según el PROMPT_VISION.
    """
    if not os.path.exists(image_path):
        return None
        
    try:
        # Usamos el modelo generativo
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        # Subir el archivo local usando la API de archivos de GenAI
        print(f"Analizando: {image_path}")
        sample_file = genai.upload_file(path=image_path)
        
        # Generar contenido
        response = model.generate_content([PROMPT_VISION, sample_file])
        
        # Borrar el archivo en la nube de Google para mantener limpieza
        genai.delete_file(sample_file.name)
        
        # Extraer el texto de la respuesta (que debería ser un JSON)
        result_text = response.text.strip()
        
        # Limpiar backticks de markdown si el modelo los añade
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()
            
        return json.loads(result_text)
        
    except Exception as e:
        print(f"Error procesando imagen {image_path} con Gemini: {e}")
        return None

def extract_visual_features():
    """
    Lee el CSV con los datos y las rutas locales de las imágenes,
    invoca a Gemini para la imagen principal y añade las nuevas variables predictivas.
    """
    csv_path = RAW_DIR / "propiedades_raw.csv"
    if not os.path.exists(csv_path):
        print("El archivo CSV crudo no existe. Ejecuta el scraper primero.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Intentar cargar CSV enriquecido previo para no repetir la llamada a la IA
    out_csv = PROCESSED_DIR / "propiedades_enrich_vlm.csv"
    if os.path.exists(out_csv):
        df_existing_enrich = pd.read_csv(out_csv)
        # Sincronizamos con el raw actual (nuevos pisos)
        # Hacemos un merge left para mantener todos los crudos y ver cuáles ya tienen VLM
        df = df.merge(
            df_existing_enrich[['id_inmueble', 'vlm_estado_cocina', 'vlm_luminosidad', 'vlm_calidad_materiales', 'vlm_terraza']], 
            on='id_inmueble', 
            how='left'
        )
    else:
        df['vlm_estado_cocina'] = None
        df['vlm_luminosidad'] = None
        df['vlm_calidad_materiales'] = None
        df['vlm_terraza'] = None
    
    if not setup_gemini():
        return
        
    for index, row in df.iterrows():
        # Si ya tiene valor, nos lo saltamos para no gastar API ni tiempo
        if pd.notna(row.get('vlm_estado_cocina')):
            continue

        img_rel_path = row.get('local_img_1')
        
        if pd.notna(img_rel_path):
            img_abs_path = DATA_DIR / img_rel_path
            
            features = analyze_image_with_gemini(str(img_abs_path))
            
            if features:
                df.at[index, 'vlm_estado_cocina'] = features.get('estado_cocina', 'no_aplicable')
                df.at[index, 'vlm_luminosidad'] = features.get('luminosidad_estimada', 'no_determinable')
                df.at[index, 'vlm_calidad_materiales'] = features.get('calidad_materiales', 'no_determinable')
                df.at[index, 'vlm_terraza'] = features.get('tiene_terraza_visible', 0)
            else:
                df.at[index, 'vlm_estado_cocina'] = 'no_aplicable'
                df.at[index, 'vlm_luminosidad'] = 'no_determinable'
                df.at[index, 'vlm_calidad_materiales'] = 'no_determinable'
                df.at[index, 'vlm_terraza'] = 0
        else:
            df.at[index, 'vlm_estado_cocina'] = 'no_aplicable'
            df.at[index, 'vlm_luminosidad'] = 'no_determinable'
            df.at[index, 'vlm_calidad_materiales'] = 'no_determinable'
            df.at[index, 'vlm_terraza'] = 0
            
    # Guardar en processed
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n¡Extracción VLM Completada! Datos enriquecidos guardados/actualizados en: {out_csv}")
    
    return df

if __name__ == "__main__":
    pass
