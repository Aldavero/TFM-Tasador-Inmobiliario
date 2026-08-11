import os
import json
import time
import pandas as pd
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# Configuracion de rutas
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "propiedades_limpias.csv"
ETIQUETADAS_CSV = PROCESSED_DIR / "propiedades_etiquetadas.csv"
CHECKPOINT_VLM = PROCESSED_DIR / "vlm_checkpoint.json"

# Cargar variables de entorno (API Key)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

PROMPT_TASADOR = """
Actúa como un tasador inmobiliario experto en Madrid.
A continuación te proporciono hasta 5 fotografías de una misma propiedad en venta, y adicionalmente un texto con la descripción comercial del anuncio.

Analiza la información y evalúa EXCLUSIVAMENTE tres variables. 
Debes responder ÚNICAMENTE con un objeto JSON válido con las siguientes claves y restricciones de valores:

1. "estado_conservacion": Solo puede ser uno de estos valores: ["A reformar", "Buen estado", "Lujo"].
   - "A reformar": Se ven muebles muy antiguos, baños/cocinas de los años 80/90, paredes desconchadas o vacías en mal estado.
   - "Buen estado": Piso normal, habitable, decoración estándar o moderna sencilla.
   - "Lujo": Reformado por arquitecto, diseño de interiores premium, domótica visible.

2. "calidad_materiales": Solo puede ser uno de estos valores: ["Básica", "Premium"].
   - "Básica": Suelos de sintasol/terrazo antiguo, cocinas con electrodomésticos blancos, ventanas de aluminio viejo.
   - "Premium": Suelos de madera noble o porcelánico de gran formato, cocinas con isla o integradas modernas.

3. "barrio_deducido": Lee la descripción adjunta y deduce a qué barrio o distrito oficial de Madrid Capital pertenece la propiedad. Devuelve solo el nombre del barrio (ej. "Mirasierra", "Salamanca", "Chamberí"). Si la descripción no aporta ninguna pista sobre la ubicación exacta, devuelve exactamente la palabra "Desconocido".
   
Si las imágenes no muestran el interior con claridad, asigna los valores que consideres más probables o "Buen estado" y "Básica" por defecto.

Respuesta esperada (ejemplo estricto sin markdown):
{
  "estado_conservacion": "Buen estado",
  "calidad_materiales": "Básica",
  "barrio_deducido": "Mirasierra"
}
"""

def init_gemini():
    if not API_KEY:
        print("ERROR: No se ha encontrado GEMINI_API_KEY en el archivo .env")
        return None
    genai.configure(api_key=API_KEY)
    # Usamos gemini-flash-latest que es el alias universal con cuota activa
    return genai.GenerativeModel('gemini-flash-latest')

def get_images(row):
    paths = []
    for col in ['local_img_1', 'local_img_2', 'local_img_3', 'local_img_4', 'local_img_5']:
        img_path = str(row.get(col, ''))
        if pd.notna(img_path) and img_path and img_path != 'None':
            # img_path es algo como "raw/images/id_1.jpg"
            full_path = BASE_DIR / "data" / img_path
            if full_path.exists():
                paths.append(full_path)
    return paths

def analyze_property(model, img_paths, descripcion_texto=""):
    imgs = []
    for path in img_paths:
        try:
            imgs.append(Image.open(path))
        except Exception:
            pass
            
    if not imgs and not descripcion_texto:
        return {'estado_conservacion': 'Desconocido', 'calidad_materiales': 'Desconocido', 'barrio_deducido': 'Desconocido'}

    try:
        texto_input = f"Descripción del anuncio:\n{descripcion_texto}"
        response = model.generate_content([PROMPT_TASADOR, texto_input] + imgs)
        
        try:
            # Intentar obtener el texto. A veces Gemini lo bloquea por seguridad y falla al acceder a .text
            txt = response.text
        except Exception as e_text:
            print(f"\n[Error] Gemini bloqueó la respuesta (posible filtro de seguridad). Respuesta: {response}")
            return {'estado_conservacion': 'Error_Seguridad', 'calidad_materiales': 'Error_Seguridad', 'barrio_deducido': 'Error_Seguridad'}
            
        try:
            txt_clean = txt.replace('```json', '').replace('```', '').strip()
            # Extraer solo lo que esté entre corchetes por si añade texto basura
            if '{' in txt_clean and '}' in txt_clean:
                txt_clean = txt_clean[txt_clean.find('{'):txt_clean.rfind('}')+1]
            return json.loads(txt_clean)
        except Exception as parse_error:
            print(f"\n[Error JSON] No se pudo parsear. Respuesta original de Gemini:\n{txt}")
            return {'estado_conservacion': 'Error_JSON', 'calidad_materiales': 'Error_JSON', 'barrio_deducido': 'Error_JSON'}
            
    except Exception as e:
        error_msg = str(e)
        print(f"\n[Error API] Fallo en la comunicación con Gemini: {error_msg}")
        if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
            return {'estado_conservacion': 'RATE_LIMIT', 'calidad_materiales': 'RATE_LIMIT', 'barrio_deducido': 'RATE_LIMIT'}
        return {'estado_conservacion': 'Error_API', 'calidad_materiales': 'Error_API', 'barrio_deducido': 'Error_API'}

def run_vlm_pipeline(limit=None):
    print("="*60)
    print("INICIANDO EXTRACCIÓN MULTIMODAL CON GEMINI (VLM)")
    print("="*60)
    
    if not CLEAN_CSV.exists():
        print(f"ERROR: No existe {CLEAN_CSV}. Ejecuta 3_data_cleaner.py primero.")
        return
        
    df = pd.read_csv(CLEAN_CSV)
    model = init_gemini()
    if not model:
        return
        
    # Crear columnas si no existen
    if 'estado_conservacion' not in df.columns:
        df['estado_conservacion'] = None
    if 'calidad_materiales' not in df.columns:
        df['calidad_materiales'] = None
    if 'barrio_deducido' not in df.columns:
        df['barrio_deducido'] = None

    # Recuperar etiquetas previas si existen
    if ETIQUETADAS_CSV.exists():
        df_old = pd.read_csv(ETIQUETADAS_CSV)
        if 'estado_conservacion' in df_old.columns and 'id_inmueble' in df_old.columns:
            estado_map = df_old.dropna(subset=['estado_conservacion']).set_index('id_inmueble')['estado_conservacion'].to_dict()
            calidad_map = df_old.dropna(subset=['calidad_materiales']).set_index('id_inmueble')['calidad_materiales'].to_dict()
            
            df['estado_conservacion'] = df['id_inmueble'].map(estado_map).combine_first(df['estado_conservacion'])
            df['calidad_materiales'] = df['id_inmueble'].map(calidad_map).combine_first(df['calidad_materiales'])
            
            if 'barrio_deducido' in df_old.columns:
                barrio_map = df_old.dropna(subset=['barrio_deducido']).set_index('id_inmueble')['barrio_deducido'].to_dict()
                df['barrio_deducido'] = df['id_inmueble'].map(barrio_map).combine_first(df['barrio_deducido'])

    # Cargar checkpoint para saber donde nos quedamos (por si falla a medias)
    procesados = set()
    if CHECKPOINT_VLM.exists():
        with open(CHECKPOINT_VLM, 'r') as f:
            procesados = set(json.load(f))
            
    # Filtrar las filas que necesitan procesamiento
    filas_a_procesar = df[
        (df['estado_conservacion'].isna() | df['barrio_deducido'].isna()) & 
        (~df['id_inmueble'].astype(str).isin(procesados))
    ]
    
    if limit is not None:
        filas_a_procesar = filas_a_procesar.head(limit)
    
    total = len(filas_a_procesar)
    print(f"Propiedades pendientes de analizar por IA: {total}")
    
    count = 0
    for idx, row in filas_a_procesar.iterrows():
        id_inm = str(row['id_inmueble'])
        img_paths = get_images(row)
        desc = str(row.get('descripcion_limpia', ''))
        
        print(f"[{count+1}/{total}] Analizando ID: {id_inm} ({len(img_paths)} imgs, con descripcion)... ", end="")
        
        result = analyze_property(model, img_paths, desc)
        
        # Manejo especial si la API nos bloquea por limites gratuitos
        if result.get('estado_conservacion') == 'RATE_LIMIT':
            print("\n[!] Limite de cuota API alcanzado. Esperando 60 segundos...")
            time.sleep(60)
            # Reintentar la misma imagen una vez
            result = analyze_property(model, img_paths, desc)
            if result.get('estado_conservacion') == 'RATE_LIMIT':
                print("--- Sigues bloqueado, guardando progreso y parando. Ejecuta mas tarde. ---")
                break
        
        df.at[idx, 'estado_conservacion'] = result.get('estado_conservacion', 'Error')
        df.at[idx, 'calidad_materiales'] = result.get('calidad_materiales', 'Error')
        df.at[idx, 'barrio_deducido'] = result.get('barrio_deducido', 'Desconocido')
        
        print(f"{result.get('estado_conservacion')} | {result.get('barrio_deducido')}")
        
        # Solo guardar en checkpoint si no fue un error grave
        if result.get('estado_conservacion') not in ['Error', 'RATE_LIMIT']:
            procesados.add(id_inm)
            with open(CHECKPOINT_VLM, 'w') as f:
                json.dump(list(procesados), f)
            
        count += 1
        
        # Esperar 5 segundos para respetar el límite gratuito de 15 RPM (Peticiones por minuto) de Gemini
        time.sleep(5)
        
        # Guardado intermedio (sin dropear nada todavia para no liar el mapping de indices)
        if count % 10 == 0:
            df.to_csv(ETIQUETADAS_CSV, index=False)
            print("--- Guardado intermedio ---")

    # Guardado final
    with open(CHECKPOINT_VLM, 'w') as f:
        json.dump(list(procesados), f)
        
    # --- LIMPIEZA DE FILAS BASURA (Desconocidas) ---
    print("\nAplicando filtro de Zonas Desconocidas...")
    df_clean = df[~df['barrio_deducido'].isin(['Desconocido', 'Error_Seguridad', 'Error_API', 'Error_JSON', 'RATE_LIMIT', 'Error', None])]
    eliminadas = len(df) - len(df_clean)
    
    df_clean.to_csv(ETIQUETADAS_CSV, index=False)
    print("\n" + "="*60)
    print(f"PROCESO VLM COMPLETADO. Guardado en: {ETIQUETADAS_CSV}")
    print(f"Propiedades ELIMINADAS por no poder deducir su barrio: {eliminadas}")
    print("="*60)

if __name__ == "__main__":
    # Límite de 1450 para no superar la cuota gratuita diaria de 1500 peticiones de Gemini
    run_vlm_pipeline(limit=1450)
