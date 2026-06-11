import os
import shutil
import pandas as pd
import torch
from pathlib import Path
from PIL import Image

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("Error: No se encontró la librería 'transformers'. Instálala con 'pip install transformers'")
    exit(1)

# Rutas
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "data" / "raw" / "images"
DISCARD_DIR = BASE_DIR / "data" / "raw" / "discarded_images"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CSV_PATH = PROCESSED_DIR / "propiedades_limpias.csv"

# Categorias Zero-Shot para CLIP
# NOTA: CLIP fue entrenado en inglés, por lo que las etiquetas en inglés tienen muchisima mas precision
LABELS = [
    "a photo of the interior of a house, living room, kitchen, bedroom, bathroom, terrace, balcony, or private backyard garden", # Indice 0 (VALIDO)
    "a photo of the exterior of a building, a public street, a public park, or nature",           # Indice 1 (INVALIDO)
    "a 2D or 3D architectural floor plan or map",                                   # Indice 2 (INVALIDO)
    "a commercial real estate agency logo, brand name, or pure text",               # Indice 3 (INVALIDO)
    "a photo of a community swimming pool, gym, or shared common area"              # Indice 4 (INVALIDO)
]

def init_clip():
    print("Cargando modelo CLIP (puede tardar la primera vez)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo de Inferencia: {device}")
    
    # Cargamos el modelo estandar de OpenAI
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

def filter_images(model, processor, device, test_mode=True):
    print("="*60)
    print("INICIANDO PURGA ZERO-SHOT CON CLIP")
    print("="*60)
    
    os.makedirs(DISCARD_DIR, exist_ok=True)
    
    all_images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = len(all_images)
    print(f"Se encontraron {total} imagenes en total.")
    
    # En modo test, solo procesamos unas cuantas al azar
    if test_mode:
        import random
        all_images = random.sample(all_images, min(20, total))
        total = len(all_images)
        print(f"MODO TEST ACTIVADO: Solo se analizarán {total} imagenes al azar.")
        
    stats = {"interior": 0, "exterior": 0, "plano": 0, "logo": 0, "comunes": 0, "errores": 0}
    
    for i, img_name in enumerate(all_images):
        img_path = IMAGES_DIR / img_name
        try:
            image = Image.open(img_path).convert("RGB")
            
            inputs = processor(text=LABELS, images=image, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]
            
            best_idx = probs.argmax()
            confidence = probs[best_idx] * 100
            
            if best_idx == 0:
                stats["interior"] += 1
                # print(f"[{i+1}/{total}] {img_name} -> INTERIOR ({confidence:.1f}%)")
            else:
                if best_idx == 1: stats["exterior"] += 1
                elif best_idx == 2: stats["plano"] += 1
                elif best_idx == 3: stats["logo"] += 1
                elif best_idx == 4: stats["comunes"] += 1
                
                print(f"[{i+1}/{total}] BASURA DETECTADA: {img_name} -> {LABELS[best_idx]} ({confidence:.1f}%)")
                
                # Mover la imagen si NO es modo test
                if not test_mode:
                    shutil.move(str(img_path), str(DISCARD_DIR / img_name))
                    
        except Exception as e:
            print(f"Error procesando {img_name}: {e}")
            stats["errores"] += 1

    print("\n" + "="*60)
    print(f"RESUMEN DE PURGA (Total Analizadas: {total})")
    print(f"[OK] Interiores válidos: {stats['interior']}")
    print(f"[BASURA] Exteriores descartados: {stats['exterior']}")
    print(f"[BASURA] Planos/Mapas descartados: {stats['plano']}")
    print(f"[BASURA] Logos descartados: {stats['logo']}")
    print(f"[BASURA] Zonas Comunes descartadas: {stats['comunes']}")
    print(f"[AVISO] Errores de lectura: {stats['errores']}")
    print("="*60)

    # Solo actualizar CSV si estamos en modo real
    if not test_mode and CSV_PATH.exists():
        print("Actualizando dataset tabular para borrar referencias a fotos descartadas...")
        df = pd.read_csv(CSV_PATH)
        for col in ['local_img_1', 'local_img_2', 'local_img_3', 'local_img_4', 'local_img_5']:
            if col in df.columns:
                # Comprobar para cada celda si el archivo existe en la carpeta RAW. Si no existe, ponerlo a nulo
                df[col] = df[col].apply(lambda x: x if (pd.notna(x) and (BASE_DIR / "data" / str(x)).exists()) else None)
        df.to_csv(CSV_PATH, index=False)
        print(f"Dataset {CSV_PATH.name} actualizado y guardado.")

def run_image_filtering():
    print("\n--- INICIANDO MODULO 5: PURGA VISUAL CLIP ---")
    model, processor, device = init_clip()
    filter_images(model, processor, device, test_mode=False)

if __name__ == "__main__":
    # Cambia test_mode a False cuando quieras hacer la purga real
    TEST_MODE = False
    model, processor, device = init_clip()
    filter_images(model, processor, device, test_mode=TEST_MODE)
