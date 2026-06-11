import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Rutas de base
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = RAW_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"

# URL objetivo para pisos (estructura genérica basada en portales populares)
BASE_URL = "https://www.pisos.com/venta/pisos-madrid/"
# Si es necesario hacer scraping de una propiedad específica (para pruebas)
TEST_URLS = [
    "https://www.pisos.com/comprar/piso-ibiza28009-64199921765_528715/",
    "https://www.pisos.com/comprar/piso-guindalera28028-63366549974_440000/"
]

# Selectores CSS (Plantilla genérica a ajustar según el portal)
SELECTORS = {
    "list_items": "div.ad-preview",                     # Contenedor de cada anuncio en la lista
    "link_detail": "a.ad-preview__title",               # Link a la ficha de la propiedad
    "title": "h1.title",                                # Título del inmueble
    "price": "div.price",                               # Precio
    "features": "div.basicdata-item",                   # Habitaciones, m2, baños (generalmente en una lista)
    "description": "div.description",                   # Texto descriptivo
    "images": "img.photo",                              # Selector para extraer los enlaces de las imágenes (src)
    "next_page": "a.pagination__next"                   # Botón de siguiente página
}

# Configuración anti-bot
MIN_SLEEP = 2.0
MAX_SLEEP = 5.0
TIMEOUT_PAGE = 60000 # milisegundos

# Configuración de IA (Gemini)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Usamos 'gemini-flash-latest' que es el alias universal de Google para su modelo gratuito actual sin las restricciones de limit: 0
GEMINI_MODEL_NAME = "gemini-flash-latest"

PROMPT_VISION = """
Eres un tasador inmobiliario experto en Madrid. Analiza la imagen proporcionada de un piso y extrae las siguientes características en formato JSON estricto.
Solo puedes usar los valores indicados. No devuelvas ningún texto extra, solo el objeto JSON válido.

1. "estado_cocina": ["reformada", "antigua", "no_visible", "no_aplicable"]
2. "luminosidad_estimada": ["alta", "media", "baja", "no_determinable"]
3. "calidad_materiales": ["premium", "media", "basica", "no_determinable"]
4. "tiene_terraza_visible": [1, 0]

Ejemplo de respuesta:
{
  "estado_cocina": "reformada",
  "luminosidad_estimada": "alta",
  "calidad_materiales": "media",
  "tiene_terraza_visible": 0
}
"""
