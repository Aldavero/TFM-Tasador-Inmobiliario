import time
import random
import uuid
import re
from playwright.sync_api import sync_playwright
import pandas as pd
import os
from config import SELECTORS, MIN_SLEEP, MAX_SLEEP, TIMEOUT_PAGE, RAW_DIR

def human_delay(min_s=None, max_s=None):
    """Genera una pausa aleatoria para simular comportamiento humano."""
    lo = min_s or MIN_SLEEP
    hi = max_s or MAX_SLEEP
    time.sleep(random.uniform(lo, hi))


# Variables globales para control desde run_tanda.py
existing_urls_global = set()
start_page_global = 1
current_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
last_page_visited = 1

def discover_listing_urls(page, existing_urls, max_pages=50, target_new=40):
    """
    FASE 1: Navega por las paginas de resultados de pisos.com para la provincia de Madrid
    buscando hasta alcanzar el target de URLs NUEVAS. Ahora lo hace por distritos para evitar bloqueos.
    """
    new_urls = []
    global last_page_visited
    
    DISTRITOS = [
        "madrid_capital_centro", "madrid_capital_arganzuela", "madrid_capital_retiro",
        "madrid_capital_salamanca", "madrid_capital_chamartin", "madrid_capital_tetuan",
        "madrid_capital_chamberi", "madrid_capital_fuencarral_el_pardo",
        "madrid_capital_moncloa_aravaca", "madrid_capital_latina", "madrid_capital_carabanchel",
        "madrid_capital_usera", "madrid_capital_puente_de_vallecas", "madrid_capital_moratalaz",
        "madrid_capital_ciudad_lineal", "madrid_capital_hortaleza", "madrid_capital_villaverde",
        "madrid_capital_villa_de_vallecas", "madrid_capital_vicalvaro", "madrid_capital_san_blas",
        "madrid_capital_barajas"
    ]
    
    ALL_PAGES = []
    for d in DISTRITOS:
        ALL_PAGES.append(f"https://www.pisos.com/venta/pisos-{d}/")
        for i in range(2, 60): # 60 paginas por distrito es seguro
            ALL_PAGES.append(f"https://www.pisos.com/venta/pisos-{d}/{i}/")
            
    # Usamos start_page_global como un indice maestro que recorre ALL_PAGES
    idx_inicio = max(0, start_page_global - 1)
    # Por si llegamos al final de todas las paginas posibles
    if idx_inicio >= len(ALL_PAGES):
        idx_inicio = 0 
        
    search_pages = ALL_PAGES[idx_inicio : idx_inicio + max_pages]

    for search_url in search_pages:
        # Actualizamos para que run_tanda.py sepa donde nos quedamos
        last_page_visited = idx_inicio + search_pages.index(search_url) + 1
        if len(new_urls) >= target_new:
            break

        try:
            print(f"  [Descubrimiento] Paginando: {search_url}")
            page.goto(search_url, timeout=TIMEOUT_PAGE, wait_until="domcontentloaded")
            human_delay(2, 4)

            hrefs = page.eval_on_selector_all(
                "a[href]",
                "els => els.map(e => e.href)"
            )

            for href in hrefs:
                is_detail_page = (
                    "pisos.com" in href
                    and ("/comprar/" in href or "/venta/piso-" in href
                         or "/venta/casa-" in href or "/venta/chalet-" in href)
                    and "_" in href.rstrip("/").split("/")[-1]
                    and href.rstrip("/").split("_")[-1].isdigit()
                )
                if is_detail_page and href not in existing_urls and href not in new_urls:
                    new_urls.append(href)

            print(f"  [Descubrimiento] URLs NUEVAS acumuladas: {len(new_urls)} / {target_new}")
            human_delay(1, 3)

        except Exception as e:
            print(f"  [Error Descubrimiento] {search_url}: {e}")

    print(f"\n  Total de anuncios nuevos encontrados: {len(new_urls)}")
    return new_urls[:target_new]


def scrape_single_property(page, url):
    """
    FASE 2: Scraping de la ficha individual de un inmueble.
    Extrae precio, m2, habitaciones, banos, planta, barrio, descripcion e imagenes.
    """
    try:
        page.goto(url, timeout=TIMEOUT_PAGE, wait_until="domcontentloaded")
        human_delay()

        id_inmueble = str(uuid.uuid4())

        # --- Titulo (desde meta og:title, mas fiable que el H1 del DOM) ---
        try:
            title = page.locator("meta[property='og:title']").first.get_attribute("content") or "Sin titulo"
            title = title.strip()
        except Exception:
            title = "Sin titulo"

        # --- Precio ---
        try:
            price = page.locator(SELECTORS["price"]).first.inner_text().strip()
        except Exception:
            price = "0"

        # --- Descripcion ---
        try:
            description = page.locator(SELECTORS["description"]).first.inner_text().strip()
        except Exception:
            description = ""

        # --- Barrio (extraído inteligentemente del título) ---
        try:
            if title and " en " in title and " por " in title:
                barrio = title.split(" en ")[-1].split(" por ")[0].strip()
            elif title and " en " in title:
                barrio = title.split(" en ")[-1].strip()
            else:
                barrio = ""
        except Exception:
            barrio = ""

        # --- Caracteristicas (m2, habitaciones, banos, planta) ---
        m2, habitaciones, banos, planta = "", "", "", ""
        try:
            features_text = page.locator(
                "li:has-text('m'), span:has-text('m')"
            ).all_inner_texts()
            for ft in features_text:
                m2_match = re.search(r'(\d[\d.,]+)\s*m', ft)
                if m2_match:
                    m2 = m2_match.group(1)
                    break
        except Exception:
            pass
        try:
            for selector in ["li:has-text('hab')", "span:has-text('hab')",
                             "li:has-text('dormit')", "span:has-text('dormit')"]:
                texts = page.locator(selector).all_inner_texts()
                for t in texts:
                    match = re.search(r'(\d+)\s*(hab|dormit)', t, re.IGNORECASE)
                    if match:
                        habitaciones = match.group(1)
                        break
                if habitaciones:
                    break
        except Exception:
            pass
        try:
            for selector in ["li:has-text('bano')", "span:has-text('bano')",
                             "li:has-text('\u00f1o')", "span:has-text('\u00f1o')"]:
                texts = page.locator(selector).all_inner_texts()
                for t in texts:
                    match = re.search(r'(\d+)\s*ba', t, re.IGNORECASE)
                    if match:
                        banos = match.group(1)
                        break
                if banos:
                    break
        except Exception:
            pass
        try:
            texts = page.locator("li:has-text('planta'), span:has-text('planta')").all_inner_texts()
            for t in texts:
                match = re.search(r'(\w[\w.]+)\s*planta', t, re.IGNORECASE)
                if match:
                    planta = match.group(1)
                    break
        except Exception:
            pass

        # --- Imagenes (hasta 5, priorizando CDN de pisos imghs) ---
        image_urls = []
        try:
            meta_img = page.locator("meta[property='og:image']").first.get_attribute("content")
            if meta_img and "pisos-logo" not in meta_img:
                image_urls.append(meta_img)
        except Exception:
            pass
        try:
            img_locators = page.locator("img[src*='imghs']").all()
            for img in img_locators:
                src = img.get_attribute("src")
                if src and src not in image_urls and "pisos-logo" not in src:
                    image_urls.append(src)
                if len(image_urls) >= 5:
                    break
        except Exception:
            pass

        return {
            "id_inmueble": id_inmueble,
            "url_origen": url,
            "titulo": title,
            "precio": price,
            "m2": m2,
            "habitaciones": habitaciones,
            "banos": banos,
            "planta": planta,
            "barrio": barrio,
            "descripcion": description,
            "img_url_1": image_urls[0] if len(image_urls) > 0 else None,
            "img_url_2": image_urls[1] if len(image_urls) > 1 else None,
            "img_url_3": image_urls[2] if len(image_urls) > 2 else None,
            "img_url_4": image_urls[3] if len(image_urls) > 3 else None,
            "img_url_5": image_urls[4] if len(image_urls) > 4 else None,
        }

    except Exception as e:
        print(f"  [Error Ficha] {url}: {e}")
        return None


def scrape_properties(max_pages=2, max_properties=10):
    """
    Orquesta el scraping completo:
    1. Descubre URLs automaticamente desde las paginas de resultados de Madrid Capital.
    2. Scrape los datos de cada ficha individual.
    """
    scraped_data = []

    # --- Cargar datos existentes para hacer el scraping incremental ---
    csv_path = RAW_DIR / "propiedades_raw.csv"
    existing_urls = existing_urls_global.copy()
    
    if os.path.exists(csv_path) and not existing_urls:
        df_existing = pd.read_csv(csv_path)
        if 'url_origen' in df_existing.columns:
            existing_urls = set(df_existing['url_origen'].dropna().tolist())
            print(f"[INFO] Se encontraron {len(existing_urls)} propiedades ya extraidas en iteraciones anteriores.")
    else:
        # Si venimos de run_tanda.py, pasamos el df vacio al principio para no cargarlo en memoria cada vez,
        # run_tanda.py se encarga de appendear.
        df_existing = pd.DataFrame()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=current_user_agent,
            viewport={"width": 1920, "height": 1080},
            locale="es-ES"
        )
        page = context.new_page()
        # Stealth: oculta que es un navegador automatizado
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        # --- FASE 1: Descubrimiento automatico de URLs ---
        print("[FASE 1] Descubriendo anuncios en la Provincia de Madrid desde pisos.com...")
        new_urls = discover_listing_urls(page, existing_urls, max_pages=max_pages, target_new=max_properties)
        
        if not new_urls:
            print("No se encontraron anuncios NUEVOS. Puede que ya tengas todos los de estas paginas.")
            browser.close()
            return pd.DataFrame()

        # --- FASE 2: Scraping de cada ficha ---
        print(f"\n[FASE 2] Scraping de {len(new_urls)} fichas individuales NUEVAS...")
        for i, url in enumerate(new_urls):
            print(f"  [{i+1}/{len(new_urls)}] {url}")
            data = scrape_single_property(page, url)
            if data:
                scraped_data.append(data)
            human_delay()

        browser.close()

    df_new = pd.DataFrame(scraped_data)
    
    # Combinar con los existentes si los hay
    if not df_new.empty and os.path.exists(csv_path):
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    elif not df_new.empty:
        df_final = df_new
    else:
        df_final = df_existing if os.path.exists(csv_path) else pd.DataFrame()

    print(f"\n  Scraping completado: {len(df_new)} propiedades NUEVAS extraidas con exito.")
    return df_final


if __name__ == "__main__":
    print("Prueba del scraper automatico (5 propiedades, 1 pagina de resultados)...")
    df = scrape_properties(max_pages=1, max_properties=5)
    print(df[["titulo", "precio", "m2", "habitaciones", "barrio"]].to_string())
