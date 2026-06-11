import sys
from importlib import import_module

scraper = import_module("1_scraper")
data_manager = import_module("2_data_manager")
vlm_extractor = import_module("3_vlm_extractor")

def run_pipeline(max_pages=2, max_properties=20):
    print("="*55)
    print("INICIANDO PIPELINE DE DATOS - TFM V2")
    print(f"  Objetivo: hasta {max_properties} propiedades de Madrid Capital")
    print("="*55)

    # PASO 1: Scraping automatico (descubrimiento + extraccion de fichas)
    print("\n[PASO 1] Ejecutando Web Scraping automatico en Madrid Capital...")
    df_raw = scraper.scrape_properties(max_pages=max_pages, max_properties=max_properties)

    if df_raw.empty:
        print("Error: No se extrajeron propiedades. Revisa la conexion o los selectores.")
        sys.exit(1)

    print(f"Propiedades extraidas con exito: {len(df_raw)}")

    # PASO 2: Descarga de Imagenes y Trazabilidad
    print("\n[PASO 2] Descargando imagenes y creando CSV trazado...")
    df_with_images = data_manager.process_and_download_images(df_raw)

    # PASO 3: Feature Engineering Visual con Gemini
    print("\n[PASO 3] Extraccion de Caracteristicas Visuales con Gemini (VLM)...")
    df_final = vlm_extractor.extract_visual_features()

    if df_final is not None:
        print("="*55)
        print("PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*55)
        cols_show = [c for c in ['titulo','precio','m2','habitaciones','barrio',
                                  'vlm_estado_cocina','vlm_luminosidad','vlm_calidad_materiales']
                     if c in df_final.columns]
        print(df_final[cols_show].to_string())
    else:
        print("\nEl pipeline termino con advertencias en la fase de IA (revisa API KEY).")

if __name__ == "__main__":
    # max_pages: cuantas paginas de resultados de pisos.com paginar
    # max_properties: cuantos pisos en total extraer
    run_pipeline(max_pages=8, max_properties=300)
