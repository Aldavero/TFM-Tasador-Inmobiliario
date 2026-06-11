import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_CSV = BASE_DIR / "data_pipeline" / "data" / "processed" / "propiedades_etiquetadas.csv"
MODEL_PATH = BASE_DIR / "app" / "modelo_madrid_global.joblib"
ENCODING_PATH = BASE_DIR / "app" / "barrios_encoding.json"
ESTADO_ENCODING_PATH = BASE_DIR / "app" / "estado_encoding.json"
CALIDAD_ENCODING_PATH = BASE_DIR / "app" / "calidad_encoding.json"

MIN_MUESTRAS_BARRIO = 15

def main():
    print("="*60)
    print("INICIANDO ENTRENAMIENTO MODELO MONOLÍTICO (TARGET ENCODING)")
    print("="*60)
    
    if not CLEAN_CSV.exists():
        print(f"ERROR: Dataset no encontrado en {CLEAN_CSV}")
        return

    df = pd.read_csv(CLEAN_CSV)
    print(f"Total propiedades originales: {len(df)}")
    
    # 1. Limpieza extra
    df = df.dropna(subset=['barrio_limpio', 'precio_limpio', 'm2', 'estado_conservacion', 'calidad_materiales'])
    df['precio_m2'] = df['precio_limpio'] / df['m2']
    
    # 2. Análisis de frecuencias para detectar barrios escasos
    conteos_barrio = df['barrio_limpio'].value_counts()
    barrios_frecuentes = conteos_barrio[conteos_barrio >= MIN_MUESTRAS_BARRIO].index.tolist()
    
    # Asignar "Otros" a los barrios con pocas muestras
    df['barrio_agrupado'] = df['barrio_limpio'].apply(lambda x: x if x in barrios_frecuentes else "Otros")
    print(f"Barrios únicos detectados (>= {MIN_MUESTRAS_BARRIO} casas): {len(barrios_frecuentes)}")
    print(f"Casas agrupadas en 'Otros' por falta de datos: {len(df[df['barrio_agrupado'] == 'Otros'])}")

    # 3. Target Encoding Geográfico
    stats_df = df.groupby('barrio_agrupado').agg(
        precio_m2_medio=('precio_m2', 'mean'),
        superficie_media=('m2', 'mean'),
        habs_media=('habitaciones', 'mean')
    ).reset_index()
    
    barrios_encoding = {}
    for _, row in stats_df.iterrows():
        barrios_encoding[row['barrio_agrupado']] = {
            "precio_m2_medio": float(row['precio_m2_medio']),
            "superficie_media": float(row['superficie_media']),
            "habs_media": float(row['habs_media'])
        }
        
    os.makedirs(ENCODING_PATH.parent, exist_ok=True)
    with open(ENCODING_PATH, 'w', encoding='utf-8') as f:
        json.dump(barrios_encoding, f, indent=4, ensure_ascii=False)
    print(f"Diccionario dinámico de barrios guardado en: {ENCODING_PATH}")
    
    # 4. Feature Engineering Dinámico
    def apply_encoding(row, key):
        return barrios_encoding[row['barrio_agrupado']][key]

    df['target_encoding_m2'] = df.apply(lambda r: apply_encoding(r, 'precio_m2_medio'), axis=1)
    df['ratio_metros_zona'] = df['m2'] / df.apply(lambda r: apply_encoding(r, 'superficie_media'), axis=1)
    df['ratio_hab_zona'] = df['habitaciones'] / df.apply(lambda r: apply_encoding(r, 'habs_media'), axis=1)
    
    # Manejar posibles NaNs si alguna casa agrupada no tenia habitaciones informadas
    df['ratio_hab_zona'] = df['ratio_hab_zona'].fillna(1.0)
    
    # 3.5 Target Encoding Multimodal (Estado y Calidad) — Ratio relativo al precio medio global
    # En vez de usar el precio €/m² absoluto (que está contaminado por el barrio de cada casa),
    # calculamos cuánto multiplica/divide cada estado respecto a la media global.
    # Así: Lujo ~ 1.8, Buen estado ~ 1.0, A reformar ~ 0.85 (siempre ordenado lógicamente).
    precio_m2_global = df['precio_m2'].mean()

    estado_stats = df.groupby('estado_conservacion')['precio_m2'].mean()
    estado_encoding = {estado: float(precio / precio_m2_global) for estado, precio in estado_stats.items()}
    
    calidad_stats = df.groupby('calidad_materiales')['precio_m2'].mean()
    # Filtrar posibles valores basura (ej. "Error" generado por fallos de Gemini)
    calidad_encoding = {
        calidad: float(precio / precio_m2_global)
        for calidad, precio in calidad_stats.items()
        if calidad not in ['Error', 'None', 'nan']
    }
    
    print("\n=== TARGET ENCODING MULTIMODAL (Ratio vs. media global) ===")
    print(f"Precio m² global medio: {precio_m2_global:.0f} €/m²")
    for estado, ratio in sorted(estado_encoding.items(), key=lambda x: x[1]):
        print(f"  {estado}: x{ratio:.3f}")
    
    with open(ESTADO_ENCODING_PATH, 'w', encoding='utf-8') as f:
        json.dump(estado_encoding, f, indent=4, ensure_ascii=False)
    with open(CALIDAD_ENCODING_PATH, 'w', encoding='utf-8') as f:
        json.dump(calidad_encoding, f, indent=4, ensure_ascii=False)
        
    df['target_encoding_estado'] = df['estado_conservacion'].map(estado_encoding).fillna(1.0)
    df['target_encoding_calidad'] = df['calidad_materiales'].map(calidad_encoding).fillna(1.0)

    
    # Target
    df['log_precio'] = np.log1p(df['precio_limpio'])
    
    # 13 Nuevas variables predictoras
    features = [
        'target_encoding_m2', 'm2', 'habitaciones', 'banos', 
        'tiene_ascensor', 'tiene_terraza', 'tiene_piscina', 'tiene_garaje', 'tiene_trastero',
        'ratio_metros_zona', 'ratio_hab_zona',
        'target_encoding_estado', 'target_encoding_calidad'
    ]
    
    # Reglas lógicas (1=Positivo): Todas las features deben sumar valor matemáticamente
    reglas_monotonicas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
    
    X = df[features]
    y = df['log_precio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    modelo_global = HistGradientBoostingRegressor(
        monotonic_cst=reglas_monotonicas,
        min_samples_leaf=5,
        max_iter=500,
        random_state=42
    )
    
    print("\nEntrenando Modelo Maestro Global...")
    modelo_global.fit(X_train_scaled, y_train)

    y_pred_log = modelo_global.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred_log)))
    
    bundle = {
        'modelo': modelo_global,
        'scaler': scaler_X,
        'rmse': rmse
    }
    
    joblib.dump(bundle, MODEL_PATH)
    print(f"MODELO GLOBAL ENTRENADO -> RMSE Test: {rmse:,.0f} EUR")
    print(f"Archivo del modelo maestro guardado en: {MODEL_PATH}")

if __name__ == "__main__":
    main()
