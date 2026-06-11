# 🏡 TasIA Multimodal: Valoración Inmobiliaria Inteligente

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn)
![Gemini](https://img.shields.io/badge/Gemini_VLM-AI_Labeling-4285F4?style=for-the-badge&logo=google)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=for-the-badge)

**Trabajo de Fin de Máster (TFM) — Máster CEU**

TasIA Multimodal es una plataforma avanzada de valoración inmobiliaria (PropTech) que combina **Datos Tabulares (Machine Learning)** y **Visión Computacional (Deep Learning)** para estimar el precio de mercado de viviendas en Madrid.

A diferencia de los tasadores automáticos tradicionales (AVMs) que solo usan metros cuadrados y ubicación, TasIA extrae características cualitativas de las fotografías del inmueble (estado de conservación y calidad de materiales) para ajustar el precio final, simulando el ojo crítico de un tasador humano.

---

## 🚀 Características Principales

* **Motor Predictivo Híbrido:** Fusión de un modelo de Gradient Boosting (`HistGradientBoostingRegressor`) para datos geoespaciales y una Red Neuronal Convolucional (`ResNet50`) para extracción de *features* visuales.
* **Simulador de House Flipping:** Calcula la rentabilidad (ROI) esperada de comprar una vivienda "A reformar", actualizarla y venderla, descontando los costes de obra estimados.
* **Explicabilidad (XAI):** Transparencia algorítmica total. La app desglosa en cascada cuántos euros suma o resta cada variable (ubicación, estado, materiales, extras).
* **Informes Oficiales PDF:** Generación dinámica en memoria (RAM) de un informe de tasación B2B descargable.
* **Inteligencia Geoespacial:** Mapas interactivos 2D y 3D (PyDeck HexagonLayer) para analizar densidades y "skylines" de precio en el mercado madrileño.

---

## 📊 Resultados del Modelo

Los resultados a continuación corresponden al conjunto de **Test (20% del dataset)** reservado de forma estricta antes del entrenamiento:

### Modelo Tabular — HistGradientBoostingRegressor
| Métrica | Valor |
|--------|-------|
| **RMSE (Test Set)** | ~485,795 € |
| **Iteraciones (árboles)** | 500 |
| **Features utilizadas** | 13 |
| **Split Train / Test** | 80% / 20% (random_state=42) |
| **Variable objetivo** | `log(precio + 1)` (transformación logarítmica para normalizar la distribución) |

> **Nota:** El RMSE elevado (en valor absoluto) es habitual en Real Estate de lujo de Madrid, donde los precios de outliers superan los 2–5M €. La transformación logarítmica del target reduce el impacto de estos extremos durante el entrenamiento.

### Modelo Visual — ResNet50 (CNN Fine-Tuning)
| Métrica | Valor |
|--------|-------|
| **Arquitectura** | ResNet50 (pre-entrenada ImageNet) |
| **Clases** | 3 (`A reformar`, `Buen estado`, `Lujo`) |
| **Optimización** | Adam (lr=0.001) + Dropout(0.5) |
| **Early Stopping** | Paciencia = 5 épocas (máx. 30) |
| **Accuracy (Test Set)** | **69%** |
| **F1-Score Macro (Test Set)** | **0.5964** |
| **División de datos** | Estratificada por propiedad (no por foto) para evitar Data Leakage |

**Resultados por clase (Test Set — 287 imágenes):**

| Clase | Precision | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| A reformar | 1.00 | 0.29 | 0.45 | 24 |
| Buen estado | 0.74 | 0.79 | **0.77** | 178 |
| Lujo | 0.56 | 0.59 | 0.57 | 85 |

> **Nota:** El bajo Recall en "A reformar" (0.29) es esperable dado el fuerte desbalanceo de clases (solo el 8% del dataset pertenece a esa categoría). La precisión perfecta (1.00) indica que cuando el modelo predice "A reformar", siempre acierta. Esto refuerza la utilidad del modelo: los falsos negativos son inofensivos (la app los trata como "Buen estado"), pero los falsos positivos son inexistentes, evitando sobreestimaciones de reforma innecesarias.

### Dataset
| Parámetro | Valor |
|-----------|-------|
| **Total propiedades** | 553 |
| **Fuente de datos** | Portal inmobiliario (Madrid, 2026) |
| **Etiquetado** | Automático via Gemini VLM |
| **Distribución de clases** | Buen estado: 358 (64.7%) · Lujo: 150 (27.1%) · A reformar: 45 (8.1%) |
| **Precio medio** | ~984,689 € |
| **Precio mediano** | ~625,000 € |

---

## 🏗️ Arquitectura MLOps

El proyecto sigue una arquitectura modular en tres grandes fases:

```
┌─────────────────────────────────────────────────────────────────┐
│                         FASE 1: DATA PIPELINE                   │
│  Scraper → Limpieza → Descarga de Imágenes → Etiquetado (VLM)  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                       FASE 2: MODEL TRAINING                     │
│   ┌─────────────────────┐       ┌──────────────────────────┐    │
│   │   Rama Tabular       │       │     Rama Visual           │    │
│   │  HistGradBoost       │       │  ResNet50 Fine-Tuning     │    │
│   │  (13 features)       │       │  (3 clases de estado)     │    │
│   └──────────┬──────────┘       └─────────────┬────────────┘    │
│              └─────────────┬───────────────────┘                 │
│                    Fusión Multimodal (Inference)                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                         FASE 3: APP (STREAMLIT)                  │
│  Tasador Pro · Dashboard Insights · Mapa 3D · XAI · PDF         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Estructura del Repositorio

```text
📦 TFM v2
 ┣ 📂 app/                  # Interfaz web y frontend
 ┃ ┣ 📜 app.py              # Dashboard interactivo principal en Streamlit
 ┃ ┣ 📜 modelo_madrid_global.joblib  # Modelo tabular serializado
 ┃ ┣ 📜 barrios_encoding.json        # Target encoding geográfico
 ┃ ┣ 📜 estado_encoding.json         # Encoding de estado de conservación
 ┃ ┣ 📜 calidad_encoding.json        # Encoding de calidad de materiales
 ┃ ┗ 🖼️ tasia_logo.png      # Branding corporativo
 ┣ 📂 data_pipeline/        # Scripts de extracción y transformación de datos
 ┃ ┣ 📜 1_scraper.py        # Extracción de datos del portal inmobiliario
 ┃ ┣ 📜 2_data_manager.py   # Gestión de base de datos SQLite
 ┃ ┣ 📜 3_data_cleaner.py   # Limpieza y normalización del dataset tabular
 ┃ ┣ 📜 4_vlm_extractor.py  # Etiquetado automatizado de imágenes (Gemini VLM)
 ┃ ┗ 📜 5_image_filter.py   # Filtro y estandarización visual
 ┣ 📂 model_training/       # Scripts de entrenamiento algorítmico
 ┃ ┣ 📜 cnn_model.py        # Definición de la arquitectura PyTorch (ResNet50)
 ┃ ┣ 📜 dataset.py          # Dataloaders y Data Augmentation para imágenes
 ┃ ┣ 📜 train.py            # Entrenamiento CNN con Early Stopping
 ┃ ┣ 📜 train_tabular.py    # Entrenamiento del modelo HistGradientBoosting
 ┃ ┗ 📜 cnn_model_pesos.pth # Pesos del mejor modelo CNN (checkpoint)
 ┣ 📂 notebooks_v2/         # Cuadernos Jupyter experimentales
 ┃ ┗ 📜 1_EDA_Limpieza.ipynb # Análisis Exploratorio de Datos (EDA)
 ┗ 📜 README.md             # Este documento
```

---

## ⚙️ Instalación y Ejecución

### 1. Requisitos Previos
* Python 3.10 o superior.
* Entorno virtual (recomendado `venv`).

```bash
# Navegar a la carpeta del proyecto
cd "TFM v2"

# Crear y activar entorno virtual (Windows)
python -m venv venv
.\venv\Scripts\activate
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

> El archivo `requirements.txt` contiene todas las librerías necesarias para ejecutar la aplicación.

### 3. Variables de Entorno
Para el pipeline de datos (etiquetado con Gemini VLM), configura el archivo `data_pipeline/.env`:
```env
GEMINI_API_KEY=tu_clave_api_aqui
```

### 4. Lanzar la Aplicación Web
```bash
streamlit run app/app.py
```
La aplicación se abrirá en `http://localhost:8501`.

### 5. Re-entrenar los Modelos (opcional)

```bash
# Modelo Tabular (HistGradientBoosting)
python model_training/train_tabular.py

# Modelo Visual (ResNet50 CNN)
python model_training/train.py
```

---

## 🔬 Decisiones Técnicas Clave

| Decisión | Justificación |
|----------|---------------|
| **Target Encoding Geográfico** | Codifica el barrio como su precio/m² medio histórico. Es más robusto que One-Hot-Encoding con 60+ categorías y evita el maldito *curse of dimensionality*. |
| **Transformación `log(precio)`** | La distribución del precio es muy asimétrica (long-tail). El logaritmo la normaliza y reduce el peso desproporcionado de los outliers de lujo. |
| **Restricciones Monótonas** | El modelo tiene `monotonic_cst=[1,1,...]` para garantizar que más baños, más m² y mejor estado **siempre** suben el precio. Evita predicciones absurdas. |
| **Split por Propiedad (no por foto)** | La CNN divide el dataset por propiedad completa. Si dividiéramos por foto, las 5 imágenes de una misma casa podrían quedar en Train y Test simultáneamente, generando *Data Leakage*. |
| **ResNet50 + Fine-Tuning** | Aprovecha el conocimiento visual pre-entrenado en ImageNet (formas, texturas) y solo re-entrena la capa final para adaptarla a nuestra tarea de clasificación del estado. |

---

## 🎓 Sobre el Proyecto

**Autor:** Jorge Aldavero Romero  
**Institución:** Máster CEU  
**Año:** 2025–2026  
**Dominio:** PropTech · Real Estate · Machine Learning · Computer Vision

Este repositorio combina las disciplinas de Data Engineering, Machine Learning, Deep Learning y Business Intelligence aplicadas al sector inmobiliario de Madrid.
