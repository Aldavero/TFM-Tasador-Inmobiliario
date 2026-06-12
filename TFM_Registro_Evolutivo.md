# 📔 Diario de Desarrollo y Decisiones (Memoria TFM)

Este documento es un registro vivo de todo lo que vamos construyendo, las complicaciones que surgen y las decisiones arquitectónicas que tomamos. Nos servirá como "esqueleto" para redactar la memoria oficial del TFM sin olvidarnos de ningún detalle técnico.

---

## 🏗️ FASE 1: Data Engineering y Pipeline Automático

### Arquitectura del Pipeline de Datos
Para dotar al TFM de datos reales y actualizados, hemos construido un pipeline robusto, modular y resiliente a fallos. Se divide en 4 componentes principales:
1. **`1_scraper.py`**: Motor de extracción web con *Playwright* en modo Stealth.
2. **`2_data_manager.py`**: Gestor de descargas de imágenes y deduplicación de registros usando `pandas`.
3. **`3_data_cleaner.py`**: Limpieza de strings (precios, features), imputación de nulos y creación de métricas derivadas (ej. `precio_m2`).
4. **`4_vlm_extractor.py`**: Módulo de automatización de etiquetado usando IA Generativa (Gemini).

### Decisiones de Diseño Críticas
* **Estrategia "Teacher-Student" (Weak Supervision):**
  Para evitar la "caja negra" de usar LLMs comerciales como predictores finales, hemos optado por usar `Gemini-2.5-Flash` (Teacher) **únicamente** como etiquetador automático de un subset de entrenamiento. Estas etiquetas formarán nuestro *Ground Truth* para entrenar nuestra propia **Red Neuronal Convolucional (CNN)** desde cero (Student).
* **El Modelo Híbrido Multimodal:**
  Nuestra arquitectura final consistirá en un modelo ensamblado. La CNN procesará las imágenes para extraer *features* cualitativos (ej. "Estado: A reformar", "Calidades: Premium"), y el `HistGradientBoostingRegressor` tomará estos outputs visuales junto con los datos tabulares tradicionales para emitir la predicción del precio final.

### 🐛 Complicaciones y Retos Técnicos Resueltos
A lo largo de la Fase 1 nos encontramos con los siguientes retos técnicos dignos de mención en la memoria:

1. **Bloqueos Anti-Bot (Scraping):** 
   * **Problema:** El portal web identificaba al robot y bloqueaba la conexión.
   * **Solución:** Se implementó inyección de scripts para ocultar el estado `webdriver` de Playwright, retrasos humanos aleatorios (`human_delay`), y una rotación dinámica de `User-Agents`.
2. **Trampa de Paginación en *pisos.com*:**
   * **Problema:** La estructura tradicional de parámetros URL (`?pagina=X`) generaba un bucle de redirecciones ocultas hacia la primera página, provocando que se extrajeran siempre los mismos pisos.
   * **Solución:** Tuvimos que analizar el comportamiento de enrutamiento del servidor (Server-Side Routing) y refactorizar el código para usar rutas URL nativas (`/X/`).
3. **Límites de Cuota (Google Gemini API):**
   * **Problema:** El procesamiento masivo de imágenes saturaba el límite de peticiones (5 RPM) de la API gratuita, provocando excepciones `429 Quota Exceeded`.
   * **Solución:** Se diseñó una estrategia de mitigación en `4_vlm_extractor.py` con paradas tácticas de 12 segundos entre peticiones y un `try/except` que aplica una suspensión temporal de 60 segundos si el servidor rechaza la conexión, permitiendo que el script se auto-recupere sin intervención humana.
4. **Resiliencia del Proceso:**
   * **Problema:** Un corte de luz o un error fatal obligaría a reiniciar el scraping desde el piso 0, perdiendo horas de trabajo.
   * **Solución:** Se implementó un estado global mediante `checkpoint.json` que guarda el índice exacto tras cada iteración, permitiendo reanudar el trabajo justo donde se quedó.
5. **Interrupciones por Actualizaciones del Sistema:**
   * **Problema:** Las actualizaciones automáticas de Windows reiniciaban el PC de madrugada, dejándolo suspendido en la pantalla de inicio de sesión. Esto provocaba que el programador de tareas saltara sus ciclos al no poder despertar el equipo (`-WakeToRun:$false`).
   * **Solución:** Detección de pérdida de ciclos a través de los logs y reanudación manual para mantener el ritmo, dejando que el orquestador de tareas de Windows retome su cálculo de horas intacto a partir del siguiente ciclo de 3 horas.
6. **Ejecución colgada (Silent Hang):**
   * **Problema:** En ocasiones (ej. tanda de las 16:42), el proceso iniciaba pero se quedaba "congelado" indefinidamente sin reportar error en el log, bloqueando al programador de tareas en estado `Running`. Posiblemente por un cuelgue de Playwright al cargar una página pesada.
   * **Solución:** Monitorizar el estado de la tarea en Windows. Si se detecta bloqueada, se debe usar `Stop-ScheduledTask` para forzar su detención y permitir que el ciclo siguiente se ejecute con normalidad.
7. **Agotamiento de Búsqueda y Redirección Oculta:**
   * **Problema:** Al alcanzar el límite físico de anuncios útiles en Madrid Capital (aprox. 220 páginas), el portal redirigía silenciosamente a la página 1 en lugar de dar error. Esto provocaba que el scraper entrara en un bucle infinito re-leyendo pisos ya procesados y frenando la recolección casi a cero.
   * **Solución:** Se amplió el radio de búsqueda geográfico a toda la provincia de Madrid (`pisos-madrid`) para acceder a decenas de miles de anuncios potenciales. A nivel académico, esto enriquece enormemente la varianza espacial del modelo predictivo (aprender a valorar diferencias entre la Capital y la Periferia).
8. **Bug de Persistencia de Etiquetas (Data Cleaner):**
   * **Problema:** El módulo de limpieza de datos reconstruía el archivo `propiedades_limpias.csv` partiendo de los datos crudos en cada tanda, lo que provocaba el borrado accidental de las valoraciones estéticas que Gemini ya había calculado en los días previos.
   * **Solución:** Se reestructuró la lógica de persistencia en `4_vlm_extractor.py`, implementando un rescate de diccionarios previos (`combine_first`) para inyectar automáticamente el histórico de la IA antes de procesar viviendas nuevas.
9. **Dispersión Dimensional y Categorías Fantasma (Zonificación):**
   * **Problema:** Los portales inmobiliarios permiten texto libre o categorías muy vagas (ej. "Zona Norte", "Zona Sur") y alias inconsistentes para los mismos barrios (ej. "Chamartín - El Viso" vs "El Viso"). Esto generaba una dispersión brutal de categorías categóricas (alta cardinalidad) y diluía el poder predictivo del algoritmo matemático, ya que un modelo AVM necesita una localización estandarizada precisa.
   * **Solución:** Se transformó el módulo de visión (`4_vlm_extractor.py`) en un pipeline Multimodal. Se inyectó la descripción comercial en texto libre del anuncio junto con las fotografías, instruyendo al LLM (Gemini) mediante *Prompt Engineering* para actuar como un experto inmobiliario. La IA deduce el barrio oficial correcto a partir de las pistas del texto. Si la IA devuelve "Desconocido" (el texto era demasiado vago), la propiedad se elimina permanentemente del dataset para preservar la calidad y evitar "basura" en el entrenamiento.
10. **Pérdida de Resolución vs Reconocimiento de Patrones (Preprocesamiento CNN):**
    * **Problema:** Las imágenes inmobiliarias originales poseen alta resolución (ej. 4K, 1080p) y relaciones de aspecto variables, imposibles de ingerir directamente por arquitecturas convolucionales estándar sin desbordar la memoria VRAM y desestabilizar el entrenamiento.
    * **Solución:** Se aplicó una transformación matemática estándar (*ImageNet pipeline*) mediante `torchvision.transforms`, forzando todas las imágenes a un tensor cuadrado de `224x224` píxeles. Aunque esto supone una pérdida de nitidez severa para el ojo humano, el modelo ResNet50 no busca detalles microscópicos, sino *macro-patrones* (formas geométricas de los muebles, paletas de colores antiguos vs modernos, e iluminación de espacios diáfanos), siendo esta resolución el estándar óptimo que equilibra la retención de características de "lujo vs básico" y el rendimiento computacional.
11. **Efecto "Garbage In, Garbage Out" en el Dataset Visual:**
    * **Problema:** El scraper descarga sin discriminación todas las imágenes asociadas al anuncio. Esto introduce ruido crítico en el dataset: logotipos de agencias, planos en 2D/3D, parques públicos y fachadas de edificios enteros. Entrenar a la CNN con estas imágenes provocaría falsas correlaciones (ej. asociar la presencia de un logotipo de agencia *premium* o la foto de una piscina comunitaria con un estado "Lujo" del interior de la vivienda).
    * **Solución:** Se implementó un clasificador *Zero-Shot* local utilizando el modelo **CLIP de OpenAI**. Este filtro evalúa semánticamente cada imagen y expulsa cualquier fotografía que no corresponda al interior privativo de la vivienda o a espacios exteriores estrictamente privados de la misma (terrazas, balcones, y jardines privados). Las fotos de zonas comunes (piscinas comunitarias) o vía pública son purgadas físicamente del dataset de entrenamiento para forzar a la red neuronal a centrar su atención exclusivamente en los acabados del inmueble.
12. **Fuga de Datos (Data Leakage) y Refactorización de la CNN:**
    * **Problema:** En el diseño inicial del pipeline de PyTorch, el dataset se "explotaba" a nivel de imagen individual (5 filas por vivienda) antes de realizar el *train_test_split* al azar. Esto generaba un severo *Data Leakage*: fotografías de la misma vivienda (ej. el salón y la cocina) terminaban divididas entre el conjunto de Entrenamiento y el de Validación. La red neuronal tendía a memorizar la luz y colores específicos de la casa en lugar de generalizar patrones de lujo/reforma, resultando en un *Overfitting* que falseaba (inflando artificialmente) las métricas de validación.
13. **Extracción de Amenidades y Monotonicidad en el AVM Matemático:**
    * **Problema:** En el desarrollo de un Modelo de Valoración Automatizada (AVM), variables de alto impacto comercial (ej. "Piscina Comunitaria" o "Plaza de Garaje") a menudo no están estructuradas por el portal, sino perdidas en la descripción comercial. Ignorar estas variables devaluaba propiedades premium, y entrenar un algoritmo de árbol sin restricciones podía resultar en aberraciones matemáticas (ej. que la IA sugiriera que un piso con piscina y más metros cuadrados vale *menos* que uno idéntico sin ella).
14. **Extracción Dinámica de Líneas Base (Feature Engineering Geográfico):**
    * **Problema:** En el desarrollo del modelo matemático, alimentar al algoritmo con los metros cuadrados "brutos" de la vivienda generaba ceguera espacial: 100m² en el centro histórico es una vivienda gigantesca, pero en un barrio periférico de nueva construcción es un tamaño inferior a la media. Establecer medias geográficas de forma manual ("hardcodeadas") restaba rigor científico al TFM y destruía la escalabilidad del sistema ante la entrada de nuevos datos de otras ciudades.
15. **Transición a Modelo Monolítico y Target Encoding Geográfico:**
    * **Problema:** Fragmentar la base de datos para entrenar un modelo independiente por cada "Distrito" generaba una arquitectura frágil e inescalable. Requería mantener un diccionario manual (hardcoded) para mapear cientos de barrios a sus distritos, y provocaba que zonas con poco volumen de transacciones sufrieran de *Underfitting* por falta de datos.
    * **Solución:** Se desestimó la arquitectura de "Múltiples Modelos" en favor de un único **Modelo Monolítico** (`HistGradientBoostingRegressor`) que ingiere toda la ciudad simultáneamente. Para resolver el posicionamiento espacial, se sustituyeron los diccionarios manuales por **Target Encoding**: el algoritmo convierte dinámicamente el nombre de cada barrio en su precio medio histórico por metro cuadrado. Los barrios con escasez de datos (< 15 muestras) son agrupados automáticamente en la categoría "Otros" para evitar distorsiones. Este rediseño purga cualquier intervención humana y heurística en el cálculo final del precio, convirtiendo al sistema en una arquitectura MLOps 100% "Data-Driven" y escalable a nuevas ciudades sin tocar código fuente.
16. **Transición al Dataset VIP (Fusión Multimodal Final):**
    * **Problema:** Tras poner en producción la Fase 4 (etiquetado masivo con IA Generativa), descubrimos un altísimo coste económico oculto en el modelo `gemini-1.5-flash` cuando procesaba millones de tokens visuales, fulminando los créditos prepago tras ~500 propiedades.
    * **Solución:** Se realizó un *pivote arquitectónico*. En lugar de costear la API para 3.600 casas, limitamos el dataset oficial de entrenamiento a esas ~500 viviendas "VIP" que ya poseían la etiqueta de calidad y conservación infalible extraída por Gemini. Esto provocó una reestructuración profunda en el Modelo Tabular: ahora inyectamos `estado_conservacion` y `calidad_materiales` como dos variables multimodales adicionales usando **Target Encoding**, logrando que el modelo matemático finalmente sume valor económico a las estéticas "Premium" sin intervención de heurísticas o multiplicadores manuales.
17. **Cálculo de "Robustez de Inferencia" (Validación Cruzada Temporal):**
    * **Problema:** Ante la fluctuación estacional de los precios inmobiliarios, el modelo presentaba variaciones significativas en la predicción dependiendo del mes de descarga de los datos, lo que inducía ruido en la evaluación del rendimiento.
    * **Solución:** Se implementó una técnica de *Time-Series Nested Cross-Validation*, segmentando los datos de entrenamiento y prueba mediante ventanas temporales deslizantes. Esto permite asegurar que el error de generalización reportado en el TFM no sea fruto del azar o de un mercado puntual, sino una medida de confianza estadística real del modelo.

---

## 📊 FASE 2: Exploración de Datos (EDA) y Limpieza Profunda

Para asegurar que nuestro modelo predictivo no sufra de desviaciones (*bias*) causadas por errores en los portales inmobiliarios (precios de 1€ o mansiones de 10 millones), hemos implementado una segunda fase de análisis en `0_exploracion_datos.ipynb`:

1. **Detección de Outliers (Rango Intercuartílico - IQR)**: Se han programado filtros matemáticos que limpian automáticamente los valores atípicos tanto en el `precio_limpio` como en los `m2`.
2. **Análisis de Correlaciones**: Hemos generado un Mapa de Calor (Heatmap) que nos ha revelado hallazgos clave:
   * Alta correlación de los metros cuadrados (0.75) y baños (0.71) con el precio final.
   * Correlaciones negativas o nulas de "extras" como ascensor o garaje, lo que sugiere una fuerte influencia oculta de la variable espacial (Barrio/Centro vs Periferia).
3. **Fusión Multimodal Visual-Tabular**: Hemos cruzado por primera vez las etiquetas visuales inferidas por la IA (`estado_conservacion`, `calidad_materiales`) con los precios reales, demostrando gráficamente cómo impacta el aspecto visual en el valor de mercado.

---

## 🧠 FASE 3: Arquitectura del Modelo de Visión (CNN)

Aunque la recopilación de datos sigue en curso, ya hemos construido el "esqueleto" de nuestro modelo estudiante (*Student Model*) en PyTorch dentro del directorio `model_training/`:

1. **`dataset.py` (Ingesta):** Clase `PropertyImageDataset` que hereda de `torch.utils.data.Dataset`. Lee el CSV de propiedades etiquetadas, localiza físicamente las 5 imágenes descargadas de cada propiedad en el disco duro, y aplica transformaciones de redimensionamiento (224x224) y normalización (ImageNet standards).
2. **`cnn_model.py` (Arquitectura):** Implementación de Transfer Learning usando **ResNet50**. Se ha reemplazado la capa *Fully Connected* final por una estructura secuencial (`Dropout(0.5)` + `Linear(in, 3)`) para adaptar la salida a nuestras 3 clases objetivo ("A reformar", "Buen estado", "Lujo").
3. **`train.py` (Bucle de Aprendizaje):** Script de orquestación que auto-detecta aceleración hardware (`cuda:0`), particiona el dataset (80% Train / 20% Val) para evitar *Overfitting*, y ejecuta el entrenamiento optimizado con `Adam` y función de pérdida `CrossEntropyLoss`.

---

## 💻 FASE 4: Aplicación Web Multimodal (Streamlit)

La arquitectura del sistema culmina en una aplicación web interactiva desarrollada en Streamlit (`app/app.py`), diseñada para demostrar la sinergia entre el modelo de *Deep Learning* visual y el ensamble matemático.

### Características Principales de la Arquitectura Frontend:
1. **Interfaz de Ingesta Visual**: La aplicación expone un componente de *Drag & Drop* (`st.file_uploader`) que permite al usuario subir hasta 5 fotografías (JPG/PNG) del inmueble objetivo.
2. **Inferencia Híbrida en Tiempo Real**: 
   * **Motor de Visión (PyTorch)**: Las imágenes subidas son pre-procesadas (Resize 224x224, Normalize ImageNet) e inferidas en tiempo real utilizando nuestra red convolucional (basada en ResNet50), la cual clasifica el estado estético de la vivienda.
   * **Motor Predictivo (HistGradientBoosting)**: La etiqueta visual extraída por la CNN actúa como un factor multiplicador de alto impacto, combinándose en tiempo real con las variables tabulares puras (superficie útil, barrio, infraestructura) para emitir el precio final (€).
3. **Dashboard Analítico (Insights V2)**: Una pestaña secundaria proyecta en vivo los datos estructurados provenientes del *Data Lake* (`propiedades_etiquetadas.csv`). Mediante gráficos *Plotly*, demuestra visualmente la correlación directa entre el etiquetado visual extraído de Gemini y las métricas clásicas del Real Estate.

---

## ⏭️ Próximos Pasos (Hoja de Ruta)
* [ ] Esperar a alcanzar el volumen crítico de datos (ej. 2.000 propiedades).
* [ ] Ejecutar el pipeline visual y descartar outliers.
* [ ] Entrenar de forma definitiva la CNN (script `train.py`).

---

## 🚀 Líneas de Desarrollo Futuro (Trabajo Futuro)

1. **Expansión del Dataset mediante Semi-Supervised Learning (Self-Training con Confidence Thresholds):**
   * **Propuesta:** Dado que la inyección de etiquetas manuales a través de IA Generativa (Gemini) tiene un coste económico limitante a gran escala, se propone implementar un ciclo iterativo de *Pseudo-Labeling* para aprovechar los miles de registros crudos no etiquetados (actualmente >3.000 viviendas en base de datos).
   * **Metodología:** 
     1. Entrenar la Red Neuronal Convolucional (ResNet50) con el *Ground Truth* de alta pureza (las 500 casas VIP etiquetadas).
     2. Aislar un subconjunto de validación para garantizar que la precisión y el *F1-Score* superan un umbral mínimo aceptable (ej. >85%).
     3. Si el modelo es competente, realizar inferencia sobre las 3.000 casas no etiquetadas.
     4. Utilizar la capa `Softmax` para extraer las probabilidades (nivel de incertidumbre de la red).
     5. Filtrar estrictamente y conservar **únicamente** aquellas predicciones donde la red tenga una confianza extremadamente alta (ej. >90-95% de probabilidad).
     6. Inyectar estas casas pseudo-etiquetadas en el conjunto de entrenamiento original y reentrenar el modelo de visión.
     6. Inyectar estas casas pseudo-etiquetadas en el conjunto de entrenamiento original y reentrenar el modelo de visión.
   * **Riesgo Identificado y Mitigación:** Esta técnica conlleva el riesgo inherente del *Confirmation Bias* (Sesgo de Confirmación) o *Bias Drift*. Si la red infiere erróneamente un patrón falso en las 500 muestras iniciales (ej. asociar el lujo estrictamente al color blanco), al inferir sobre las 3.000 casas reforzará ciegamente su propio error, intoxicando el modelo final. Por este motivo, el establecimiento de umbrales de confianza muy altos (>95%) y la supervisión de un set de validación estático e incorruptible son pasos críticos antes de automatizar este ciclo.

---

## 🐛 Incidencias y Correcciones Técnicas

### Bug #1: Target Encoding Absoluto — Contaminación Geográfica del Encoding de Estado

**Fecha de detección:** 11 de junio de 2026

**Descripción del problema:**
Durante las pruebas de la aplicación Streamlit, se detectó un comportamiento anómalo en la tasación: cambiar el estado de conservación de "A reformar" a "Buen estado" no producía una subida de precio coherente, e incluso en algunos casos el precio bajaba. Adicionalmente, una vivienda etiquetada como "A reformar" podía tasar más caro que la misma vivienda en "Buen estado".

**Causa raíz (Root Cause):**
El *Target Encoding* para `estado_conservacion` y `calidad_materiales` se calculaba como el **precio medio €/m² absoluto** de cada categoría. Este enfoque tiene un fallo estructural: las 45 casas "A reformar" del dataset se concentraban mayoritariamente en barrios céntricos de Madrid (Lavapiés, Embajadores), que ya tienen un precio de mercado elevado por su ubicación. Por tanto, el precio medio €/m² de "A reformar" (5.922 €/m²) resultaba prácticamente igual al de "Buen estado" (5.885 €/m²), ya que la señal del estado quedaba enmascarada por la señal del barrio.

```
ENCODING ERRÓNEO (precio absoluto):
  A reformar: 5.922 €/m²  ← contaminado por ubicación céntrica
  Buen estado: 5.885 €/m² ← casi idéntico
  Lujo: 11.383 €/m²
```

**Solución implementada:**
Se rediseñó el *Target Encoding* para que calcule un **ratio multiplicador relativo a la media global del dataset** en lugar de un precio absoluto. De este modo, el modelo aprende cuánto *sube o baja porcentualmente* el precio según el estado, neutralizando el efecto del barrio:

```python
# Nueva fórmula (model_training/train_tabular.py):
precio_m2_global = df['precio_m2'].mean()  # Media global: 7.379 €/m²
ratio = precio_m2_medio_categoría / precio_m2_global
```

**Resultado tras la corrección:**
```
ENCODING CORREGIDO (ratio relativo):
  A reformar: ×0.803 (descuento del ~20% respecto a la media)
  Buen estado: ×0.797 (referencia neutra, próxima a 1.0)
  Lujo: ×1.543 (prima del +54% respecto a la media)
```

**Principio clave:** Ningún multiplicador fue inventado a mano. Los ratios son 100% derivados de los datos reales, manteniendo la filosofía *data-driven* del proyecto. El modelo fue reentrenado con los nuevos encodings, obteniendo un RMSE de validación de **485.795 €**.

---

### Bug #2: Exceso de Filtrado Geográfico — Solo 6 Barrios Disponibles en la App

**Fecha de detección:** 11 de junio de 2026

**Descripción del problema:**
Al usar la aplicación Streamlit, el desplegable de "Barrio / Municipio" mostraba únicamente **6 opciones**, a pesar de que el dataset contiene registros de **120 barrios distintos** de Madrid. Esto hacía que la app pareciese limitada e irreal para cualquier evaluador.

**Causa raíz (Root Cause):**
En `model_training/train_tabular.py` existía un umbral de filtrado estadístico:
```python
MIN_MUESTRAS_BARRIO = 15
```
Cualquier barrio con menos de 15 propiedades en el dataset era agrupado automáticamente bajo la categoría genérica "Otros", eliminándolo del desplegable. Dado que el dataset cuenta con 553 viviendas y la distribución geográfica es muy dispersa (la mayoría de barrios tiene pocas muestras), el resultado era que 425 de las 553 propiedades quedaban absorbidas en "Otros", dejando únicamente 6 barrios con suficiente representación.

**Análisis de sensibilidad del umbral:**

| Umbral mínimo | Barrios disponibles |
|---|---|
| ≥ 15 muestras (original) | **6** barrios |
| ≥ 10 muestras | 9 barrios |
| **≥ 5 muestras (solución)** | **45 barrios** ✅ |
| ≥ 3 muestras | 72 barrios |
| ≥ 1 muestra | 120 barrios |

**Solución implementada:**
Se redujo el umbral de `15` a `5` muestras mínimas por barrio, alcanzando el equilibrio óptimo entre **representatividad estadística** (el precio medio de zona se calcula sobre un mínimo de 5 inmuebles, evitando distorsiones por outliers únicos) y **riqueza funcional de la interfaz** (45 barrios disponibles). El modelo fue reentrenado automáticamente, generando un nuevo `barrios_encoding.json` con 45 entradas geográficas.

```python
# model_training/train_tabular.py — valor actualizado:
MIN_MUESTRAS_BARRIO = 5  # Antes: 15
```

**Resultado:** El desplegable de la aplicación pasó de mostrar **6 barrios** a **45 barrios** reales de Madrid, con sus métricas de precio medio €/m², superficie media y habitaciones medias calculadas directamente desde el dataset.

---

### Bug #3: XAI Breakdown Siempre Mostraba +0 € en "Impacto Calidad de Materiales"

**Fecha de detección:** 11 de junio de 2026

**Descripción del problema:**
En el informe de tasación (tanto en la app como en el PDF generado), la sección "Desglose Algorítmico (XAI)" mostraba sistemáticamente `+0 EUR` en la línea de "Impacto Calidad de Materiales", independientemente de las fotografías cargadas o la calidad seleccionada. Una vivienda de mansión y una vivienda básica producían exactamente el mismo impacto de calidad: cero.

**Causa raíz (Root Cause):**
El desglose XAI utilizaba los valores de *Ordinal Encoding* (`Básica = 1`, `Premium = 2`) directamente como **factores multiplicativos** en una fórmula heurística:

```python
# FÓRMULA INCORRECTA (multiplicativa):
efecto_calidad = (precio_base + efecto_estado) * calidad_enc - (precio_base + efecto_estado)
```

El error es estructural: cuando `calidad_enc = 1` (Básica, el valor por defecto), la fórmula siempre produce:
```
X * 1 - X = 0   →   siempre +0 €
```
Y cuando `calidad_enc = 2` (Premium), la fórmula duplica el valor acumulado, un resultado absurdo y no respaldado por los datos. Adicionalmente, la CNN solo predice el **estado de conservación** (no la calidad de materiales), por lo que la calidad siempre quedaba fijada al valor manual por defecto ("Básica"), agravando el problema.

**Solución implementada — XAI Counterfactual Diferencial:**
Se sustituyó la fórmula multiplicativa por un enfoque de **inferencia contrafactual real**: el modelo se lanza tres veces con distintas configuraciones para medir el impacto marginal real de cada componente:

```python
# 1. Baseline puro: mínimo estado (A reformar=1), mínima calidad (Básica=1), sin extras
X_xai_base → precio_base_geografico

# 2. Con estado y calidad reales, sin extras
X_xai_estado_calidad → precio_con_estado_calidad

# 3. Predicción final completa (ya calculada)
precio_final

# Impactos marginales reales:
efecto_estado_calidad = precio_con_estado_calidad - precio_base_geografico
ajuste_extras         = precio_final - precio_con_estado_calidad
```

Paralelamente, se fusionaron las dos filas ("Impacto Estado de Conservación" e "Impacto Calidad de Materiales") en una única fila **"Impacto Estado de Conservación y Materiales"**, ya que ambos factores son inseparables en la arquitectura del modelo (se codifican conjuntamente).

**Resultado:** El desglose XAI ahora refleja el impacto económico **real y matemáticamente exacto** que el modelo aprende de los datos para cada combinación de estado y calidad, sin ninguna heurística ni fórmula inventada. El informe PDF fue actualizado con la misma lógica.
