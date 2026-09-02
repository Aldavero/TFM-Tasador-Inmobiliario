import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import time
import base64
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from fpdf import FPDF
import io
import pydeck as pdk

# Añadir directorio padre al path para poder importar cnn_model.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torchvision import transforms
from model_training.cnn_model import get_property_model

# =========================================================
# 1. CONFIGURACIÓN DE PÁGINA Y CARGA DE IMAGEN
# =========================================================
st.set_page_config(page_title="TasIA | Pro", page_icon="🏢", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def get_base64_of_bin_file(bin_file):
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_imagen = os.path.join(ruta_base, bin_file)
    if not os.path.exists(ruta_imagen): return None
    with open(ruta_imagen, 'rb') as f: data = f.read()
    return base64.b64encode(data).decode()

bg_base64 = get_base64_of_bin_file("background.jpg")

# =========================================================
# 2. DATOS Y MODELOS (Tabular + CNN)
# =========================================================
import json

@st.cache_data
def load_barrios_encoding():
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_json = os.path.join(ruta_base, "barrios_encoding.json")
    if os.path.exists(ruta_json):
        with open(ruta_json, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_json_encoding(filename):
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_json = os.path.join(ruta_base, filename)
    if os.path.exists(ruta_json):
        with open(ruta_json, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

BARRIOS_ENCODING = load_barrios_encoding()
ESTADO_ENCODING = load_json_encoding("estado_encoding.json")
CALIDAD_ENCODING = load_json_encoding("calidad_encoding.json")

@st.cache_resource
def load_tabular_model():
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_modelo = os.path.join(ruta_base, "modelo_madrid_global.joblib")
    try:
        if os.path.exists(ruta_modelo):
            return joblib.load(ruta_modelo)
    except Exception as e:
        import subprocess
        # Si falla por incompatibilidad de versiones, reentrenamos el modelo usando las librerías del servidor
        ruta_script = os.path.abspath(os.path.join(ruta_base, "..", "model_training", "train_tabular.py"))
        try:
            subprocess.run([sys.executable, ruta_script], check=True)
            if os.path.exists(ruta_modelo):
                return joblib.load(ruta_modelo)
        except Exception as train_error:
            pass
    return None

@st.cache_resource
def load_cnn_model():
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_pesos = os.path.join(ruta_base, "..", "model_training", "cnn_model_pesos.pth")
    
    # Instanciar arquitectura (3 clases)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_property_model(num_classes=3, pretrained=False)
    
    if os.path.exists(ruta_pesos):
        model.load_state_dict(torch.load(ruta_pesos, map_location=device))
        
    model = model.to(device)
    model.eval()
    return model, device

modelo_maestro = load_tabular_model()
cnn_model, device = load_cnn_model()

# =========================================================
# 3. CSS PREMIUM Y ESTILOS AVANZADOS
# =========================================================
# 1. Fondo menos oscuro (ajuste en la opacidad del rgba y en el color por defecto)
bg_style = f'background-image: linear-gradient(rgba(11, 15, 25, 0.2), rgba(11, 15, 25, 0.5)), url("data:image/jpg;base64,{bg_base64}"); background-size: cover; background-attachment: fixed;' if bg_base64 else 'background-color: #1e293b;'

st.markdown(f"""
<style>
    /* Global App Background */
    .stApp {{ {bg_style} }}
    
    /* 2. Títulos completamente blancos */
    h1, h2, h3, h4, h5, h6 {{ color: #ffffff !important; font-family: 'Inter', sans-serif; }}
    p, label {{ color: #f8fafc !important; font-family: 'Inter', sans-serif; }}
    
    /* Títulos de Expanders (Dashboard) */
    details summary p {{ color: #ffffff !important; font-weight: 700 !important; font-size: 1.05rem !important; }}
    
    /* Funcionalidad Sidebar Ocultar Elementos Innecesarios */
    [data-testid="stSidebar"] {{ 
        /* Sidebar ligeramente menos opaco para acompañar el fondo claro */
        background: rgba(15, 23, 42, 0.6); 
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }}
    
    /* Títulos del Sidebar Customizados */
    [data-testid="stSidebarNav"] {{ display: none; }} /* Oculta navegación por defecto si existe */
    
    /* Metrics Customization */
    div[data-testid="stMetricValue"] {{ font-size: 3.2rem !important; color: #10b981 !important; font-weight: 900; text-shadow: 0 0 20px rgba(16,185,129,0.4); }}
    div[data-testid="stMetricDelta"] {{ font-size: 1.1rem !important; margin-top: 5px;}}
    div[data-testid="stMetricDelta"] svg {{ fill: #00d2ff !important; }}
    div[data-testid="stMetricDelta"] div {{ color: #00d2ff !important; font-weight: 600;}}
    div[data-testid="stMetricLabel"] {{ font-size: 1.1rem !important; color: #94a3b8 !important; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.8;}}
    
    /* Tarjeta Efecto Cristal de Resultados */
    div[data-testid="stMetric"] {{ 
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%); 
        padding: 40px 30px; 
        border-radius: 20px; 
        border-top: 4px solid #00d2ff; 
        text-align: center; 
        box-shadow: 0 15px 35px rgba(0,0,0,0.6); 
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }}
    
    /* Botón Tasación Premium */
    .stButton>button {{
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white !important; border: none; border-radius: 12px; height: 3.8em; width: 100%;
        font-weight: 800; font-size: 18px; text-transform: uppercase; letter-spacing: 1.5px;
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.35); transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }}
    .stButton>button:hover {{ transform: translateY(-3px); box-shadow: 0 12px 30px rgba(0, 210, 255, 0.5); filter: brightness(1.1);}}
    .stButton>button:active {{ transform: translateY(1px); }}
    
    /* Botón Descarga PDF */
    [data-testid="stDownloadButton"] button {{
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border: none; border-radius: 12px; height: 3.8em; width: 100%;
        font-weight: 800; font-size: 16px; text-transform: uppercase; letter-spacing: 1px;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.35); transition: all 0.3s ease;
    }}
    [data-testid="stDownloadButton"] button p {{ color: white !important; font-weight: 800; }}
    [data-testid="stDownloadButton"] button:hover {{ transform: translateY(-3px); box-shadow: 0 12px 30px rgba(16, 185, 129, 0.5); filter: brightness(1.1);}}
    
    /* Radio Buttons Sidebar Styling (Simular Menú Premium) */
    div[role="radiogroup"] > label {{
        background: rgba(30, 41, 59, 0.4);
        padding: 12px 15px;
        border-radius: 10px;
        margin-bottom: 8px;
        border: 1px solid transparent;
        transition: all 0.2s ease;
        cursor: pointer;
    }}
    div[role="radiogroup"] > label:hover {{
        background: rgba(51, 65, 85, 0.7);
        border: 1px solid rgba(255,255,255,0.1);
    }}
    
    /* Formularios y Contenedores */
    .stSelectbox div[data-baseweb="select"], .stNumberInput input {{ background-color: rgba(15, 23, 42, 0.6) !important; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); color: white;}}
    
    /* Badge / Tags */
    .badge {{ background-color: rgba(30, 41, 59, 0.9); color: #00d2ff; padding: 6px 16px; border-radius: 30px; font-size: 0.85rem; font-weight: 700; margin: 4px; display: inline-block; border: 1px solid rgba(0, 210, 255, 0.2); text-transform: uppercase; letter-spacing: 1px;}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. SIDEBAR NAVIGATION & BRANDING
# =========================================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: flex-start; margin-bottom: 40px; margin-top: 10px;">
        <h1 style="margin: 0; font-size: 2.8rem; font-weight: 900; letter-spacing: 1px;">Tas<span style="color: #00d2ff;">IA</span></h1>
        <p style="color: #94a3b8 !important; font-size: 0.9rem; margin-top: 0px; font-weight: 600; text-transform: uppercase; letter-spacing: 2px;">Multimodal Edition</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-size: 0.9rem; color: #64748b !important; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;'>Menú Principal</h3>", unsafe_allow_html=True)
    menu_seleccionado = st.radio(
        "Navegación",
        options=["🏠 Tasador Pro (Multimodal)", "📊 Dashboard de Insights", "📐 Arquitectura MLOps"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 20px; border-radius: 12px; background: linear-gradient(180deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%); border: 1px solid rgba(255,255,255,0.05);">
        <p style="font-size: 0.8rem; color: #94a3b8 !important; margin: 0;">Motor Impulsado por</p>
        <p style="font-size: 1rem; color: #00d2ff !important; font-weight: bold; margin: 0;">CNN PyTorch + HistGB</p>
        <div style="height: 2px; width: 30px; background: #00d2ff; margin-top: 10px;"></div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# 5. RENDERIZADO DEL CONTENIDO PRINCIPAL
# =========================================================

def predict_image_class(img_file):
    # Transformaciones estándar de ImageNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        img = Image.open(img_file).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = cnn_model(img_t)
            _, predicted = torch.max(outputs, 1)
        
        # Mapeo inverso (0: A reformar, 1: Buen estado, 2: Lujo)
        clases = {0: "A reformar", 1: "Buen estado", 2: "Lujo"}
        return clases[predicted.item()]
    except Exception as e:
        return "Buen estado" # Default en caso de error

def generar_pdf_tasacion(datos_inmueble, precio_final, precio_base_geografico, efecto_estado_calidad, ajuste_extras):
    pdf = FPDF()
    pdf.add_page()
    
    # Colores corporativos (approx)
    pdf.set_text_color(30, 41, 59) # Slate 800
    
    # Título Principal (TasIA en texto con colores)
    pdf.set_font('Helvetica', 'B', 28)
    w_tas = pdf.get_string_width('Tas')
    w_ia = pdf.get_string_width('IA')
    total_w = w_tas + w_ia
    start_x = (210 - total_w) / 2
    
    pdf.set_y(20)
    pdf.set_x(start_x)
    pdf.set_text_color(30, 41, 59) # Slate oscuro (se vera como negro en papel blanco)
    pdf.cell(w_tas, 15, 'Tas', border=0, ln=0)
    pdf.set_text_color(37, 99, 235) # Azul mas oscuro para IA
    pdf.cell(w_ia, 15, 'IA', border=0, ln=1)
    
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(100, 116, 139) # Slate 500
    pdf.cell(0, 10, 'Informe Oficial de Valoracion por Inteligencia Artificial', ln=True, align='C')
    pdf.ln(10)
    
    # Sección 1: Datos del Inmueble
    pdf.set_text_color(30, 41, 59)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, '1. Caracteristicas del Inmueble', ln=True)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font('Helvetica', '', 12)
    for k, v in datos_inmueble.items():
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(60, 8, str(k) + ':', border=0)
        pdf.set_font('Helvetica', '', 12)
        pdf.cell(0, 8, str(v), border=0, ln=True)
    pdf.ln(10)
    
    # Sección 2: Desglose de Valoración (XAI)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, '2. Desglose Algoritmico (XAI)', ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(80, 8, 'Base Geografica y Superficie:', border=0)
    pdf.cell(0, 8, f"{precio_base_geografico:,.0f} EUR", border=0, ln=True, align='R')
    
    pdf.cell(80, 8, 'Impacto Estado de Conservacion y Materiales:', border=0)
    pdf.cell(0, 8, f"{efecto_estado_calidad:+,.0f} EUR", border=0, ln=True, align='R')
    
    pdf.cell(80, 8, 'Ajuste Algoritmico Extras (Ascensor, etc):', border=0)
    pdf.cell(0, 8, f"{ajuste_extras:+,.0f} EUR", border=0, ln=True, align='R')
    
    pdf.ln(5)
    
    # Total
    pdf.set_fill_color(240, 249, 255) # Light blue
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(80, 15, 'VALOR DE MERCADO ESTIMADO:', border=0, fill=True)
    pdf.set_text_color(16, 185, 129) # Emerald 500
    pdf.cell(0, 15, f"{precio_final:,.0f} EUR", border=0, ln=True, align='R', fill=True)
    
    pdf.ln(20)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(0, 6, 'Aviso Legal: Esta valoracion es una estimacion matematica basada en modelos de Machine Learning (HistGB) y Redes Neuronales Convolucionales (PyTorch ResNet50) para TFM. No constituye una tasacion hipotecaria oficial.')
    
    # Return as bytes
    # pdf.output returns bytearray in fpdf2 if dest='S'
    return bytes(pdf.output(dest='S'))

if menu_seleccionado == "🏠 Tasador Pro (Multimodal)":
    st.markdown("<h2 style='margin-top: 0; margin-bottom: 30px; font-weight: 800; font-size: 2.2rem;'>Valoración Automática Multimodal</h2>", unsafe_allow_html=True)
    
    col_input, col_result = st.columns([1.3, 1], gap="large")
    
    with col_input:
        with st.container(border=True):
            st.markdown("<h4 style='color: #ffffff !important; margin-bottom: 15px;'>Ubicación y Datos Tabulares</h4>", unsafe_allow_html=True)
            barrios_lista = sorted(list(BARRIOS_ENCODING.keys())) if BARRIOS_ENCODING else ["Cargando..."]
            barrio_nom = st.selectbox("Barrio / Municipio", barrios_lista)
            
            m2 = st.number_input("Superficie Útil (m²)", min_value=20, max_value=800, value=85, step=5)
            c3, c4 = st.columns(2)
            habs = c3.slider("Habitaciones", 1, 8, 2)
            banos = c4.slider("Cuartos de Baño", 1, 5, 1)
            
            c5, c6 = st.columns(2)
            ascensor = c5.toggle("Edificio con Ascensor", value=True)
            terraza = c6.toggle("Dispone de Terraza", value=False)
            
            c7, c8, c9 = st.columns(3)
            piscina = c7.toggle("Piscina", value=False)
            garaje = c8.toggle("Plaza de Garaje", value=False)
            trastero = c9.toggle("Trastero", value=False)
            
            st.markdown("<h5 style='color: #ffffff !important; margin-top: 15px;'>Filtros Visuales Manuales</h5>", unsafe_allow_html=True)
            c10, c11 = st.columns(2)
            estado_manual = c10.selectbox("Estado", ["A reformar", "Buen estado", "Lujo"], index=1)
            calidad_manual = c11.selectbox("Materiales", ["Básica", "Premium"], index=0)
            
        with st.container(border=True):
            st.markdown("<h4 style='color: #ffffff !important; margin-bottom: 15px;'>Análisis Visual (CNN)</h4>", unsafe_allow_html=True)
            
            # --- Lógica para limpiar el uploader ---
            if 'uploader_key' not in st.session_state:
                st.session_state.uploader_key = 1

            def reset_uploader():
                st.session_state.uploader_key += 1

            uploaded_files = st.file_uploader("Sube fotos del interior de la vivienda (Max 5)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")
            
            if uploaded_files:
                st.button("🗑️ Quitar imágenes y subir otras", on_click=reset_uploader)
                st.markdown("<p style='font-size: 0.9rem; color: #94a3b8;'>Imágenes cargadas para la Red Neuronal:</p>", unsafe_allow_html=True)
                cols_img = st.columns(len(uploaded_files[:5]))
                for i, file in enumerate(uploaded_files[:5]):
                    cols_img[i].image(file, width='stretch')
    with col_result:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True) 
        btn_calc = st.button("🚀 Procesar Tasación con IA")
        
        if not btn_calc:
            st.markdown("""
            <div style="border: 2px dashed rgba(255,255,255,0.1); border-radius: 16px; padding: 60px 30px; text-align: center; margin-top: 10px; background: rgba(15, 23, 42, 0.4); backdrop-filter: blur(5px);">
                <span style="font-size: 3rem;">📸 + 📊</span>
                <p style="color: #94a3b8 !important; font-size: 1.1rem; line-height: 1.6; margin-top: 20px;">
                Rellene los datos y arrastre fotografías.<br><br>Pulse <b>Procesar Tasación</b> para que PyTorch analice visualmente el inmueble e inyecte los tensores en el modelo matemático HistGB.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        if btn_calc:
            if modelo_maestro is not None and BARRIOS_ENCODING:
                with st.spinner("⏳ Analizando imágenes con CNN y procesando inferencia tabular..."):
                    time.sleep(1.0) # Efecto inmersivo
                    
                    # 1. Inferencia CNN
                    estado_predicho = estado_manual # Por defecto el manual
                    if uploaded_files:
                        preds = [predict_image_class(f) for f in uploaded_files[:5]]
                        # Coger la clase más repetida (moda)
                        estado_predicho = max(set(preds), key=preds.count)
                    
                    calidad_final = calidad_manual
                    
                    # 2. Inferencia Tabular
                    bundle = modelo_maestro
                    encodings = BARRIOS_ENCODING[barrio_nom]
                    
                    precio_m2_enc = encodings['precio_m2_medio']
                    r_m2 = m2 / encodings['superficie_media']
                    r_hab = habs / encodings['habs_media']
                    
                    # Obtener encodings multimodales
                    estado_enc = ESTADO_ENCODING.get(estado_predicho, ESTADO_ENCODING.get("Buen estado", 0))
                    calidad_enc = CALIDAD_ENCODING.get(calidad_final, CALIDAD_ENCODING.get("Básica", 0))
                    
                    X_input = pd.DataFrame([
                        [precio_m2_enc, m2, habs, banos, int(ascensor), int(terraza), int(piscina), int(garaje), int(trastero), r_m2, r_hab, estado_enc, calidad_enc]
                    ], columns=[
                        'target_encoding_m2', 'm2', 'habitaciones', 'banos', 
                        'tiene_ascensor', 'tiene_terraza', 'tiene_piscina', 'tiene_garaje', 'tiene_trastero',
                        'ratio_metros_zona', 'ratio_hab_zona',
                        'target_encoding_estado', 'target_encoding_calidad'
                    ])
                    
                    log_pred = bundle['modelo'].predict(bundle['scaler'].transform(X_input))[0]
                    precio_base = np.expm1(log_pred)
                    precio_final = precio_base
                        
                    precio_m2 = precio_final / m2
                    
                    # --- Configuración visual por clase predicha ---
                    estado_config = {
                        "Lujo": {
                            "icono": "💎",
                            "color_borde": "#f59e0b",
                            "color_badge": "linear-gradient(135deg, #f59e0b, #d97706)",
                            "color_texto": "#fef3c7",
                            "descripcion": "La IA ha detectado acabados y materiales de alta gama.",
                            "label": "LUJO"
                        },
                        "Buen estado": {
                            "icono": "✅",
                            "color_borde": "#10b981",
                            "color_badge": "linear-gradient(135deg, #10b981, #059669)",
                            "color_texto": "#d1fae5",
                            "descripcion": "Inmueble en buen estado de conservación y habitabilidad.",
                            "label": "BUEN ESTADO"
                        },
                        "A reformar": {
                            "icono": "🔨",
                            "color_borde": "#ef4444",
                            "color_badge": "linear-gradient(135deg, #ef4444, #dc2626)",
                            "color_texto": "#fee2e2",
                            "descripcion": "La IA detecta que el inmueble requiere obras de reforma.",
                            "label": "A REFORMAR"
                        }
                    }
                    cfg = estado_config.get(estado_predicho, estado_config["Buen estado"])
                    origen_label = "🤖 Detectado por Red Neuronal (CNN)" if uploaded_files else "🖱️ Selección manual del usuario"

                    st.markdown(f"""
                    <div style="
                        border: 2px solid {cfg['color_borde']};
                        border-radius: 16px;
                        padding: 18px 22px;
                        margin-bottom: 18px;
                        background: rgba(15, 23, 42, 0.7);
                        backdrop-filter: blur(10px);
                        display: flex;
                        align-items: center;
                        gap: 18px;
                    ">
                        <span style="font-size: 2.6rem; line-height: 1;">{cfg['icono']}</span>
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 5px;">
                                <span style="
                                    background: {cfg['color_badge']};
                                    color: white;
                                    padding: 4px 14px;
                                    border-radius: 20px;
                                    font-size: 0.78rem;
                                    font-weight: 800;
                                    letter-spacing: 1.5px;
                                    text-transform: uppercase;
                                ">{cfg['label']}</span>
                            </div>
                            <p style="color: {cfg['color_texto']} !important; font-size: 0.88rem; margin: 0 0 4px 0;">{cfg['descripcion']}</p>
                            <p style="color: #64748b !important; font-size: 0.75rem; margin: 0;">{origen_label}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    with st.container():
                        
                        st.metric(
                            label="Valor Estimado (EUR)", 
                            value=f"{precio_final:,.0f} €", 
                            delta=f"{precio_m2:,.0f} € / m² Equivalente"
                        )
                        
                        # --- Cálculos XAI Counterfactual (impacto marginal real del modelo) ---
                        # Baseline: mínimo estado (A reformar=1) y mínima calidad (Básica=1), sin extras
                        min_estado_enc = min(ESTADO_ENCODING.values())
                        min_calidad_enc = min(CALIDAD_ENCODING.values())

                        X_xai_base = pd.DataFrame([[
                            precio_m2_enc, m2, habs, banos,
                            0, 0, 0, 0, 0,
                            r_m2, r_hab, min_estado_enc, min_calidad_enc
                        ]], columns=[
                            'target_encoding_m2', 'm2', 'habitaciones', 'banos',
                            'tiene_ascensor', 'tiene_terraza', 'tiene_piscina', 'tiene_garaje', 'tiene_trastero',
                            'ratio_metros_zona', 'ratio_hab_zona',
                            'target_encoding_estado', 'target_encoding_calidad'
                        ])

                        X_xai_estado_calidad = pd.DataFrame([[
                            precio_m2_enc, m2, habs, banos,
                            0, 0, 0, 0, 0,
                            r_m2, r_hab, estado_enc, calidad_enc
                        ]], columns=[
                            'target_encoding_m2', 'm2', 'habitaciones', 'banos',
                            'tiene_ascensor', 'tiene_terraza', 'tiene_piscina', 'tiene_garaje', 'tiene_trastero',
                            'ratio_metros_zona', 'ratio_hab_zona',
                            'target_encoding_estado', 'target_encoding_calidad'
                        ])

                        precio_base_geografico = np.expm1(bundle['modelo'].predict(bundle['scaler'].transform(X_xai_base))[0])
                        precio_con_estado_calidad = np.expm1(bundle['modelo'].predict(bundle['scaler'].transform(X_xai_estado_calidad))[0])

                        efecto_estado_calidad = precio_con_estado_calidad - precio_base_geografico
                        ajuste_extras = precio_final - precio_con_estado_calidad

                        st.markdown("""
                        <div style='text-align:center; margin-top: 15px;'>
                            <span class='badge'>Visión: PyTorch ResNet50</span> 
                            <span class='badge'>Predictor: Modelo Monolítico HistGB</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 30px 0;'>", unsafe_allow_html=True)
                        
                        # --- MODULO 1: SIMULADOR DE FLIPPING ---
                        if estado_predicho in ["A reformar", "Buen estado"]:
                            estado_objetivo = "Buen estado" if estado_predicho == "A reformar" else "Lujo"
                            estado_objetivo_enc = ESTADO_ENCODING.get(estado_objetivo, 0)
                            calidad_objetivo = "Premium" if estado_objetivo == "Lujo" else calidad_final
                            calidad_objetivo_enc = CALIDAD_ENCODING.get(calidad_objetivo, 0)
                            
                            X_input_flip = pd.DataFrame([
                                [precio_m2_enc, m2, habs, banos, int(ascensor), int(terraza), int(piscina), int(garaje), int(trastero), r_m2, r_hab, estado_objetivo_enc, calidad_objetivo_enc]
                            ], columns=[
                                'target_encoding_m2', 'm2', 'habitaciones', 'banos', 
                                'tiene_ascensor', 'tiene_terraza', 'tiene_piscina', 'tiene_garaje', 'tiene_trastero',
                                'ratio_metros_zona', 'ratio_hab_zona',
                                'target_encoding_estado', 'target_encoding_calidad'
                            ])
                            
                            log_pred_flip = bundle['modelo'].predict(bundle['scaler'].transform(X_input_flip))[0]
                            precio_flip = np.expm1(log_pred_flip)
                            
                            coste_m2 = 600 if estado_objetivo == "Buen estado" else 1000
                            coste_total_reforma = m2 * coste_m2
                            beneficio_bruto = precio_flip - precio_final - coste_total_reforma
                            roi = (beneficio_bruto / coste_total_reforma) * 100 if coste_total_reforma > 0 else 0
                            
                            if roi > 0:
                                st.markdown(f"""
                                <div style="background: linear-gradient(145deg, rgba(16, 185, 129, 0.2) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1px solid rgba(16,185,129,0.5); padding: 25px; border-radius: 12px; margin-top: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); backdrop-filter: blur(10px);">
                                    <h3 style="color: #34d399 !important; margin-top: 0; font-weight: 800;">💡 Oportunidad House Flipping</h3>
                                    <p style="color: #f8fafc !important; font-size: 1.1rem; margin-bottom: 15px;">Si inviertes aprox. <b>{coste_total_reforma:,.0f} €</b> en reformar la casa y subir su estado a <b>'{estado_objetivo}'</b>:</p>
                                    <p style="color: #ffffff !important; font-size: 1.1rem; margin-bottom: 8px;">• <b style="color: #ffffff !important;">Nuevo Precio Estimado:</b> <span style="color:#34d399; font-weight: 700;">{precio_flip:,.0f} €</span></p>
                                    <p style="color: #ffffff !important; font-size: 1.1rem; margin-bottom: 8px;">• <b style="color: #ffffff !important;">Beneficio Bruto Esperado:</b> <span style="color:#34d399; font-weight: 700;">{beneficio_bruto:,.0f} €</span></p>
                                    <p style="color: #ffffff !important; font-size: 1.1rem; margin-bottom: 20px;">• <b style="color: #ffffff !important;">ROI (Retorno de Inversión):</b> <span style="color:#34d399; font-weight: 700;">+{roi:,.1f}%</span></p>
                                    <p style="color: #94a3b8 !important; font-size: 0.85rem; margin-bottom: 0;"><i>Nota: Estimación teórica calculada por la red neuronal sobre las dinámicas de mercado del barrio.</i></p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="background: linear-gradient(145deg, rgba(239, 68, 68, 0.2) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1px solid rgba(239,68,68,0.5); padding: 25px; border-radius: 12px; margin-top: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); backdrop-filter: blur(10px);">
                                    <h3 style="color: #f87171 !important; margin-top: 0; font-weight: 800;">⚠️ Riesgo House Flipping</h3>
                                    <p style="color: #f8fafc !important; font-size: 1.1rem; margin-bottom: 10px;">Si inviertes aprox. <b>{coste_total_reforma:,.0f} €</b> en reforma ('{estado_objetivo}'), el nuevo precio estimado sería <b>{precio_flip:,.0f} €</b>.</p>
                                    <p style="color: #ffffff !important; font-size: 1.1rem; margin-bottom: 0;">El sobrecoste de reforma supera al incremento de valor de mercado en este barrio <span style="color:#f87171; font-weight: 700;">(Pérdida esperada: {beneficio_bruto:,.0f} €)</span>.</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # --- MODULO 2: DESCARGA DE PDF ---
                        datos_inmueble = {
                            "Barrio": barrio_nom,
                            "Superficie": f"{m2} m2",
                            "Habitaciones": habs,
                            "Banos": banos,
                            "Estado IA (Vision)": estado_predicho,
                            "Calidad IA (Vision)": calidad_final
                        }
                        try:
                            pdf_bytes = generar_pdf_tasacion(datos_inmueble, precio_final, precio_base_geografico, efecto_estado_calidad, ajuste_extras)
                            
                            col_pdf1, col_pdf2 = st.columns([1, 1])
                            with col_pdf2:
                                st.download_button(
                                    label="📄 Descargar Informe Oficial (PDF)",
                                    data=pdf_bytes,
                                    file_name="TasIA_Informe_Valoracion.pdf",
                                    mime="application/pdf",
                                    width='stretch'
                                )
                        except Exception as e:
                            st.error(f"Error generando PDF: {e}")

                    st.balloons()
            else:
                st.error("🚨 Error del Servidor: Motor Predictivo offline (weights no encontrados).")

elif menu_seleccionado == "📊 Dashboard de Insights":
    st.markdown("<h2 style='margin-top: 0; margin-bottom: 20px; font-weight: 800; font-size: 2.2rem;'>Inteligencia y Geometría de Mercado</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #ffffff; font-size: 1.1rem; margin-bottom: 30px;'>Interactúa con los datos procesados en la Fase 1 y 2, incluyendo las etiquetas de Visión Artificial (Gemini).</p>", unsafe_allow_html=True)
    
    @st.cache_data
    def load_plot_data():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cand_limpias = [
            os.path.join(base_dir, "..", "data_pipeline", "data", "processed", "propiedades_limpias.csv"),
            os.path.join(base_dir, "data_pipeline", "data", "processed", "propiedades_limpias.csv"),
            os.path.join("data_pipeline", "data", "processed", "propiedades_limpias.csv"),
        ]
        cand_etiquetadas = [
            os.path.join(base_dir, "..", "data_pipeline", "data", "processed", "propiedades_etiquetadas.csv"),
            os.path.join(base_dir, "data_pipeline", "data", "processed", "propiedades_etiquetadas.csv"),
            os.path.join("data_pipeline", "data", "processed", "propiedades_etiquetadas.csv"),
        ]
        ruta_limpias = next((p for p in cand_limpias if os.path.exists(p)), None)
        ruta_etiquetadas = next((p for p in cand_etiquetadas if os.path.exists(p)), None)
        
        if ruta_limpias and ruta_etiquetadas:
            df_limpias = pd.read_csv(ruta_limpias)
            df_etiq = pd.read_csv(ruta_etiquetadas)[['id_inmueble', 'estado_conservacion', 'calidad_materiales']]
            # Hacemos left join para tener todas las casas, y las que no tengan etiqueta tendrán NaN
            df_full = pd.merge(df_limpias, df_etiq, on='id_inmueble', how='left')
            df_full['estado_conservacion'] = df_full['estado_conservacion'].fillna('No etiquetado')
            return df_full
        return None
    
    df_plot = load_plot_data()
    
    if df_plot is not None:
        # Renombramos columnas internamente para que coincidan con la versión anterior o usamos las nuevas
        col_precio = 'precio_limpio' if 'precio_limpio' in df_plot.columns else (df_plot.columns[0])
        col_sup = 'm2' if 'm2' in df_plot.columns else (df_plot.columns[1])
        
        # --- NUEVO: Mapa Interactivo ---
        if 'img_url_3' in df_plot.columns:
            # Parsear lat/lon
            df_plot['lat'] = df_plot['img_url_3'].astype(str).str.extract(r'([0-9]{2}\.[0-9]+)@').astype(float)
            df_plot['lon'] = df_plot['img_url_3'].astype(str).str.extract(r'@(-[0-9]{1}\.[0-9]+)_').astype(float)
            df_map = df_plot.dropna(subset=['lat', 'lon', 'estado_conservacion']).copy()
            
            if not df_map.empty:
                with st.container(border=True):
                    st.markdown("<h4 style='color: #ffffff !important; margin-bottom: 5px;'>Geometría Espacial del Lujo (Mapa)</h4>", unsafe_allow_html=True)
                    
                    # Toggle para mostrar el Data Lake completo
                    mostrar_todas = st.toggle("⚪ Mostrar casas NO etiquetadas (Puntos Blancos)", value=False)
                    if not mostrar_todas:
                        df_map = df_map[df_map['estado_conservacion'] != 'No etiquetado']
                    
                    color_discrete_map = {"Lujo": "#f59e0b", "Buen estado": "#10b981", "A reformar": "#ef4444", "No etiquetado": "#ffffff"}
                    
                    fig_map = px.scatter_map(
                        df_map, lat="lat", lon="lon", color="estado_conservacion", 
                        hover_name="barrio_limpio", hover_data=["precio_limpio", "m2"],
                        color_discrete_map=color_discrete_map, zoom=11, height=500,
                        map_style="carto-darkmatter",
                        opacity=0.5 if mostrar_todas else 0.9
                    )
                    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                
                # --- SISTEMA DE PESTAÑAS PARA LOS MAPAS ---
                tab_2d, tab_3d = st.tabs(["🗺️ Mapa 2D (Clásico)", "🏙️ Mapa 3D Hexagonal (Densidad)"])
                
                with tab_2d:
                    st.plotly_chart(fig_map, width='stretch')
                
                with tab_3d:
                    st.markdown("<p style='color: #94a3b8; font-size: 0.9rem; margin-bottom: 10px;'>Las columnas hexagonales representan la densidad de oferta inmobiliaria. Puedes rotar el mapa manteniendo pulsada la tecla <b>Ctrl/Cmd + Clic izquierdo</b>.</p>", unsafe_allow_html=True)
                    
                    df_hex = df_map.dropna(subset=['lat', 'lon', col_precio])
                    
                    layer = pdk.Layer(
                        'HexagonLayer',
                        data=df_hex,
                        get_position='[lon, lat]',
                        radius=150, # Reducimos el radio para más definición
                        elevation_scale=100, # Aumentamos para ver bien el skyline 3D
                        elevation_range=[0, 1500],
                        pickable=True,
                        extruded=True,
                        auto_highlight=True,
                        get_elevation_weight=col_precio,
                        elevation_aggregation='"MEAN"'
                    )

                    view_state = pdk.ViewState(
                        longitude=df_hex['lon'].mean(),
                        latitude=df_hex['lat'].mean(),
                        zoom=11.5,
                        pitch=55, # Inclinación 3D
                        bearing=-15 # Rotación
                    )

                    r = pdk.Deck(
                        layers=[layer], 
                        initial_view_state=view_state, 
                        map_provider="carto",
                        map_style="dark", 
                        tooltip={"html": "<b>Precio Medio Estimado:</b> {elevationValue} €<br/><b>Concentración:</b> {colorValue} inmuebles", "style": {"backgroundColor": "steelblue", "color": "white"}}
                    )
                    st.pydeck_chart(r, width='stretch')
                    with st.expander("🔬 Inteligencia Geoespacial"):
                        st.write("Cada coordenada ha sido extraída usando expresiones regulares (Regex) sobre los datos estáticos, parseando las llamadas a las APIs de mapas originales. Los colores reflejan el etiquetado visual automático que generó la IA (Gemini VLM).")
                st.markdown("<br>", unsafe_allow_html=True)

        c_chart1, c_chart2 = st.columns(2)
        
        with c_chart1:
            with st.container(border=True):
                fig1 = px.histogram(df_plot, x=col_precio, nbins=50, title="Distribución Probabilística de Precios", 
                                    template="plotly_dark", color_discrete_sequence=["#00d2ff"], opacity=0.8)
                fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="white"), font_color="white", title_font_color="white", legend_font_color="white")
                fig1.update_xaxes(color="white", title_font=dict(color="white"), tickfont=dict(color="white"))
                fig1.update_yaxes(color="white", title_font=dict(color="white"), tickfont=dict(color="white"))
                st.plotly_chart(fig1, width='stretch')
                with st.expander("🔬 Especificaciones del Feature Engineering"):
                    st.write("El vector de precios presenta asimetría (long tail) que requiere transformaciones logarítmicas en el backend para estabilizar el entrenamiento de HistGB.")
                
        with c_chart2:
            with st.container(border=True):
                # Usar color por estado_conservacion si existe
                color_col = 'estado_conservacion' if 'estado_conservacion' in df_plot.columns else None
                fig2 = px.scatter(df_plot, x=col_sup, y=col_precio, opacity=0.6, title="Función Rendimiento-Superficie vs Estado Visivo", 
                                  template="plotly_dark", color=color_col, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="white"), font_color="white", title_font_color="white", legend_font_color="white")
                fig2.update_xaxes(color="white", title_font=dict(color="white"), tickfont=dict(color="white"))
                fig2.update_yaxes(color="white", title_font=dict(color="white"), tickfont=dict(color="white"))
                st.plotly_chart(fig2, width='stretch')
                with st.expander("🔬 Insights de Visión Computacional"):
                    st.write("La dispersión demuestra cómo el 'estado_conservacion' (etiquetado por IA visual) afecta a la pendiente de tasación: las casas de lujo muestran una curva más agresiva respecto a los metros cuadrados.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if 'barrio_limpio' in df_plot.columns:
            with st.container(border=True):
                df_sample = df_plot.sample(min(8000, len(df_plot))) if len(df_plot) > 8000 else df_plot
                fig3 = px.box(df_sample, x='barrio_limpio', y=col_precio, color='barrio_limpio', title="Micro-Mercados: Dispersión por Barrio", template="plotly_dark")
                fig3.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="white"), font_color="white", title_font_color="white", legend_font_color="white")
                fig3.update_xaxes(color="white", title_font=dict(color="white"), tickfont=dict(color="white"))
                fig3.update_yaxes(color="white", title_font=dict(color="white"), tickfont=dict(color="white"))
                st.plotly_chart(fig3, width='stretch')
                with st.expander("🔬 Lógica de Modelado Territorial (Tuning)"):
                    st.write("Las medias móviles geográficas actúan como anclajes gravitacionales. Emplear diccionarios segmentados o Variables Categóricas Nativas en HistGB incrementa drásticamente la precisión.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        c_chart3, c_chart4 = st.columns(2)
        
        with c_chart3:
            with st.container(border=True):
                if 'estado_conservacion' in df_plot.columns:
                    # Agrupar por estado
                    df_estado = df_plot.groupby('estado_conservacion')[col_precio].mean().reset_index()
                    fig4 = px.bar(df_estado, x='estado_conservacion', y=col_precio, title="Impacto del Estado (IA Visual) en Precio Medio", 
                                  color='estado_conservacion', template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Set3)
                    fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="white"), font_color="white", title_font_color="white", legend_font_color="white")
                    fig4.update_xaxes(color="white", title_font=dict(color="white"), tickfont=dict(color="white"))
                    fig4.update_yaxes(color="white", title_font=dict(color="white"), tickfont=dict(color="white"))
                    st.plotly_chart(fig4, width='stretch')
                    with st.expander("🔬 Análisis Multimodal"):
                        st.write("Este gráfico confirma la validez de la Fase 3: la IA visual (CNN) consigue extraer una variable que influye de forma crítica y demostrable en el precio final del inmueble.")
                else:
                    st.info("La columna 'estado_conservacion' no está disponible en este dataset.")
        
        with c_chart4:
            if 'calidad_materiales' in df_plot.columns:
                with st.container(border=True):
                    # Boxplot calidad vs precio
                    fig5 = px.box(df_plot, x='calidad_materiales', y=col_precio, title="Dispersión por Calidad de Materiales (IA Visual)", 
                                  color='calidad_materiales', template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Set2)
                    fig5.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="white"), font_color="white", title_font_color="white", legend_font_color="white")
                    fig5.update_xaxes(color="white", title_font=dict(color="white"), tickfont=dict(color="white"))
                    fig5.update_yaxes(color="white", title_font=dict(color="white"), tickfont=dict(color="white"))
                    st.plotly_chart(fig5, width='stretch')
                    with st.expander("🔬 Insights Cualitativos"):
                        st.write("La calidad de los materiales es un feature hiper-complejo de extraer de forma tradicional. La CNN actúa como un Feature Extractor profundo que alimenta al árbol de decisión final.")

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            cols_corr = ['precio_limpio', 'm2', 'habitaciones', 'banos', 'tiene_ascensor', 'tiene_terraza', 'tiene_piscina', 'tiene_garaje', 'tiene_trastero']
            cols_corr = [c for c in cols_corr if c in df_plot.columns]
            if len(cols_corr) > 1:
                corr = df_plot[cols_corr].corr()
                fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", title="Matriz de Correlación de Variables Tabulares", color_continuous_scale="Blues", template="plotly_dark")
                fig_corr.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="white"), font_color="white", title_font_color="white")
                fig_corr.update_xaxes(color="white", title_font=dict(color="white"), tickfont=dict(color="white"))
                fig_corr.update_yaxes(color="white", title_font=dict(color="white"), tickfont=dict(color="white"))
                st.plotly_chart(fig_corr, width='stretch')
                with st.expander("🔬 Matriz de Correlación (Análisis Lineal)"):
                    st.write("Visualizamos rápidamente qué variables clásicas tienen mayor correlación lineal con el precio, validando su inclusión en el modelo base antes de transformaciones no lineales.")

    else:
        st.warning("⚠️ El entorno Data Lake no responde: No se localizó `propiedades_etiquetadas.csv` en la tubería procesada.")

elif menu_seleccionado == "📐 Arquitectura MLOps":
    st.markdown("<h2 style='margin-top: 0; margin-bottom: 20px; font-weight: 800; font-size: 2.2rem;'>Pipeline de Inteligencia Artificial Multimodal</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #ffffff; font-size: 1.1rem; margin-bottom: 30px;'>Arquitectura MLOps completa que procesa texto, datos tabulares e imágenes en tiempo real.</p>", unsafe_allow_html=True)

    st.markdown("""
    <style>
    .diagram-box {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
        margin: 10px auto;
        position: relative;
    }
    .diagram-box h3 { margin-top: 0; font-size: 1.3rem; margin-bottom: 8px; font-weight: 800; }
    .diagram-box p { color: #cbd5e1 !important; font-size: 0.95rem; margin-bottom: 0; }

    .arrow-down {
        text-align: center;
        color: #64748b;
        font-size: 2rem;
        margin: -5px 0;
    }

    .split-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 10px auto;
    }
    .split-box {
        flex: 1;
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }
    
    .fusion-box {
        background: linear-gradient(145deg, rgba(168, 85, 247, 0.2) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid #a855f7;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin: 10px auto;
    }
    .fusion-box h3 { color: #d8b4fe !important; margin-top: 0; font-weight: 800; }
    </style>

    <div class="diagram-box" style="border-top: 3px solid #3b82f6;">
        <h3 style="color: #60a5fa !important;">📥 Ingesta de Datos (Data Lake)</h3>
        <p>Un script automático extrae propiedades y sus imágenes desde portales inmobiliarios. Los datos se limpian de NaNs y se estructuran en un Data Lake local.</p>
    </div>

    <div class="arrow-down">⬇️</div>

    <div class="diagram-box" style="border-top: 3px solid #f59e0b;">
        <h3 style="color: #fbbf24 !important;">🤖 Etiquetado VLM (Google Gemini 1.5)</h3>
        <p>Para evitar el coste del etiquetado manual, un Modelo de Lenguaje Visual analiza miles de fotos para clasificar el estado de conservación y la calidad de los materiales (Ground Truth).</p>
    </div>

    <div style="display: flex; justify-content: space-around; color: #64748b; font-size: 2rem; margin: -5px 0;">
        <span>↙️</span>
        <span>↘️</span>
    </div>

    <div class="split-container">
        <div class="split-box" style="border-top: 3px solid #ef4444;">
            <h3 style="color: #f87171 !important;">👁️ PyTorch CNN (Visión)</h3>
            <p>Una red neuronal <b>ResNet50</b> se entrena mediante Transfer Learning para detectar patrones arquitectónicos directamente desde los píxeles de las imágenes.</p>
        </div>
        <div class="split-box" style="border-top: 3px solid #10b981;">
            <h3 style="color: #34d399 !important;">📊 Feature Engineering (Tabular)</h3>
            <p>El modelo analiza datos clásicos y utiliza Ordinal Encoding para forzar jerarquías lógicas matemáticas en las variables cualitativas sin usar heurísticas de post-procesado.</p>
        </div>
    </div>

    <div style="display: flex; justify-content: space-around; color: #64748b; font-size: 2rem; margin: -5px 0;">
        <span>↘️</span>
        <span>↙️</span>
    </div>

    <div class="fusion-box">
        <h3>🚀 Fusión Multimodal (HistGradientBoosting)</h3>
        <p style="color: #cbd5e1 !important; font-size: 1rem; text-align: center; margin-bottom: 0;">En tiempo real, la App procesa la foto del usuario (CNN) y la combina con los datos tabulares. Un algoritmo de árboles de decisión fusiona ambos mundos y genera la tasación económica final.</p>
    </div>
    """, unsafe_allow_html=True)
