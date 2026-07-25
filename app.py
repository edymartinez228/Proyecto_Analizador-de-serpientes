import streamlit as st
from PIL import Image
import numpy as np

# Importar funciones de utilidad
from utils.model_utils import (
    load_presence_model,
    load_venom_model,
    load_species_model,
    predict_presence,
    predict_venom,
    predict_species
)

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Analizador de Serpientes",
    page_icon="🐍",
    layout="centered",
    initial_sidebar_state="collapsed"  # Mantiene la barra lateral colapsada
)

# Estilos CSS opcionales para ocultar por completo el botón del sidebar
st.markdown("""
    <style>
        [data-testid="collapsedControl"] { display: none; }
    </style>
""", unsafe_allow_callbacks=True)

# --- CARGA DE MODELOS CON CACHÉ ---
@st.cache_resource
def get_presence_model():
    return load_presence_model("models/modelo_presencia.h5")

@st.cache_resource
def get_venom_model():
    return load_venom_model("models/modelo_veneno.weights.h5")

@st.cache_resource
def get_species_model():
    return load_species_model("models/modelo_especie.h5")

# --- INTERFAZ PRINCIPAL ---
st.title("🐍 Analizador Identificador de Serpientes")
st.write("Sube una imagen o toma una fotografía para detectar si hay una serpiente, evaluar si es venenosa e identificar su especie.")

st.divider()

# --- SELECCIÓN DE ENTRADA EN LA PANTALLA PRINCIPAL ---
col_mode, col_gradcam = st.columns([2, 1])

with col_mode:
    input_method = st.radio(
        "Selecciona el método de entrada de imagen:",
        ["📁 Subir Archivo", "📷 Usar Cámara"],
        horizontal=True
    )

with col_gradcam:
    show_gradcam = st.checkbox("Mostrar mapa de calor Grad-CAM", value=False)

image_file = None

if input_method == "📁 Subir Archivo":
    image_file = st.file_uploader(
        "Sube una imagen de una serpiente (JPG, PNG, JPEG)", 
        type=["jpg", "png", "jpeg"]
    )
else:
    image_file = st.camera_input("Toma una fotografía de la serpiente")

# Parámetros internos prefijados (ya no requeridos en la interfaz)
PRESENCE_THRESHOLD = 0.50
VENOM_THRESHOLD = 0.50

# --- PROCESAMIENTO Y EJECUCIÓN DEL PIPELINE ---
if image_file is not None:
    st.divider()
    st.subheader("🖼️ Imagen a Analizar")
    
    # Mostrar la imagen cargada
    image = Image.open(image_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    with st.spinner("Analizando la imagen con los modelos de IA..."):
        try:
            # 1. Cargar Modelos
            presence_model = get_presence_model()
            venom_model = get_venom_model()
            species_model = get_species_model()
            
            # Convertir imagen a formato procesable si es necesario
            image_np = np.array(image.convert("RGB"))

            # 2. Paso 1: Detección de Presencia
            has_snake, presence_prob = predict_presence(presence_model, image_np, PRESENCE_THRESHOLD)

            st.subheader("📊 Resultados del Análisis")

            if not has_snake:
                st.warning(f"⚠️ No se detectó ninguna serpiente en la imagen (Confianza de detección: {presence_prob*100:.1f}%).")
            else:
                st.success(f"✅ Serpiente detectada con una confianza del {presence_prob*100:.1f}%.")
                
                # 3. Paso 2: Análisis de Veneno
                is_venomous, venom_prob = predict_venom(venom_model, image_np, VENOM_THRESHOLD)
                
                col_venom, col_species = st.columns(2)
                
                with col_venom:
                    st.metric(
                        label="Peligrosidad / Veneno",
                        value="VENENOSA ⚠️" if is_venomous else "NO VENENOSA 🟢",
                        delta=f"{venom_prob*100:.1f}% probabilidad"
                    )
                
                # 4. Paso 3: Clasificación de Especie
                species_name, species_prob = predict_species(species_model, image_np)
                
                with col_species:
                    st.metric(
                        label="Especie Detectada",
                        value=species_name,
                        delta=f"{species_prob*100:.1f}% coincidencia"
                    )

                # Opción Grad-CAM si está habilitada
                if show_gradcam:
                    st.subheader("🔥 Mapa de Atención (Grad-CAM)")
                    st.info("Visualización de las regiones de la imagen en las que se enfocó la red neuronal.")
                    # Si tienes integrada la función de gradcam en model_utils, se invoca aquí

        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
