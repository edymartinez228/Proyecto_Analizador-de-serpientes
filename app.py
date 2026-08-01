import os
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# Importar funciones de utilidad desde utils/model_utils.py
from utils.model_utils import (
    load_presence_model,
    load_venom_model,
    load_species_model,
    predict_presence,
    predict_venom,
    predict_species,
    cross_validate_venom_risk,
    generate_gradcam
)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Analizador de Serpientes",
    page_icon="🐍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Estilos CSS para ocultar el control de la barra lateral
st.markdown("""
    <style>
        [data-testid="collapsedControl"] { display: none; }
    </style>
""", unsafe_allow_html=True)


# --- CARGA DE MODELOS CON CACHÉ Y BÚSQUEDA FLEXIBLE ---

@st.cache_resource
def get_presence_model():
    paths = [
        "models/best.pt",
        "models/modelo_serpiente.pt",
        "models/modelo_presencia.pt",
        "models/detector_serpientes_best.pt"
    ]
    selected_path = next((p for p in paths if os.path.exists(p)), None)
    
    if not selected_path:
        existing = os.listdir("models") if os.path.exists("models") else "Carpeta 'models' no encontrada"
        raise FileNotFoundError(f"❌ No se encontró el modelo YOLOv8 de presencia (.pt). Archivos disponibles en 'models/': {existing}")
        
    return load_presence_model(selected_path)


@st.cache_resource
def get_venom_model():
    path = "models/modelo_veneno.weights.h5"
    if not os.path.exists(path):
        existing = os.listdir("models") if os.path.exists("models") else "Carpeta 'models' no encontrada"
        raise FileNotFoundError(f"❌ No se encontró '{path}'. Archivos disponibles en 'models/': {existing}")
        
    return load_venom_model(path)


@st.cache_resource
def get_species_model():
    paths = [
        "modelo_especie_out/modelo_especie.pth",
        "models/modelo_especie.pth",
        "models/modelo_especie.pt",
        "models/modelo_especie_efficientnet.pth"
    ]
    selected_path = next((p for p in paths if os.path.exists(p)), None)
    
    if not selected_path:
        existing = os.listdir("models") if os.path.exists("models") else "Carpeta 'models' no encontrada"
        raise FileNotFoundError(f"❌ No se encontró el modelo PyTorch de especies (.pth/.pt). Archivos revisados: {paths}")
        
    return load_species_model(selected_path)


# --- INTERFAZ PRINCIPAL ---

st.title("🐍 Analizador e Identificador de Serpientes")
st.write("Sube una imagen o toma una fotografía para detectar si hay una serpiente, evaluar si es venenosa e identificar su especie exacta.")

st.divider()

# Opciones de control
col_mode, col_gradcam = st.columns([2, 1])

with col_mode:
    input_method = st.radio(
        "Selecciona el método de entrada de imagen:",
        ["📁 Subir Archivo", "📷 Usar Cámara"],
        horizontal=True
    )

with col_gradcam:
    show_gradcam = st.checkbox("Mostrar mapa Grad-CAM", value=False)

image_file = None

if input_method == "📁 Subir Archivo":
    image_file = st.file_uploader(
        "Sube una imagen de una serpiente (JPG, PNG, JPEG)", 
        type=["jpg", "png", "jpeg"]
    )
else:
    image_file = st.camera_input("Toma una fotografía de la serpiente")

# Umbrales de confianza configurables
PRESENCE_THRESHOLD = 0.60
VENOM_THRESHOLD = 0.50

# --- PROCESAMIENTO Y EJECUCIÓN DEL PIPELINE ---

if image_file is not None:
    st.divider()
    st.subheader("🖼️ Imagen a Analizar")
    
    # 1. Cargar imagen y normalizar orientación EXIF (típico en dispositivos móviles)
    image = Image.open(image_file).convert("RGB")
    image = ImageOps.exif_transpose(image)
    
    # Renderizar en la interfaz respetando el tamaño de pantalla
    st.image(image, caption="Imagen cargada", use_container_width=True)
    
    with st.spinner("Analizando la imagen con los modelos de IA..."):
        try:
            # 2. Convertir a NumPy en espacio RGB puro
            image_np = np.array(image)

            # 3. Cargar modelo de presencia y evaluar
            presence_model = get_presence_model()
            has_snake, presence_prob = predict_presence(presence_model, image_np, PRESENCE_THRESHOLD)

            st.subheader("📊 Resultados del Análisis")

            if not has_snake:
                st.warning(f"⚠️ No se detectó ninguna serpiente en la imagen (Confianza de coincidencia: {presence_prob*100:.1f}%).")
                st.info("💡 **Consejo:** Acércate un poco más al objetivo o asegúrate de enfocar bien al reptil y contar con buena iluminación.")
            
            else:
                st.success(f"✅ Serpiente detectada con una confianza del {presence_prob*100:.1f}%.")
                
                # Cargar modelos de clasificación
                venom_model = get_venom_model()
                species_model = get_species_model()

                # 4. Análisis de Veneno
                is_venomous, venom_prob = predict_venom(venom_model, image_np, VENOM_THRESHOLD)
                
                # 5. Clasificación de Especie con Ranking Top-K
                species_name, species_prob, top_predictions = predict_species(species_model, image_np, top_k=3)
                
                # 6. Métricas Principales en Pantalla
                col_venom, col_species = st.columns(2)
                
                with col_venom:
                    st.metric(
                        label="Peligrosidad / Veneno",
                        value="VENENOSA ⚠️" if is_venomous else "NO VENENOSA 🟢",
                        delta=f"{venom_prob*100:.1f}% probabilidad"
                    )
                
                with col_species:
                    st.metric(
                        label="Especie Detectada",
                        value=species_name,
                        delta=f"{species_prob*100:.1f}% coincidencia"
                    )

                # 7. Validación Cruzada de Seguridad (Especie vs Veneno)
                top_raw_name = top_predictions[0]["raw_name"]
                safety_check = cross_validate_venom_risk(top_raw_name, is_venomous)
                if safety_check["warning_triggered"]:
                    st.warning(safety_check["warning_message"])

                # 8. Desglose del Top de Predicciones de Especie
                with st.expander("📊 Ver Top de Coincidencias de Especies", expanded=True):
                    st.write("Ranking de las especies más probables identificadas por el modelo:")
                    for idx, pred in enumerate(top_predictions, 1):
                        prob_percent = pred['probability'] * 100
                        st.write(f"**{idx}. {pred['spanish_name']}** (`{pred['raw_name']}`) — {prob_percent:.2f}%")
                        st.progress(min(pred['probability'], 1.0))

                # 9. Mapa de Atención (Grad-CAM)
                if show_gradcam:
                    st.divider()
                    st.subheader("🔥 Mapa de Atención (Grad-CAM)")
                    st.info("Visualización de las regiones clave en las que se enfocó el modelo para determinar la especie.")
                
                    with st.spinner("Generando mapa de calor Grad-CAM..."):
                        cam_image = generate_gradcam(species_model, image_np)
                
                        col_orig, col_cam = st.columns(2)
                        with col_orig:
                            st.image(image, caption="Imagen Original", use_container_width=True)
                        with col_cam:
                            st.image(cam_image, caption="Mapa de Calor (Atención)", use_container_width=True)

        except Exception as e:
            st.error(f"Ocurrió un error durante el procesamiento: {str(e)}")
