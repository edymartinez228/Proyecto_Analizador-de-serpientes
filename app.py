"""
app.py
------
Aplicación Streamlit para reconocimiento de especies de serpientes.

Pipeline:
    1. Detección de presencia (¿hay una serpiente en la imagen?)
    2. Clasificación de especie + Grad-CAM (explicabilidad)
    3. Clasificación de veneno (venenosa / no venenosa)

Ejecutar localmente con:
    streamlit run app.py
"""

import os
import numpy as np
from PIL import Image
import streamlit as st

from utils.model_utils import (
    load_presence_model,
    load_species_model,
    load_venom_model,
    load_class_names,
    preprocess_for_torch,
    predict_presence,
    predict_species,
    predict_venom,
    GradCAM,
    overlay_heatmap,
)

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🐍 Reconocimiento de Serpientes",
    page_icon="🐍",
    layout="wide",
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

PRESENCE_MODEL_PATH = os.path.join(MODELS_DIR, "modelo_serpiente.pth")
SPECIES_MODEL_PATH = os.path.join(MODELS_DIR, "modelo_especie.pth")
VENOM_MODEL_PATH = os.path.join(MODELS_DIR, "modelo_veneno.weights.h5")
CLASS_NAMES_PATH = os.path.join(ASSETS_DIR, "class_names.json")

PRESENCE_THRESHOLD_DEFAULT = 0.60
VENOM_THRESHOLD_DEFAULT = 0.50


# ---------------------------------------------------------------------------
# Carga de modelos en caché (se ejecuta una sola vez por sesión de servidor)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo de presencia...")
def get_presence_model():
    return load_presence_model(PRESENCE_MODEL_PATH)


@st.cache_resource(show_spinner="Cargando modelo de especies...")
def get_species_model_and_labels():
    class_names = load_class_names(CLASS_NAMES_PATH)
    model = load_species_model(SPECIES_MODEL_PATH, num_classes=len(class_names))
    return model, class_names


@st.cache_resource(show_spinner="Cargando modelo de veneno...")
def get_venom_model():
    return load_venom_model(VENOM_MODEL_PATH)


# ---------------------------------------------------------------------------
# Barra lateral: entrada de imagen y parámetros
# ---------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("🐍 Panel de control")
    st.sidebar.markdown("Sube una imagen o captúrala con tu cámara para analizarla.")

    source = st.sidebar.radio("Fuente de la imagen", ["Subir archivo", "Usar cámara"])

    image_file = None
    if source == "Subir archivo":
        image_file = st.sidebar.file_uploader(
            "Selecciona una imagen", type=["jpg", "jpeg", "png"]
        )
    else:
        image_file = st.sidebar.camera_input("Captura una foto")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Parámetros del pipeline")
    presence_threshold = st.sidebar.slider(
        "Umbral de confianza (presencia)", 0.0, 1.0, PRESENCE_THRESHOLD_DEFAULT, 0.05
    )
    venom_threshold = st.sidebar.slider(
        "Umbral de confianza (veneno)", 0.0, 1.0, VENOM_THRESHOLD_DEFAULT, 0.05
    )
    show_gradcam = st.sidebar.checkbox("Mostrar mapa de calor Grad-CAM", value=True)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "⚠️ Esta herramienta es un apoyo informativo y **no reemplaza** el "
        "criterio de un especialista en herpetología ni atención médica de urgencia."
    )

    return image_file, presence_threshold, venom_threshold, show_gradcam


# ---------------------------------------------------------------------------
# Bloques de interfaz para cada etapa del pipeline
# ---------------------------------------------------------------------------
def render_presence_alert(confidence: float):
    st.error(
        f"🚫 **No se detectó ninguna serpiente en la imagen** "
        f"(confianza: {confidence:.1%}). Intenta con otra imagen más clara."
    )


def render_species_result(image_rgb, species_name, confidence, top3, cam_overlay, show_gradcam):
    st.success(f"✅ Serpiente detectada")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image_rgb, caption="Imagen original", use_container_width=True)

    with col2:
        if show_gradcam and cam_overlay is not None:
            st.image(cam_overlay, caption="Grad-CAM · zona de atención del modelo", use_container_width=True)
        else:
            st.image(image_rgb, caption="Imagen analizada", use_container_width=True)

    st.subheader(f"🔬 Especie identificada: **{species_name}**")
    st.metric("Confianza del modelo", f"{confidence:.1%}")

    st.markdown("**Top-3 predicciones:**")
    top3_data = {name: prob for name, prob in top3}
    st.bar_chart(top3_data)


def render_venom_card(is_venomous: bool, confidence: float):
    st.markdown("### 🧪 Resultado de toxicidad")
    if is_venomous:
        st.markdown(
            f"""
            <div style="background-color:#ffe5e5; padding:20px; border-radius:12px;
                        border:2px solid #d10000; text-align:center;">
                <h2 style="color:#d10000; margin:0;">⚠️ VENENOSA</h2>
                <p style="font-size:18px; margin:5px 0 0 0;">Confianza: {confidence:.1%}</p>
                <p style="margin-top:10px;">Evita manipular al animal y mantén distancia de seguridad.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#e6ffe9; padding:20px; border-radius:12px;
                        border:2px solid #1e8e3e; text-align:center;">
                <h2 style="color:#1e8e3e; margin:0;">✅ NO VENENOSA</h2>
                <p style="font-size:18px; margin:5px 0 0 0;">Confianza: {confidence:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "Este resultado es una estimación del modelo. Ante cualquier mordedura, "
        "acude siempre a un servicio médico de emergencia sin importar la predicción."
    )


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def run_pipeline(image_file, presence_threshold, venom_threshold, show_gradcam):
    pil_image = Image.open(image_file).convert("RGB")
    image_rgb = np.array(pil_image)

    presence_model = get_presence_model()
    species_model, class_names = get_species_model_and_labels()
    venom_model = get_venom_model()

    tensor = preprocess_for_torch(image_rgb)

    # --- Paso 1: presencia ---
    with st.spinner("Analizando presencia de serpiente..."):
        is_snake, presence_confidence = predict_presence(
            presence_model, tensor, threshold=presence_threshold
        )

    if not is_snake:
        render_presence_alert(presence_confidence)
        st.image(image_rgb, caption="Imagen analizada", use_container_width=True)
        return

    # --- Paso 2: especie + Grad-CAM ---
    with st.spinner("Identificando especie..."):
        species_name, species_confidence, top3, pred_idx = predict_species(
            species_model, tensor, class_names
        )

        cam_overlay = None
        if show_gradcam:
            target_layer = species_model.features[-1]
            gradcam = GradCAM(species_model, target_layer)
            cam_tensor = preprocess_for_torch(image_rgb).requires_grad_(True)
            cam = gradcam.generate(cam_tensor, class_idx=pred_idx)
            cam_overlay = overlay_heatmap(image_rgb, cam)

    render_species_result(
        image_rgb, species_name, species_confidence, top3, cam_overlay, show_gradcam
    )

    # --- Paso 3: veneno ---
    with st.spinner("Evaluando toxicidad..."):
        is_venomous, venom_confidence = predict_venom(
            venom_model, image_rgb, threshold=venom_threshold
        )

    render_venom_card(is_venomous, venom_confidence)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("🐍 Sistema de Reconocimiento de Especies de Serpientes")
    st.markdown(
        "Sube o captura una imagen para detectar la presencia de una serpiente, "
        "identificar su especie y evaluar si es venenosa."
    )

    image_file, presence_threshold, venom_threshold, show_gradcam = render_sidebar()

    if image_file is None:
        st.info("👈 Sube una imagen o usa la cámara desde la barra lateral para comenzar.")
        return

    run_pipeline(image_file, presence_threshold, venom_threshold, show_gradcam)


if __name__ == "__main__":
    main()
