import os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# Importar funciones de utilidad desde utils/model_utils.py
from utils.model_utils import (
    cross_validate_venom_risk,
    generate_gradcam,
    load_species_model,
    load_venom_model,
    predict_species,
    predict_venom,
)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Snakely | Analizador de Serpientes",
    page_icon="🐍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- ESTILOS CSS CUSTOM (MODERNO Y ELEGANTE) ---
st.markdown(
    """
    <style>
        /* Ocultar barra lateral */
        [data-testid="collapsedControl"] { display: none; }
        
        .main {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Hero Header */
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #10B981 0%, #059669 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            font-size: 1rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }

        /* Custom Cards para Métricas */
        .metric-card {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(229, 231, 235, 0.3);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
            backdrop-filter: blur(8px);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .metric-card.danger {
            border-left: 5px solid #EF4444;
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.05) 0%, rgba(255, 255, 255, 0) 100%);
        }
        .metric-card.safe {
            border-left: 5px solid #10B981;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(255, 255, 255, 0) 100%);
        }
        
        .card-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #6B7280;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .card-value {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .card-badge {
            display: inline-block;
            font-size: 0.8rem;
            padding: 3px 10px;
            border-radius: 20px;
            font-weight: 600;
        }
        .badge-danger { background-color: #FEE2E2; color: #991B1B; }
        .badge-safe { background-color: #D1FAE5; color: #065F46; }
        .badge-info { background-color: #E0F2FE; color: #075985; }

        /* Contenedores envolventes de Recomendaciones */
        .recom-container-do {
            background-color: rgba(16, 185, 129, 0.05);
            border: 2px solid #10B981;
            border-radius: 12px;
            padding: 20px;
            height: 100%;
        }
        .recom-container-dont {
            background-color: rgba(239, 68, 68, 0.05);
            border: 2px solid #EF4444;
            border-radius: 12px;
            padding: 20px;
            height: 100%;
        }
        .recom-title-do {
            color: #065F46;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 12px;
        }
        .recom-title-dont {
            color: #991B1B;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 12px;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# --- CARGA DE MODELOS CON CACHÉ ---

@st.cache_resource
def get_venom_model():
    path = "models/modelo_veneno.weights.h5"
    if not os.path.exists(path):
        existing = (
            os.listdir("models")
            if os.path.exists("models")
            else "Carpeta 'models' no encontrada"
        )
        raise FileNotFoundError(
            f"❌ No se encontró '{path}'. Archivos en 'models/': {existing}"
        )
    return load_venom_model(path)


@st.cache_resource
def get_species_model():
    path = "models/modelo_especie.pth"
    if not os.path.exists(path):
        existing = (
            os.listdir("models")
            if os.path.exists("models")
            else "Carpeta 'models' no encontrada"
        )
        raise FileNotFoundError(
            f"❌ No se encontró '{path}'. Archivos en 'models/': {existing}"
        )
    return load_species_model(path)


# --- INTERFAZ PRINCIPAL ---

st.markdown('<div class="hero-title">🐍 Snakely AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Plataforma inteligente para la identificación de especies y diagnóstico de peligrosidad en ofidios.</div>',
    unsafe_allow_html=True,
)

col_upload, col_gradcam = st.columns([2.2, 1.2], gap="medium")

with col_upload:
    image_file = st.file_uploader(
        "Sube una imagen (JPG, PNG)", type=["jpg", "png", "jpeg"]
    )

with col_gradcam:
    st.write(" ")
    st.write(" ")
    show_gradcam = st.checkbox("🔥 Visualizar Grad-CAM", value=False)

VENOM_THRESHOLD = 0.50

# --- PROCESAMIENTO Y EJECUCIÓN DEL PIPELINE ---

if image_file is not None:
    st.markdown("---")

    # 1. Mostrar la imagen cargada
    image = Image.open(image_file).convert("RGB")
    image = ImageOps.exif_transpose(image)

    col_img_left, col_img_center, col_img_right = st.columns([1, 2, 1])
    with col_img_center:
        st.image(
            image,
            caption="Imagen de la muestra",
            use_container_width=True,
        )

    # Inferencia con redes neuronales
    with st.status("🔮 Analizando imagen con redes neuronales...", expanded=True) as status:
        try:
            image_np = np.array(image)

            status.write("🧠 Cargando arquitecturas de IA...")
            venom_model = get_venom_model()
            species_model = get_species_model()

            status.write("⚡ Evaluando patrones de toxicidad...")
            is_venomous, venom_prob, venom_recommendations = predict_venom(
                venom_model, image_np, VENOM_THRESHOLD
            )

            status.write("🔍 Identificando taxón y especie...")
            species_name, species_prob, top_predictions = predict_species(
                species_model, image_np, top_k=5
            )

            status.update(
                label="✅ Análisis completado con éxito",
                state="complete",
                expanded=False,
            )

        except Exception as e:
            status.update(label="❌ Error durante la inferencia", state="error")
            st.error(f"Detalle del error: {str(e)}")
            st.stop()

    st.write("")

    # --- 2. LÓGICA DE DETECCIÓN DE CONTRADICCIÓN ENTRE MODELOS ---
    top_raw_name = top_predictions[0]["raw_name"]
    species_lower = top_raw_name.lower()

    # Importar o evaluar si pertenece a la lista de palabras de especies venenosas
    from utils.model_utils import VENOMOUS_KEYWORDS
    is_species_known_venomous = any(kw in species_lower for kw in VENOMOUS_KEYWORDS)

    # CONTRADICCIÓN TIPO A: Especie Taxonómicamente NO Venenosa + Modelo de Veneno indica VENENOSA
    is_false_positive_risk = (not is_species_known_venomous) and is_venomous

    # CONTRADICCIÓN TIPO B: Especie Taxonómicamente VENENOSA + Modelo de Veneno indica NO VENENOSA
    is_false_negative_risk = is_species_known_venomous and (not is_venomous)

    has_contradiction = is_false_positive_risk or is_false_negative_risk

    # EL MENSAJE DE ADVERTENCIA DINÁMICO (Aplica a CUALQUIER especie)
    if has_contradiction:
        if is_false_positive_risk:
            st.error(
                f"🚨 **ADVERTENCIA: MODELOS CONTRADICTORIOS (POSIBLE FALSO POSITIVO DE VENENO)**\n\n"
                f"Se ha detectado una discrepancia: La especie fue identificada como **{species_name}** "
                f"(especie clasificada biológicamente como **NO VENENOSA**), pero el detector de toxicidad "
                f"registró una probabilidad del **{venom_prob*100:.1f}%** de características de veneno.\n\n"
                f"📌 **Criterio de Precaución Extrema:** Dado que la morfología, ángulos de luz o patrones del ejemplar pueden "
                f"haber causado una confusión en el modelo taxonómico o en el de veneno, el sistema prioriza la seguridad.\n\n"
                f"👉 **Recomendación:** No te confíes. Trata al ejemplar como potencialmente peligroso y mantén la distancia."
            )
        elif is_false_negative_risk:
            st.warning(
                f"⚠️ **ADVERTENCIA DE PRECAUCIÓN: MODELOS CONTRADICTORIOS**\n\n"
                f"El detector de veneno registró un nivel bajo ({venom_prob*100:.1f}%), pero la especie identificada "
                f"es **{species_name}**, la cual pertenece a un grupo **POTENCIALMENTE VENENOSO**.\n\n"
                f"👉 **Recomendación:** Se aplican protocolos de seguridad de forma preventiva."
            )
        st.write("")
    # --- 3. RESULTADOS DESTACADOS (TARJETAS) ---
    col_venom, col_species = st.columns(2, gap="medium")

    # Tarjeta de Veneno
    with col_venom:
        card_class = "danger" if is_venomous else "safe"
        badge_class = "badge-danger" if is_venomous else "badge-safe"
        status_text = "VENENOSA ⚠️" if is_venomous else "NO VENENOSA 🟢"

        st.markdown(
            f"""
            <div class="metric-card {card_class}">
                <div class="card-label">Diagnóstico de Peligrosidad (Prioritario)</div>
                <div class="card-value">{status_text}</div>
                <span class="card-badge {badge_class}">{venom_prob*100:.1f}% Indicador de Veneno</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tarjeta de Especie
    with col_species:
        st.markdown(
            f"""
            <div class="metric-card safe" style="border-left: 5px solid #0284C7;">
                <div class="card-label">Especie Predominante</div>
                <div class="card-value" style="font-size: 1.2rem;">{species_name}</div>
                <span class="card-badge badge-info">{species_prob*100:.1f}% Coincidencia</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # --- 4. RECOMENDACIONES EN CUADROS VERDE Y ROJO ENVOLVENTES ---
    recommendations_to_display = (
        safety_check["recommendations"]
        if safety_check.get("warning_triggered") and safety_check.get("recommendations")
        else venom_recommendations
    )

    if recommendations_to_display:
        st.write("")
        st.markdown("### 🚑 Protocolo de Primeros Auxilios")

        col_do, col_dont = st.columns(2, gap="medium")

        # Cuadro Verde Envolvente (Acciones Recomendadas)
        with col_do:
            do_items_html = "".join(
                [f"<li style='margin-bottom: 6px;'>{item}</li>" for item in recommendations_to_display.get("que_hacer", [])]
            )
            st.markdown(
                f"""
                <div class="recom-container-do">
                    <div class="recom-title-do">✅ Qué HACER</div>
                    <ul style="margin: 0; padding-left: 20px; color: #047857;">
                        {do_items_html}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Cuadro Rojo Envolvente (Acciones Prohibidas)
        with col_dont:
            dont_items_html = "".join(
                [f"<li style='margin-bottom: 6px;'>{item}</li>" for item in recommendations_to_display.get("nunca_hacer", [])]
            )
            st.markdown(
                f"""
                <div class="recom-container-dont">
                    <div class="recom-title-dont">❌ Lo que NUNCA debes hacer</div>
                    <ul style="margin: 0; padding-left: 20px; color: #B91C1C;">
                        {dont_items_html}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.write("")

    # --- 5. PESTAÑAS (TABS) PARA DETALLES AVANZADOS ---
    tab_rankings, tab_gradcam = st.tabs(
        ["📊 Ranking de Especies", "🔥 Mapa de Atención (Grad-CAM)"]
    )

    with tab_rankings:
        st.caption(
            "Distribución de probabilidad de las 5 especies más afines detectadas por la red:"
        )
        for idx, pred in enumerate(top_predictions, 1):
            prob_percent = pred["probability"] * 100
            st.write(
                f"**{idx}. {pred['spanish_name']}** _({pred['raw_name']})_"
            )
            st.progress(
                min(pred["probability"], 1.0), text=f"{prob_percent:.2f}% Coincidencia"
            )

    with tab_gradcam:
        if show_gradcam:
            st.caption(
                "Las regiones más cálidas (rojas/amarillas) indican en qué partes de la imagen se basó el modelo para clasificar la especie."
            )
            with st.spinner("Generando interpretabilidad visual..."):
                cam_image = generate_gradcam(species_model, image_np)

                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    st.image(
                        image,
                        caption="Original",
                        use_container_width=True,
                    )
                with c2:
                    st.image(
                        cam_image,
                        caption="Enfoque de la IA",
                        use_container_width=True,
                    )
        else:
            st.info(
                "💡 Activa la casilla **'Visualizar Grad-CAM'** en la parte superior para desplegar la interpretabilidad visual del modelo."
            )
