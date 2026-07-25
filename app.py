import os
import streamlit as st

# --- CARGA DE MODELOS CON CACHÉ Y RUTAS EXACTAS ---
@st.cache_resource
def get_presence_model():
    # Intenta buscar el archivo de presencia o usa el de veneno si es un pipeline unificado
    paths = [
        "models/modelo_presencia.keras",
        "models/modelo_presencia.weights.h5",
        "models/modelo_presencia.h5",
        "models/modelo_veneno.weights.h5"  # Fallback si usas el mismo modelo base
    ]
    selected_path = next((p for p in paths if os.path.exists(p)), None)
    
    if not selected_path:
        archivos_locales = os.listdir("models") if os.path.exists("models") else "Carpeta models no encontrada"
        raise FileNotFoundError(f"❌ No se encontró modelo de presencia. Archivos en models/: {archivos_locales}")
        
    return load_presence_model(selected_path)

@st.cache_resource
def get_venom_model():
    path = "models/modelo_veneno.weights.h5"
    if not os.path.exists(path):
        archivos_locales = os.listdir("models") if os.path.exists("models") else "Carpeta models no encontrada"
        raise FileNotFoundError(f"❌ No se encontró '{path}'. Archivos en models/: {archivos_locales}")
    return load_venom_model(path)

@st.cache_resource
def get_species_model():
    paths = [
        "models/modelo_especie.keras",
        "models/modelo_especie.weights.h5",
        "models/modelo_especie.h5"
    ]
    selected_path = next((p for p in paths if os.path.exists(p)), None)
    
    if not selected_path:
        archivos_locales = os.listdir("models") if os.path.exists("models") else "Carpeta models no encontrada"
        raise FileNotFoundError(f"❌ No se encontró modelo de especie. Archivos en models/: {archivos_locales}")
        
    return load_species_model(selected_path)
