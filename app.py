"""
===============================================================================
 SNAKELY AI  ·  Analizador de ofidios
 Identificación taxonómica + evaluación de riesgo toxicológico
===============================================================================

Archivo único y autosuficiente. Contiene:
  · bootstrap del tema nativo de Streamlit
  · sistema de diseño (tokens + CSS)
  · librería de iconos SVG inline
  · componentes de UI reutilizables
  · pipeline de inferencia e interfaz

Ejecutar:  streamlit run app.py

Dependencias del proyecto:  utils/model_utils.py  ·  models/*.h5|*.pth
===============================================================================
"""

from __future__ import annotations

import base64
import io
import os
import pathlib
import re
import time
from datetime import datetime

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

from utils.model_utils import (
    VENOMOUS_KEYWORDS,
    cross_validate_venom_risk,
    generate_gradcam,
    load_species_model,
    load_venom_model,
    predict_species,
    predict_venom,
)

# =============================================================================
#  1 · BOOTSTRAP DEL TEMA NATIVO
#     Streamlit lee .streamlit/config.toml al arrancar el proceso. Se genera si
#     no existe para que este archivo siga siendo autosuficiente.
# =============================================================================

_CONFIG_TOML = """[theme]
base = "dark"
primaryColor = "#10E098"
backgroundColor = "#0A0E14"
secondaryBackgroundColor = "#111721"
textColor = "#E6EDF7"
font = "sans serif"

[client]
toolbarMode = "minimal"
showErrorDetails = true

[server]
maxUploadSize = 25
"""


def _bootstrap_theme() -> None:
    try:
        cfg = pathlib.Path(".streamlit") / "config.toml"
        if not cfg.exists():
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_text(_CONFIG_TOML, encoding="utf-8")
    except OSError:
        pass  # entorno de solo lectura: el CSS propio ya cubre la apariencia


_bootstrap_theme()

st.set_page_config(
    page_title="Snakely · Analizador de Ofidios",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
#  2 · SISTEMA DE DISEÑO
# =============================================================================

THEME = {
    "bg": "#080B11",
    "surface": "#10161F",
    "surface_2": "#161E2A",
    "surface_3": "#1C2634",
    "border": "rgba(148, 163, 184, 0.13)",
    "border_hi": "rgba(148, 163, 184, 0.24)",
    "text": "#E8EFF9",
    "muted": "#93A1B5",
    "dim": "#64748B",
    "accent": "#10E098",
    "accent_2": "#6EE7F9",
    "danger": "#FF4D5E",
    "warning": "#FFB020",
    "info": "#3B9EFF",
    "violet": "#A78BFA",
}

# Niveles de riesgo. Los cortes se derivan del umbral activo para que la escala
# siga siendo coherente cuando el usuario mueve el control de sensibilidad.
_TIER_META = [
    ("MÍNIMO", THEME["accent"], "Sin indicios morfológicos de toxicidad"),
    ("BAJO", "#8CE99A", "Indicios débiles, no concluyentes"),
    ("ELEVADO", THEME["warning"], "Rasgos compatibles con especie venenosa"),
    ("CRÍTICO", THEME["danger"], "Alta compatibilidad con especie venenosa"),
]


def tier_cuts(threshold: float) -> list[float]:
    return [0.0, threshold * 0.5, threshold, threshold + (1 - threshold) * 0.55]


def risk_tier(prob: float, threshold: float) -> tuple[str, str, str]:
    result = _TIER_META[0]
    for cut, meta in zip(tier_cuts(threshold), _TIER_META):
        if prob >= cut:
            result = meta
    return result


_ROOT_VARS = f"""
:root {{
    --bg: {THEME["bg"]};
    --surface: {THEME["surface"]};
    --surface-2: {THEME["surface_2"]};
    --surface-3: {THEME["surface_3"]};
    --border: {THEME["border"]};
    --border-hi: {THEME["border_hi"]};
    --text: {THEME["text"]};
    --muted: {THEME["muted"]};
    --dim: {THEME["dim"]};
    --accent: {THEME["accent"]};
    --accent-2: {THEME["accent_2"]};
    --danger: {THEME["danger"]};
    --warning: {THEME["warning"]};
    --info: {THEME["info"]};
    --violet: {THEME["violet"]};
    --radius: 16px;
    --radius-sm: 11px;
    --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
}}
"""

# CSS en cadena plana: sin llaves escapadas, mucho menos frágil de mantener.
_CSS = """
/* ---------------------------------------------------------------- base --- */
[data-testid="collapsedControl"], #MainMenu, footer, header { display: none !important; }

html, body, .stApp, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}

.stApp {
    background:
        radial-gradient(1200px 560px at 8% -12%, rgba(16, 224, 152, 0.11), transparent 62%),
        radial-gradient(1000px 480px at 100% -4%, rgba(59, 158, 255, 0.09), transparent 58%),
        radial-gradient(760px 420px at 50% 108%, rgba(167, 139, 250, 0.06), transparent 60%),
        var(--bg);
    color: var(--text);
}
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; }

.block-container { max-width: 1240px; padding: 2.4rem 2rem 5rem; }

h1, h2, h3, h4, h5, h6 { color: var(--text) !important; letter-spacing: -0.02em; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { color: var(--text); }

/* ---------------------------------------------------------- animaciones --- */
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes sweep { 0% { background-position: -180% 0; } 100% { background-position: 280% 0; } }
@keyframes pulse-ring { 0% { box-shadow: 0 0 0 0 rgba(255,77,94,0.35); } 70% { box-shadow: 0 0 0 12px rgba(255,77,94,0); } 100% { box-shadow: 0 0 0 0 rgba(255,77,94,0); } }
@keyframes dash { from { stroke-dashoffset: var(--circ); } }

.rise   { animation: rise .5s cubic-bezier(.22,1,.36,1) both; }
.rise-1 { animation-delay: .05s; }
.rise-2 { animation-delay: .12s; }
.rise-3 { animation-delay: .19s; }
.rise-4 { animation-delay: .26s; }

/* ---------------------------------------------------------------- hero --- */
.hero {
    position: relative; overflow: hidden;
    border: 1px solid var(--border);
    border-radius: 22px;
    background:
        linear-gradient(155deg, rgba(16,224,152,0.09) 0%, rgba(16,22,31,0.9) 42%, rgba(16,22,31,0.98) 100%);
    padding: 34px 38px;
    margin-bottom: 16px;
}
.hero::before {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(148,163,184,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148,163,184,0.05) 1px, transparent 1px);
    background-size: 46px 46px;
    mask-image: radial-gradient(560px 300px at 88% 12%, #000, transparent 72%);
    -webkit-mask-image: radial-gradient(560px 300px at 88% 12%, #000, transparent 72%);
}
.hero::after {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(16,224,152,0.55), rgba(110,231,249,0.35), transparent);
    background-size: 200% 100%;
    animation: sweep 5.5s linear infinite;
}
.hero > * { position: relative; }

.eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--accent);
    background: rgba(16,224,152,0.08);
    border: 1px solid rgba(16,224,152,0.26);
    padding: 6px 13px; border-radius: 999px; margin-bottom: 18px;
}
.hero-title {
    font-size: clamp(2.1rem, 4.6vw, 3rem); font-weight: 800;
    letter-spacing: -0.042em; line-height: 1.02; margin: 0 0 12px;
}
.hero-title em {
    font-style: normal;
    background: linear-gradient(96deg, var(--accent) 0%, var(--accent-2) 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { font-size: 1.03rem; color: var(--muted); max-width: 660px; line-height: 1.65; margin: 0; }

.hero-meta {
    display: flex; flex-wrap: wrap; gap: 10px;
    margin-top: 24px; padding-top: 20px; border-top: 1px solid var(--border);
}
.chip {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 0.77rem; font-weight: 500; color: var(--muted);
    background: rgba(148,163,184,0.06);
    border: 1px solid var(--border);
    padding: 7px 13px; border-radius: 999px;
}
.chip b { color: var(--text); font-weight: 600; }

/* ------------------------------------------------------------- advertencia --- */
/* Banner de aviso legal. Va justo tras el hero, visible antes de cualquier
   resultado, para que el usuario lo lea desde el primer segundo. */
.advisory {
    display: flex; gap: 16px; align-items: flex-start;
    border-radius: var(--radius); padding: 18px 22px; margin-bottom: 26px;
    border: 1px solid rgba(255,176,32,0.32);
    background: linear-gradient(120deg, rgba(255,176,32,0.09) 0%, rgba(16,22,31,0.4) 60%);
}
.advisory-ico {
    flex-shrink: 0; width: 34px; height: 34px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(255,176,32,0.14);
}
.advisory-t {
    font-size: 0.68rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase;
    color: #FFCC70; margin-bottom: 6px;
}
.advisory-d { font-size: 0.87rem; line-height: 1.62; color: var(--muted); }
.advisory-d b { color: var(--text); font-weight: 600; }

/* ------------------------------------------------------------ secciones --- */
.section { display: flex; align-items: center; gap: 11px; margin: 38px 0 16px; }
.section .ttl { font-size: 1.06rem; font-weight: 700; letter-spacing: -0.018em; color: var(--text); }
.section .num {
    font-family: var(--mono); font-size: 0.68rem; font-weight: 600;
    color: var(--dim); background: rgba(148,163,184,0.08);
    border: 1px solid var(--border); padding: 2px 7px; border-radius: 6px;
}
.section .rule { flex: 1; height: 1px; background: linear-gradient(90deg, var(--border-hi), transparent); }

/* ------------------------------------------------------------- tarjetas --- */
.card {
    position: relative; overflow: hidden; height: 100%;
    background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 26px;
    transition: border-color .22s ease, transform .22s ease;
}
.card:hover { border-color: var(--border-hi); }
.card::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--info); }
.card.t-danger::before { background: var(--danger); }
.card.t-safe::before   { background: linear-gradient(180deg, var(--accent), #0B9E6E); }
.card.t-warn::before   { background: var(--warning); }
.card.t-danger { background: linear-gradient(140deg, rgba(255,77,94,0.09) 0%, var(--surface) 62%); }
.card.t-safe   { background: linear-gradient(140deg, rgba(16,224,152,0.08) 0%, var(--surface) 62%); }
.card.t-info   { background: linear-gradient(140deg, rgba(59,158,255,0.08) 0%, var(--surface) 62%); }

.card-label {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--dim); margin-bottom: 14px;
}
.card-value { font-size: 1.34rem; font-weight: 800; letter-spacing: -0.028em; line-height: 1.18; color: var(--text); }
.card-sub { font-size: 0.81rem; color: var(--dim); font-style: italic; margin-top: 4px; }

.badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: var(--mono); font-size: 0.72rem; font-weight: 600;
    padding: 5px 11px; border-radius: 999px; margin-top: 12px;
}
.badge-danger { background: rgba(255,77,94,0.13); color: #FF95A0; border: 1px solid rgba(255,77,94,0.3); }
.badge-safe   { background: rgba(16,224,152,0.12); color: #63EDBD; border: 1px solid rgba(16,224,152,0.3); }
.badge-warn   { background: rgba(255,176,32,0.13); color: #FFCC70; border: 1px solid rgba(255,176,32,0.3); }
.badge-info   { background: rgba(59,158,255,0.12); color: #8AC7FF; border: 1px solid rgba(59,158,255,0.3); }

/* ----------------------------------------------------------- donut svg --- */
.donut-wrap { display: flex; align-items: center; gap: 24px; }
.donut { position: relative; flex-shrink: 0; }
.donut svg { display: block; transform: rotate(-90deg); }
.donut .arc { animation: dash 1.1s cubic-bezier(.22,1,.36,1) both; }
.donut-center {
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 1px;
}
.donut-pct { font-family: var(--mono); font-size: 1.42rem; font-weight: 700; letter-spacing: -0.03em; }
.donut-cap { font-size: 0.56rem; font-weight: 700; letter-spacing: 0.16em; color: var(--dim); }
.donut-side { min-width: 0; }

/* --------------------------------------------------------- escala tiers --- */
.tiers { display: flex; gap: 4px; margin-top: 16px; }
.tier { flex: 1; }
.tier-bar { height: 4px; border-radius: 999px; background: rgba(148,163,184,0.14); transition: all .3s ease; }
.tier.on .tier-bar { box-shadow: 0 0 10px currentColor; }
.tier-lbl {
    font-family: var(--mono); font-size: 0.56rem; font-weight: 600; letter-spacing: 0.08em;
    color: var(--dim); margin-top: 6px; text-align: center;
}
.tier.on .tier-lbl { color: var(--text); }

/* -------------------------------------------------------------- alertas --- */
.alert {
    display: flex; gap: 16px; align-items: flex-start;
    border-radius: var(--radius); padding: 22px 24px; border: 1px solid;
}
.alert-critical { background: rgba(255,77,94,0.07); border-color: rgba(255,77,94,0.34); animation: pulse-ring 2.6s ease-out 3; }
.alert-warn     { background: rgba(255,176,32,0.07); border-color: rgba(255,176,32,0.34); }
.alert-ico {
    flex-shrink: 0; width: 38px; height: 38px; border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
}
.alert-critical .alert-ico { background: rgba(255,77,94,0.13); }
.alert-warn .alert-ico { background: rgba(255,176,32,0.13); }
.alert-title { font-size: 0.7rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 10px; }
.alert-critical .alert-title { color: #FF95A0; }
.alert-warn .alert-title { color: #FFCC70; }
.alert-body { font-size: 0.9rem; line-height: 1.68; color: var(--muted); }
.alert-body b { color: var(--text); font-weight: 600; }
.alert-body p { margin: 0 0 10px; }
.alert-body p:last-child { margin: 0; }

/* ------------------------------------------------------------- consenso --- */
.consensus { display: flex; flex-direction: column; gap: 2px; }
.sig {
    display: flex; align-items: center; gap: 13px; padding: 13px 4px;
    border-bottom: 1px solid rgba(148,163,184,0.07);
}
.sig:last-child { border-bottom: none; }
.sig-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.sig-name { flex: 1; font-size: 0.86rem; font-weight: 500; color: var(--muted); }
.sig-val { font-family: var(--mono); font-size: 0.78rem; font-weight: 600; }

/* ------------------------------------------------------------ protocolo --- */
.proto { border-radius: var(--radius); padding: 24px 26px; height: 100%; border: 1px solid; }
.proto-do   { background: rgba(16,224,152,0.045); border-color: rgba(16,224,152,0.28); }
.proto-dont { background: rgba(255,77,94,0.045); border-color: rgba(255,77,94,0.28); }
.proto-head {
    display: flex; align-items: center; gap: 10px;
    font-size: 0.95rem; font-weight: 700; letter-spacing: -0.012em;
    padding-bottom: 14px; margin-bottom: 6px; border-bottom: 1px solid var(--border);
}
.proto-do .proto-head { color: #63EDBD; }
.proto-dont .proto-head { color: #FF95A0; }
.proto ul { list-style: none; margin: 0; padding: 0; }
.proto li {
    display: flex; gap: 11px; align-items: flex-start;
    font-size: 0.885rem; line-height: 1.6; color: var(--muted); padding: 10px 0;
}
.proto li + li { border-top: 1px solid rgba(148,163,184,0.07); }
.proto li .bl { margin-top: 2px; flex-shrink: 0; }

/* -------------------------------------------------------------- ranking --- */
.rank {
    display: flex; align-items: center; gap: 18px;
    padding: 14px 18px; border-radius: var(--radius-sm);
    background: var(--surface); border: 1px solid var(--border); margin-bottom: 8px;
    transition: border-color .2s ease, background .2s ease;
}
.rank:hover { border-color: var(--border-hi); background: var(--surface-2); }
.rank.lead { border-color: rgba(59,158,255,0.34); background: rgba(59,158,255,0.055); }
.rank-idx {
    width: 28px; height: 28px; border-radius: 9px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 0.72rem; font-weight: 700;
    color: var(--dim); background: rgba(148,163,184,0.09);
}
.rank.lead .rank-idx { background: rgba(59,158,255,0.2); color: #8AC7FF; }
.rank-main { flex: 1; min-width: 0; }
.rank-name { font-size: 0.92rem; font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rank-lat { font-size: 0.755rem; color: var(--dim); font-style: italic; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rank-viz { width: 32%; flex-shrink: 0; }
.rank-bar { height: 5px; border-radius: 999px; background: rgba(148,163,184,0.13); overflow: hidden; }
.rank-bar > i { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--info), var(--accent-2)); }
.rank.lead .rank-bar > i { background: linear-gradient(90deg, var(--info), var(--accent-2)); }
.rank-pct { font-family: var(--mono); font-size: 0.76rem; font-weight: 600; color: var(--muted); text-align: right; margin-top: 7px; }

/* --------------------------------------------------------- empty state --- */
.feat {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 24px; height: 100%;
    transition: border-color .22s ease, transform .22s ease;
}
.feat:hover { border-color: var(--border-hi); transform: translateY(-2px); }
.feat-ico {
    width: 40px; height: 40px; border-radius: 12px; margin-bottom: 16px;
    display: flex; align-items: center; justify-content: center;
}
.feat-t { font-size: 0.96rem; font-weight: 700; color: var(--text); margin-bottom: 7px; letter-spacing: -0.015em; }
.feat-d { font-size: 0.855rem; line-height: 1.6; color: var(--muted); }

.steps { display: flex; flex-direction: column; gap: 0; }
.step { display: flex; gap: 16px; align-items: flex-start; padding: 15px 0; }
.step + .step { border-top: 1px solid var(--border); }
.step-n {
    width: 26px; height: 26px; border-radius: 8px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 0.7rem; font-weight: 700;
    background: rgba(16,224,152,0.11); color: var(--accent);
    border: 1px solid rgba(16,224,152,0.24);
}
.step-t { font-size: 0.9rem; font-weight: 600; color: var(--text); }
.step-d { font-size: 0.83rem; color: var(--muted); margin-top: 3px; line-height: 1.55; }

/* ---------------------------------------------------------- metadatos --- */
.meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden; }
.meta-cell { background: var(--surface); padding: 14px 16px; }
.meta-k { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase; color: var(--dim); margin-bottom: 5px; }
.meta-v { font-family: var(--mono); font-size: 0.86rem; font-weight: 600; color: var(--text); }

/* --------------------------------------------------------------- nota --- */
.note {
    display: flex; gap: 13px; align-items: flex-start;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 16px 19px;
    font-size: 0.835rem; line-height: 1.62; color: var(--dim);
}
.note b { color: var(--muted); font-weight: 600; }

/* -------------------------------------------------------- umbral en vivo --- */
/* Texto dinámico bajo el slider: traduce el número a lenguaje llano para que
   quede claro qué controla el umbral de decisión. */
.threshold-live {
    margin-top: 12px; padding: 11px 13px;
    background: rgba(148,163,184,0.05); border: 1px dashed var(--border-hi);
    border-radius: var(--radius-sm); font-size: 0.78rem; line-height: 1.55; color: var(--muted);
}
.threshold-live b { color: var(--text); font-weight: 600; }

/* ------------------------------------------------- widgets de streamlit --- */
[data-testid="stFileUploader"] section {
    background: var(--surface); border: 1.5px dashed rgba(148,163,184,0.22);
    border-radius: var(--radius); padding: 26px; transition: all .22s ease;
}
[data-testid="stFileUploader"] section:hover { border-color: rgba(16,224,152,0.48); background: var(--surface-2); }
[data-testid="stFileUploader"] label, [data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span { color: var(--muted) !important; }
[data-testid="stFileUploader"] button {
    background: rgba(16,224,152,0.1) !important; border: 1px solid rgba(16,224,152,0.32) !important;
    color: var(--accent) !important; font-weight: 600 !important; border-radius: 9px !important;
}
[data-testid="stFileUploader"] button:hover { background: rgba(16,224,152,0.18) !important; }

.stCheckbox label p, .stSlider label p { color: var(--muted) !important; font-size: 0.86rem !important; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--dim) !important; font-family: var(--mono); }

[data-testid="stImage"] img { border-radius: var(--radius); border: 1px solid var(--border); }
[data-testid="stImageCaption"] {
    color: var(--dim) !important; font-family: var(--mono);
    font-size: 0.68rem !important; letter-spacing: 0.1em; text-align: center;
}

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 10px 10px 0 0; padding: 11px 20px;
    font-size: 0.875rem; font-weight: 600; color: var(--dim);
}
.stTabs [data-baseweb="tab"]:hover { color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--accent) !important; background: rgba(16,224,152,0.07) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

[data-testid="stStatusWidget"], [data-testid="stExpander"] details,
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stStatusWidget"] p, [data-testid="stExpander"] summary p,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    color: var(--muted) !important; font-size: 0.86rem !important;
}

[data-testid="stAlert"] {
    background: rgba(255,77,94,0.07) !important; border: 1px solid rgba(255,77,94,0.3) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stAlert"] p { color: #FFB3BB !important; }
[data-testid="stSpinner"] p { color: var(--muted) !important; }

[data-testid="stDownloadButton"] button {
    background: rgba(148,163,184,0.07) !important; border: 1px solid var(--border-hi) !important;
    color: var(--muted) !important; border-radius: 10px !important;
    font-size: 0.83rem !important; font-weight: 600 !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: rgba(16,224,152,0.45) !important; color: var(--accent) !important;
    background: rgba(16,224,152,0.07) !important;
}

[data-baseweb="tooltip"], [data-baseweb="popover"] > div {
    background: var(--surface-3) !important; color: var(--text) !important; border-radius: 9px !important;
}
hr { border-color: var(--border) !important; }
[data-testid="stCaptionContainer"] p { color: var(--dim) !important; font-size: 0.83rem !important; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.2); border-radius: 999px; border: 2px solid var(--bg); }
::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,0.34); }
::selection { background: rgba(16,224,152,0.3); color: #fff; }

@media (max-width: 860px) {
    .block-container { padding: 1.4rem 1rem 3rem; }
    .hero { padding: 26px 22px; }
    .donut-wrap { flex-direction: column; align-items: flex-start; gap: 16px; }
    .rank-viz { width: 26%; }
}
"""


def minify_css(css: str) -> str:
    """Comprime el CSS a una sola línea.

    Es imprescindible: el parser de markdown de Streamlit cierra un bloque HTML
    en cuanto encuentra una línea en blanco, de modo que un <style> con saltos
    de línea acaba imprimiéndose como texto plano en la página.
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)          # comentarios
    css = re.sub(r"\s+", " ", css)                            # saltos y sangrías
    css = re.sub(r"\s*([{}:;,>])\s*", r"\1", css)             # espacio alrededor de símbolos
    css = re.sub(r";}", "}", css)                             # último punto y coma
    return css.strip()


_FONTS = (
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    "family=Inter:wght@400;500;600;700;800&"
    'family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">'
)

st.markdown(
    _FONTS + f"<style>{minify_css(_ROOT_VARS + _CSS)}</style>",
    unsafe_allow_html=True,
)

# =============================================================================
#  3 · LIBRERÍA DE ICONOS
# =============================================================================

_ICONS = {
    "shield": '<path d="M12 2 4 5.5v6c0 5 3.4 9.2 8 10.5 4.6-1.3 8-5.5 8-10.5v-6L12 2Z"/>',
    "shield_alert": (
        '<path d="M12 2 4 5.5v6c0 5 3.4 9.2 8 10.5 4.6-1.3 8-5.5 8-10.5v-6L12 2Z"/>'
        '<path d="M12 8v4"/><path d="M12 16h.01"/>'
    ),
    "dna": (
        '<path d="M4 3c0 6 16 6 16 12"/><path d="M20 3c0 6-16 6-16 12"/>'
        '<path d="M4 21c0-2.2 16-2.2 16 0"/><path d="M7 6h10"/><path d="M8 17h8"/>'
    ),
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 9 5-5 5 5"/><path d="M12 4v12"/>',
    "activity": '<path d="M3 12h4l3 8 4-16 3 8h4"/>',
    "layers": '<path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/>',
    "flame": '<path d="M12 22c4 0 7-2.6 7-6.5 0-4-3-6-4.5-9.5C13 9 11 8 11 5 8 7 5 10 5 15.5 5 19.4 8 22 12 22Z"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "cross": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "alert": (
        '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>'
        '<path d="M12 9v4"/><path d="M12 17h.01"/>'
    ),
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-5"/><path d="M12 8h.01"/>',
    "kit": (
        '<rect x="2" y="7" width="20" height="14" rx="2"/>'
        '<path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M12 11v6"/><path d="M9 14h6"/>'
    ),
    "chart": '<path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6"/><rect x="12" y="8" width="3" height="10"/><rect x="17" y="4" width="3" height="14"/>',
    "eye": '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
    "cpu": (
        '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v3"/><path d="M15 2v3"/>'
        '<path d="M9 19v3"/><path d="M15 19v3"/><path d="M2 9h3"/><path d="M2 15h3"/>'
        '<path d="M19 9h3"/><path d="M19 15h3"/>'
    ),
    "spark": (
        '<path d="M12 2v6"/><path d="m5 5 4 4"/><path d="M2 12h6"/><path d="m5 19 4-4"/>'
        '<path d="M12 22v-6"/><path d="m19 19-4-4"/><path d="M22 12h-6"/><path d="m19 5-4 4"/>'
    ),
    "scan": (
        '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
        '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M3 12h18"/>'
    ),
    "sliders": '<path d="M4 21v-7"/><path d="M4 10V3"/><path d="M12 21v-9"/><path d="M12 8V3"/><path d="M20 21v-5"/><path d="M20 12V3"/><path d="M1 14h6"/><path d="M9 8h6"/><path d="M17 16h6"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/>',
    "link": '<path d="M9 17H7A5 5 0 0 1 7 7h2"/><path d="M15 7h2a5 5 0 0 1 0 10h-2"/><path d="M8 12h8"/>',
}


def ico(name: str, size: int = 18, color: str = "currentColor", w: float = 1.8) -> str:
    """SVG inline listo para inyectar en HTML."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{w}" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:-0.15em;flex-shrink:0">{_ICONS.get(name, _ICONS["info"])}</svg>'
    )


# =============================================================================
#  4 · COMPONENTES DE UI
# =============================================================================

def html(markup: str) -> None:
    """Inyecta HTML en una sola línea.

    Streamlit pasa el markup por un renderizador de markdown: cualquier línea
    en blanco rompe el bloque HTML y el resto se muestra como texto. Colapsar
    los saltos de línea evita ese fallo en todos los componentes.
    """
    st.markdown(re.sub(r"\s*\n\s*", " ", markup).strip(), unsafe_allow_html=True)


def section(icon_name: str, title: str, num: str = "", color: str = THEME["accent"]) -> None:
    tag = f'<span class="num">{num}</span>' if num else ""
    html(
        f'<div class="section">{ico(icon_name, 17, color)}'
        f'<span class="ttl">{title}</span>{tag}<span class="rule"></span></div>'
    )


def donut(prob: float, color: str, radius: int = 54, stroke: int = 9) -> str:
    """Anillo de progreso SVG animado."""
    circ = 2 * 3.14159265 * radius
    offset = circ * (1 - min(max(prob, 0.0), 1.0))
    size = (radius + stroke) * 2
    c = radius + stroke
    return (
        f'<div class="donut" style="width:{size}px;height:{size}px">'
        f'<svg width="{size}" height="{size}">'
        f'<circle cx="{c}" cy="{c}" r="{radius}" fill="none" '
        f'stroke="rgba(148,163,184,0.13)" stroke-width="{stroke}"/>'
        f'<circle class="arc" cx="{c}" cy="{c}" r="{radius}" fill="none" stroke="{color}" '
        f'stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}" '
        f'style="--circ:{circ:.2f}px"/></svg>'
        f'<div class="donut-center">'
        f'<div class="donut-pct" style="color:{color}">{prob * 100:.0f}<span style="font-size:.8rem">%</span></div>'
        f'<div class="donut-cap">ÍNDICE</div></div></div>'
    )


def tier_scale(prob: float, threshold: float) -> str:
    active = risk_tier(prob, threshold)[0]
    cells = []
    for (label, color, _), cut in zip(_TIER_META, tier_cuts(threshold)):
        on = label == active
        bar = color if on else "rgba(148,163,184,0.14)"
        cells.append(
            f'<div class="tier {"on" if on else ""}" style="color:{color}" '
            f'title="a partir del {cut * 100:.0f}%">'
            f'<div class="tier-bar" style="background:{bar}"></div>'
            f'<div class="tier-lbl">{label}</div></div>'
        )
    return f'<div class="tiers">{"".join(cells)}</div>'


def alert_box(kind: str, title: str, paragraphs: list[str]) -> None:
    crit = kind == "critical"
    color = THEME["danger"] if crit else THEME["warning"]
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    html(
        f'<div class="alert alert-{"critical" if crit else "warn"} rise">'
        f'<div class="alert-ico">{ico("shield_alert" if crit else "alert", 20, color)}</div>'
        f'<div><div class="alert-title">{title}</div>'
        f'<div class="alert-body">{body}</div></div></div>'
    )


def protocol_box(kind: str, title: str, items: list[str]) -> None:
    is_do = kind == "do"
    color = THEME["accent"] if is_do else THEME["danger"]
    bullet = ico("check" if is_do else "cross", 14, color, 2.5)
    lis = "".join(f'<li><span class="bl">{bullet}</span><span>{i}</span></li>' for i in items)
    html(
        f'<div class="proto proto-{"do" if is_do else "dont"} rise rise-2">'
        f'<div class="proto-head">{ico("shield" if is_do else "alert", 17, color)}<span>{title}</span></div>'
        f"<ul>{lis}</ul></div>"
    )


def signal_row(name: str, value: str, color: str) -> str:
    return (
        f'<div class="sig"><span class="sig-dot" style="background:{color};'
        f'box-shadow:0 0 9px {color}"></span>'
        f'<span class="sig-name">{name}</span>'
        f'<span class="sig-val" style="color:{color}">{value}</span></div>'
    )


def meta_grid(pairs: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="meta-cell"><div class="meta-k">{k}</div><div class="meta-v">{v}</div></div>'
        for k, v in pairs
    )
    return f'<div class="meta-grid rise rise-1">{cells}</div>'


def to_b64(img) -> str:
    """PIL.Image o ndarray → data URI PNG."""
    if not isinstance(img, Image.Image):
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255 if arr.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def comparison_slider(before, after, height: int = 430) -> None:
    """Comparador interactivo original / Grad-CAM con divisor arrastrable."""
    a, b = to_b64(before), to_b64(after)
    components.html(
        f"""
        <style>
          .cmp {{ position:relative; width:100%; height:{height}px; border-radius:16px;
                  overflow:hidden; border:1px solid rgba(148,163,184,0.16);
                  background:#10161F; user-select:none; touch-action:none;
                  font-family:'JetBrains Mono',monospace; }}
          .cmp img {{ position:absolute; inset:0; width:100%; height:100%;
                      object-fit:contain; pointer-events:none; }}
          .cmp .top {{ clip-path: inset(0 0 0 var(--x, 50%)); }}
          .cmp .bar {{ position:absolute; top:0; bottom:0; width:2px; left:var(--x,50%);
                       background:rgba(232,239,249,.9); box-shadow:0 0 14px rgba(16,224,152,.55);
                       cursor:ew-resize; }}
          .cmp .knob {{ position:absolute; top:50%; left:50%; width:38px; height:38px;
                        transform:translate(-50%,-50%); border-radius:50%;
                        background:rgba(16,22,31,.92); border:2px solid rgba(232,239,249,.9);
                        display:flex; align-items:center; justify-content:center; }}
          .cmp .tag {{ position:absolute; bottom:12px; font-size:9px; letter-spacing:.14em;
                       font-weight:700; padding:5px 10px; border-radius:999px;
                       background:rgba(8,11,17,.72); border:1px solid rgba(148,163,184,.2); }}
          .cmp .l {{ left:12px; color:#93A1B5; }}
          .cmp .r {{ right:12px; color:#10E098; }}
        </style>
        <div class="cmp" id="cmp">
          <img src="{a}">
          <img class="top" src="{b}">
          <div class="bar" id="bar">
            <div class="knob">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#E8EFF9"
                   stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
                <path d="m9 6-6 6 6 6"/><path d="m15 6 6 6-6 6"/>
              </svg>
            </div>
          </div>
          <div class="tag l">ORIGINAL</div>
          <div class="tag r">GRAD-CAM</div>
        </div>
        <script>
          (function() {{
            const box = document.getElementById('cmp');
            let dragging = false;
            const move = (e) => {{
              const r = box.getBoundingClientRect();
              const cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
              const pct = Math.max(0, Math.min(100, (cx / r.width) * 100));
              box.style.setProperty('--x', pct + '%');
              document.getElementById('bar').style.left = pct + '%';
            }};
            const down = (e) => {{ dragging = true; move(e); }};
            box.addEventListener('mousedown', down);
            box.addEventListener('touchstart', down, {{passive:true}});
            window.addEventListener('mouseup', () => dragging = false);
            window.addEventListener('touchend', () => dragging = false);
            window.addEventListener('mousemove', (e) => dragging && move(e));
            window.addEventListener('touchmove', (e) => dragging && move(e), {{passive:true}});
          }})();
        </script>
        """,
        height=height + 14,
    )


# =============================================================================
#  5 · CARGA DE MODELOS
# =============================================================================

def _require(path: str) -> str:
    if not os.path.exists(path):
        found = ", ".join(os.listdir("models")) if os.path.isdir("models") else "no existe la carpeta 'models'"
        raise FileNotFoundError(f"No se encontró '{path}'. Contenido de 'models/': {found}")
    return path


@st.cache_resource(show_spinner=False)
def get_venom_model():
    return load_venom_model(_require("models/modelo_veneno.weights.h5"))


@st.cache_resource(show_spinner=False)
def get_species_model():
    return load_species_model(_require("models/modelo_especie.pth"))


# =============================================================================
#  6 · CABECERA
# =============================================================================

st.session_state.setdefault("runs", 0)

html(
    f"""
    <div class="hero rise">
        <div class="eyebrow">{ico("spark", 12, THEME["accent"], 2.2)} Computer Vision · Herpetología</div>
        <h1 class="hero-title">Snakely <em>AI</em></h1>
        <p class="hero-sub">
            Identificación taxonómica y evaluación de riesgo toxicológico en ofidios,
            con validación cruzada entre modelos independientes y trazabilidad visual
            de la inferencia.
        </p>
        <div class="hero-meta">
            <span class="chip">{ico("cpu", 14, THEME["accent"])} Doble arquitectura <b>CNN</b></span>
            <span class="chip">{ico("shield", 14, THEME["info"])} Validación cruzada de <b>seguridad</b></span>
            <span class="chip">{ico("eye", 14, THEME["violet"])} Interpretabilidad <b>Grad-CAM</b></span>
            <span class="chip">{ico("kit", 14, THEME["warning"])} Protocolo de <b>primeros auxilios</b></span>
        </div>
    </div>
    """
)

# --- AVISO LEGAL: arriba del todo, antes de subir cualquier imagen -----------
# Se muestra siempre, no solo en el estado vacío, para que sea lo primero que
# se lee al entrar a la aplicación.
html(
    f"""
    <div class="advisory rise rise-1">
        <div class="advisory-ico">{ico("alert", 18, THEME["warning"])}</div>
        <div>
            <div class="advisory-t">Léelo antes de continuar</div>
            <div class="advisory-d">
                Los resultados son <b>estimaciones probabilísticas</b> generadas por modelos de
                aprendizaje profundo y pueden contener errores. No sustituyen el criterio de un
                herpetólogo ni la atención médica profesional.
                <b>Ante una mordedura, acude de inmediato al centro de salud más cercano</b> —
                no esperes a confirmar la especie con esta herramienta.
            </div>
        </div>
    </div>
    """
)

# =============================================================================
#  7 · ENTRADA Y PARÁMETROS
# =============================================================================

section("upload", "Muestra de análisis", "01")

col_up, col_cfg = st.columns([2.3, 1], gap="large")

with col_up:
    image_file = st.file_uploader(
        "Arrastra una fotografía nítida del ejemplar  ·  JPG · PNG · JPEG",
        type=["jpg", "png", "jpeg"],
    )

with col_cfg:
    with st.container(border=True):
        html(
            f'<div class="card-label" style="margin-bottom:10px">'
            f'{ico("sliders", 13, THEME["info"])} Sensibilidad del sistema</div>'
        )
        venom_threshold = st.slider(
            "Umbral de decisión de toxicidad",
            min_value=0.30,
            max_value=0.70,
            value=0.50,
            step=0.05,
            help="El detector de veneno no da un sí/no: da un porcentaje de indicios "
            "de toxicidad. Este control decide a partir de qué porcentaje ese número "
            "se traduce en la alerta de 'venenosa'.",
        )
        # Texto en vivo: traduce el número del slider a lenguaje llano, para que
        # quede claro qué controla exactamente (no todo el mundo lee el help del ?).
        if venom_threshold <= 0.35:
            stance = "muy conservador"
            trade_off = "avisará con más facilidad, a costa de más falsas alarmas"
        elif venom_threshold <= 0.55:
            stance = "equilibrado"
            trade_off = "es el punto medio recomendado para uso general"
        else:
            stance = "estricto"
            trade_off = "solo avisará ante indicios claros, con más riesgo de pasar por alto una venenosa"
        html(
            f'<div class="threshold-live">Con <b>{venom_threshold * 100:.0f}%</b>, el sistema '
            f'marcará un ejemplar como <b>peligroso</b> en cuanto el detector de veneno supere ese '
            f'porcentaje de indicios. Es un ajuste <b>{stance}</b>: {trade_off}.</div>'
        )
        show_gradcam = st.checkbox("Generar mapa de atención (Grad-CAM)", value=True)

# ------------------------------- estado vacío --------------------------------

if image_file is None:
    section("layers", "Cómo funciona", "02", THEME["info"])

    f1, f2, f3 = st.columns(3, gap="large")
    features = [
        (f1, "dna", THEME["info"], "Clasificador taxonómico",
         "Una red convolucional identifica la especie y devuelve las cinco candidatas "
         "más probables con su nivel de confianza."),
        (f2, "shield_alert", THEME["danger"], "Detector de toxicidad",
         "Un segundo modelo, independiente del anterior, estima la probabilidad de que "
         "el ejemplar presente rasgos de especie venenosa."),
        (f3, "scan", THEME["accent"], "Validación cruzada",
         "Ambos dictámenes se contrastan. Ante cualquier discrepancia, el sistema "
         "resuelve siempre a favor de la hipótesis más segura."),
    ]
    for col, name, color, title, desc in features:
        with col:
            html(
                f'<div class="feat rise rise-1">'
                f'<div class="feat-ico" style="background:{color}1F">{ico(name, 20, color)}</div>'
                f'<div class="feat-t">{title}</div><div class="feat-d">{desc}</div></div>'
            )

    st.write("")
    c_steps, c_tips = st.columns([1.15, 1], gap="large")

    with c_steps:
        steps = [
            ("Sube la fotografía", "Formatos JPG, PNG o JPEG. Se corrige automáticamente la orientación EXIF."),
            ("Ajusta la sensibilidad", "Baja el umbral si prefieres un criterio más conservador."),
            ("Revisa el dictamen", "Índice de toxicidad, especie predominante y consenso entre modelos."),
            ("Consulta el protocolo", "Acciones recomendadas y prohibidas ante una mordedura."),
        ]
        rows = "".join(
            f'<div class="step"><div class="step-n">{i}</div>'
            f'<div><div class="step-t">{t}</div><div class="step-d">{d}</div></div></div>'
            for i, (t, d) in enumerate(steps, 1)
        )
        html(f'<div class="card rise rise-2"><div class="card-label">{ico("activity", 13, THEME["accent"])} '
             f'Flujo de trabajo</div><div class="steps">{rows}</div></div>')

    with c_tips:
        tips = [
            "Encuadra el cuerpo completo y, si es posible, la cabeza.",
            "Evita contraluces, reflejos y fondos muy saturados.",
            "Prioriza fotografías enfocadas y con buena resolución.",
            "Una sola serpiente por imagen mejora la precisión.",
        ]
        lis = "".join(
            f'<li><span class="bl">{ico("check", 14, THEME["accent"], 2.5)}</span><span>{t}</span></li>'
            for t in tips
        )
        html(f'<div class="proto proto-do rise rise-3">'
             f'<div class="proto-head">{ico("eye", 17, THEME["accent"])}'
             f'<span>Cómo obtener mejores resultados</span></div><ul>{lis}</ul></div>')

    st.stop()

# =============================================================================
#  8 · INFERENCIA
# =============================================================================

raw_bytes = image_file.getvalue()
image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
image_np = np.array(image)

col_a, col_b, col_c = st.columns([1, 1.55, 1])
with col_b:
    st.image(image, caption="MUESTRA CARGADA", use_container_width=True)

t0 = time.perf_counter()
with st.status("Ejecutando pipeline de inferencia…", expanded=True) as status:
    try:
        status.write("Cargando arquitecturas neuronales…")
        venom_model = get_venom_model()
        species_model = get_species_model()

        status.write("Evaluando patrones morfológicos de toxicidad…")
        is_venomous, venom_prob, venom_recommendations = predict_venom(
            venom_model, image_np, venom_threshold
        )

        status.write("Identificando taxón y especie…")
        species_name, species_prob, top_predictions = predict_species(
            species_model, image_np, top_k=5
        )

        status.update(label="Análisis completado", state="complete", expanded=False)
    except Exception as exc:  # noqa: BLE001
        status.update(label="Error durante la inferencia", state="error")
        st.error(f"Detalle técnico: {exc}")
        st.stop()

elapsed = time.perf_counter() - t0
st.session_state["runs"] += 1

# ------------------------- validación cruzada --------------------------------

top_raw = top_predictions[0]["raw_name"]

safety = cross_validate_venom_risk(
    species_raw_name=top_raw,
    is_venomous_pred=is_venomous,
    species_prob=species_prob,
)

known_venomous = any(kw in top_raw.lower() for kw in VENOMOUS_KEYWORDS)
false_positive_risk = (not known_venomous) and is_venomous
false_negative_risk = known_venomous and (not is_venomous)
contradiction = false_positive_risk or false_negative_risk
final_is_venomous = safety.get("final_is_venomous", is_venomous)

# El dictamen final puede elevar el nivel mostrado por encima del índice bruto.
display_prob = max(venom_prob, venom_threshold + 0.01) if final_is_venomous else venom_prob
tier_label, tier_color, tier_desc = risk_tier(display_prob, venom_threshold)

# ------------------------------ metadatos ------------------------------------

w, h = image.size
html(
    meta_grid(
        [
            ("Archivo", image_file.name[:22] + ("…" if len(image_file.name) > 22 else "")),
            ("Resolución", f"{w}×{h} px"),
            ("Peso", f"{len(raw_bytes) / 1024:.0f} KB"),
            ("Inferencia", f"{elapsed:.2f} s"),
            ("Umbral", f"{venom_threshold:.2f}"),
            ("Análisis", f"#{st.session_state['runs']:03d}"),
        ]
    )
)

# =============================================================================
#  9 · DICTAMEN
# =============================================================================

section("activity", "Dictamen del sistema", "02", tier_color)

verdict = "POTENCIALMENTE VENENOSA" if final_is_venomous else "SIN INDICIOS DE VENENO"
card_tone = "t-danger" if final_is_venomous else "t-safe"
badge_tone = "badge-danger" if final_is_venomous else "badge-safe"

col_v, col_s = st.columns([1.05, 1], gap="large")

with col_v:
    html(
        f"""
        <div class="card {card_tone} rise rise-1">
            <div class="card-label">{ico("shield_alert", 13, tier_color)} Diagnóstico de peligrosidad · prioritario</div>
            <div class="donut-wrap">
                {donut(display_prob, tier_color)}
                <div class="donut-side">
                    <div class="card-value" style="font-size:1.16rem">{verdict}</div>
                    <div class="card-sub" style="font-style:normal">{tier_desc}</div>
                    <span class="badge {badge_tone}">{ico("activity", 12, "currentColor")} NIVEL {tier_label}</span>
                </div>
            </div>
            {tier_scale(display_prob, venom_threshold)}
        </div>
        """
    )

with col_s:
    html(
        f"""
        <div class="card t-info rise rise-2">
            <div class="card-label">{ico("dna", 13, THEME["info"])} Especie predominante</div>
            <div class="donut-wrap">
                {donut(species_prob, THEME["info"])}
                <div class="donut-side">
                    <div class="card-value" style="font-size:1.16rem">{species_name}</div>
                    <div class="card-sub">{top_raw}</div>
                    <span class="badge badge-info">{ico("chart", 12, "currentColor")}
                        {"GRUPO VENENOSO" if known_venomous else "GRUPO NO VENENOSO"}</span>
                </div>
            </div>
        </div>
        """
    )

# ------------------------- alertas de contradicción --------------------------

if contradiction:
    st.write("")
    if false_positive_risk:
        alert_box(
            "critical",
            "Modelos contradictorios · posible falso positivo de veneno",
            [
                f"La especie fue identificada como <b>{species_name}</b>, clasificada "
                f"biológicamente como <b>no venenosa</b>, pero el detector de toxicidad "
                f"registró un <b>{venom_prob * 100:.1f}%</b> de rasgos compatibles con veneno.",
                "<b>Criterio de precaución extrema.</b> La morfología del ejemplar, el ángulo "
                "de captura o las condiciones de luz pueden haber inducido error en el modelo "
                "taxonómico o en el de toxicidad. El sistema resuelve a favor de la seguridad.",
                "<b>Recomendación:</b> trata al ejemplar como potencialmente peligroso y "
                "mantén la distancia.",
            ],
        )
    else:
        alert_box(
            "warn",
            "Modelos contradictorios · protocolo preventivo activado",
            [
                f"El detector de veneno registró un nivel bajo (<b>{venom_prob * 100:.1f}%</b>), "
                f"pero la especie identificada es <b>{species_name}</b>, perteneciente a un grupo "
                f"<b>potencialmente venenoso</b>.",
                "<b>Recomendación:</b> se aplican los protocolos de seguridad de forma preventiva.",
            ],
        )

# =============================================================================
#  10 · CONSENSO DE MODELOS
# =============================================================================

section("scan", "Consenso entre modelos", "03", THEME["violet"])

agree_color = THEME["danger"] if contradiction else THEME["accent"]
agree_text = "DISCREPANCIA" if contradiction else "CONVERGENTE"
conf_color = (
    THEME["accent"] if species_prob >= 0.70
    else THEME["warning"] if species_prob >= 0.45
    else THEME["danger"]
)
conf_text = (
    "ALTA" if species_prob >= 0.70 else "MEDIA" if species_prob >= 0.45 else "BAJA"
)

col_sig, col_dl = st.columns([1.6, 1], gap="large")

with col_sig:
    signals = (
        signal_row(
            "Detector de toxicidad",
            f"{venom_prob * 100:.1f}% · {'POSITIVO' if is_venomous else 'NEGATIVO'}",
            THEME["danger"] if is_venomous else THEME["accent"],
        )
        + signal_row(
            "Clasificador taxonómico",
            f"{species_prob * 100:.1f}% · CONFIANZA {conf_text}",
            conf_color,
        )
        + signal_row(
            "Grupo biológico de la especie",
            "VENENOSO" if known_venomous else "NO VENENOSO",
            THEME["danger"] if known_venomous else THEME["accent"],
        )
        + signal_row("Validación cruzada", agree_text, agree_color)
        + signal_row(
            "Dictamen final aplicado",
            "PELIGROSA" if final_is_venomous else "NO PELIGROSA",
            THEME["danger"] if final_is_venomous else THEME["accent"],
        )
    )
    html(
        f'<div class="card rise rise-1" style="border-left-color:{THEME["violet"]}">'
        f'<div class="card-label">{ico("layers", 13, THEME["violet"])} Señales del pipeline</div>'
        f'<div class="consensus">{signals}</div></div>'
    )

with col_dl:
    interp = (
        "Los dos modelos discrepan. El sistema ha aplicado el criterio más conservador "
        "y elevado el nivel de riesgo por precaución."
        if contradiction
        else "Ambos modelos coinciden en su dictamen, lo que refuerza la fiabilidad del "
        "resultado. Aun así, considera el margen de error inherente a toda predicción."
    )
    html(
        f'<div class="card {"t-danger" if contradiction else "t-safe"} rise rise-2">'
        f'<div class="card-label">{ico("info", 13, agree_color)} Interpretación</div>'
        f'<div class="card-value" style="font-size:1.05rem;margin-bottom:10px">{agree_text}</div>'
        f'<div style="font-size:.875rem;line-height:1.65;color:var(--muted)">{interp}</div></div>'
    )

# =============================================================================
#  11 · PROTOCOLO DE PRIMEROS AUXILIOS
# =============================================================================

recs = (
    safety["recommendations"]
    if safety.get("warning_triggered") and safety.get("recommendations")
    else venom_recommendations
)

if recs:
    section("kit", "Protocolo de primeros auxilios", "04", THEME["danger"])
    c_do, c_dont = st.columns(2, gap="large")
    with c_do:
        protocol_box("do", "Acciones recomendadas", recs.get("que_hacer", []))
    with c_dont:
        protocol_box("dont", "Acciones prohibidas", recs.get("nunca_hacer", []))

# =============================================================================
#  12 · DETALLE TÉCNICO
# =============================================================================

section("layers", "Detalle técnico", "05", THEME["info"])

tab_rank, tab_cam = st.tabs(["Ranking de especies", "Mapa de atención"])

with tab_rank:
    st.caption(
        "Distribución de probabilidad sobre las cinco especies más afines detectadas "
        "por el clasificador taxonómico."
    )
    rows = []
    for i, pred in enumerate(top_predictions, 1):
        pct = min(pred["probability"], 1.0) * 100
        rows.append(
            f'<div class="rank {"lead" if i == 1 else ""}">'
            f'<div class="rank-idx">{i:02d}</div>'
            f'<div class="rank-main"><div class="rank-name">{pred["spanish_name"]}</div>'
            f'<div class="rank-lat">{pred["raw_name"]}</div></div>'
            f'<div class="rank-viz"><div class="rank-bar"><i style="width:{pct:.2f}%"></i></div>'
            f'<div class="rank-pct">{pct:.2f}%</div></div></div>'
        )
    html("".join(rows))

    tail = 1.0 - sum(min(p["probability"], 1.0) for p in top_predictions)
    if tail > 0.005:
        html(
            f'<div class="note" style="margin-top:12px">{ico("info", 15, THEME["dim"])}'
            f'<span>El <b>{tail * 100:.1f}%</b> restante de la masa de probabilidad se '
            f"reparte entre el resto de clases del modelo.</span></div>"
        )

with tab_cam:
    if show_gradcam:
        st.caption(
            "Arrastra el divisor para comparar. Las regiones cálidas concentran el mayor "
            "peso en la decisión del clasificador de especie."
        )
        try:
            with st.spinner("Generando interpretabilidad visual…"):
                cam_image = generate_gradcam(species_model, image_np)
            comparison_slider(image, cam_image)
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo generar el Grad-CAM: {exc}")
    else:
        html(
            f'<div class="note" style="margin-top:12px">{ico("flame", 16, THEME["warning"])}'
            f"<span>Activa <b>Generar mapa de atención (Grad-CAM)</b> en el panel de "
            f"sensibilidad para desplegar la interpretabilidad visual del modelo.</span></div>"
        )

# =============================================================================
#  13 · INFORME DESCARGABLE
# =============================================================================

def build_report() -> str:
    ts = datetime.now().strftime("%d/%m/%Y %H:%M")
    rank_html = "".join(
        f"<tr><td>{i:02d}</td><td>{p['spanish_name']}</td>"
        f"<td><i>{p['raw_name']}</i></td><td>{p['probability'] * 100:.2f}%</td></tr>"
        for i, p in enumerate(top_predictions, 1)
    )
    do_html = "".join(f"<li>{x}</li>" for x in (recs or {}).get("que_hacer", []))
    dont_html = "".join(f"<li>{x}</li>" for x in (recs or {}).get("nunca_hacer", []))
    warn = (
        '<p class="warn"><b>Aviso:</b> los modelos arrojaron dictámenes contradictorios. '
        "El sistema aplicó el criterio más conservador.</p>"
        if contradiction
        else ""
    )
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Informe Snakely · {ts}</title><style>
body{{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;max-width:760px;margin:40px auto;
padding:0 22px;color:#12181F;line-height:1.65}}
h1{{font-size:1.5rem;margin:0 0 4px;letter-spacing:-.02em}}
h2{{font-size:1rem;margin:30px 0 10px;text-transform:uppercase;letter-spacing:.09em;color:#5D6B80}}
.sub{{color:#6B7787;font-size:.88rem;margin:0 0 26px}}
.box{{border:1px solid #E2E7EE;border-left:4px solid {tier_color};border-radius:10px;
padding:16px 20px;margin-bottom:12px}}
.k{{font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;color:#6B7787}}
.v{{font-size:1.15rem;font-weight:700;margin-top:3px}}
table{{width:100%;border-collapse:collapse;font-size:.9rem}}
th,td{{text-align:left;padding:8px 6px;border-bottom:1px solid #EDF0F4}}
th{{font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:#6B7787}}
ul{{margin:0;padding-left:20px}} li{{margin-bottom:5px;font-size:.92rem}}
.warn{{background:#FFF4E5;border:1px solid #FFD9A0;border-radius:9px;padding:12px 16px;font-size:.9rem}}
.foot{{margin-top:36px;padding-top:16px;border-top:1px solid #E2E7EE;font-size:.8rem;color:#6B7787}}
</style></head><body>
<h1>Informe de análisis · Snakely AI</h1>
<p class="sub">Generado el {ts} · Archivo: {image_file.name} · Umbral: {venom_threshold:.2f}</p>
{warn}
<div class="box"><div class="k">Diagnóstico de peligrosidad</div>
<div class="v">{verdict} — nivel {tier_label}</div>
<div style="font-size:.88rem;color:#6B7787">Índice de toxicidad: {venom_prob * 100:.1f}% · {tier_desc}</div></div>
<div class="box" style="border-left-color:#3B9EFF"><div class="k">Especie predominante</div>
<div class="v">{species_name}</div>
<div style="font-size:.88rem;color:#6B7787"><i>{top_raw}</i> · {species_prob * 100:.1f}% de coincidencia</div></div>
<h2>Ranking de especies</h2>
<table><tr><th>#</th><th>Nombre común</th><th>Nombre científico</th><th>Prob.</th></tr>{rank_html}</table>
<h2>Acciones recomendadas</h2><ul>{do_html or "<li>Sin recomendaciones disponibles.</li>"}</ul>
<h2>Acciones prohibidas</h2><ul>{dont_html or "<li>Sin recomendaciones disponibles.</li>"}</ul>
<p class="foot">Resultados generados por modelos de aprendizaje profundo. Son estimaciones
probabilísticas y pueden contener errores. No sustituyen el criterio de un herpetólogo ni la
atención médica profesional. Ante una mordedura, acude de inmediato al centro de salud más cercano.</p>
</body></html>"""


st.write("")
col_note, col_btn = st.columns([2.2, 1], gap="large")

with col_note:
    html(
        f'<div class="note">{ico("info", 16, THEME["dim"])}'
        f"<span><b>Recordatorio.</b> Este dictamen es orientativo y puede contener errores. "
        f"Ante una mordedura, acude de inmediato al centro de salud más cercano.</span></div>"
    )

with col_btn:
    st.download_button(
        "⤓  Descargar informe (HTML)",
        data=build_report(),
        file_name=f"snakely_informe_{datetime.now():%Y%m%d_%H%M}.html",
        mime="text/html",
        use_container_width=True,
    )
