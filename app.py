"""
===============================================================================
 SNAKELY AI  ·  Ficha de campo digital para identificación de ofidios
 Clasificación taxonómica + evaluación de riesgo toxicológico
===============================================================================

Archivo único y autosuficiente. Contiene:
  · bootstrap del tema nativo de Streamlit
  · sistema de diseño «ficha de campo» (tokens + CSS)
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
import math
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
# =============================================================================

_CONFIG_TOML = """[theme]
base = "dark"
primaryColor = "#3FBF8F"
backgroundColor = "#0B0E12"
secondaryBackgroundColor = "#12161C"
textColor = "#E8E4DB"
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
    page_title="Snakely · Ficha de campo",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
#  2 · SISTEMA DE DISEÑO
#     Paleta «tinta y ocre»: fondo de tinta cálida, texto hueso, filetes ocre
#     de cuaderno técnico y solo dos colores semánticos (jade / bermellón).
# =============================================================================

THEME = {
    "bg": "#0B0E12",
    "surface": "#12161C",
    "surface_2": "#171C24",
    "surface_3": "#1E242D",
    "line": "rgba(224, 164, 88, 0.17)",     # filete ocre de la ficha
    "line_soft": "rgba(224, 164, 88, 0.09)",
    "rule": "rgba(232, 228, 219, 0.10)",
    "text": "#E8E4DB",                       # hueso
    "muted": "#9C978A",
    "dim": "#6E6C64",
    "ochre": "#E0A458",                      # color de anotación
    "jade": "#3FBF8F",                       # seguro
    "jade_soft": "#7FD9B4",
    "vermilion": "#E2504E",                  # peligro
    "vermilion_soft": "#F08D8B",
    "amber": "#EFB458",                      # precaución
    "steel": "#5B9FD6",                      # taxonomía
    "plum": "#9B8AC4",                       # consenso
}

# --- modos de sensibilidad, en lenguaje llano ---------------------------------
SENSITIVITY = {
    "Precavido": {
        "threshold": 0.35,
        "color": THEME["amber"],
        "short": "Prioriza no pasar por alto ninguna serpiente venenosa.",
        "long": (
            "Con este modo basta un <b>35%</b> de indicios para que el ejemplar se marque "
            "como peligroso. Es el criterio más seguro en campo: reduce al mínimo el riesgo "
            "de omitir una especie venenosa, a costa de más falsas alarmas."
        ),
    },
    "Equilibrado": {
        "threshold": 0.50,
        "color": THEME["jade"],
        "short": "Punto medio entre falsas alarmas y omisiones.",
        "long": (
            "Se marca como peligroso a partir de un <b>50%</b> de indicios. Es el ajuste "
            "por defecto y el recomendado para uso general: equilibra el coste de una falsa "
            "alarma con el de una omisión."
        ),
    },
    "Estricto": {
        "threshold": 0.65,
        "color": THEME["steel"],
        "short": "Solo señala los casos con indicios claros.",
        "long": (
            "Hace falta un <b>65%</b> de indicios para marcar el ejemplar como peligroso. "
            "Genera muy pocas falsas alarmas, pero <b>aumenta el riesgo de dar por inocuo "
            "un ejemplar venenoso</b>. Úsalo solo con fines de estudio, nunca para decidir "
            "si acercarte a una serpiente."
        ),
    },
}

# --- niveles de riesgo, con cortes derivados del umbral activo ----------------
_TIER_META = [
    ("MÍNIMO", THEME["jade"], "Sin indicios morfológicos de toxicidad"),
    ("BAJO", THEME["jade_soft"], "Indicios débiles, no concluyentes"),
    ("ELEVADO", THEME["amber"], "Rasgos compatibles con especie venenosa"),
    ("CRÍTICO", THEME["vermilion"], "Alta compatibilidad con especie venenosa"),
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
    --line: {THEME["line"]};
    --line-soft: {THEME["line_soft"]};
    --rule: {THEME["rule"]};
    --text: {THEME["text"]};
    --muted: {THEME["muted"]};
    --dim: {THEME["dim"]};
    --ochre: {THEME["ochre"]};
    --jade: {THEME["jade"]};
    --vermilion: {THEME["vermilion"]};
    --amber: {THEME["amber"]};
    --steel: {THEME["steel"]};
    --plum: {THEME["plum"]};
    --radius: 4px;
    --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
    --serif: 'Spectral', Georgia, 'Times New Roman', serif;
}}
"""

# El CSS se escribe en una cadena plana y se minifica antes de inyectarlo:
# una línea en blanco dentro de un bloque HTML hace que Streamlit lo imprima
# como texto en lugar de aplicarlo.
_CSS = """
/* ================================================================= base === */
[data-testid="collapsedControl"], #MainMenu, footer, header { display: none !important; }

html, body, .stApp, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
    font-variant-numeric: tabular-nums;
}

/* Papel milimetrado tenue: el sustrato de toda la ficha. */
.stApp {
    background-color: var(--bg);
    background-image:
        linear-gradient(var(--line-soft) 1px, transparent 1px),
        linear-gradient(90deg, var(--line-soft) 1px, transparent 1px),
        radial-gradient(900px 500px at 82% -6%, rgba(224,164,88,0.055), transparent 62%),
        radial-gradient(760px 460px at 6% 4%, rgba(63,191,143,0.05), transparent 60%);
    background-size: 26px 26px, 26px 26px, 100% 100%, 100% 100%;
    color: var(--text);
}
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; }

.block-container { max-width: 1200px; padding: 2rem 2rem 5rem; }
h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--text); }

/* Utilidades tipográficas de la ficha. */
.mono { font-family: var(--mono); }
.tag {
    font-family: var(--mono); font-size: 0.62rem; font-weight: 500;
    letter-spacing: 0.2em; text-transform: uppercase; color: var(--ochre);
}
.tag-dim { color: var(--dim); }

/* =========================================================== animaciones === */
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes dash { from { stroke-dashoffset: var(--circ); } }
@keyframes blink { 0%, 100% { opacity: .25; } 50% { opacity: 1; } }
.rise   { animation: rise .5s cubic-bezier(.22,1,.36,1) both; }
.rise-1 { animation-delay: .05s; } .rise-2 { animation-delay: .12s; }
.rise-3 { animation-delay: .19s; } .rise-4 { animation-delay: .26s; }

/* ============================================== marco de lámina (plate) === */
.plate {
    position: relative; background: var(--surface);
    border: 1px solid var(--line); border-radius: var(--radius);
}
/* Segundo filete interior: el doble encuadre de las láminas científicas. */
.plate::before {
    content: ""; position: absolute; inset: 5px; pointer-events: none;
    border: 1px solid var(--line-soft); border-radius: 2px;
}
/* Puntos de registro en las cuatro esquinas. */
.plate::after {
    content: ""; position: absolute; inset: 5px; pointer-events: none;
    background:
        radial-gradient(circle at 0 0,     var(--ochre) 0 1.6px, transparent 1.7px),
        radial-gradient(circle at 100% 0,  var(--ochre) 0 1.6px, transparent 1.7px),
        radial-gradient(circle at 0 100%,  var(--ochre) 0 1.6px, transparent 1.7px),
        radial-gradient(circle at 100% 100%, var(--ochre) 0 1.6px, transparent 1.7px);
    opacity: .55;
}
.plate > * { position: relative; z-index: 1; }

/* Regla graduada: marca menor cada 7px, mayor cada 35px. */
.ruler {
    height: 9px; width: 100%;
    background-image:
        repeating-linear-gradient(90deg, var(--line) 0 1px, transparent 1px 7px),
        repeating-linear-gradient(90deg, var(--ochre) 0 1px, transparent 1px 35px);
    background-size: 100% 4px, 100% 9px;
    background-repeat: no-repeat;
    background-position: 0 0, 0 0;
    opacity: .5;
}

/* ================================================================ hero === */
.hero { padding: 0; overflow: hidden; margin-bottom: 14px; }
.hero-body { padding: 30px 34px 26px; display: flex; gap: 30px; align-items: flex-start; }
.hero-left { flex: 1; min-width: 0; }
.hero-kicker { display: flex; align-items: center; gap: 9px; margin-bottom: 14px; }
.hero-kicker .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--jade); animation: blink 2.4s ease-in-out infinite; }
.hero-title {
    font-family: var(--serif); font-size: clamp(2.2rem, 5vw, 3.15rem);
    font-weight: 600; letter-spacing: -0.022em; line-height: 1; margin: 0 0 4px;
    color: var(--text);
}
.hero-title i { font-style: italic; color: var(--ochre); font-weight: 400; }
.hero-latin { font-family: var(--serif); font-style: italic; font-size: 0.94rem; color: var(--dim); margin: 0 0 16px; }
.hero-sub { font-size: 0.97rem; color: var(--muted); max-width: 560px; line-height: 1.7; margin: 0; }
.hero-file {
    flex-shrink: 0; width: 190px; border-left: 1px solid var(--line); padding-left: 22px;
}
.hero-file .row { display: flex; justify-content: space-between; gap: 10px; padding: 7px 0; border-bottom: 1px dotted var(--rule); }
.hero-file .row:last-child { border-bottom: none; }
.hero-file .k { font-family: var(--mono); font-size: 0.6rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--dim); }
.hero-file .v { font-family: var(--mono); font-size: 0.68rem; color: var(--muted); text-align: right; }
.hero-caps { display: flex; flex-wrap: wrap; gap: 0; border-top: 1px solid var(--line); }
.hero-cap {
    flex: 1 1 190px; display: flex; align-items: center; gap: 10px;
    padding: 15px 20px; border-right: 1px solid var(--line);
    font-size: 0.775rem; color: var(--muted);
}
.hero-cap:last-child { border-right: none; }
.hero-cap b { color: var(--text); font-weight: 600; }

/* ============================================================== aviso === */
.advisory {
    display: flex; align-items: stretch; gap: 0; margin-bottom: 6px;
    background: linear-gradient(90deg, rgba(226,80,78,0.075), rgba(18,22,28,0.5) 70%);
    border: 1px solid rgba(226,80,78,0.3); border-radius: var(--radius);
    overflow: hidden;
}
/* Banda de peligro diagonal: el sello de la ficha. */
.advisory-stripe {
    width: 13px; flex-shrink: 0;
    background: repeating-linear-gradient(45deg,
        rgba(226,80,78,0.85) 0 5px, rgba(11,14,18,0.9) 5px 10px);
}
.advisory-body { padding: 15px 20px; display: flex; gap: 15px; align-items: flex-start; }
.advisory-ico { margin-top: 1px; flex-shrink: 0; }
.advisory-t {
    font-family: var(--mono); font-size: 0.63rem; font-weight: 700; letter-spacing: 0.19em;
    text-transform: uppercase; color: var(--vermilion); margin-bottom: 6px;
}
.advisory-d { font-size: 0.855rem; line-height: 1.62; color: var(--muted); }
.advisory-d b { color: var(--text); font-weight: 600; }

/* =========================================================== secciones === */
.sec { display: flex; align-items: baseline; gap: 12px; margin: 34px 0 15px; }
.sec-n {
    font-family: var(--mono); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
    color: var(--ochre); flex-shrink: 0;
}
.sec-t { font-size: 1.02rem; font-weight: 700; letter-spacing: -0.012em; color: var(--text); flex-shrink: 0; }
.sec-rule {
    flex: 1; height: 4px; align-self: center;
    background-image: repeating-linear-gradient(90deg, var(--line) 0 2px, transparent 2px 7px);
}
.sec-tag {
    font-family: var(--mono); font-size: 0.6rem; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--dim); flex-shrink: 0;
}

/* ============================================================ tarjetas === */
.card { padding: 22px 24px; height: 100%; }
.card-head {
    display: flex; align-items: center; gap: 9px; margin-bottom: 16px;
    padding-bottom: 11px; border-bottom: 1px dotted var(--rule);
}
.card-idx {
    width: 20px; height: 20px; flex-shrink: 0; border: 1px solid var(--line);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 0.6rem; font-weight: 700; color: var(--ochre);
}
.card-t {
    font-family: var(--mono); font-size: 0.63rem; font-weight: 600; letter-spacing: 0.17em;
    text-transform: uppercase; color: var(--muted);
}
.card-value { font-size: 1.2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.22; color: var(--text); }
.card-latin { font-family: var(--serif); font-style: italic; font-size: 0.87rem; color: var(--dim); margin-top: 3px; }
.card-note { font-size: 0.83rem; color: var(--muted); line-height: 1.6; margin-top: 7px; }

.tone-danger { border-color: rgba(226,80,78,0.34); background: linear-gradient(160deg, rgba(226,80,78,0.07), var(--surface) 55%); }
.tone-safe   { border-color: rgba(63,191,143,0.3);  background: linear-gradient(160deg, rgba(63,191,143,0.06), var(--surface) 55%); }
.tone-info   { border-color: rgba(91,159,214,0.28); background: linear-gradient(160deg, rgba(91,159,214,0.06), var(--surface) 55%); }
.tone-plum   { border-color: rgba(155,138,196,0.28); }

/* Sello de caucho del dictamen. */
.stamp {
    position: absolute; top: 16px; right: 18px; z-index: 2;
    font-family: var(--mono); font-size: 0.6rem; font-weight: 700; letter-spacing: 0.15em;
    padding: 5px 10px; border: 2px solid currentColor; border-radius: 3px;
    transform: rotate(-7deg); opacity: .8; pointer-events: none;
}

.pill {
    display: inline-flex; align-items: center; gap: 6px; margin-top: 12px;
    font-family: var(--mono); font-size: 0.66rem; font-weight: 600; letter-spacing: 0.1em;
    padding: 5px 10px; border: 1px solid currentColor; border-radius: 2px;
}

/* =============================================================== dial === */
.dial-wrap { display: flex; align-items: center; gap: 22px; }
.dial { position: relative; flex-shrink: 0; }
.dial svg { display: block; }
.dial .arc { animation: dash 1.15s cubic-bezier(.22,1,.36,1) both; }
.dial-c { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.dial-n { font-family: var(--mono); font-size: 1.5rem; font-weight: 700; letter-spacing: -0.04em; line-height: 1; }
.dial-u { font-family: var(--mono); font-size: 0.54rem; letter-spacing: 0.19em; color: var(--dim); margin-top: 4px; }
.dial-side { min-width: 0; flex: 1; }

/* ======================================================= escala de nivel === */
.tiers { display: flex; margin-top: 18px; border-top: 1px dotted var(--rule); padding-top: 13px; }
.tier { flex: 1; padding-right: 6px; }
.tier-bar { height: 3px; background: var(--rule); }
.tier-lbl {
    font-family: var(--mono); font-size: 0.55rem; font-weight: 600; letter-spacing: 0.1em;
    color: var(--dim); margin-top: 7px;
}
.tier-cut { font-family: var(--mono); font-size: 0.52rem; color: var(--dim); opacity: .6; margin-top: 2px; }
.tier.on .tier-lbl { color: var(--text); }

/* ============================================================= alertas === */
.alert { display: flex; gap: 16px; align-items: flex-start; padding: 20px 22px; }
.alert-ico { flex-shrink: 0; width: 34px; height: 34px; border: 1px solid currentColor; display: flex; align-items: center; justify-content: center; }
.alert-t { font-family: var(--mono); font-size: 0.64rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 9px; }
.alert-b { font-size: 0.885rem; line-height: 1.7; color: var(--muted); }
.alert-b b { color: var(--text); font-weight: 600; }
.alert-b p { margin: 0 0 9px; } .alert-b p:last-child { margin: 0; }

/* ============================================================ consenso === */
.sig { display: flex; align-items: center; gap: 13px; padding: 11px 0; border-bottom: 1px dotted var(--rule); }
.sig:last-child { border-bottom: none; }
.sig-i { font-family: var(--mono); font-size: 0.58rem; color: var(--dim); width: 20px; flex-shrink: 0; }
.sig-n { font-size: 0.85rem; color: var(--muted); flex-shrink: 0; }
.sig-lead { flex: 1; height: 1px; background-image: repeating-linear-gradient(90deg, var(--rule) 0 2px, transparent 2px 5px); }
.sig-v { font-family: var(--mono); font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; flex-shrink: 0; }

/* =========================================================== protocolo === */
.proto { padding: 22px 24px; height: 100%; }
.proto-do   { border-color: rgba(63,191,143,0.3); }
.proto-dont { border-color: rgba(226,80,78,0.3); }
.proto ul { list-style: none; margin: 0; padding: 0; }
.proto li { display: flex; gap: 12px; align-items: flex-start; font-size: 0.87rem; line-height: 1.62; color: var(--muted); padding: 10px 0; }
.proto li + li { border-top: 1px dotted var(--rule); }
.proto li .bl { margin-top: 2px; flex-shrink: 0; }

/* ============================================================= ranking === */
.rank { display: flex; align-items: center; gap: 14px; padding: 13px 4px; border-bottom: 1px dotted var(--rule); }
.rank:last-child { border-bottom: none; }
.rank-i { font-family: var(--mono); font-size: 0.66rem; color: var(--dim); flex-shrink: 0; width: 42px; }
.rank.lead .rank-i { color: var(--ochre); }
.rank-n { font-size: 0.9rem; font-weight: 600; color: var(--text); flex-shrink: 0; max-width: 40%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rank.lead .rank-n { color: var(--text); }
.rank-l { font-family: var(--serif); font-style: italic; font-size: 0.8rem; color: var(--dim); flex-shrink: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rank-lead-dots { flex: 1; height: 1px; min-width: 18px; background-image: repeating-linear-gradient(90deg, var(--rule) 0 2px, transparent 2px 5px); }
.rank-bar { width: 90px; height: 3px; background: var(--rule); flex-shrink: 0; }
.rank-bar > i { display: block; height: 100%; background: var(--steel); }
.rank.lead .rank-bar > i { background: var(--ochre); }
.rank-p { font-family: var(--mono); font-size: 0.73rem; font-weight: 600; color: var(--muted); width: 52px; text-align: right; flex-shrink: 0; }

/* =========================================================== lamina img === */
.spec { padding: 12px; }
.spec-frame { position: relative; line-height: 0; background: #05070A; overflow: hidden; }
.spec-frame img { width: 100%; display: block; }
.cm { position: absolute; width: 17px; height: 17px; border: 1.5px solid var(--ochre); opacity: .8; }
.cm.tl { top: 9px; left: 9px; border-right: none; border-bottom: none; }
.cm.tr { top: 9px; right: 9px; border-left: none; border-bottom: none; }
.cm.bl { bottom: 9px; left: 9px; border-right: none; border-top: none; }
.cm.br { bottom: 9px; right: 9px; border-left: none; border-top: none; }
.xh { position: absolute; top: 50%; left: 50%; width: 26px; height: 26px; transform: translate(-50%,-50%); opacity: .3; }
.xh::before, .xh::after { content: ""; position: absolute; background: var(--ochre); }
.xh::before { top: 50%; left: 0; right: 0; height: 1px; }
.xh::after { left: 50%; top: 0; bottom: 0; width: 1px; }
.spec-label { display: flex; justify-content: space-between; gap: 12px; padding: 11px 4px 3px; }
.spec-label div { font-family: var(--mono); font-size: 0.6rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--dim); }
.spec-label div b { color: var(--muted); font-weight: 500; }

/* ============================================================== fichas === */
.feat { padding: 22px; height: 100%; }
.feat-n { font-family: var(--mono); font-size: 0.6rem; letter-spacing: 0.2em; color: var(--ochre); margin-bottom: 14px; }
.feat-t { font-size: 0.95rem; font-weight: 700; color: var(--text); margin: 12px 0 7px; letter-spacing: -0.012em; }
.feat-d { font-size: 0.845rem; line-height: 1.62; color: var(--muted); }

.step { display: flex; gap: 15px; align-items: flex-start; padding: 13px 0; }
.step + .step { border-top: 1px dotted var(--rule); }
.step-n { font-family: var(--mono); font-size: 0.66rem; font-weight: 700; color: var(--ochre); flex-shrink: 0; width: 22px; padding-top: 1px; }
.step-t { font-size: 0.88rem; font-weight: 600; color: var(--text); }
.step-d { font-size: 0.82rem; color: var(--muted); margin-top: 3px; line-height: 1.55; }

/* =========================================================== registro === */
.reg { display: grid; grid-template-columns: repeat(auto-fit, minmax(115px, 1fr)); }
.reg-c { padding: 13px 16px; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); }
.reg-k { font-family: var(--mono); font-size: 0.56rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--dim); margin-bottom: 5px; }
.reg-v { font-family: var(--mono); font-size: 0.82rem; font-weight: 600; color: var(--text); }

/* =============================================================== nota === */
.note { display: flex; gap: 12px; align-items: flex-start; padding: 14px 18px; font-size: 0.825rem; line-height: 1.6; color: var(--dim); }
.note b { color: var(--muted); font-weight: 600; }

/* ================================================ widgets de streamlit === */
[data-testid="stFileUploader"] section {
    background: var(--surface); border: 1px dashed var(--line);
    border-radius: var(--radius); padding: 26px; transition: all .2s ease;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--ochre); background: var(--surface-2); }
[data-testid="stFileUploader"] label, [data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span { color: var(--muted) !important; }
[data-testid="stFileUploader"] button {
    background: transparent !important; border: 1px solid var(--line) !important;
    color: var(--ochre) !important; font-family: var(--mono) !important;
    font-size: 0.72rem !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; border-radius: 2px !important;
}
[data-testid="stFileUploader"] button:hover { border-color: var(--ochre) !important; background: rgba(224,164,88,0.07) !important; }

/* Selector de sensibilidad como control segmentado. */
div[role="radiogroup"] { gap: 0 !important; border: 1px solid var(--line); border-radius: 2px; overflow: hidden; }
div[role="radiogroup"] > label {
    flex: 1; margin: 0 !important; padding: 9px 6px !important;
    justify-content: center; border-right: 1px solid var(--line);
    transition: background .18s ease;
}
div[role="radiogroup"] > label:last-child { border-right: none; }
div[role="radiogroup"] > label:hover { background: rgba(224,164,88,0.06); }
div[role="radiogroup"] > label > div:first-child { display: none !important; }
div[role="radiogroup"] > label p {
    font-family: var(--mono) !important; font-size: 0.68rem !important;
    letter-spacing: 0.11em !important; text-transform: uppercase !important;
    color: var(--dim) !important; text-align: center;
}
div[role="radiogroup"] > label:has(input:checked) { background: rgba(224,164,88,0.11); }
div[role="radiogroup"] > label:has(input:checked) p { color: var(--ochre) !important; font-weight: 700 !important; }

.stCheckbox label p { color: var(--muted) !important; font-size: 0.84rem !important; }

[data-testid="stImage"] img { border: 1px solid var(--line); }
[data-testid="stImageCaption"] { color: var(--dim) !important; font-family: var(--mono); font-size: 0.62rem !important; letter-spacing: 0.16em; text-align: center; }

.stTabs [data-baseweb="tab-list"] { gap: 0; background: transparent; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 0; padding: 11px 20px;
    font-family: var(--mono); font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase; color: var(--dim);
}
.stTabs [data-baseweb="tab"]:hover { color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--ochre) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--ochre) !important; height: 2px; }

[data-testid="stStatusWidget"], [data-testid="stExpander"] details,
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important; border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stStatusWidget"] p, [data-testid="stExpander"] summary p,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    color: var(--muted) !important; font-family: var(--mono) !important; font-size: 0.76rem !important;
}

[data-testid="stAlert"] { background: rgba(226,80,78,0.07) !important; border: 1px solid rgba(226,80,78,0.3) !important; border-radius: var(--radius) !important; }
[data-testid="stAlert"] p { color: var(--vermilion) !important; }
[data-testid="stSpinner"] p { color: var(--muted) !important; font-family: var(--mono) !important; font-size: 0.76rem !important; }

[data-testid="stDownloadButton"] button {
    background: transparent !important; border: 1px solid var(--line) !important;
    color: var(--muted) !important; border-radius: 2px !important;
    font-family: var(--mono) !important; font-size: 0.7rem !important;
    letter-spacing: 0.13em !important; text-transform: uppercase !important;
}
[data-testid="stDownloadButton"] button:hover { border-color: var(--ochre) !important; color: var(--ochre) !important; background: rgba(224,164,88,0.07) !important; }

[data-baseweb="tooltip"], [data-baseweb="popover"] > div { background: var(--surface-3) !important; color: var(--text) !important; border-radius: 2px !important; }
hr { border-color: var(--line) !important; }
[data-testid="stCaptionContainer"] p { color: var(--dim) !important; font-size: 0.8rem !important; }

::-webkit-scrollbar { width: 11px; height: 11px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(224,164,88,0.2); border: 3px solid var(--bg); }
::-webkit-scrollbar-thumb:hover { background: rgba(224,164,88,0.36); }
::selection { background: rgba(224,164,88,0.3); color: #fff; }

@media (max-width: 900px) {
    .block-container { padding: 1.2rem 0.9rem 3rem; }
    .hero-body { flex-direction: column; gap: 20px; padding: 22px; }
    .hero-file { width: 100%; border-left: none; border-top: 1px solid var(--line); padding: 14px 0 0; }
    .dial-wrap { flex-direction: column; align-items: flex-start; gap: 16px; }
    .rank-l, .rank-bar { display: none; }
    .rank-n { max-width: 70%; }
}
"""


def minify_css(css: str) -> str:
    """Comprime el CSS a una sola línea.

    Imprescindible: el parser de markdown de Streamlit cierra un bloque HTML en
    cuanto encuentra una línea en blanco, de modo que un <style> con saltos de
    línea acaba imprimiéndose como texto plano en la página.
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"\s+", " ", css)
    css = re.sub(r"\s*([{}:;,>])\s*", r"\1", css)
    css = re.sub(r";}", "}", css)
    return css.strip()


st.markdown(
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    "family=Inter:wght@400;500;600;700&"
    "family=JetBrains+Mono:wght@400;500;600;700&"
    'family=Spectral:ital,wght@0,600;1,400;1,500&display=swap" rel="stylesheet">'
    f"<style>{minify_css(_ROOT_VARS + _CSS)}</style>",
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
    "scan": (
        '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
        '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M3 12h18"/>'
    ),
    "compass": '<circle cx="12" cy="12" r="10"/><path d="m16.2 7.8-2.9 6.4-6.4 2.9 2.9-6.4 6.4-2.9Z"/>',
    "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z"/>',
    "ruler": '<path d="M3 15 15 3l6 6L9 21l-6-6Z"/><path d="m7 11 2 2"/><path d="m11 7 2 2"/><path d="m5 13 2 2"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>',
}


def ico(name: str, size: int = 16, color: str = "currentColor", w: float = 1.7) -> str:
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
    """Inyecta HTML colapsado en una línea (ver nota en minify_css)."""
    st.markdown(re.sub(r"\s*\n\s*", " ", markup).strip(), unsafe_allow_html=True)


def sec(num: str, title: str, tag: str = "") -> None:
    html(
        f'<div class="sec"><span class="sec-n">§{num}</span>'
        f'<span class="sec-t">{title}</span><span class="sec-rule"></span>'
        f'<span class="sec-tag">{tag}</span></div>'
    )


def dial(prob: float, color: str, radius: int = 52, stroke: int = 7) -> str:
    """Esfera graduada: anillo de progreso con corona de marcas."""
    p = min(max(prob, 0.0), 1.0)
    circ = 2 * math.pi * radius
    pad = stroke + 12
    size = (radius + pad) * 2
    c = radius + pad

    ticks = []
    for i in range(60):
        ang = math.radians(i * 6 - 90)
        major = i % 5 == 0
        r1 = radius + stroke / 2 + 4
        r2 = r1 + (6 if major else 3)
        ticks.append(
            f'<line x1="{c + r1 * math.cos(ang):.2f}" y1="{c + r1 * math.sin(ang):.2f}" '
            f'x2="{c + r2 * math.cos(ang):.2f}" y2="{c + r2 * math.sin(ang):.2f}" '
            f'stroke="{THEME["ochre"]}" stroke-width="{1.3 if major else 0.8}" '
            f'opacity="{0.55 if major else 0.25}"/>'
        )

    return (
        f'<div class="dial" style="width:{size}px;height:{size}px">'
        f'<svg width="{size}" height="{size}">{"".join(ticks)}'
        f'<g transform="rotate(-90 {c} {c})">'
        f'<circle cx="{c}" cy="{c}" r="{radius}" fill="none" stroke="{THEME["rule"]}" stroke-width="{stroke}"/>'
        f'<circle class="arc" cx="{c}" cy="{c}" r="{radius}" fill="none" stroke="{color}" '
        f'stroke-width="{stroke}" stroke-linecap="butt" stroke-dasharray="{circ:.2f}" '
        f'stroke-dashoffset="{circ * (1 - p):.2f}" style="--circ:{circ:.2f}px"/></g></svg>'
        f'<div class="dial-c"><div class="dial-n" style="color:{color}">{p * 100:.0f}'
        f'<span style="font-size:.72rem">%</span></div>'
        f'<div class="dial-u">ÍNDICE</div></div></div>'
    )


def tier_scale(prob: float, threshold: float) -> str:
    active = risk_tier(prob, threshold)[0]
    cells = []
    for (label, color, _), cut in zip(_TIER_META, tier_cuts(threshold)):
        on = label == active
        cells.append(
            f'<div class="tier {"on" if on else ""}">'
            f'<div class="tier-bar" style="background:{color if on else "var(--rule)"}"></div>'
            f'<div class="tier-lbl" style="{f"color:{color}" if on else ""}">{label}</div>'
            f'<div class="tier-cut">≥{cut * 100:.0f}%</div></div>'
        )
    return f'<div class="tiers">{"".join(cells)}</div>'


def alert_box(kind: str, title: str, paragraphs: list[str]) -> None:
    crit = kind == "critical"
    color = THEME["vermilion"] if crit else THEME["amber"]
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    html(
        f'<div class="plate alert rise" style="border-color:{color}55;'
        f'background:linear-gradient(160deg,{color}12,var(--surface) 60%)">'
        f'<div class="alert-ico" style="color:{color}">{ico("shield_alert" if crit else "alert", 17, color)}</div>'
        f'<div><div class="alert-t" style="color:{color}">{title}</div>'
        f'<div class="alert-b">{body}</div></div></div>'
    )


def protocol_box(kind: str, idx: str, title: str, items: list[str]) -> None:
    is_do = kind == "do"
    color = THEME["jade"] if is_do else THEME["vermilion"]
    bullet = ico("check" if is_do else "cross", 13, color, 2.4)
    lis = "".join(f'<li><span class="bl">{bullet}</span><span>{i}</span></li>' for i in items)
    html(
        f'<div class="plate proto proto-{"do" if is_do else "dont"} rise rise-2">'
        f'<div class="card-head"><span class="card-idx" style="color:{color};border-color:{color}55">{idx}</span>'
        f'<span class="card-t" style="color:{color}">{title}</span></div>'
        f"<ul>{lis}</ul></div>"
    )


def signal(i: int, name: str, value: str, color: str) -> str:
    return (
        f'<div class="sig"><span class="sig-i">{i:02d}</span>'
        f'<span class="sig-n">{name}</span><span class="sig-lead"></span>'
        f'<span class="sig-v" style="color:{color}">{value}</span></div>'
    )


def to_b64(img, max_w: int = 1200) -> str:
    """PIL.Image o ndarray → data URI PNG, reescalado para no inflar el HTML."""
    if not isinstance(img, Image.Image):
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255 if arr.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    img = img.convert("RGB")
    if img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def specimen_plate(img: Image.Image, left: str, right: str) -> str:
    """Lámina de espécimen: marcas de encuadre, retícula central y cartela."""
    return (
        f'<div class="plate spec rise">'
        f'<div class="spec-frame"><img src="{to_b64(img)}">'
        f'<i class="cm tl"></i><i class="cm tr"></i><i class="cm bl"></i><i class="cm br"></i>'
        f'<i class="xh"></i></div>'
        f'<div class="spec-label"><div>{left}</div><div>{right}</div></div></div>'
    )


def comparison_slider(before, after, height: int = 440) -> None:
    """Comparador interactivo original / Grad-CAM con divisor arrastrable."""
    a, b = to_b64(before), to_b64(after)
    components.html(
        f"""
        <style>
          .cmp {{ position:relative; width:100%; height:{height}px; overflow:hidden;
                  border:1px solid rgba(224,164,88,0.17); background:#05070A;
                  user-select:none; touch-action:none; font-family:'JetBrains Mono',monospace; }}
          .cmp img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:contain; pointer-events:none; }}
          .cmp .top {{ clip-path: inset(0 0 0 var(--x, 50%)); }}
          .cmp .bar {{ position:absolute; top:0; bottom:0; width:1px; left:var(--x,50%);
                       background:#E0A458; box-shadow:0 0 12px rgba(224,164,88,.6); cursor:ew-resize; }}
          .cmp .knob {{ position:absolute; top:50%; left:50%; width:34px; height:34px;
                        transform:translate(-50%,-50%); border:1.5px solid #E0A458;
                        background:rgba(11,14,18,.9); display:flex; align-items:center; justify-content:center; }}
          .cmp .cm {{ position:absolute; width:15px; height:15px; border:1.5px solid #E0A458; opacity:.75; }}
          .cmp .tl {{ top:9px; left:9px; border-right:none; border-bottom:none; }}
          .cmp .br {{ bottom:9px; right:9px; border-left:none; border-top:none; }}
          .cmp .tag {{ position:absolute; bottom:11px; font-size:9px; letter-spacing:.18em; font-weight:600;
                       padding:4px 9px; background:rgba(11,14,18,.8); border:1px solid rgba(224,164,88,.25); }}
          .cmp .l {{ left:32px; color:#9C978A; }}
          .cmp .r {{ right:32px; color:#E0A458; }}
        </style>
        <div class="cmp" id="cmp">
          <img src="{a}"><img class="top" src="{b}">
          <div class="bar" id="bar"><div class="knob">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#E0A458"
                 stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <path d="m9 6-6 6 6 6"/><path d="m15 6 6 6-6 6"/></svg>
          </div></div>
          <i class="cm tl"></i><i class="cm br"></i>
          <div class="tag l">ORIGINAL</div><div class="tag r">GRAD-CAM</div>
        </div>
        <script>
          (function() {{
            const box = document.getElementById('cmp'); let drag = false;
            const move = (e) => {{
              const r = box.getBoundingClientRect();
              const cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
              const pct = Math.max(0, Math.min(100, (cx / r.width) * 100));
              box.style.setProperty('--x', pct + '%');
              document.getElementById('bar').style.left = pct + '%';
            }};
            const down = (e) => {{ drag = true; move(e); }};
            box.addEventListener('mousedown', down);
            box.addEventListener('touchstart', down, {{passive:true}});
            window.addEventListener('mouseup', () => drag = false);
            window.addEventListener('touchend', () => drag = false);
            window.addEventListener('mousemove', (e) => drag && move(e));
            window.addEventListener('touchmove', (e) => drag && move(e), {{passive:true}});
          }})();
        </script>
        """,
        height=height + 12,
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
now = datetime.now()

html(
    f"""
    <div class="plate hero rise">
        <div class="ruler"></div>
        <div class="hero-body">
            <div class="hero-left">
                <div class="hero-kicker"><span class="dot"></span>
                    <span class="tag">Ficha de campo digital · Herpetología</span></div>
                <h1 class="hero-title">Snakely <i>AI</i></h1>
                <p class="hero-latin">Serpentes · identificación asistida por visión artificial</p>
                <p class="hero-sub">
                    Clasificación taxonómica y evaluación de riesgo toxicológico a partir de una
                    fotografía, con validación cruzada entre dos modelos independientes y
                    trazabilidad visual de la inferencia.
                </p>
            </div>
            <div class="hero-file">
                <div class="row"><span class="k">Expediente</span><span class="v">SNK·{now:%Y}</span></div>
                <div class="row"><span class="k">Fecha</span><span class="v">{now:%d.%m.%Y}</span></div>
                <div class="row"><span class="k">Sesión</span><span class="v">{now:%H:%M}</span></div>
                <div class="row"><span class="k">Registros</span><span class="v">{st.session_state["runs"]:03d}</span></div>
            </div>
        </div>
        <div class="hero-caps">
            <div class="hero-cap">{ico("dna", 15, THEME["steel"])}<span>Clasificador <b>taxonómico</b></span></div>
            <div class="hero-cap">{ico("shield_alert", 15, THEME["vermilion"])}<span>Detector de <b>toxicidad</b></span></div>
            <div class="hero-cap">{ico("scan", 15, THEME["jade"])}<span>Validación <b>cruzada</b></span></div>
            <div class="hero-cap">{ico("eye", 15, THEME["plum"])}<span>Mapa de <b>atención</b></span></div>
        </div>
    </div>
    """
)

# --- AVISO: primero de todo, antes de cualquier resultado ---------------------

html(
    f"""
    <div class="advisory rise rise-1">
        <div class="advisory-stripe"></div>
        <div class="advisory-body">
            <div class="advisory-ico">{ico("alert", 19, THEME["vermilion"])}</div>
            <div>
                <div class="advisory-t">Léelo antes de usar la herramienta</div>
                <div class="advisory-d">
                    Los resultados son <b>estimaciones probabilísticas</b> de modelos de aprendizaje
                    profundo y pueden equivocarse. No sustituyen el criterio de un herpetólogo ni la
                    atención médica profesional. <b>Ante una mordedura, acude de inmediato al centro
                    de salud más cercano</b> y no esperes a confirmar la especie.
                </div>
            </div>
        </div>
    </div>
    """
)

# =============================================================================
#  7 · ENTRADA Y PARÁMETROS
# =============================================================================

sec("01", "Muestra de análisis", "Entrada")

col_up, col_cfg = st.columns([2.1, 1.1], gap="large")

with col_up:
    image_file = st.file_uploader(
        "Arrastra una fotografía nítida del ejemplar  ·  JPG · PNG · JPEG",
        type=["jpg", "png", "jpeg"],
    )

with col_cfg:
    with st.container(border=True):
        html(
            f'<div class="card-head" style="margin-bottom:12px">'
            f'<span class="card-idx">{ico("ruler", 11, THEME["ochre"])}</span>'
            f'<span class="card-t">Sensibilidad del aviso</span></div>'
            f'<div style="font-size:.82rem;color:var(--muted);line-height:1.55;margin-bottom:10px">'
            f"Define cuántos indicios hacen falta para que el sistema marque un ejemplar "
            f"como peligroso.</div>"
        )
        mode = st.radio(
            "Modo de sensibilidad",
            list(SENSITIVITY),
            index=1,
            horizontal=True,
            label_visibility="collapsed",
        )
        cfg = SENSITIVITY[mode]
        html(
            f'<div style="margin-top:10px;padding-top:11px;border-top:1px dotted var(--rule);'
            f'font-size:.82rem;line-height:1.6;color:var(--muted)">'
            f'<b style="color:{cfg["color"]};font-family:var(--mono);font-size:.68rem;'
            f'letter-spacing:.14em">{mode.upper()}</b> · {cfg["short"]}</div>'
        )
        show_gradcam = st.checkbox("Generar mapa de atención (Grad-CAM)", value=True)

venom_threshold = cfg["threshold"]

# ------------------------------- estado vacío --------------------------------

if image_file is None:
    sec("02", "Cómo funciona", "Manual")

    cols = st.columns(3, gap="large")
    features = [
        ("dna", THEME["steel"], "Clasificador taxonómico",
         "Una red convolucional identifica la especie y devuelve las cinco candidatas "
         "más probables con su nivel de confianza."),
        ("shield_alert", THEME["vermilion"], "Detector de toxicidad",
         "Un segundo modelo, independiente del anterior, estima la probabilidad de que "
         "el ejemplar presente rasgos de especie venenosa."),
        ("scan", THEME["jade"], "Validación cruzada",
         "Ambos dictámenes se contrastan. Ante cualquier discrepancia, el sistema resuelve "
         "siempre a favor de la hipótesis más segura."),
    ]
    for i, (col, (name, color, title, desc)) in enumerate(zip(cols, features), 1):
        with col:
            html(
                f'<div class="plate feat rise rise-1">'
                f'<div class="feat-n">MÓDULO {i:02d}</div>{ico(name, 22, color, 1.5)}'
                f'<div class="feat-t">{title}</div><div class="feat-d">{desc}</div></div>'
            )

    st.write("")
    c_steps, c_tips = st.columns([1.1, 1], gap="large")

    with c_steps:
        steps = [
            ("Sube la fotografía", "Formatos JPG, PNG o JPEG. Se corrige automáticamente la orientación EXIF."),
            ("Elige la sensibilidad", "«Precavido» avisa antes; «Estricto» solo ante indicios claros."),
            ("Revisa el dictamen", "Índice de toxicidad, especie predominante y consenso entre modelos."),
            ("Consulta el protocolo", "Acciones recomendadas y prohibidas ante una mordedura."),
        ]
        rows = "".join(
            f'<div class="step"><div class="step-n">{i:02d}</div>'
            f'<div><div class="step-t">{t}</div><div class="step-d">{d}</div></div></div>'
            for i, (t, d) in enumerate(steps, 1)
        )
        html(
            f'<div class="plate card rise rise-2"><div class="card-head">'
            f'<span class="card-idx">{ico("compass", 11, THEME["ochre"])}</span>'
            f'<span class="card-t">Procedimiento</span></div>{rows}</div>'
        )

    with c_tips:
        tips = [
            "Encuadra el cuerpo completo y, si es posible, la cabeza.",
            "Evita contraluces, reflejos y fondos muy saturados.",
            "Prioriza fotografías enfocadas y con buena resolución.",
            "Una sola serpiente por imagen mejora la precisión.",
        ]
        lis = "".join(
            f'<li><span class="bl">{ico("check", 13, THEME["jade"], 2.4)}</span><span>{t}</span></li>'
            for t in tips
        )
        html(
            f'<div class="plate proto proto-do rise rise-3"><div class="card-head">'
            f'<span class="card-idx" style="color:{THEME["jade"]};border-color:{THEME["jade"]}55">'
            f'{ico("eye", 11, THEME["jade"])}</span>'
            f'<span class="card-t" style="color:{THEME["jade"]}">Notas de campo</span></div>'
            f"<ul>{lis}</ul></div>"
        )

    st.stop()

# =============================================================================
#  8 · INFERENCIA
# =============================================================================

raw_bytes = image_file.getvalue()
image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
image_np = np.array(image)
w, h = image.size

col_a, col_b, col_c = st.columns([1, 1.7, 1])
with col_b:
    html(
        specimen_plate(
            image,
            f"Espécimen · <b>{image_file.name[:26]}</b>",
            f"<b>{w}×{h}</b> px · {len(raw_bytes) / 1024:.0f} KB",
        )
    )

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

# ------------------------------ registro -------------------------------------

reg = [
    ("Registro", f"#{st.session_state['runs']:03d}"),
    ("Hora", f"{datetime.now():%H:%M:%S}"),
    ("Resolución", f"{w}×{h}"),
    ("Inferencia", f"{elapsed:.2f} s"),
    ("Modo", mode.upper()),
    ("Umbral", f"{venom_threshold:.2f}"),
]
html(
    '<div class="plate rise rise-1" style="overflow:hidden"><div class="reg">'
    + "".join(
        f'<div class="reg-c"><div class="reg-k">{k}</div><div class="reg-v">{v}</div></div>'
        for k, v in reg
    )
    + "</div></div>"
)

# =============================================================================
#  9 · DICTAMEN
# =============================================================================

sec("02", "Dictamen del sistema", "Resultado")

verdict = "POTENCIALMENTE VENENOSA" if final_is_venomous else "SIN INDICIOS DE VENENO"
tone = "tone-danger" if final_is_venomous else "tone-safe"

col_v, col_s = st.columns([1.06, 1], gap="large")

with col_v:
    html(
        f"""
        <div class="plate card {tone} rise rise-1">
            <div class="stamp" style="color:{tier_color}">{tier_label}</div>
            <div class="card-head">
                <span class="card-idx" style="color:{tier_color};border-color:{tier_color}55">A</span>
                <span class="card-t">Peligrosidad · dictamen prioritario</span>
            </div>
            <div class="dial-wrap">
                {dial(display_prob, tier_color)}
                <div class="dial-side">
                    <div class="card-value">{verdict}</div>
                    <div class="card-note">{tier_desc}</div>
                    <span class="pill" style="color:{tier_color}">
                        {ico("activity", 12, tier_color)} NIVEL {tier_label}</span>
                </div>
            </div>
            {tier_scale(display_prob, venom_threshold)}
        </div>
        """
    )

with col_s:
    grp_color = THEME["vermilion"] if known_venomous else THEME["jade"]
    html(
        f"""
        <div class="plate card tone-info rise rise-2">
            <div class="card-head">
                <span class="card-idx" style="color:{THEME["steel"]};border-color:{THEME["steel"]}55">B</span>
                <span class="card-t">Determinación taxonómica</span>
            </div>
            <div class="dial-wrap">
                {dial(species_prob, THEME["steel"])}
                <div class="dial-side">
                    <div class="card-value">{species_name}</div>
                    <div class="card-latin">{top_raw}</div>
                    <span class="pill" style="color:{grp_color}">
                        {ico("dna", 12, grp_color)}
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
            "Modelos contradictorios · posible falso positivo",
            [
                f"La especie fue identificada como <b>{species_name}</b>, clasificada "
                f"biológicamente como <b>no venenosa</b>, pero el detector de toxicidad "
                f"registró un <b>{venom_prob * 100:.1f}%</b> de rasgos compatibles con veneno.",
                "<b>Criterio de precaución extrema.</b> La morfología del ejemplar, el ángulo "
                "de captura o las condiciones de luz pueden haber inducido error en cualquiera "
                "de los dos modelos. El sistema resuelve a favor de la seguridad.",
                "<b>Recomendación:</b> trata al ejemplar como potencialmente peligroso y "
                "mantén la distancia.",
            ],
        )
    else:
        alert_box(
            "warn",
            "Modelos contradictorios · protocolo preventivo",
            [
                f"El detector de veneno registró un nivel bajo (<b>{venom_prob * 100:.1f}%</b>), "
                f"pero la especie identificada es <b>{species_name}</b>, perteneciente a un grupo "
                f"<b>potencialmente venenoso</b>.",
                "<b>Recomendación:</b> se aplican los protocolos de seguridad de forma preventiva.",
            ],
        )

# =============================================================================
#  10 · CONSENSO
# =============================================================================

sec("03", "Consenso entre modelos", "Auditoría")

agree_color = THEME["vermilion"] if contradiction else THEME["jade"]
agree_text = "DISCREPANCIA" if contradiction else "CONVERGENTE"
conf_color = (
    THEME["jade"] if species_prob >= 0.70
    else THEME["amber"] if species_prob >= 0.45
    else THEME["vermilion"]
)
conf_text = "ALTA" if species_prob >= 0.70 else "MEDIA" if species_prob >= 0.45 else "BAJA"

col_sig, col_int = st.columns([1.55, 1], gap="large")

with col_sig:
    rows = (
        signal(1, "Detector de toxicidad",
               f"{venom_prob * 100:.1f}% · {'POSITIVO' if is_venomous else 'NEGATIVO'}",
               THEME["vermilion"] if is_venomous else THEME["jade"])
        + signal(2, "Clasificador taxonómico",
                 f"{species_prob * 100:.1f}% · {conf_text}", conf_color)
        + signal(3, "Grupo biológico de la especie",
                 "VENENOSO" if known_venomous else "NO VENENOSO",
                 THEME["vermilion"] if known_venomous else THEME["jade"])
        + signal(4, "Validación cruzada", agree_text, agree_color)
        + signal(5, "Dictamen final aplicado",
                 "PELIGROSA" if final_is_venomous else "NO PELIGROSA",
                 THEME["vermilion"] if final_is_venomous else THEME["jade"])
    )
    html(
        f'<div class="plate card tone-plum rise rise-1"><div class="card-head">'
        f'<span class="card-idx" style="color:{THEME["plum"]};border-color:{THEME["plum"]}55">C</span>'
        f'<span class="card-t">Señales del pipeline</span></div>{rows}</div>'
    )

with col_int:
    interp = (
        "Los dos modelos discrepan. El sistema ha aplicado el criterio más conservador "
        "y ha elevado el nivel de riesgo por precaución."
        if contradiction
        else "Ambos modelos coinciden, lo que refuerza la fiabilidad del resultado. Aun así, "
        "considera el margen de error inherente a toda predicción."
    )
    html(
        f'<div class="plate card {"tone-danger" if contradiction else "tone-safe"} rise rise-2">'
        f'<div class="card-head">'
        f'<span class="card-idx" style="color:{agree_color};border-color:{agree_color}55">D</span>'
        f'<span class="card-t">Lectura</span></div>'
        f'<div class="card-value" style="font-size:1.05rem;color:{agree_color}">{agree_text}</div>'
        f'<div class="card-note">{interp}</div></div>'
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
    sec("04", "Protocolo de primeros auxilios", "Emergencia")
    c_do, c_dont = st.columns(2, gap="large")
    with c_do:
        protocol_box("do", "E", "Acciones recomendadas", recs.get("que_hacer", []))
    with c_dont:
        protocol_box("dont", "F", "Acciones prohibidas", recs.get("nunca_hacer", []))

# =============================================================================
#  12 · DETALLE TÉCNICO
# =============================================================================

sec("05", "Detalle técnico", "Anexo")

tab_rank, tab_cam = st.tabs(["Ranking de especies", "Mapa de atención"])

with tab_rank:
    st.caption(
        "Distribución de probabilidad sobre las cinco especies más afines detectadas "
        "por el clasificador taxonómico."
    )
    rows = []
    total = len(top_predictions)
    for i, pred in enumerate(top_predictions, 1):
        pct = min(pred["probability"], 1.0) * 100
        rows.append(
            f'<div class="rank {"lead" if i == 1 else ""}">'
            f'<span class="rank-i">{i:02d}/{total:02d}</span>'
            f'<span class="rank-n">{pred["spanish_name"]}</span>'
            f'<span class="rank-l">{pred["raw_name"]}</span>'
            f'<span class="rank-lead-dots"></span>'
            f'<span class="rank-bar"><i style="width:{pct:.2f}%"></i></span>'
            f'<span class="rank-p">{pct:.2f}%</span></div>'
        )
    html(f'<div class="plate card">{"".join(rows)}</div>')

    tail = 1.0 - sum(min(p["probability"], 1.0) for p in top_predictions)
    if tail > 0.005:
        html(
            f'<div class="plate note" style="margin-top:12px">{ico("info", 15, THEME["dim"])}'
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
            f'<div class="plate note" style="margin-top:12px">{ico("flame", 15, THEME["amber"])}'
            f"<span>Activa <b>Generar mapa de atención (Grad-CAM)</b> en el panel de "
            f"sensibilidad para desplegar la interpretabilidad visual del modelo.</span></div>"
        )

# =============================================================================
#  13 · CIERRE E INFORME
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
<title>Ficha Snakely · {ts}</title><style>
body{{font-family:Georgia,'Times New Roman',serif;max-width:740px;margin:44px auto;padding:0 24px;
color:#14181D;line-height:1.62}}
h1{{font-size:1.5rem;margin:0 0 4px;letter-spacing:-.02em}}
h2{{font-size:.74rem;margin:30px 0 10px;text-transform:uppercase;letter-spacing:.17em;
color:#8A7A5E;font-family:ui-monospace,monospace}}
.sub{{color:#6B7280;font-size:.85rem;margin:0 0 24px;font-family:ui-monospace,monospace}}
.rule{{height:2px;background:#14181D;margin:0 0 22px}}
.box{{border:1px solid #DDD6C8;border-left:4px solid {tier_color};padding:15px 19px;margin-bottom:11px}}
.k{{font-size:.63rem;text-transform:uppercase;letter-spacing:.16em;color:#8A7A5E;
font-family:ui-monospace,monospace}}
.v{{font-size:1.14rem;font-weight:700;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:.88rem}}
th,td{{text-align:left;padding:8px 6px;border-bottom:1px dotted #DDD6C8}}
th{{font-size:.63rem;text-transform:uppercase;letter-spacing:.14em;color:#8A7A5E;
font-family:ui-monospace,monospace}}
ul{{margin:0;padding-left:20px}} li{{margin-bottom:5px;font-size:.9rem}}
.warn{{border:1px solid #E0A458;background:#FDF6EA;padding:11px 15px;font-size:.88rem}}
.foot{{margin-top:34px;padding-top:15px;border-top:1px solid #DDD6C8;font-size:.78rem;color:#6B7280}}
</style></head><body>
<h1>Ficha de análisis · Snakely AI</h1>
<p class="sub">{ts} · {image_file.name} · modo {mode.upper()} (umbral {venom_threshold:.2f})</p>
<div class="rule"></div>
{warn}
<div class="box"><div class="k">Dictamen de peligrosidad</div>
<div class="v">{verdict} — nivel {tier_label}</div>
<div style="font-size:.86rem;color:#6B7280">Índice de toxicidad: {venom_prob * 100:.1f}% · {tier_desc}</div></div>
<div class="box" style="border-left-color:#5B9FD6"><div class="k">Determinación taxonómica</div>
<div class="v">{species_name}</div>
<div style="font-size:.86rem;color:#6B7280"><i>{top_raw}</i> · {species_prob * 100:.1f}% de coincidencia</div></div>
<h2>Ranking de especies</h2>
<table><tr><th>#</th><th>Nombre común</th><th>Nombre científico</th><th>Prob.</th></tr>{rank_html}</table>
<h2>Acciones recomendadas</h2><ul>{do_html or "<li>Sin recomendaciones disponibles.</li>"}</ul>
<h2>Acciones prohibidas</h2><ul>{dont_html or "<li>Sin recomendaciones disponibles.</li>"}</ul>
<p class="foot">Resultados generados por modelos de aprendizaje profundo. Son estimaciones
probabilísticas y pueden contener errores. No sustituyen el criterio de un herpetólogo ni la
atención médica profesional. Ante una mordedura, acude de inmediato al centro de salud más cercano.</p>
</body></html>"""


st.write("")
col_note, col_btn = st.columns([2.3, 1], gap="large")

with col_note:
    html(
        f'<div class="plate note">{ico("book", 15, THEME["ochre"])}'
        f"<span><b>Recordatorio.</b> Este dictamen es orientativo. Ante una mordedura, acude "
        f"de inmediato al centro de salud más cercano y no esperes a confirmar la especie.</span></div>"
    )

with col_btn:
    st.download_button(
        "Descargar ficha (HTML)",
        data=build_report(),
        file_name=f"snakely_ficha_{datetime.now():%Y%m%d_%H%M}.html",
        mime="text/html",
        use_container_width=True,
    )
