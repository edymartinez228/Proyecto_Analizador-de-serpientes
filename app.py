"""
Snakely AI — Plataforma de identificación de ofidios y diagnóstico de peligrosidad.

Archivo único y autosuficiente: incluye el tema nativo de Streamlit, el sistema
de diseño CSS, la librería de iconos SVG y toda la interfaz.

Ejecutar:  streamlit run app.py
"""

import os
import pathlib

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from utils.model_utils import (
    cross_validate_venom_risk,
    generate_gradcam,
    load_species_model,
    load_venom_model,
    predict_species,
    predict_venom,
)

# ==========================================================================
#  BOOTSTRAP DEL TEMA NATIVO
#  Streamlit lee .streamlit/config.toml al arrancar el proceso. Este bloque lo
#  genera si no existe, de modo que el archivo siga siendo autosuficiente:
#  a partir del siguiente arranque los widgets nativos (menús, tooltips,
#  selectores) heredan la paleta oscura de la aplicación.
# ==========================================================================

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
maxUploadSize = 20
"""


def _bootstrap_theme() -> None:
    try:
        cfg = pathlib.Path(".streamlit") / "config.toml"
        if not cfg.exists():
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_text(_CONFIG_TOML, encoding="utf-8")
    except OSError:
        # Entornos de solo lectura (p. ej. algunos despliegues): el CSS propio
        # ya cubre la apariencia, así que se ignora en silencio.
        pass


_bootstrap_theme()

# ==========================================================================
#  CONFIGURACIÓN DE PÁGINA
# ==========================================================================

st.set_page_config(
    page_title="Snakely · Analizador de Ofidios",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

VENOM_THRESHOLD = 0.50

# ==========================================================================
#  SISTEMA DE DISEÑO — TOKENS
# ==========================================================================

THEME = {
    "bg": "#0A0E14",
    "surface": "#111721",
    "surface_2": "#161E2B",
    "border": "#1F2937",
    "border_soft": "rgba(148, 163, 184, 0.12)",
    "text": "#E6EDF7",
    "text_muted": "#8B98AC",
    "text_dim": "#5D6B80",
    "accent": "#10E098",
    "accent_dim": "#0B9E6E",
    "danger": "#FF4D5E",
    "danger_dim": "#B02532",
    "warning": "#FFB020",
    "info": "#3B9EFF",
}

# ==========================================================================
#  LIBRERÍA DE ICONOS (SVG inline, stroke heredado vía currentColor)
# ==========================================================================

_ICON_PATHS = {
    "shield": '<path d="M12 2 4 5.5v6c0 5 3.4 9.2 8 10.5 4.6-1.3 8-5.5 8-10.5v-6L12 2Z"/>',
    "shield_alert": (
        '<path d="M12 2 4 5.5v6c0 5 3.4 9.2 8 10.5 4.6-1.3 8-5.5 8-10.5v-6L12 2Z"/>'
        '<path d="M12 8v4"/><path d="M12 16h.01"/>'
    ),
    "dna": (
        '<path d="M4 3c0 6 16 6 16 12"/><path d="M20 3c0 6-16 6-16 12"/>'
        '<path d="M4 21c0-2.2 16-2.2 16 0"/><path d="M7 6h10"/><path d="M8 17h8"/>'
    ),
    "upload": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<path d="m7 9 5-5 5 5"/><path d="M12 4v12"/>'
    ),
    "activity": '<path d="M3 12h4l3 8 4-16 3 8h4"/>',
    "layers": (
        '<path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5"/>'
        '<path d="m3 17 9 5 9-5"/>'
    ),
    "flame": (
        '<path d="M12 22c4 0 7-2.6 7-6.5 0-4-3-6-4.5-9.5C13 9 11 8 11 5 8 7 5 10 5 15.5 5 19.4 8 22 12 22Z"/>'
    ),
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "cross": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "alert": (
        '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>'
        '<path d="M12 9v4"/><path d="M12 17h.01"/>'
    ),
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-5"/><path d="M12 8h.01"/>',
    "kit": (
        '<rect x="2" y="7" width="20" height="14" rx="2"/>'
        '<path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>'
        '<path d="M12 11v6"/><path d="M9 14h6"/>'
    ),
    "chart": (
        '<path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6"/>'
        '<rect x="12" y="8" width="3" height="10"/><rect x="17" y="4" width="3" height="14"/>'
    ),
    "eye": (
        '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/>'
        '<circle cx="12" cy="12" r="3"/>'
    ),
    "cpu": (
        '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v3"/>'
        '<path d="M15 2v3"/><path d="M9 19v3"/><path d="M15 19v3"/><path d="M2 9h3"/>'
        '<path d="M2 15h3"/><path d="M19 9h3"/><path d="M19 15h3"/>'
    ),
    "spark": '<path d="M12 2v6"/><path d="m5 5 4 4"/><path d="M2 12h6"/><path d="m5 19 4-4"/><path d="M12 22v-6"/><path d="m19 19-4-4"/><path d="M22 12h-6"/><path d="m19 5-4 4"/>',
}


def icon(name: str, size: int = 18, color: str = "currentColor", stroke: float = 1.8) -> str:
    """Devuelve un SVG inline listo para inyectar en HTML."""
    path = _ICON_PATHS.get(name, _ICON_PATHS["info"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:-0.15em;flex-shrink:0">{path}</svg>'
    )


# ==========================================================================
#  ESTILOS
# ==========================================================================

def inject_styles() -> None:
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: {THEME["bg"]};
                --surface: {THEME["surface"]};
                --surface-2: {THEME["surface_2"]};
                --border: {THEME["border"]};
                --border-soft: {THEME["border_soft"]};
                --text: {THEME["text"]};
                --muted: {THEME["text_muted"]};
                --dim: {THEME["text_dim"]};
                --accent: {THEME["accent"]};
                --accent-dim: {THEME["accent_dim"]};
                --danger: {THEME["danger"]};
                --warning: {THEME["warning"]};
                --info: {THEME["info"]};
                --radius: 14px;
            }}

            [data-testid="collapsedControl"], #MainMenu, footer, header {{ display: none !important; }}

            html, body, [class*="css"], .stApp {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                -webkit-font-smoothing: antialiased;
            }}

            .stApp {{
                background:
                    radial-gradient(1100px 520px at 12% -10%, rgba(16, 224, 152, 0.10), transparent 60%),
                    radial-gradient(900px 460px at 95% 0%, rgba(59, 158, 255, 0.08), transparent 55%),
                    var(--bg);
                color: var(--text);
            }}

            .block-container {{
                max-width: 1180px;
                padding-top: 2.6rem;
                padding-bottom: 4rem;
            }}

            /* ---------- HERO ---------- */
            .hero {{
                border: 1px solid var(--border-soft);
                border-radius: 20px;
                background: linear-gradient(150deg, rgba(16,224,152,0.07) 0%, rgba(17,23,33,0.85) 45%, rgba(17,23,33,0.95) 100%);
                padding: 30px 34px;
                margin-bottom: 26px;
                position: relative;
                overflow: hidden;
            }}
            .hero::after {{
                content: "";
                position: absolute; inset: 0;
                background: linear-gradient(90deg, transparent, rgba(16,224,152,0.35), transparent);
                height: 1px; top: 0;
            }}
            .hero-eyebrow {{
                display: inline-flex; align-items: center; gap: 7px;
                font-size: 0.68rem; font-weight: 700; letter-spacing: 0.16em;
                text-transform: uppercase; color: var(--accent);
                background: rgba(16,224,152,0.09);
                border: 1px solid rgba(16,224,152,0.25);
                padding: 5px 11px; border-radius: 999px; margin-bottom: 16px;
            }}
            .hero-title {{
                font-size: 2.6rem; font-weight: 800; letter-spacing: -0.035em;
                line-height: 1.05; margin: 0 0 10px 0; color: var(--text);
            }}
            .hero-title span {{
                background: linear-gradient(95deg, var(--accent) 0%, #6EE7F9 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            }}
            .hero-sub {{
                font-size: 1.02rem; color: var(--muted); max-width: 640px;
                line-height: 1.6; margin: 0;
            }}
            .hero-meta {{
                display: flex; flex-wrap: wrap; gap: 22px; margin-top: 22px;
                padding-top: 18px; border-top: 1px solid var(--border-soft);
            }}
            .hero-meta-item {{
                display: flex; align-items: center; gap: 8px;
                font-size: 0.8rem; color: var(--dim); font-weight: 500;
            }}
            .hero-meta-item b {{ color: var(--muted); font-weight: 600; }}

            /* ---------- SECCIONES ---------- */
            .section-head {{
                display: flex; align-items: center; gap: 10px;
                margin: 30px 0 14px 0;
            }}
            .section-head .st {{
                font-size: 1.05rem; font-weight: 700; letter-spacing: -0.015em;
                color: var(--text);
            }}
            .section-head .rule {{
                flex: 1; height: 1px;
                background: linear-gradient(90deg, var(--border-soft), transparent);
            }}

            /* ---------- TARJETAS ---------- */
            .card {{
                background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
                border: 1px solid var(--border-soft);
                border-radius: var(--radius);
                padding: 22px 24px;
                position: relative; overflow: hidden; height: 100%;
                transition: border-color .2s ease, transform .2s ease;
            }}
            .card::before {{
                content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
                background: var(--info);
            }}
            .card.is-danger::before {{ background: linear-gradient(180deg, var(--danger), var(--danger)); }}
            .card.is-safe::before   {{ background: linear-gradient(180deg, var(--accent), var(--accent-dim)); }}
            .card.is-danger {{ background: linear-gradient(135deg, rgba(255,77,94,0.08) 0%, var(--surface) 60%); }}
            .card.is-safe   {{ background: linear-gradient(135deg, rgba(16,224,152,0.07) 0%, var(--surface) 60%); }}
            .card.is-info   {{ background: linear-gradient(135deg, rgba(59,158,255,0.07) 0%, var(--surface) 60%); }}

            .card-label {{
                display: flex; align-items: center; gap: 8px;
                font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em;
                text-transform: uppercase; color: var(--dim); margin-bottom: 12px;
            }}
            .card-value {{
                font-size: 1.62rem; font-weight: 800; letter-spacing: -0.03em;
                line-height: 1.15; color: var(--text); margin-bottom: 4px;
            }}
            .card-value.sm {{ font-size: 1.24rem; }}
            .card-sub {{
                font-size: 0.82rem; color: var(--dim); font-style: italic;
                margin-bottom: 14px;
            }}

            .badge {{
                display: inline-flex; align-items: center; gap: 6px;
                font-size: 0.74rem; font-weight: 600; letter-spacing: 0.02em;
                padding: 4px 11px; border-radius: 999px;
                font-family: 'JetBrains Mono', monospace;
            }}
            .badge-danger {{ background: rgba(255,77,94,0.13); color: #FF8A96; border: 1px solid rgba(255,77,94,0.28); }}
            .badge-safe   {{ background: rgba(16,224,152,0.12); color: #5EEBBB; border: 1px solid rgba(16,224,152,0.28); }}
            .badge-info   {{ background: rgba(59,158,255,0.12); color: #85C4FF; border: 1px solid rgba(59,158,255,0.28); }}

            /* ---------- GAUGE ---------- */
            .gauge {{ margin-top: 16px; }}
            .gauge-track {{
                position: relative; height: 7px; border-radius: 999px;
                background: rgba(148,163,184,0.13); overflow: visible;
            }}
            .gauge-fill {{
                height: 100%; border-radius: 999px;
                background: linear-gradient(90deg, var(--accent) 0%, var(--warning) 55%, var(--danger) 100%);
            }}
            .gauge-mark {{
                position: absolute; top: -5px; width: 2px; height: 17px;
                background: rgba(230,237,247,0.55); border-radius: 2px;
            }}
            .gauge-legend {{
                display: flex; justify-content: space-between;
                font-size: 0.68rem; color: var(--dim); margin-top: 8px;
                font-family: 'JetBrains Mono', monospace;
            }}

            /* ---------- ALERTAS ---------- */
            .alert {{
                border-radius: var(--radius); padding: 20px 22px;
                border: 1px solid; margin-bottom: 8px;
                display: flex; gap: 15px; align-items: flex-start;
            }}
            .alert-critical {{ background: rgba(255,77,94,0.07); border-color: rgba(255,77,94,0.32); }}
            .alert-warn     {{ background: rgba(255,176,32,0.07); border-color: rgba(255,176,32,0.32); }}
            .alert-ico {{ margin-top: 2px; }}
            .alert-title {{
                font-size: 0.72rem; font-weight: 800; letter-spacing: 0.13em;
                text-transform: uppercase; margin-bottom: 9px;
            }}
            .alert-critical .alert-title {{ color: #FF8A96; }}
            .alert-warn .alert-title {{ color: #FFC963; }}
            .alert-body {{ font-size: 0.9rem; line-height: 1.65; color: var(--muted); }}
            .alert-body b {{ color: var(--text); font-weight: 600; }}
            .alert-body p {{ margin: 0 0 9px 0; }}
            .alert-body p:last-child {{ margin-bottom: 0; }}

            /* ---------- PROTOCOLO ---------- */
            .proto {{
                border-radius: var(--radius); padding: 22px 24px; height: 100%;
                border: 1px solid;
            }}
            .proto-do   {{ background: rgba(16,224,152,0.05); border-color: rgba(16,224,152,0.28); }}
            .proto-dont {{ background: rgba(255,77,94,0.05); border-color: rgba(255,77,94,0.28); }}
            .proto-head {{
                display: flex; align-items: center; gap: 9px;
                font-size: 0.95rem; font-weight: 700; letter-spacing: -0.01em;
                padding-bottom: 13px; margin-bottom: 14px;
                border-bottom: 1px solid var(--border-soft);
            }}
            .proto-do .proto-head   {{ color: #5EEBBB; }}
            .proto-dont .proto-head {{ color: #FF8A96; }}
            .proto ul {{ list-style: none; margin: 0; padding: 0; }}
            .proto li {{
                display: flex; gap: 10px; align-items: flex-start;
                font-size: 0.88rem; line-height: 1.55; color: var(--muted);
                padding: 7px 0;
            }}
            .proto li + li {{ border-top: 1px solid rgba(148,163,184,0.07); }}
            .proto li .bullet {{ margin-top: 2px; }}

            /* ---------- RANKING ---------- */
            .rank-row {{
                display: flex; align-items: center; gap: 16px;
                padding: 13px 16px; border-radius: 11px;
                background: var(--surface); border: 1px solid var(--border-soft);
                margin-bottom: 9px;
            }}
            .rank-row.top {{ border-color: rgba(59,158,255,0.35); background: rgba(59,158,255,0.05); }}
            .rank-idx {{
                width: 26px; height: 26px; border-radius: 8px; flex-shrink: 0;
                display: flex; align-items: center; justify-content: center;
                font-size: 0.74rem; font-weight: 700; color: var(--dim);
                background: rgba(148,163,184,0.09); font-family: 'JetBrains Mono', monospace;
            }}
            .rank-row.top .rank-idx {{ background: rgba(59,158,255,0.18); color: #85C4FF; }}
            .rank-main {{ flex: 1; min-width: 0; }}
            .rank-name {{
                font-size: 0.92rem; font-weight: 600; color: var(--text);
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }}
            .rank-latin {{
                font-size: 0.76rem; color: var(--dim); font-style: italic;
                margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }}
            .rank-bar-wrap {{ width: 34%; flex-shrink: 0; }}
            .rank-bar {{
                height: 5px; border-radius: 999px;
                background: rgba(148,163,184,0.13); overflow: hidden;
            }}
            .rank-bar > div {{
                height: 100%; border-radius: 999px;
                background: linear-gradient(90deg, var(--info), #6EE7F9);
            }}
            .rank-pct {{
                font-size: 0.78rem; font-weight: 600; color: var(--muted);
                font-family: 'JetBrains Mono', monospace; text-align: right;
                margin-top: 6px;
            }}

            /* ---------- NOTA / DISCLAIMER ---------- */
            .note {{
                display: flex; gap: 11px; align-items: flex-start;
                background: var(--surface); border: 1px solid var(--border-soft);
                border-radius: 11px; padding: 15px 18px;
                font-size: 0.83rem; line-height: 1.6; color: var(--dim);
            }}
            .note b {{ color: var(--muted); font-weight: 600; }}

            /* ---------- WIDGETS NATIVOS ---------- */
            [data-testid="stFileUploader"] section {{
                background: var(--surface); border: 1.5px dashed rgba(148,163,184,0.24);
                border-radius: var(--radius); padding: 22px; transition: all .2s ease;
            }}
            [data-testid="stFileUploader"] section:hover {{
                border-color: rgba(16,224,152,0.45); background: var(--surface-2);
            }}
            [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] small {{
                color: var(--muted) !important;
            }}
            [data-testid="stFileUploader"] button {{
                background: rgba(16,224,152,0.10) !important;
                border: 1px solid rgba(16,224,152,0.32) !important;
                color: var(--accent) !important; font-weight: 600 !important;
                border-radius: 9px !important;
            }}

            .stCheckbox label p {{ color: var(--muted) !important; font-size: 0.88rem !important; }}

            [data-testid="stImage"] img {{
                border-radius: var(--radius);
                border: 1px solid var(--border-soft);
            }}
            [data-testid="stImageCaption"] {{
                color: var(--dim) !important; font-size: 0.78rem !important;
                text-align: center; letter-spacing: 0.02em;
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: 6px; background: transparent;
                border-bottom: 1px solid var(--border-soft);
            }}
            .stTabs [data-baseweb="tab"] {{
                background: transparent; border-radius: 9px 9px 0 0;
                padding: 10px 18px; font-size: 0.88rem; font-weight: 600;
                color: var(--dim);
            }}
            .stTabs [aria-selected="true"] {{
                color: var(--accent) !important;
                background: rgba(16,224,152,0.07) !important;
            }}
            .stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--accent) !important; }}

            [data-testid="stStatusWidget"], [data-testid="stExpander"] details {{
                background: var(--surface) !important;
                border: 1px solid var(--border-soft) !important;
                border-radius: var(--radius) !important;
            }}

            [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {{
                gap: 0.4rem;
            }}
            [data-testid="stVerticalBlockBorderWrapper"] {{
                border-radius: var(--radius) !important;
            }}

            /* Fallback: fuerza la paleta oscura aunque aún no exista config.toml */
            [data-testid="stAppViewContainer"], [data-testid="stMain"],
            [data-testid="stHeader"], [data-testid="stToolbar"] {{
                background: transparent !important;
            }}
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li,
            [data-testid="stMarkdownContainer"] span {{
                color: var(--text);
            }}
            h1, h2, h3, h4, h5, h6 {{ color: var(--text) !important; }}

            [data-testid="stStatusWidget"] p,
            [data-testid="stExpander"] summary p,
            [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {{
                color: var(--muted) !important; font-size: 0.87rem !important;
            }}
            [data-testid="stExpander"] summary svg {{ fill: var(--muted) !important; }}

            [data-testid="stAlert"] {{
                background: rgba(255,77,94,0.07) !important;
                border: 1px solid rgba(255,77,94,0.30) !important;
                border-radius: var(--radius) !important;
                color: var(--text) !important;
            }}
            [data-testid="stAlert"] p {{ color: #FFB3BB !important; }}

            [data-testid="stSpinner"] p {{ color: var(--muted) !important; }}
            [data-baseweb="tooltip"], [data-baseweb="popover"] div {{
                background: var(--surface-2) !important; color: var(--text) !important;
                border-radius: 9px !important;
            }}

            ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
            ::-webkit-scrollbar-track {{ background: var(--bg); }}
            ::-webkit-scrollbar-thumb {{
                background: rgba(148,163,184,0.22); border-radius: 999px;
                border: 2px solid var(--bg);
            }}
            ::-webkit-scrollbar-thumb:hover {{ background: rgba(148,163,184,0.35); }}
            ::selection {{ background: rgba(16,224,152,0.28); color: #FFFFFF; }}

            hr {{ border-color: var(--border-soft) !important; }}
            .stCaption, [data-testid="stCaptionContainer"] p {{
                color: var(--dim) !important; font-size: 0.83rem !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()


# ==========================================================================
#  COMPONENTES DE UI
# ==========================================================================

def section(icon_name: str, title: str, color: str = THEME["accent"]) -> None:
    st.markdown(
        f'<div class="section-head">{icon(icon_name, 17, color)}'
        f'<span class="st">{title}</span><span class="rule"></span></div>',
        unsafe_allow_html=True,
    )


def alert_box(kind: str, title: str, paragraphs: list) -> None:
    is_critical = kind == "critical"
    cls = "alert-critical" if is_critical else "alert-warn"
    ico = icon(
        "shield_alert" if is_critical else "alert",
        21,
        THEME["danger"] if is_critical else THEME["warning"],
    )
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    st.markdown(
        f'<div class="alert {cls}"><div class="alert-ico">{ico}</div>'
        f'<div><div class="alert-title">{title}</div>'
        f'<div class="alert-body">{body}</div></div></div>',
        unsafe_allow_html=True,
    )


def protocol_box(kind: str, title: str, items: list) -> None:
    is_do = kind == "do"
    color = THEME["accent"] if is_do else THEME["danger"]
    bullet = icon("check" if is_do else "cross", 14, color, 2.4)
    head_ico = icon("shield" if is_do else "alert", 17, color)
    lis = "".join(
        f'<li><span class="bullet">{bullet}</span><span>{it}</span></li>' for it in items
    )
    st.markdown(
        f'<div class="proto proto-{"do" if is_do else "dont"}">'
        f'<div class="proto-head">{head_ico}<span>{title}</span></div>'
        f"<ul>{lis}</ul></div>",
        unsafe_allow_html=True,
    )


# ==========================================================================
#  CARGA DE MODELOS
# ==========================================================================

def _resolve_model(path: str):
    if not os.path.exists(path):
        existing = (
            ", ".join(os.listdir("models"))
            if os.path.exists("models")
            else "la carpeta 'models' no existe"
        )
        raise FileNotFoundError(
            f"No se encontró '{path}'. Contenido de 'models/': {existing}"
        )
    return path


@st.cache_resource(show_spinner=False)
def get_venom_model():
    return load_venom_model(_resolve_model("models/modelo_veneno.weights.h5"))


@st.cache_resource(show_spinner=False)
def get_species_model():
    return load_species_model(_resolve_model("models/modelo_especie.pth"))


# ==========================================================================
#  HERO
# ==========================================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-eyebrow">{icon("spark", 12, THEME["accent"], 2.2)} Computer Vision · Herpetología</div>
        <h1 class="hero-title">Snakely <span>AI</span></h1>
        <p class="hero-sub">
            Plataforma de identificación taxonómica y evaluación de riesgo toxicológico en ofidios,
            con validación cruzada entre modelos y trazabilidad visual de la inferencia.
        </p>
        <div class="hero-meta">
            <div class="hero-meta-item">{icon("cpu", 14, THEME["text_dim"])}<span>Doble arquitectura&nbsp;<b>CNN</b></span></div>
            <div class="hero-meta-item">{icon("shield", 14, THEME["text_dim"])}<span>Validación cruzada de&nbsp;<b>seguridad</b></span></div>
            <div class="hero-meta-item">{icon("eye", 14, THEME["text_dim"])}<span>Interpretabilidad&nbsp;<b>Grad-CAM</b></span></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
#  ENTRADA
# ==========================================================================

section("upload", "Muestra de análisis")

col_upload, col_options = st.columns([2.4, 1], gap="large")

with col_upload:
    image_file = st.file_uploader(
        "Arrastra una fotografía del ejemplar (JPG · PNG · JPEG)",
        type=["jpg", "png", "jpeg"],
        label_visibility="visible",
    )

with col_options:
    with st.container(border=True):
        st.markdown(
            f'<div class="card-label" style="margin-bottom:6px">'
            f'{icon("layers", 13, THEME["info"])} Opciones de análisis</div>',
            unsafe_allow_html=True,
        )
        show_gradcam = st.checkbox("Generar mapa de atención (Grad-CAM)", value=False)

if image_file is None:
    st.markdown(
        f'<div style="margin-top:26px" class="note">{icon("info", 16, THEME["info"])}'
        f"<span><b>Uso responsable.</b> Snakely es una herramienta de apoyo a la decisión "
        f"y no sustituye el criterio de un herpetólogo ni la atención médica profesional. "
        f"Ante una mordedura, acude de inmediato al centro de salud más cercano.</span></div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ==========================================================================
#  INFERENCIA
# ==========================================================================

image = ImageOps.exif_transpose(Image.open(image_file).convert("RGB"))
image_np = np.array(image)

col_a, col_b, col_c = st.columns([1, 1.6, 1])
with col_b:
    st.image(image, caption="MUESTRA CARGADA", use_container_width=True)

with st.status("Ejecutando pipeline de inferencia…", expanded=True) as status:
    try:
        status.write("Cargando arquitecturas neuronales…")
        venom_model = get_venom_model()
        species_model = get_species_model()

        status.write("Evaluando patrones morfológicos de toxicidad…")
        is_venomous, venom_prob, venom_recommendations = predict_venom(
            venom_model, image_np, VENOM_THRESHOLD
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

# ==========================================================================
#  VALIDACIÓN CRUZADA
# ==========================================================================

from utils.model_utils import VENOMOUS_KEYWORDS  # noqa: E402

top_raw_name = top_predictions[0]["raw_name"]

safety_check = cross_validate_venom_risk(
    species_raw_name=top_raw_name,
    is_venomous_pred=is_venomous,
    species_prob=species_prob,
)

species_lower = top_raw_name.lower()
is_species_known_venomous = any(kw in species_lower for kw in VENOMOUS_KEYWORDS)
is_false_positive_risk = (not is_species_known_venomous) and is_venomous
is_false_negative_risk = is_species_known_venomous and (not is_venomous)
has_contradiction = is_false_positive_risk or is_false_negative_risk

final_is_venomous = safety_check.get("final_is_venomous", is_venomous)

# ==========================================================================
#  DICTAMEN
# ==========================================================================

section("activity", "Dictamen del sistema")

card_class = "is-danger" if final_is_venomous else "is-safe"
badge_class = "badge-danger" if final_is_venomous else "badge-safe"
verdict_color = THEME["danger"] if final_is_venomous else THEME["accent"]
verdict_text = "POTENCIALMENTE VENENOSA" if final_is_venomous else "SIN INDICIOS DE VENENO"
verdict_icon = icon("shield_alert" if final_is_venomous else "shield", 14, verdict_color)

gauge_pct = min(max(venom_prob, 0.0), 1.0) * 100
threshold_pct = VENOM_THRESHOLD * 100

col_venom, col_species = st.columns(2, gap="large")

with col_venom:
    st.markdown(
        f"""
        <div class="card {card_class}">
            <div class="card-label">{icon("shield_alert", 13, verdict_color)} Diagnóstico de peligrosidad · prioritario</div>
            <div class="card-value sm">{verdict_text}</div>
            <span class="badge {badge_class}">{verdict_icon} {venom_prob * 100:.1f}% índice de toxicidad</span>
            <div class="gauge">
                <div class="gauge-track">
                    <div class="gauge-fill" style="width:{gauge_pct:.1f}%"></div>
                    <div class="gauge-mark" style="left:{threshold_pct:.1f}%"></div>
                </div>
                <div class="gauge-legend">
                    <span>0%</span>
                    <span>UMBRAL {threshold_pct:.0f}%</span>
                    <span>100%</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_species:
    st.markdown(
        f"""
        <div class="card is-info">
            <div class="card-label">{icon("dna", 13, THEME["info"])} Especie predominante</div>
            <div class="card-value sm">{species_name}</div>
            <div class="card-sub">{top_raw_name}</div>
            <span class="badge badge-info">{icon("chart", 13, "#85C4FF")} {species_prob * 100:.1f}% de coincidencia</span>
            <div class="gauge">
                <div class="gauge-track">
                    <div class="gauge-fill" style="width:{min(species_prob, 1.0) * 100:.1f}%;background:linear-gradient(90deg,#3B9EFF,#6EE7F9)"></div>
                </div>
                <div class="gauge-legend">
                    <span>CONFIANZA TAXONÓMICA</span>
                    <span>{species_prob * 100:.1f}%</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================================
#  ALERTAS DE CONTRADICCIÓN
# ==========================================================================

if has_contradiction:
    st.write("")
    if is_false_positive_risk:
        alert_box(
            "critical",
            "Modelos contradictorios · posible falso positivo de veneno",
            [
                f"La especie fue identificada como <b>{species_name}</b>, clasificada "
                f"biológicamente como <b>no venenosa</b>, pero el detector de toxicidad "
                f"registró un <b>{venom_prob * 100:.1f}%</b> de características compatibles con veneno.",
                "<b>Criterio de precaución extrema.</b> La morfología del ejemplar, el ángulo de "
                "captura o las condiciones de luz pueden haber inducido error en el modelo "
                "taxonómico o en el de toxicidad. El sistema prioriza la seguridad.",
                "<b>Recomendación:</b> trata al ejemplar como potencialmente peligroso y mantén la distancia.",
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

# ==========================================================================
#  PROTOCOLO DE PRIMEROS AUXILIOS
# ==========================================================================

recommendations = (
    safety_check["recommendations"]
    if safety_check.get("warning_triggered") and safety_check.get("recommendations")
    else venom_recommendations
)

if recommendations:
    section("kit", "Protocolo de primeros auxilios", THEME["danger"])
    col_do, col_dont = st.columns(2, gap="large")
    with col_do:
        protocol_box("do", "Acciones recomendadas", recommendations.get("que_hacer", []))
    with col_dont:
        protocol_box("dont", "Acciones prohibidas", recommendations.get("nunca_hacer", []))

# ==========================================================================
#  DETALLE TÉCNICO
# ==========================================================================

section("layers", "Detalle técnico", THEME["info"])

tab_rankings, tab_gradcam = st.tabs(["Ranking de especies", "Mapa de atención"])

with tab_rankings:
    st.caption(
        "Distribución de probabilidad sobre las cinco especies más afines detectadas por la red."
    )
    rows = []
    for idx, pred in enumerate(top_predictions, 1):
        pct = min(pred["probability"], 1.0) * 100
        rows.append(
            f'<div class="rank-row {"top" if idx == 1 else ""}">'
            f'<div class="rank-idx">{idx:02d}</div>'
            f'<div class="rank-main">'
            f'<div class="rank-name">{pred["spanish_name"]}</div>'
            f'<div class="rank-latin">{pred["raw_name"]}</div>'
            f"</div>"
            f'<div class="rank-bar-wrap">'
            f'<div class="rank-bar"><div style="width:{pct:.2f}%"></div></div>'
            f'<div class="rank-pct">{pct:.2f}%</div>'
            f"</div></div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)

with tab_gradcam:
    if show_gradcam:
        st.caption(
            "Las regiones cálidas indican las zonas de la imagen con mayor peso en la "
            "decisión del clasificador de especie."
        )
        with st.spinner("Generando interpretabilidad visual…"):
            cam_image = generate_gradcam(species_model, image_np)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.image(image, caption="ENTRADA ORIGINAL", use_container_width=True)
        with c2:
            st.image(cam_image, caption="ENFOQUE DEL MODELO", use_container_width=True)
    else:
        st.markdown(
            f'<div class="note" style="margin-top:12px">{icon("flame", 16, THEME["warning"])}'
            f"<span>Activa <b>Generar mapa de atención (Grad-CAM)</b> en el panel superior "
            f"para desplegar la interpretabilidad visual del modelo.</span></div>",
            unsafe_allow_html=True,
        )

# ==========================================================================
#  PIE
# ==========================================================================

st.write("")
st.markdown(
    f'<div class="note">{icon("info", 16, THEME["text_dim"])}'
    f"<span><b>Aviso.</b> Los resultados son estimaciones probabilísticas generadas por "
    f"modelos de aprendizaje profundo y pueden contener errores. No sustituyen el criterio "
    f"de un herpetólogo ni la atención médica profesional. Ante una mordedura, acude de "
    f"inmediato al centro de salud más cercano.</span></div>",
    unsafe_allow_html=True,
)
