import streamlit as st
import base64
import random
import io
from pathlib import Path
from textwrap import dedent
from html import escape
from preprocessing import preprocess, load_dermaga, limit_to_n_days
from config import CONFIG, WEIGHTS
from optimizer_s1 import run_ch1, run_love_bird_optimization
from optimizer_s2 import run_ch2, run_love_bird_s2
from visualization import plot_berth_allocation_by_category
import streamlit.components.v1 as components
from output_formatter import format_output

   
st.set_page_config(
    page_title="Ship Berth Scheduling",
    page_icon="🚢",
    layout="wide"
)

# =========================
# SESSION PAGE
# =========================
if "page" not in st.session_state:
    st.session_state["page"] = "home"


# =========================
# HELPER IMAGE
# =========================
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# =========================
# GLOBAL CSS
# =========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }

    header {
        visibility: hidden;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def show_error_modal(errors):
    st.session_state["show_modal"] = True
    st.session_state["modal_errors"] = errors

def render_modal():
    if not st.session_state.get("show_modal", False):
        return

    errors = st.session_state.get("modal_errors", [])
    error_items = "".join([f"<li>{e}</li>" for e in errors])

    st.markdown(f"""
    <style>
    .modal-overlay {{
        position: fixed;
        top: 0; 
        left: 0;
        width: 100vw; 
        height: 100vh;
        background: rgba(0,0,0,0.4);
        z-index: 999998;
    }}

    :root {{
        --modal-pad-top: clamp(20px, 4.5vw, 30px);
        --modal-pad-x: clamp(18px, 4vw, 36px);
        --modal-icon-size: clamp(42px, 9vw, 54px);
        --modal-icon-mb: clamp(10px, 2.2vw, 14px);
        --modal-title-size: clamp(16px, 3.8vw, 21px);
        --modal-title-mb: clamp(6px, 1.6vw, 8px);
        --modal-body-max-h: clamp(60px, 16vw, 90px);
        --modal-body-mb: 6px;
        --modal-footer-h: clamp(52px, 9vw, 64px);
        --modal-pad-bottom: calc(var(--modal-footer-h) + 8px);
        --modal-btn-w: clamp(130px, 32vw, 170px);
        --modal-btn-h: clamp(36px, 7vw, 42px);

        --modal-box-h: calc(
            var(--modal-pad-top) + var(--modal-icon-size) + var(--modal-icon-mb) +
            (var(--modal-title-size) * 1.3) + var(--modal-title-mb) +
            var(--modal-body-max-h) + var(--modal-body-mb) +
            var(--modal-pad-bottom)
        );
        --modal-ok-offset: calc(
            (var(--modal-box-h) / 2) - (var(--modal-footer-h) / 2) - (var(--modal-btn-h) / 2)
        );
    }}

    .modal-box {{
        position: fixed;
        top: 50%; 
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        border-radius: 18px;

        width: min(420px, 88vw);
        min-height: var(--modal-box-h);
        padding: var(--modal-pad-top) var(--modal-pad-x) var(--modal-pad-bottom) var(--modal-pad-x);
        box-sizing: border-box;

        z-index: 999999;
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
        font-family: 'Poppins', sans-serif;
        text-align: center;
    }}

    .modal-icon {{
        width: var(--modal-icon-size);
        height: var(--modal-icon-size);
        background: #e02424;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto var(--modal-icon-mb) auto;
        font-size: clamp(20px, 4.6vw, 28px);
        color: white;
        box-shadow: 0 4px 16px rgba(224,36,36,0.4);
    }}

    .modal-title {{
        font-size: var(--modal-title-size);
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: var(--modal-title-mb);
    }}

    .modal-body {{
        font-size: clamp(12px, 3vw, 14px);
        color: #555;
        line-height: 1.6;
        text-align: left;
        margin-bottom: var(--modal-body-mb);

        max-height: var(--modal-body-max-h);
        overflow-y: scroll;
        padding-right: 14px;
    }}

    .modal-body ul {{
        padding-left: 18px;
        margin: 0;
    }}

    .modal-body li {{
        margin-bottom: 5px;
    }}

    .modal-body::-webkit-scrollbar {{
        width: 8px;
    }}

    .modal-body::-webkit-scrollbar-track {{
        background: #e6e6e6;
        border-radius: 10px;
    }}

    .modal-body::-webkit-scrollbar-thumb {{
        background: #9d9d9d;
        border-radius: 10px;
    }}

    .modal-body::-webkit-scrollbar-thumb:hover {{
        background: #7f7f7f;
    }}

    .modal-body {{
        scrollbar-width: thin;
        scrollbar-color: #9d9d9d #e6e6e6;
    }}

    .modal-footer {{
        position: absolute;
        left: 0;
        bottom: 0;
        width: 100%;
        height: var(--modal-footer-h);
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 0 0 18px 18px;
        background: white;
    }}

    .st-key-close_modal button {{
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, var(--modal-ok-offset));

        z-index: 9999999;
        width: var(--modal-btn-w) !important;
        height: var(--modal-btn-h) !important;

        background: #e02424 !important;
        border: none !important;
        border-radius: 21px !important;
        box-shadow: 0 4px 14px rgba(224,36,36,0.35) !important;

        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    .st-key-close_modal button p {{
        font-size: clamp(13px, 3.2vw, 15px) !important;
        font-weight: 600 !important;
        color: white !important;
        margin: 0 !important;
    }}

    .st-key-close_modal button:hover {{
        background: #c41e1e !important;
        box-shadow: 0 6px 18px rgba(224,36,36,0.45) !important;
        border: none !important;
    }}

    .st-key-close_modal button:focus {{
        background: #e02424 !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(224,36,36,0.35) !important;
    }}
    </style>

    <div class="modal-overlay"></div>

    <div class="modal-box">
        <div class="modal-icon">✕</div>
        <div class="modal-title">Invalid Data</div>
        <div class="modal-body">
            <ul>{error_items}</ul>
        </div>
        <div class="modal-footer"></div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("OK", key="close_modal"):
        st.session_state["show_modal"] = False
        st.rerun()


# =========================
# ARRIVAL LIMIT MODAL (> MAX_ARRIVAL_DAYS)
# =========================
def show_arrival_limit_dialog(scenario_key: str, info: dict):
    st.session_state["show_arrival_modal"] = True
    st.session_state["arrival_modal_key"] = scenario_key
    st.session_state["arrival_modal_info"] = info


def render_arrival_limit_modal():
    if not st.session_state.get("show_arrival_modal", False):
        return

    info = st.session_state.get("arrival_modal_info", {})
    scenario_key = st.session_state.get("arrival_modal_key", "single")

    total_days = info.get("total_days", "?")
    max_days = info.get("max_days", 3)
    unique_dates = info.get("unique_dates", [])
    date_range = (
        f"{unique_dates[0]} to {unique_dates[-1]}"
        if unique_dates else ""
    )

    st.markdown(f"""
    <style>
    .arrival-modal-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(0,0,0,0.4);
        z-index: 999998;
    }}

    :root {{
        --amodal-pad-top: clamp(20px, 4.5vw, 30px);
        --amodal-pad-x: clamp(18px, 4vw, 36px);
        --amodal-icon-size: clamp(42px, 9vw, 54px);
        --amodal-icon-mb: clamp(10px, 2.2vw, 14px);
        --amodal-title-size: clamp(16px, 3.8vw, 21px);
        --amodal-title-mb: clamp(6px, 1.6vw, 8px);
        --amodal-body-max-h: clamp(90px, 24vw, 140px);
        --amodal-body-mb: 6px;
        --amodal-footer-h: clamp(52px, 9vw, 64px);
        --amodal-pad-bottom: calc(var(--amodal-footer-h) + 8px);
        --amodal-btn-w: clamp(108px, 26vw, 150px);
        --amodal-btn-h: clamp(36px, 7vw, 42px);
        --amodal-btn-gap: clamp(8px, 2.4vw, 14px);

        --amodal-box-h: calc(
            var(--amodal-pad-top) + var(--amodal-icon-size) + var(--amodal-icon-mb) +
            (var(--amodal-title-size) * 1.3) + var(--amodal-title-mb) +
            var(--amodal-body-max-h) + var(--amodal-body-mb) +
            var(--amodal-pad-bottom)
        );
        --amodal-btn-offset: calc(
            (var(--amodal-box-h) / 2) - (var(--amodal-footer-h) / 2) - (var(--amodal-btn-h) / 2)
        );
    }}

    .arrival-modal-box {{
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        border-radius: 18px;

        width: min(420px, 88vw);
        height: var(--amodal-box-h);
        padding: var(--amodal-pad-top) var(--amodal-pad-x) var(--amodal-pad-bottom) var(--amodal-pad-x);
        box-sizing: border-box;

        z-index: 999999;
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
        font-family: 'Poppins', sans-serif;
        text-align: center;
    }}

    .arrival-modal-icon {{
        width: var(--amodal-icon-size);
        height: var(--amodal-icon-size);
        background: #f5a623;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto var(--amodal-icon-mb) auto;
        font-size: clamp(20px, 4.6vw, 28px);
        font-weight: 700;
        color: white;
        box-shadow: 0 4px 16px rgba(245,166,35,0.4);
    }}

    .arrival-modal-title {{
        font-size: var(--amodal-title-size);
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: var(--amodal-title-mb);
    }}

    .arrival-modal-body {{
        font-size: clamp(12px, 3vw, 14px);
        color: #555;
        line-height: 1.55;
        text-align: left;
        margin-bottom: var(--amodal-body-mb);

        max-height: var(--amodal-body-max-h);
        overflow-y: auto;
        padding-right: 6px;
    }}

    .arrival-modal-body::-webkit-scrollbar {{
        width: 6px;
    }}

    .arrival-modal-body::-webkit-scrollbar-track {{
        background: #e6e6e6;
        border-radius: 10px;
    }}

    .arrival-modal-body::-webkit-scrollbar-thumb {{
        background: #9d9d9d;
        border-radius: 10px;
    }}

    .arrival-modal-body {{
        scrollbar-width: thin;
        scrollbar-color: #9d9d9d #e6e6e6;
    }}

    .arrival-modal-footer {{
        position: absolute;
        left: 0;
        bottom: 0;
        width: 100%;
        height: var(--amodal-footer-h);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: var(--amodal-btn-gap);
        border-radius: 0 0 18px 18px;
        background: white;
    }}

    .st-key-confirm_arrival_modal button {{
        position: fixed;
        top: 50%;
        left: calc(50% + (var(--amodal-btn-gap) / 2));
        transform: translate(0, var(--amodal-btn-offset));

        z-index: 9999999;
        width: var(--amodal-btn-w) !important;
        height: var(--amodal-btn-h) !important;

        background: #16006b !important;
        border: none !important;
        border-radius: 21px !important;
        box-shadow: 0 4px 14px rgba(22,0,107,0.3) !important;

        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    .st-key-confirm_arrival_modal button p {{
        font-size: clamp(12px, 3vw, 15px) !important;
        font-weight: 600 !important;
        color: white !important;
        margin: 0 !important;
    }}

    .st-key-confirm_arrival_modal button:hover {{
        background: #25009c !important;
        box-shadow: 0 6px 18px rgba(22,0,107,0.4) !important;
        border: none !important;
    }}

    .st-key-confirm_arrival_modal button:focus {{
        background: #16006b !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(22,0,107,0.3) !important;
    }}

    .st-key-cancel_arrival_modal button {{
        position: fixed;
        top: 50%;
        left: calc(50% - (var(--amodal-btn-gap) / 2) - var(--amodal-btn-w));
        transform: translate(0, var(--amodal-btn-offset));

        z-index: 9999999;
        width: var(--amodal-btn-w) !important;
        height: var(--amodal-btn-h) !important;

        background: white !important;
        border: 2px solid #9d9d9d !important;
        border-radius: 21px !important;
        box-shadow: none !important;

        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    .st-key-cancel_arrival_modal button p {{
        font-size: clamp(12px, 3vw, 15px) !important;
        font-weight: 600 !important;
        color: #333333 !important;
        margin: 0 !important;
    }}

    .st-key-cancel_arrival_modal button:hover {{
        background: #f3f3f3 !important;
        border: 2px solid #9d9d9d !important;
        box-shadow: none !important;
    }}

    .st-key-cancel_arrival_modal button:focus {{
        background: white !important;
        border: 2px solid #9d9d9d !important;
        box-shadow: none !important;
    }}
    </style>

    <div class="arrival-modal-overlay"></div>

    <div class="arrival-modal-box">
        <div class="arrival-modal-icon">!</div>
        <div class="arrival-modal-title">Arrival Data Exceeds {max_days} Days</div>
        <div class="arrival-modal-body">
            The uploaded arrival data spans <b>{total_days} days</b> ({date_range}).<br>
            The system can only schedule a maximum of <b>{max_days} days</b> of
            arrivals. You can trim the data to the first {max_days} days of
            arrivals, or cancel and upload a different file.
        </div>
        <div class="arrival-modal-footer"></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cancel_clicked = st.button("Cancel", key="cancel_arrival_modal")
    with col2:
        confirm_clicked = st.button("OK, Continue", key="confirm_arrival_modal")

    if cancel_clicked:
        st.session_state["show_arrival_modal"] = False
        if "uploaded_file" in st.session_state:
            del st.session_state["uploaded_file"]
        clear_arrival_state()
        st.session_state["page"] = "upload"
        st.rerun()

    if confirm_clicked:
        st.session_state["show_arrival_modal"] = False
        st.session_state[f"arrival_decision_{scenario_key}"] = "limit"
        st.rerun()


def clear_arrival_state():
    for key in [
        "show_arrival_modal", "arrival_modal_key", "arrival_modal_info",
        "arrival_info_single", "arrival_decision_single", "df_kapal_pending_single",
        "arrival_info_reberthing", "arrival_decision_reberthing", "df_kapal_pending_reberthing",
    ]:
        if key in st.session_state:
            del st.session_state[key]


# =========================
# HOME PAGE
# =========================
def home_page():
    bg_path = Path("assets/aerial-view-container-cargo-ship-sea.jpg")
    bg_image = get_base64_image(bg_path)

    st.markdown(
        f"""
        <style>
        .stApp {{
            min-height: 100vh;
            background-image:
                linear-gradient(
                    90deg,
                    rgba(255, 255, 255, 0.90) 0%,
                    rgba(255, 255, 255, 0.62) 35%,
                    rgba(255, 255, 255, 0.12) 65%,
                    rgba(255, 255, 255, 0.00) 100%
                ),
                url("data:image/jpg;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        :root {{
            --home-pad-x: clamp(20px, 7vw, 128px);
            --home-text-block-h: clamp(170px, 27vw, 270px);
        }}

        html, body {{
            height: 100%;
        }}

        [data-testid="stAppViewContainer"] {{
            min-height: 100vh;
        }}

        .home-content-wrapper {{
            position: fixed;
            top: 50%;
            left: 0;
            width: 100%;
            transform: translateY(-50%);
            z-index: 2;
            box-sizing: border-box;
            pointer-events: none;
        }}

        .home-content {{
            padding-left: var(--home-pad-x);
            padding-right: clamp(16px, 6vw, 48px);
            max-width: min(760px, 92vw);
            box-sizing: border-box;
        }}

        .hero-title {{
            font-size: clamp(28px, 4.2vw, 44px);
            font-weight: 900;
            line-height: 1.05;
            color: #000000;
            letter-spacing: 0px;
            margin-bottom: clamp(10px, 2vw, 18px);
            word-break: break-word;
        }}

        .hero-subtitle {{
            font-size: clamp(11px, 1.8vw, 22px);
            font-weight: 900;
            color: #f04b13;
            line-height: 1.35;
            margin-bottom: 8px;
            white-space: nowrap;
        }}

        div.stButton > button {{
            position: fixed !important;
            top: calc(50% + (var(--home-text-block-h) / 2)) !important;
            left: var(--home-pad-x) !important;
            transform: none !important;
            margin: 0 !important;

            width: min(clamp(180px, 32vw, 260px), calc(100vw - 2 * var(--home-pad-x))) !important;
            height: clamp(42px, 6vw, 56px) !important;

            background-color: #16006b;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0 !important;
            z-index: 3;
        }}

        div.stButton > button p {{
            font-size: clamp(13px, 2.6vw, 20px) !important;
            font-weight: 500 !important;
            color: white !important;
        }}

        div.stButton > button:hover {{
            background-color: #25009c;
            color: white;
            border: none;
        }}

        div.stButton > button:focus {{
            background-color: #16006b;
            color: white;
            border: none;
            box-shadow: none;
        }}
        </style>

        <div class="home-content-wrapper">
            <div class="home-content">
                <div class="hero-title">
                    SHIP BERTH<br>
                    SCHEDULING
                </div>
                <div class="hero-subtitle">
                    CASE STUDY OF THE JAMRUD TERMINAL
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("START SCHEDULING", key="start_scheduling"):
        st.session_state["page"] = "upload"
        st.rerun()


# =========================
# SHARED CSS TOKENS
# Definisi token --u, --header-h, --back-btn-size, --back-icon-size,
# dan --action-btn-* dipakai SAMA persis di upload_page dan scenario_page
# sehingga ukuran & posisi tombol selalu identik di kedua halaman.
# =========================
SHARED_PAGE_CSS = """
<style>
:root {
    /* ── Satu unit skala dasar, semua elemen turunan dari sini ── */
    --u: clamp(13px, 1.8vw, 24px);

    /* ── Header bar ── */
    --header-h:       clamp(64px, 9vw, 90px);
    --back-btn-size:  clamp(44px, 8vw, 60px);
    --back-icon-size: clamp(24px, 5vw, 34px);

    /* ── Action button (Select Excel / Single Berthing / Re-berthing) ──
       Semua tombol aksi di kedua halaman memakai token yang sama. */
    --action-btn-h:      calc(var(--u) * 4.2);
    --action-btn-fs:     calc(var(--u) * 1.3);
    --action-btn-fw:     600;
    --action-btn-radius: 10px;
    --action-btn-pad-x:  calc(var(--u) * 2.8);
}
</style>
"""


# =========================
# UPLOAD PAGE
# =========================
def upload_page():
    st.markdown(SHARED_PAGE_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .stApp {
            background: #f3f3f3;
        }

        .block-container {
            padding-top: 0rem !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
            padding-bottom: 0rem !important;
            max-width: 100% !important;
        }

        .top-gradient {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: var(--header-h);
            background: linear-gradient(90deg, #e97845 0%, #554797 100%);
            z-index: 1;
        }

        .upload-content {
            text-align: center;
            margin-top: calc(var(--header-h) + calc(var(--u) * 7));
        }

        .upload-title {
            font-size: clamp(18px, 3.6vw, 36px);
            font-weight: 800;
            color: #000000;
            margin-bottom: calc(var(--u) * 0.6);
            white-space: nowrap;
        }

        .upload-desc {
            font-size: var(--u);
            font-weight: 400;
            color: #000000;
            line-height: 1.45;
            margin-bottom: calc(var(--u) * 3.4);
            padding: 0 5vw;
        }

        /* ── Back button — scoped ke key-nya agar tidak bentrok dengan
           tombol lain di halaman ini ── */
        .st-key-back_home button {
            position: fixed;
            top: calc((var(--header-h) - var(--back-btn-size)) / 2);
            left: clamp(12px, 3vw, 24px);
            z-index: 9999;

            width: var(--back-btn-size) !important;
            height: var(--back-btn-size) !important;
            min-width: var(--back-btn-size) !important;

            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;

            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        .st-key-back_home button p {
            margin: 0 !important;
            font-size: var(--back-icon-size) !important;
            line-height: 1 !important;
            font-weight: 700 !important;
            color: white !important;
        }

        .st-key-back_home button:hover,
        .st-key-back_home button:focus {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        /* ── File uploader — tampil sebagai tombol ── */
        div[data-testid="stFileUploader"] {
            width: auto !important;
            max-width: 92vw !important;
            margin: 0 auto !important;
            display: flex !important;
            justify-content: center !important;
        }

        div[data-testid="stFileUploader"] > div,
        div[data-testid="stFileUploader"] section,
        div[data-testid="stFileUploader"] section > div,
        div[data-testid="stFileUploaderDropzone"],
        div[data-testid="stFileUploaderDropzoneInstructions"] {
            width: auto !important;
            box-sizing: border-box !important;
        }

        div[data-testid="stFileUploader"] label {
            display: none;
        }

        div[data-testid="stFileUploader"] section {
            border: none;
            padding: 0;
            background: transparent;
        }

        div[data-testid="stFileUploader"] section > div {
            display: none;
        }

        /* Tombol "Select Excel file" — ukuran & font sama persis dengan
           tombol skenario di scenario_page via token bersama */
        div[data-testid="stFileUploader"] button {
            position: relative !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;

            background-color: #16006b !important;
            color: transparent !important;
            border: none !important;
            border-radius: var(--action-btn-radius) !important;

            width: auto !important;
            min-width: calc(var(--u) * 17) !important;
            max-width: 92vw !important;
            height: var(--action-btn-h) !important;
            padding: 0 var(--action-btn-pad-x) !important;
            overflow: hidden !important;
            box-sizing: border-box !important;
            white-space: nowrap !important;

            transition: background-color 0.15s ease;
        }

        div[data-testid="stFileUploader"] button p,
        div[data-testid="stFileUploader"] button svg {
            display: none !important;
        }

        div[data-testid="stFileUploader"] button::after {
            content: "Select Excel file";
            color: white;
            font-size: var(--action-btn-fs);
            font-weight: var(--action-btn-fw);
            white-space: nowrap;

            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
        }

        div[data-testid="stFileUploader"] button:hover {
            background-color: #25009c !important;
            color: transparent !important;
        }

        div[data-testid="stFileUploader"] button:focus {
            background-color: #16006b !important;
            color: transparent !important;
            box-shadow: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # gradient bar
    st.markdown('<div class="top-gradient"></div>', unsafe_allow_html=True)

    # tombol back — sekarang scoped via .st-key-back_home
    if st.button("❮", key="back_home"):
        st.session_state["page"] = "home"
        st.rerun()

    # konten utama
    st.markdown(
        dedent("""
        <div class="upload-content">
            <div class="upload-title">Schedule Ship Berthing</div>
            <div class="upload-desc">
                Please ensure that the uploaded Excel file contains the following columns:<br>
                <b>Ship Name, LOA, Draft, Arrival Time, Total Service Time, and Category [Passenger, RoRo, Cargo, Other].</b><br><br>
                <b>NOTE</b>: the system can only schedule a maximum of <b>3 days</b> of arrivals.
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Select Excel file",
        type=["xlsx"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        clear_arrival_state()
        st.session_state["uploaded_file"] = uploaded_file
        st.session_state["page"] = "scenario"
        st.rerun()


# =========================
# SCENARIO PAGE
# =========================
def scenario_page():
    uploaded_name = escape(st.session_state["uploaded_file"].name)

    st.markdown(SHARED_PAGE_CSS, unsafe_allow_html=True)

    st.markdown("""
<style>
.stApp {
    background: #f3f3f3;
    min-height: 100vh;
}

.block-container {
    padding-top: 0rem !important;
    padding-left: 0rem !important;
    padding-right: 0rem !important;
    padding-bottom: 0rem !important;
    max-width: 100% !important;
}

.top-gradient {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: var(--header-h);
    background: linear-gradient(90deg, #e97845 0%, #554797 100%);
    z-index: 1;
}

/* ── shared card & button-row tokens ── */
:root {
    --card-w: min(1000px, 88vw);
    --btn-gap: calc(var(--u) * 1.2);

    --file-card-h:    calc(var(--u) * 5.2);
    --file-card-mb:   calc(var(--u) * 2.6);
    --file-card-pad-x: calc(var(--u) * 1.6);
    --remove-btn-size: calc(var(--u) * 2.6);
}

.scenario-content {
    text-align: center;
    margin-top: calc(var(--header-h) + calc(var(--u) * 4));
}

.scenario-title {
    font-size: clamp(18px, 3.6vw, 36px);
    font-weight: 800;
    color: #000000;
    margin-bottom: calc(var(--u) * 1.5);
    white-space: nowrap;
}

/* ── File card ── */
.file-card {
    width: var(--card-w);
    height: var(--file-card-h);
    margin: 0 auto var(--file-card-mb) auto;
    border: 2px solid #9d9d9d;
    border-radius: 9px;
    background: white;

    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: calc(var(--u) * 0.5);
    padding: 0 var(--file-card-pad-x);
    box-sizing: border-box;
    position: relative;
}

.file-left {
    display: flex;
    align-items: center;
    gap: calc(var(--u) * 1.2);
    min-width: 0;
    flex: 1 1 auto;
}

.excel-icon {
    width: calc(var(--u) * 2.4);
    height: calc(var(--u) * 2.4);
    flex-shrink: 0;
    background: #107c41;
    border-radius: 4px;
    color: white;
    font-size: calc(var(--u) * 1.3);
    font-weight: 700;

    display: flex;
    align-items: center;
    justify-content: center;
}

.file-name {
    font-size: var(--u);
    font-weight: 500;
    color: #000000;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.file-close {
    font-size: calc(var(--u) * 1.4);
    font-weight: 300;
    color: #333333;
    line-height: 1;
    flex-shrink: 0;
}

/* ── Back button ── */
.st-key-back_scenario button {
    position: fixed;
    top: calc((var(--header-h) - var(--back-btn-size)) / 2);
    left: clamp(12px, 3vw, 24px);
    z-index: 9999;

    width: var(--back-btn-size) !important;
    height: var(--back-btn-size) !important;
    min-width: var(--back-btn-size) !important;

    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.st-key-back_scenario button p {
    margin: 0 !important;
    font-size: var(--back-icon-size) !important;
    line-height: 1 !important;
    font-weight: 700 !important;
    color: white !important;
}

.st-key-back_scenario button:hover,
.st-key-back_scenario button:focus {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Scenario button row ── */
[data-testid="stHorizontalBlock"]:has(.st-key-single_scenario) {
    width: var(--card-w) !important;
    margin: 0 auto !important;
    gap: var(--btn-gap) !important;
    flex-wrap: nowrap !important;
    justify-content: center !important;
    box-sizing: border-box !important;
}

[data-testid="stHorizontalBlock"]:has(.st-key-single_scenario) > [data-testid="stColumn"] {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    padding: 0 !important;
}

.st-key-single_scenario,
.st-key-reberthing_scenario {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}

/* ── Tombol Single / Re-berthing — ukuran identik dengan "Select Excel file"
   via token bersama --action-btn-* ── */
.st-key-single_scenario button,
.st-key-reberthing_scenario button {
    width: 100% !important;
    height: var(--action-btn-h) !important;
    background-color: #16006b !important;
    color: white !important;
    border: none !important;
    border-radius: var(--action-btn-radius) !important;
    padding: 0 var(--action-btn-pad-x) !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-sizing: border-box !important;
}

.st-key-single_scenario button p,
.st-key-reberthing_scenario button p {
    font-size: var(--action-btn-fs) !important;
    font-weight: var(--action-btn-fw) !important;
    color: white !important;
    white-space: nowrap !important;
    margin: 0 !important;
}

.st-key-single_scenario button:hover,
.st-key-reberthing_scenario button:hover {
    background-color: #25009c !important;
    color: white !important;
    border: none !important;
}

.st-key-single_scenario button:focus,
.st-key-reberthing_scenario button:focus {
    background-color: #16006b !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Narrow viewport: tombol scenario stack vertikal ── */
@media (max-width: 700px) {
    [data-testid="stHorizontalBlock"]:has(.st-key-single_scenario) {
        flex-wrap: wrap !important;
        row-gap: var(--btn-gap) !important;
    }

    [data-testid="stHorizontalBlock"]:has(.st-key-single_scenario) > [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        width: 100% !important;
    }
}

/* ── Tombol X di file card ── */
.st-key-remove_file {
    position: absolute !important;
    top: calc(50% - (var(--remove-btn-size) / 2)) !important;
    right: calc(var(--file-card-pad-x) - calc(var(--u) * 0.4)) !important;
    width: var(--remove-btn-size) !important;
    z-index: 3;
}

.st-key-remove_file button {
    position: relative !important;

    width: var(--remove-btn-size) !important;
    height: var(--remove-btn-size) !important;
    min-width: var(--remove-btn-size) !important;

    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 0 !important;
    margin: 0 !important;

    color: transparent !important;
    font-size: 0 !important;
}

.st-key-remove_file button p {
    display: none !important;
}

.st-key-remove_file button:hover,
.st-key-remove_file button:focus,
.st-key-remove_file button:active {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

    # Gradient bar
    st.markdown('<div class="top-gradient"></div>', unsafe_allow_html=True)
    render_modal()
    render_arrival_limit_modal()

    # Tombol back
    if st.button("❮", key="back_scenario"):
        clear_arrival_state()
        st.session_state["page"] = "upload"
        st.rerun()

    # Header + file card
    scenario_html = (
        '<div class="scenario-content-wrapper">'
            '<div class="scenario-content">'
                '<div class="scenario-title">Schedule Ship Berthing</div>'

                '<div class="file-card">'
                    '<div class="file-left">'
                        '<div class="excel-icon">X</div>'
                        f'<div class="file-name">{uploaded_name}</div>'
                    '</div>'
                    '<div class="file-close">×</div>'
                '</div>'
            '</div>'
        '</div>'
    )
    st.markdown(scenario_html, unsafe_allow_html=True)

    # Invisible X button overlapping the × in file card
    if st.button(" ", key="remove_file"):
        if "uploaded_file" in st.session_state:
            del st.session_state["uploaded_file"]
        clear_arrival_state()
        st.session_state["page"] = "upload"
        st.rerun()

    LOADING_HTML = """
    <style>
    .loading-overlay {
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: rgba(255,255,255,0.92);
        z-index: 99999;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 0 24px;
        box-sizing: border-box;
        text-align: center;
    }
    .loading-title {
        font-family: 'Poppins', sans-serif;
        font-size: clamp(20px, 5vw, 32px);
        font-weight: 700;
        color: #16006b;
        margin-bottom: clamp(12px, 3vw, 20px);
    }
    .loading-sub {
        font-family: 'Poppins', sans-serif;
        font-size: clamp(14px, 3vw, 18px);
        font-weight: 400;
        color: #555;
    }
    .spinner {
        width: clamp(40px, 10vw, 60px);
        height: clamp(40px, 10vw, 60px);
        border: 6px solid #e0e0e0;
        border-top: 6px solid #16006b;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: clamp(16px, 4vw, 24px);
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    <div class="loading-overlay">
        <div class="spinner"></div>
        <div class="loading-title">Generating Schedule...</div>
        <div class="loading-sub">Please wait, this may take a moment</div>
    </div>
    """

    col_single, col_reberth = st.columns([1, 1])

    with col_single:
        if st.button("Single Berthing Scenario", key="single_scenario"):
            with st.spinner("Preprocessing..."):
                df_kapal, errors, arrival_info = preprocess(
                    st.session_state["uploaded_file"],
                    scenario="single"
                )
                df_dermaga = load_dermaga()

            pasut_keys = ['LOW_TIDE_1_START_H', 'LOW_TIDE_1_END_H','LOW_TIDE_2_START_H', 'LOW_TIDE_2_END_H']
            st.session_state["pasut_config"] = {k: CONFIG[k] for k in pasut_keys if k in CONFIG}

            if errors or df_kapal.empty:
                show_error_modal(errors if errors else ["Data could not be processed."])
                st.rerun()
                return

            st.session_state["df_kapal_pending_single"] = df_kapal
            st.session_state["arrival_info_single"] = arrival_info
            st.session_state.pop("arrival_decision_single", None)
            if arrival_info.get("exceeds"):
                show_arrival_limit_dialog("single", arrival_info)
            st.rerun()

    if (
        st.session_state.get("df_kapal_pending_single") is not None
        and (
            not st.session_state.get("arrival_info_single", {}).get("exceeds")
            or st.session_state.get("arrival_decision_single") == "limit"
        )
    ):
        df_kapal = st.session_state.pop("df_kapal_pending_single")
        arrival_info = st.session_state.pop("arrival_info_single", {})
        decision = st.session_state.pop("arrival_decision_single", None)

        if decision == "limit":
            df_kapal = limit_to_n_days(df_kapal, arrival_info.get("max_days", 3))

        df_dermaga = load_dermaga()
        run_seed = random.randint(1, 50)
        loading = st.empty()
        loading.markdown(LOADING_HTML, unsafe_allow_html=True)

        try:
            ch1, metrics_ch1 = run_ch1(
                df_kapal_raw=df_kapal,
                df_dermaga=df_dermaga,
                population_size=50,
                base_seed=run_seed,
                verbose=True
            )

            lb1, metrics_df1, history_lb1, _ = run_love_bird_optimization(
                df_kapal=df_kapal,
                df_dermaga=df_dermaga,
                initial_solutions=ch1,
                population_size=50,
                max_generations=100,
                seed=run_seed
            )

            df_schedule = lb1["df_schedule"]
            metrics = {
                "fitness": lb1.get("fitness"),
                "total_wait": lb1.get("total_wait"),
                "n_late": lb1.get("n_late"),
                "assigned": lb1.get("assigned"),
                "running_time_s": lb1.get("running_time_s"),
                "running_time_m": lb1.get("running_time_m"),
            }

        except Exception as e:
            loading.empty()
            st.error(f"Optimizer failed to run: {e}")
            return

        st.session_state["scenario"] = "single"
        st.session_state["run_seed"] = run_seed
        st.session_state["df_kapal_preprocessed"] = df_kapal
        st.session_state["df_dermaga"] = df_dermaga
        st.session_state["ch1"] = ch1
        st.session_state["metrics_ch1"] = metrics_ch1
        st.session_state["lb1"] = lb1
        st.session_state["metrics_df1"] = metrics_df1
        st.session_state["history_lb1"] = history_lb1
        st.session_state["result"] = df_schedule
        st.session_state["metrics"] = metrics
        st.session_state["page"] = "result"
        loading.empty()
        st.rerun()

    with col_reberth:
        if st.button("Re-berthing Scenario", key="reberthing_scenario"):
            with st.spinner("Preprocessing..."):
                df_kapal, errors, arrival_info = preprocess(
                    st.session_state["uploaded_file"],
                    scenario="reberthing"
                )
                df_dermaga = load_dermaga()

            pasut_keys = ['LOW_TIDE_1_START_H', 'LOW_TIDE_1_END_H',
                'LOW_TIDE_2_START_H', 'LOW_TIDE_2_END_H']
            st.session_state["pasut_config"] = {k: CONFIG[k] for k in pasut_keys if k in CONFIG}
            if errors or df_kapal.empty:
                show_error_modal(errors if errors else ["Data could not be processed."])
                st.rerun()
                return

            st.session_state["df_kapal_pending_reberthing"] = df_kapal
            st.session_state["arrival_info_reberthing"] = arrival_info
            st.session_state.pop("arrival_decision_reberthing", None)
            if arrival_info.get("exceeds"):
                show_arrival_limit_dialog("reberthing", arrival_info)
            st.rerun()

    if (
        st.session_state.get("df_kapal_pending_reberthing") is not None
        and (
            not st.session_state.get("arrival_info_reberthing", {}).get("exceeds")
            or st.session_state.get("arrival_decision_reberthing") == "limit"
        )
    ):
        df_kapal = st.session_state.pop("df_kapal_pending_reberthing")
        arrival_info = st.session_state.pop("arrival_info_reberthing", {})
        decision = st.session_state.pop("arrival_decision_reberthing", None)

        if decision == "limit":
            df_kapal = limit_to_n_days(df_kapal, arrival_info.get("max_days", 3))

        df_dermaga = load_dermaga()
        run_seed = random.randint(1, 50)
        loading = st.empty()
        loading.markdown(LOADING_HTML, unsafe_allow_html=True)

        try:
            ch2, metrics_ch2 = run_ch2(
                df_kapal_raw=df_kapal,
                df_dermaga=df_dermaga,
                population_size=20,
                base_seed=run_seed,
                verbose=True
            )

            lb2, metrics_df2, history_lb2, _ = run_love_bird_s2(
                df_kapal_raw=df_kapal,
                df_dermaga=df_dermaga,
                initial_solutions=ch2,
                population_size=20,
                max_generations=30,
                seed=run_seed
            )

            df_schedule = lb2["df_schedule"]
            metrics = {
                "fitness":        lb2.get("fitness"),
                "total_wait":     lb2.get("total_wait"),
                "n_late":         lb2.get("n_late"),
                "total_reberth":  lb2.get("total_reberth"),
                "assigned":       lb2.get("assigned"),
                "running_time_s": lb2.get("running_time_s"),
                "running_time_m": lb2.get("running_time_m"),
            }

        except Exception as e:
            loading.empty()
            st.error(f"Optimizer failed to run: {e}")
            return

        st.session_state["scenario"]              = "reberthing"
        st.session_state["run_seed"]              = run_seed
        st.session_state["df_kapal_preprocessed"] = df_kapal
        st.session_state["df_dermaga"]            = df_dermaga
        st.session_state["ch2"]                   = ch2
        st.session_state["metrics_ch2"]           = metrics_ch2
        st.session_state["lb2"]                   = lb2
        st.session_state["metrics_df2"]           = metrics_df2
        st.session_state["history_lb2"]           = history_lb2
        st.session_state["result"]                = df_schedule
        st.session_state["metrics"]               = metrics
        st.session_state["page"]                  = "result"
        loading.empty()
        st.rerun()


def result_page():
    st.markdown("""
    <style>
    .stApp { background: #ffffff; }
    .block-container {
        padding-top: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
    }
    :root {
        --header-h: clamp(64px, 9vw, 90px);
        --back-btn-size: clamp(44px, 8vw, 60px);
        --back-icon-size: clamp(24px, 5vw, 34px);
    }
    .top-gradient {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: var(--header-h);
        background: linear-gradient(90deg, #e97845 0%, #554797 100%);
        z-index: 1;
    }
    .st-key-back_result button {
        position: fixed;
        top: calc((var(--header-h) - var(--back-btn-size)) / 2);
        left: clamp(12px, 3vw, 24px);
        z-index: 9999;
        width: var(--back-btn-size) !important;
        height: var(--back-btn-size) !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    .st-key-back_result button p {
        margin: 0 !important;
        font-size: var(--back-icon-size) !important;
        font-weight: 700 !important;
        color: white !important;
    }
    .st-key-back_result button:hover,
    .st-key-back_result button:focus {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="top-gradient"></div>', unsafe_allow_html=True)

    if st.button("❮", key="back_result"):
        for key in [
            "result",
            "metrics",
            "metrics_ch1",
            "metrics_df1",
            "history_lb1",
            "metrics_ch2",
            "metrics_df2",
            "history_lb2",
            "initial_solutions",
            "df_kapal_preprocessed",
            "df_dermaga",
            "ch1",
            "lb1",
            "ch2",
            "lb2",
            "run_seed",
            "pasut_config", 
        ]:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state["page"] = "scenario"
        st.rerun()

    st.markdown(
        '<div style="padding-top:calc(var(--header-h) + clamp(8px, 2.5vw, 32px)); text-align:center;">'
        '<div style="font-size:clamp(18px,3.5vw,42px); font-weight:800; color:#000; margin-bottom:clamp(14px, 2.5vw, 28px); font-family:Poppins,sans-serif; padding:0 16px;">'
        'Berthing schedules have been generated!'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # Generate gantt chart
    fig = None
    gantt_buffer = None

    if "result" in st.session_state and "df_dermaga" in st.session_state:
        import matplotlib.pyplot as plt

        pasut_keys = ['LOW_TIDE_1_START_H', 'LOW_TIDE_1_END_H',
                    'LOW_TIDE_2_START_H', 'LOW_TIDE_2_END_H']
        saved_config = st.session_state.get("pasut_config", {})
        pasut_config = saved_config if all(k in saved_config for k in pasut_keys) else None

        fig = plot_berth_allocation_by_category(
            df_result=st.session_state["result"],
            df_dermaga_info=st.session_state["df_dermaga"],
            config=pasut_config
        )

        if fig is not None:
            gantt_buffer = io.BytesIO()
            fig.savefig(gantt_buffer, format="jpg", dpi=150, bbox_inches="tight")
            gantt_buffer.seek(0)
            gantt_b64 = base64.b64encode(gantt_buffer.read()).decode()
            plt.close(fig)
        else:
            gantt_b64 = ""
    else:
        gantt_b64 = ""

    if "result" in st.session_state:
        scenario = st.session_state.get("scenario", "single")
        df_formatted = format_output(st.session_state["result"], scenario)
        table_html = df_formatted.to_html(index=False, border=0, classes="preview-table")

        excel_buffer = io.BytesIO()
        df_formatted.to_excel(excel_buffer, index=False, engine="openpyxl")
        excel_buffer.seek(0)
        excel_b64 = base64.b64encode(excel_buffer.read()).decode()
        excel_buffer.seek(0)
    else:
        df_formatted = None
        table_html = "<p>Table not available</p>"
        excel_buffer = None
        excel_b64 = ""

    components.html(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
    * {{ font-family: 'Poppins', sans-serif; box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ background: white; }}
    .wrapper {{
        display: flex;
        flex-wrap: wrap;
        gap: 22px;
        justify-content: center;
        align-items: stretch;
        padding: 10px 14px 24px 14px;
    }}
    .card {{
        height: clamp(360px, 52vw, 540px);
        background: white;
        border: 1px solid #dddddd;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.22);
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: space-between;

        flex: 1 1 320px;
    }}
    .card-gantt {{
        max-width: 460px;
        min-width: 280px;
    }}
    .card-excel {{
        max-width: 700px;
        min-width: 280px;
    }}
    .preview {{
        flex: 1;
        display: flex;
        align-items: flex-start;
        justify-content: flex-start;
        overflow: hidden;
        padding: 8px;
        min-height: 0;
    }}
    .preview img {{
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        display: block;
        margin: auto;
    }}
    .card-gantt .preview {{
        align-items: center;
        justify-content: center;
    }}
    .card-excel .preview {{
        align-items: flex-start;
        justify-content: flex-start;
    }}
    .download-bar {{
        height: clamp(56px, 9vw, 75px);
        background: #f3f3f3;
        border-top: 1px solid #e5e5e5;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 clamp(12px, 3vw, 18px);
        flex-shrink: 0;
    }}
    .file-name {{
        font-size: clamp(13px, 2.6vw, 20px);
        font-weight: 600;
        color: #000;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .dl-icon {{
        font-size: clamp(18px, 3.6vw, 26px); font-weight: 700; color: #333;
        cursor: pointer; text-decoration: none;
        flex-shrink: 0;
        margin-left: 10px;
    }}
    .preview-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 10px;
    }}
    .preview-table th {{
        background: #16006b;
        color: white;
        padding: 4px 6px;
        text-align: left;
        position: sticky;
        top: 0;
    }}
    .preview-table td {{
        padding: 3px 6px;
        border-bottom: 1px solid #eee;
        white-space: nowrap;
    }}
    .preview-table tr:nth-child(even) td {{
        background: #f9f9f9;
    }}
    .table-scroll {{
        width: 100%;
        height: 100%;
        overflow-x: auto;
        overflow-y: scroll;
    }}
    </style>

    <div class="wrapper">
        <div class="card card-gantt">
            <div class="preview">
                {"<img src='data:image/jpeg;base64," + gantt_b64 + "'/>" if gantt_b64 else "<span>Gantt chart unavailable</span>"}
            </div>
            <div class="download-bar">
                <div class="file-name">gantt_chart.jpg</div>
                {"<a class='dl-icon' href='data:image/jpeg;base64," + gantt_b64 + "' download='gantt_chart.jpg'>⬇</a>" if gantt_b64 else "<span></span>"}
            </div>
        </div>

        <div class="card card-excel">
            <div class="preview">
                <div class="table-scroll">
                    {table_html}
                </div>
            </div>
            <div class="download-bar">
                <div class="file-name">output_schedule.xlsx</div>
                {"<a class='dl-icon' href='data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64," + excel_b64 + "' download='output_schedule.xlsx'>⬇</a>" if excel_b64 else "<span></span>"}
            </div>
        </div>
    </div>
    """, height=1180, scrolling=True)


# =========================
# PAGE ROUTER
# =========================
if st.session_state["page"] == "home":
    home_page()
elif st.session_state["page"] == "upload":
    upload_page()
elif st.session_state["page"] == "scenario":
    scenario_page()
elif st.session_state["page"] == "result":
    result_page()