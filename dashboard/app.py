import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Financial Anomaly Detection",
    page_icon="💳",
    layout="wide"
)

# -------------------------------
# Design tokens
# -------------------------------
PAPER = "#EEF2F5"      # page background, cool ledger paper
SURFACE = "#FFFFFF"    # panels
INK = "#18222F"        # primary text
SLATE = "#5B6776"      # secondary text
LINE = "#D7DEE5"       # hairlines and borders
NAVY = "#102A3B"       # sidebar
TEAL = "#12756B"       # primary / normal activity
ROSE = "#C8455A"       # reserved for anomalies

# -------------------------------
# Global styles
# -------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, p, li, label, input, textarea, button {{
    font-family: 'Instrument Sans', system-ui, sans-serif;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.25rem; padding-bottom: 4rem; max-width: 1280px; }}

h1, h2, h3, h4 {{
    font-family: 'Bricolage Grotesque', 'Instrument Sans', sans-serif !important;
    color: {INK};
    letter-spacing: -0.01em;
}}
h3 {{ font-size: 1.3rem !important; font-weight: 600 !important; padding-top: 1.75rem !important; }}

/* ---------- Header ---------- */
.fa-header {{
    display: flex; align-items: flex-end; justify-content: space-between;
    gap: 2rem; padding-bottom: 1.5rem; margin-bottom: 0.5rem;
    border-bottom: 1px solid {LINE};
}}
.fa-header h1 {{
    font-size: clamp(2rem, 3.6vw, 3rem) !important; font-weight: 700 !important;
    line-height: 1.02; margin: 0; padding: 0 !important; letter-spacing: -0.025em;
    max-width: 16ch;
}}
.fa-header p {{ color: {SLATE}; font-size: 1.02rem; margin: 0.6rem 0 0; }}
.fa-trace {{ flex: 0 1 360px; min-width: 200px; }}
.fa-trace svg {{ width: 100%; height: auto; display: block; overflow: visible; }}
.fa-trace .line {{
    fill: none; stroke: {TEAL}; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round;
    stroke-dasharray: 900; stroke-dashoffset: 900;
    animation: fa-draw 1.6s cubic-bezier(.6,.1,.2,1) .2s forwards;
}}
.fa-trace .spike {{ fill: {ROSE}; opacity: 0; animation: fa-pop .4s ease-out 1.5s forwards; }}
.fa-trace .ring  {{ fill: none; stroke: {ROSE}; stroke-width: 1.5; opacity: 0;
                    animation: fa-pop .4s ease-out 1.6s forwards; }}
.fa-trace .base  {{ stroke: {LINE}; stroke-width: 1; }}
.fa-trace .tag   {{ font: 600 11px 'Instrument Sans', sans-serif; fill: {ROSE}; opacity: 0;
                    animation: fa-pop .4s ease-out 1.7s forwards; }}
@keyframes fa-draw {{ to {{ stroke-dashoffset: 0; }} }}
@keyframes fa-pop  {{ to {{ opacity: 1; }} }}
@media (prefers-reduced-motion: reduce) {{
    .fa-trace .line {{ animation: none; stroke-dashoffset: 0; }}
    .fa-trace .spike, .fa-trace .ring, .fa-trace .tag {{ animation: none; opacity: 1; }}
}}
@media (max-width: 760px) {{
    .fa-header {{ flex-direction: column; align-items: flex-start; }}
    .fa-trace {{ width: 100%; flex-basis: auto; }}
}}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{ background: {NAVY}; }}
[data-testid="stSidebar"] * {{ color: #D9E3EA; }}
[data-testid="stSidebar"] h1 {{
    font-size: 1.05rem !important; font-weight: 600 !important; color: #FFFFFF;
    padding-bottom: 0.25rem !important;
}}
[data-testid="stSidebar"] h3 {{
    font-size: 0.95rem !important; color: #FFFFFF; padding-top: 1.5rem !important;
    border-top: 1px solid rgba(255,255,255,.12); margin-top: 1.25rem;
}}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 0.45rem 0.6rem; border-radius: 6px; margin: 1px 0; width: 100%;
    transition: background .15s ease;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(255,255,255,.06); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    background: rgba(18,117,107,.35); box-shadow: inset 3px 0 0 #3FB8A6;
}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
    background: rgba(255,255,255,.04); border: 1px dashed rgba(255,255,255,.28); border-radius: 8px;
}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
    background: transparent; border: 1px solid rgba(255,255,255,.4); color: #FFFFFF;
}}
[data-testid="stSidebar"] [data-testid="stFileChip"] {{
    background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18);
}}
[data-testid="stSidebar"] [data-testid="stFileChip"] svg {{ color: #3FB8A6; fill: currentColor; }}
[data-testid="stSidebar"] small {{ color: rgba(217,227,234,.65) !important; }}

/* ---------- Metrics as one ledger strip ---------- */
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {{
    background: {SURFACE}; border: 1px solid {LINE}; border-radius: 10px;
    padding: 1.25rem 0.5rem; gap: 0;
}}
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div {{ padding: 0 1.5rem; }}
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div + div {{
    border-left: 1px solid {LINE};
}}
[data-testid="stMetricLabel"] p {{ color: {SLATE}; font-size: 0.9rem; font-weight: 500; }}
[data-testid="stMetricValue"] {{
    font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; color: {INK};
    font-size: 2.2rem; font-variant-numeric: tabular-nums; letter-spacing: -0.02em;
}}
@media (max-width: 640px) {{
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div + div {{
        border-left: none; border-top: 1px solid {LINE}; padding-top: 1rem; margin-top: 0.5rem;
    }}
}}

/* ---------- Tables, alerts, inputs ---------- */
[data-testid="stDataFrame"] {{
    border: 1px solid {LINE}; border-radius: 10px; overflow: hidden; background: {SURFACE};
}}
[data-testid="stAlert"] {{ border-radius: 8px; border: 1px solid {LINE}; background: {SURFACE}; }}
[data-testid="stSelectbox"] > div > div {{ border-radius: 8px; border-color: {LINE}; background: {SURFACE}; }}
[data-testid="stPlotlyChart"] {{
    background: {SURFACE}; border: 1px solid {LINE}; border-radius: 10px; overflow: hidden;
}}
*:focus-visible {{ outline: 2px solid {TEAL} !important; outline-offset: 2px; }}

/* ---------- Empty state ---------- */
.fa-empty {{
    display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: 2.5rem;
    background: {SURFACE}; border: 1px solid {LINE}; border-radius: 12px; padding: 2rem 2.25rem;
    margin-top: 1.5rem;
}}
.fa-empty h3 {{ padding-top: 0 !important; margin-top: 0; }}
.fa-empty p {{ color: {SLATE}; max-width: 46ch; line-height: 1.6; }}
.fa-empty ul {{ list-style: none; padding: 0; margin: 0; }}
.fa-empty li {{
    padding: 0.7rem 0; border-bottom: 1px solid {LINE}; color: {INK};
    display: flex; align-items: center; gap: 0.75rem;
}}
.fa-empty li:last-child {{ border-bottom: none; }}
.fa-empty li::before {{
    content: ""; width: 8px; height: 8px; border-radius: 50%; background: {TEAL}; flex: none;
}}
@media (max-width: 760px) {{ .fa-empty {{ grid-template-columns: 1fr; gap: 1rem; padding: 1.5rem; }} }}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------
# Chart theme
# -------------------------------
pio.templates["ledger"] = go.layout.Template(
    layout=dict(
        font=dict(family="Instrument Sans, sans-serif", color=INK, size=13),
        title=dict(font=dict(family="Bricolage Grotesque, sans-serif", size=18, color=INK), x=0.01),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=[TEAL, ROSE, "#3E6FA8", "#B88A2E", SLATE],
        xaxis=dict(showgrid=False, linecolor=LINE, ticks="outside", tickcolor=LINE,
                   title_font=dict(color=SLATE)),
        yaxis=dict(gridcolor="#E9EEF2", zeroline=False, linecolor=LINE, title_font=dict(color=SLATE)),
        bargap=0.06,
        hoverlabel=dict(bgcolor=NAVY, font=dict(color="#FFFFFF", family="Instrument Sans")),
        margin=dict(l=20, r=20, t=60, b=30),
    )
)
pio.templates.default = "ledger"

# -------------------------------
# Dashboard Header
# -------------------------------
st.markdown(
    """
<div class="fa-header">
  <div>
    <h1>Financial Anomaly Detection Dashboard</h1>
    <p>Statistical machine learning based fraud analysis</p>
  </div>
  <div class="fa-trace" aria-hidden="true">
    <svg viewBox="0 0 360 96">
      <line class="base" x1="0" y1="84" x2="360" y2="84"/>
      <polyline class="line" points="0,66 18,62 36,68 54,60 72,64 90,57 108,63 126,59 144,65 162,58
        180,61 198,56 216,62 234,14 252,60 270,57 288,63 306,58 324,61 342,55 360,59"/>
      <circle class="ring" cx="234" cy="14" r="8"/>
      <circle class="spike" cx="234" cy="14" r="3.5"/>
      <text class="tag" x="248" y="18">flagged</text>
    </svg>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Overview", "Data Analysis"]
)

# -------------------------------
# Data Upload
# -------------------------------
st.sidebar.subheader("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success(f"Loaded {uploaded_file.name}: {len(df):,} rows ready to analyse.")

    # -------------------------------
    # Dataset Overview
    # -------------------------------
    if page == "Overview":

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Transactions", f"{len(df):,}")

        with col2:
            st.metric("Total Features", len(df.columns))

        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

    # -------------------------------
    # Data Analysis
    # -------------------------------
    elif page == "Data Analysis":

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        numeric_columns = df.select_dtypes(include="number").columns

        if len(numeric_columns) > 0:

            st.subheader("Feature Distribution")

            selected_column = st.selectbox(
                "Select feature to visualize",
                numeric_columns
            )

            fig = px.histogram(
                df,
                x=selected_column,
                title=f"Distribution of {selected_column}"
            )
            fig.update_traces(marker_line_width=0, opacity=0.9)
            fig.update_layout(
                yaxis_title="Transactions",
                height=440,
                margin=dict(l=72, r=28, t=72, b=60),
            )

            st.plotly_chart(fig, use_container_width=True, theme=None)

else:
    st.markdown(
        """
<div class="fa-empty">
  <div>
    <h3>Start with a transaction file</h3>
    <p>Upload a CSV from the sidebar. Each row should be one transaction, with numeric
    columns such as amount or time.</p>
  </div>
  <ul>
    <li>Dataset overview</li>
    <li>Transaction statistics</li>
    <li>Interactive data visualization</li>
    <li>Feature distribution analysis</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )
