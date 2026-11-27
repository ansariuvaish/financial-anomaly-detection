import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Anomaly Lens | Fraud Detection", page_icon="🛡️", layout="wide")
BASE = Path(__file__).parent


# ------------------------------------------------------------------ data and scoring (unchanged logic)
@st.cache_data
def load():
    with open(BASE / "data" / "dashboard_data.json") as f:
        return json.load(f)


D = load()


@st.cache_resource
def load_bundle():
    return joblib.load(BASE / "models" / "bundle.joblib")


def score_all(raw_df, b):
    X = raw_df[b["feature_names"]].copy()
    X["Amount"] = np.log1p(X["Amount"])
    Z = b["scaler"].transform(X)
    m = b["models"]
    pca = m["PCA recon. error"]
    return {"Mahalanobis": m["Mahalanobis"].mahalanobis(Z),
            "PCA recon. error": ((Z - pca.inverse_transform(pca.transform(Z))) ** 2).sum(axis=1),
            "Gaussian Mixture": -m["Gaussian Mixture"].score_samples(Z),
            "LOF": -m["LOF"].score_samples(Z)}


@st.cache_data(show_spinner=False)
def scored(raw_df, method):
    """Same maths as score_all; cached so widget clicks do not re-score the whole file."""
    return score_all(raw_df, load_bundle())[method]


META = D["meta"]
GRID = np.array(D["grid"])
N_POS, N_NEG = META["n_test_fraud"], META["n_test_normal"]
M = pd.DataFrame(D["metrics"]).sort_values("auprc", ascending=False).reset_index(drop=True)
for c in ["precision", "recall", "f1", "tp", "fp", "fn"]:
    if c not in M.columns:
        M[c] = np.nan
BEST = M.iloc[0]

# ------------------------------------------------------------------ theme
COL = dict(bg="#0B1220", side="#111827", card="#172033", line="#243049", text="#F8FAFC",
           muted="#94A3B8", primary="#38BDF8", good="#10B981", warn="#F59E0B", bad="#EF4444")
TONES = {"primary": (COL["primary"], "rgba(56,189,248,.14)"), "good": (COL["good"], "rgba(16,185,129,.14)"),
         "warn": (COL["warn"], "rgba(245,158,11,.14)"), "bad": (COL["bad"], "rgba(239,68,68,.14)"),
         "muted": (COL["muted"], "rgba(148,163,184,.14)")}
# one fixed colour per detector, reused in every chart
DET = {"PCA recon. error": "#38BDF8", "Gaussian Mixture": "#10B981", "LOF": "#F59E0B",
       "Mahalanobis": "#A78BFA", "Isolation Forest": "#F472B6", "One-Class SVM": "#FB923C",
       "Z-score": "#94A3B8", "IQR": "#E2E8F0"}


def _svg(inner, size=18):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{inner}</svg>')


ICONS = {
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "trend": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "bell": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>',
    "x": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    "scale": '<line x1="12" x2="12" y1="3" y2="21"/><path d="M5 7h14"/><path d="m5 7-3 8a4 4 0 0 0 6 0z"/><path d="m19 7-3 8a4 4 0 0 0 6 0z"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 12 10 5 10-5"/><path d="m2 17 10 5 10-5"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
#MainMenu, footer {visibility:hidden;}
header[data-testid="stHeader"] {background:transparent;}
.stApp, .stApp [data-testid="stMarkdownContainer"], .stApp label, .stApp button, .stApp input, .stApp textarea
  {font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', sans-serif;}
.block-container {padding-top:1.6rem; padding-bottom:3rem; max-width:1400px;}

/* sidebar */
[data-testid="stSidebar"] {border-right:1px solid #243049;}
.brand {display:flex; align-items:center; gap:10px; font-size:1.3rem; font-weight:700; color:#F8FAFC;}
.brand .logo {display:inline-flex; width:34px; height:34px; border-radius:9px; align-items:center; justify-content:center;
  background:rgba(56,189,248,.15); color:#38BDF8;}
.brand small {display:block; font-size:.74rem; font-weight:400; color:#94A3B8; margin-top:1px;}
.status {display:flex; align-items:center; gap:8px; margin:14px 0 6px; font-size:.82rem; color:#94A3B8;}
.status i {width:8px; height:8px; border-radius:50%; background:#10B981; box-shadow:0 0 0 3px rgba(16,185,129,.18);}
.navlabel {font-size:.8rem; color:#94A3B8; margin:18px 0 4px; font-weight:500;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:2px;}
[data-testid="stSidebar"] label[data-baseweb="radio"] {padding:9px 12px; border-radius:9px; width:100%; margin:0;}
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {display:none;}
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {background:rgba(148,163,184,.10);}
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {background:rgba(56,189,248,.14);}
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) p {color:#38BDF8; font-weight:600;}
.sidenote {font-size:.76rem; color:#7C8AA5; line-height:1.55; margin-top:22px; padding-top:14px; border-top:1px solid #243049;}

/* page header */
.phead {padding-bottom:14px; margin-bottom:22px; border-bottom:1px solid #243049;}
.ptitle {font-size:2.35rem; font-weight:700; letter-spacing:-.02em; line-height:1.15; margin:0; color:#F8FAFC;}
.psub {color:#94A3B8; font-size:1rem; margin:8px 0 0; max-width:900px;}
.sec {font-size:1.2rem; font-weight:600; margin:2rem 0 .7rem; color:#F8FAFC;}
.sec.first {margin-top:.2rem;}
.hint {color:#94A3B8; font-size:.86rem; line-height:1.55; margin-top:2px;}

/* KPI cards */
.kpi {background:#172033; border:1px solid #243049; border-radius:12px; padding:16px 18px 15px;
  box-shadow:inset 3px 0 0 var(--ac); min-height:176px;}
.kh {display:flex; justify-content:space-between; align-items:center;}
.kl {color:#94A3B8; font-size:.86rem; font-weight:500;}
.ki {display:inline-flex; width:32px; height:32px; border-radius:9px; align-items:center; justify-content:center;}
.kv {font-size:1.95rem; font-weight:700; line-height:1.2; margin-top:10px; color:#F8FAFC; font-variant-numeric:tabular-nums;}
.ks {font-size:.8rem; color:#94A3B8; margin-top:3px;}

/* callouts */
.callout {background:#172033; border:1px solid #243049; border-left:4px solid #F59E0B; border-radius:10px;
  padding:12px 16px; color:#CBD5E1; font-size:.88rem; line-height:1.55; margin:6px 0 14px;}
.callout.info {border-left-color:#38BDF8;}
.callout b {color:#F8FAFC;}

/* leaderboard table */
table.lb {width:100%; border-collapse:separate; border-spacing:0; font-size:.92rem; background:#172033;
  border:1px solid #243049; border-radius:12px; overflow:hidden; font-variant-numeric:tabular-nums;}
table.lb th {text-align:left; color:#94A3B8; font-weight:500; font-size:.8rem; padding:13px 14px;
  background:#131C2E; border-bottom:1px solid #243049;}
table.lb td {padding:14px; border-bottom:1px solid #1E2A42; color:#E2E8F0;}
table.lb tr:last-child td {border-bottom:none;}
table.lb th, table.lb td {border-left:none; border-right:none;}
table.lb td:nth-child(2) {white-space:nowrap;}
table.lb tr.top td {background:rgba(56,189,248,.08);}
table.lb tr.top td:first-child {box-shadow:inset 3px 0 0 #38BDF8;}
.badge {padding:3px 10px; border-radius:999px; font-size:.74rem; font-weight:600; border:1px solid; white-space:nowrap;}
.ml {color:#38BDF8; border-color:rgba(56,189,248,.45); background:rgba(56,189,248,.10);}
.st {color:#CBD5E1; border-color:#3B4A66; background:rgba(148,163,184,.10);}
.best {color:#10B981; border-color:rgba(16,185,129,.5); background:rgba(16,185,129,.12); margin-left:8px;}
.bw {display:inline-block; width:110px; height:7px; background:#243049; border-radius:4px; margin-right:10px;
  overflow:hidden; vertical-align:middle;}
.bar {display:block; height:100%; background:linear-gradient(90deg,#38BDF8,#10B981);}
.dim {color:#4F5D78;} .ci {color:#7C8AA5; font-size:.78rem; margin-left:6px;}
.dot {display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:9px; vertical-align:middle;}

/* confusion-matrix legend */
.term {display:flex; gap:12px; align-items:flex-start; padding:11px 0; border-bottom:1px solid #1E2A42;}
.term:last-child {border-bottom:none;}
.term .sw {width:12px; height:12px; border-radius:3px; margin-top:5px; flex:none;}
.term b {color:#F8FAFC; font-size:.95rem;} .term span {display:block; color:#94A3B8; font-size:.85rem; line-height:1.5;}
.term em {font-style:normal; color:#F8FAFC; font-weight:600; float:right; font-variant-numeric:tabular-nums;}

/* streamlit containers */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] .ctl) {}
div[data-testid="stExpander"] {border:1px solid #243049; border-radius:12px; background:#172033; margin-bottom:10px;}
div[data-testid="stExpander"] summary {padding:14px 18px;}
div[data-testid="stExpander"] summary p {font-weight:600; font-size:1.02rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def kpi(label, value, sub="", tone="primary", icon="database"):
    fg, bg = TONES[tone]
    return (f'<div class="kpi" style="--ac:{fg}"><div class="kh"><span class="kl">{label}</span>'
            f'<span class="ki" style="color:{fg};background:{bg}">{_svg(ICONS[icon])}</span></div>'
            f'<div class="kv">{value}</div><div class="ks">{sub}</div></div>')


def cards(items):
    cols = st.columns(len(items), gap="medium")
    for col, it in zip(cols, items):
        col.markdown(kpi(*it), unsafe_allow_html=True)


def header(title, sub):
    st.markdown(f'<div class="phead"><div class="ptitle">{title}</div><div class="psub">{sub}</div></div>',
                unsafe_allow_html=True)


def section(text, first=False):
    st.markdown(f'<div class="sec{" first" if first else ""}">{text}</div>', unsafe_allow_html=True)


def callout(html, kind="warn"):
    st.markdown(f'<div class="callout {"info" if kind == "info" else ""}">{html}</div>', unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=60, b=10), font=dict(color="#CBD5E1", size=13),
                      title=dict(font=dict(size=17, color="#F8FAFC"), x=0, xanchor="left"),
                      legend=dict(orientation="h", y=-0.18, font=dict(size=12)),
                      hoverlabel=dict(bgcolor="#172033", bordercolor="#243049", font_color="#F8FAFC"))
    fig.update_xaxes(gridcolor="#223049", zerolinecolor="#223049")
    fig.update_yaxes(gridcolor="#223049", zerolinecolor="#223049")
    return fig


def leaderboard(df):
    mx = M["auprc"].max()
    out = ("<table class='lb'><tr><th>#</th><th>Method</th><th>Type</th><th>AUPRC (95% CI)</th>"
           "<th>ROC-AUC</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Alerts</th></tr>")
    dash = "<span class='dim'>-</span>"
    for i, (_, r) in enumerate(df.iterrows(), 1):
        kind = "ml" if r["type"] == "ML detector" else "st"
        has = pd.notna(r["f1"])
        is_best = r["method"] == BEST["method"]
        tag = "<span class='badge best'>Best AUPRC</span>" if is_best else ""
        row = (f"<tr class='{'top' if is_best else ''}'><td class='dim'>{i}</td>"
               f"<td><span class='dot' style='background:{DET.get(r['method'], '#94A3B8')}'></span><b>{r['method']}</b>{tag}</td>"
               f"<td><span class='badge {kind}'>{r['type']}</span></td>"
               f"<td><span class='bw'><span class='bar' style='width:{r['auprc'] / mx * 100:.0f}%'></span></span>"
               f"<b>{r['auprc']:.3f}</b><span class='ci'>[{r['ci_low']:.2f}-{r['ci_high']:.2f}]</span></td>"
               f"<td>{r['roc_auc']:.3f}</td>")
        if has:
            row += (f"<td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td><td>{r['f1']:.3f}</td>"
                    f"<td>{int(r['tp'] + r['fp'])}</td></tr>")
        else:
            row += f"<td>{dash}</td><td>{dash}</td><td>{dash}</td><td>{dash}</td></tr>"
        out += row
    return out + "</table>"


# ------------------------------------------------------------------ sidebar
PAGES = {"Dashboard": ":material/dashboard:", "Model Performance": ":material/monitoring:",
         "Threshold Analysis": ":material/tune:", "Live Transaction Scoring": ":material/bolt:",
         "Methodology": ":material/menu_book:"}
with st.sidebar:
    st.markdown(f'<div class="brand"><span class="logo">{_svg(ICONS["shield"], 20)}</span>'
                f'<div>Anomaly Lens<small>Statistical ML fraud detection</small></div></div>'
                '<div class="status"><i></i>Test results loaded</div>'
                '<div class="navlabel">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", list(PAGES), format_func=lambda p: f"{PAGES[p]} {p}", label_visibility="collapsed")
    st.markdown('<div class="sidenote">Data: ULB credit-card fraud (Kaggle).<br>All results are from a held-out '
                'test set.<br>Detectors are fit on normal transactions only.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ Dashboard
if page == "Dashboard":
    header("Fraud detection dashboard",
           f"Held-out test set of {META['n_test']:,} transactions with {N_POS} confirmed frauds "
           f"({META['fraud_rate'] * 100:.3f}% of the total).")
    top = M.dropna(subset=["f1"]).sort_values("f1", ascending=False).iloc[0]
    cards([
        ("Test transactions", f"{META['n_test']:,}", "never used for fitting", "primary", "database"),
        ("Confirmed frauds", f"{N_POS}", f"{META['fraud_rate'] * 100:.3f}% of test", "bad", "alert"),
        ("Best AUPRC", f"{BEST['auprc']:.3f}", f"{BEST['method']}, {BEST['auprc'] / META['fraud_rate']:,.0f}x random", "good", "trend"),
        ("Precision", f"{top['precision'] * 100:.1f}%", f"{top['method']} at tuned threshold", "primary", "target"),
        ("Recall", f"{top['recall'] * 100:.1f}%", f"{int(top['tp'])} of {N_POS} frauds caught", "warn", "check"),
    ])

    section("Model performance comparison")
    left, right = st.columns([3, 2], gap="large")
    s = M.sort_values("auprc")
    groups = [("Best detector", s[s["method"] == BEST["method"]], "#38BDF8"),
              ("Other ML detectors", s[(s["type"] == "ML detector") & (s["method"] != BEST["method"])], "#2B6CB0"),
              ("Statistical baselines", s[(s["type"] != "ML detector") & (s["method"] != BEST["method"])], "#64748B")]
    fig = go.Figure()
    for name, g, colr in groups:
        if len(g):
            fig.add_bar(x=g["auprc"], y=g["method"], orientation="h", name=name, marker_color=colr,
                        error_x=dict(type="data", symmetric=False, array=(g["ci_high"] - g["auprc"]).clip(lower=0),
                                     arrayminus=(g["auprc"] - g["ci_low"]).clip(lower=0), color="#94A3B8"),
                        hovertemplate="%{y}: AUPRC %{x:.3f}<extra></extra>")
    fig.update_layout(title=f"AUPRC by method (random guess = {META['fraud_rate']:.4f})", barmode="overlay")
    fig.update_xaxes(range=[0, 0.85])
    fig.update_yaxes(categoryorder="array", categoryarray=list(s["method"]))
    left.plotly_chart(style(fig, 400), use_container_width=True)

    fd = go.Figure(go.Bar(x=["Normal", "Fraud"], y=[N_NEG, N_POS], marker_color=["#2B6CB0", COL["bad"]],
                          text=[f"{N_NEG:,}", f"{N_POS:,}"], textposition="outside", cliponaxis=False,
                          hovertemplate="%{x}: %{y:,}<extra></extra>"))
    fd.update_layout(title="Fraud vs normal transactions (test set)", showlegend=False)
    fd.update_yaxes(type="log", range=[0, 5.7], dtick=1, title="Transactions (log scale)")
    right.plotly_chart(style(fd, 400), use_container_width=True)
    right.caption(f"About 1 in {1 / META['fraud_rate']:,.0f} test transactions is fraud, which is why AUPRC "
                  "is the headline metric here.")

    section("Model leaderboard")
    f1c, f2c, _sp = st.columns([1, 1, 2], gap="medium")
    show = f1c.selectbox("Show", ["All methods", "ML detector", "Statistical baseline"])
    sort_label = f2c.selectbox("Sort by", ["AUPRC", "ROC-AUC", "F1 Score"])
    sort = {"AUPRC": "auprc", "ROC-AUC": "roc_auc", "F1 Score": "f1"}[sort_label]
    df = M if show == "All methods" else M[M["type"] == show]
    df = df.sort_values(sort, ascending=False, na_position="last")
    st.markdown(leaderboard(df), unsafe_allow_html=True)
    st.caption("AUPRC and ROC-AUC: full-train models. Precision, recall, F1, alerts: models refit on 75% of train, "
               "threshold chosen on validation data (max F1), applied once to test.")

# ------------------------------------------------------------------ Model Performance
elif page == "Model Performance":
    header("Model performance",
           f"Curves computed from the test-set ranking at {len(GRID)} alert counts. "
           "Precision-recall is the honest view at a 0.17% fraud rate.")
    names = list(D["curves"].keys())
    pc, _sp = st.columns([2, 3], gap="large")
    pick = pc.multiselect("Methods to compare", names, default=[m for m in M["method"] if m in names][:4])
    if not pick:
        st.info("Select at least one method to draw the curves.")
    else:
        f_pr, f_roc = go.Figure(), go.Figure()
        for m in pick:
            tp = np.array(D["curves"][m]["tp"]); fp = np.array(D["curves"][m]["fp"])
            f_pr.add_scatter(x=tp / N_POS, y=tp / GRID, mode="lines", name=m, line=dict(color=DET.get(m), width=2.6))
            f_roc.add_scatter(x=fp / N_NEG, y=tp / N_POS, mode="lines", name=m, line=dict(color=DET.get(m), width=2.6))
        f_pr.add_hline(y=META["fraud_rate"], line_dash="dash", line_color="#64748B")
        f_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="random", line=dict(color="#64748B", dash="dash"))
        f_pr.update_layout(title="Precision-recall curve (test set)", xaxis_title="Recall", yaxis_title="Precision")
        f_roc.update_layout(title="ROC curve (test set)", xaxis_title="False positive rate", yaxis_title="True positive rate")
        f_roc.update_xaxes(range=[0, 1])
        l, r = st.columns(2, gap="large")
        l.plotly_chart(style(f_pr, 520), use_container_width=True)
        l.caption("Higher and further right is better. Shows how many of the alerts are real frauds (precision) "
                  "as more of the frauds are caught (recall). The dashed line is a random guesser.")
        r.plotly_chart(style(f_roc, 520), use_container_width=True)
        r.caption("Shows frauds caught against normal transactions wrongly flagged. At a 0.17% fraud rate the "
                  "false positive rate stays tiny, so ROC looks good even for weak detectors.")

        section("Summary metrics for the selected methods")
        t = M[M["method"].isin(pick)]
        rows = "".join(f"<tr><td><span class='dot' style='background:{DET.get(r['method'], '#94A3B8')}'></span>"
                       f"<b>{r['method']}</b></td><td>{r['auprc']:.3f}</td><td>{r['roc_auc']:.3f}</td></tr>"
                       for _, r in t.iterrows())
        st.markdown(f"<table class='lb'><tr><th>Method</th><th>AUPRC</th><th>ROC-AUC</th></tr>{rows}</table>",
                    unsafe_allow_html=True)

# ------------------------------------------------------------------ Threshold Analysis
elif page == "Threshold Analysis":
    header("Threshold analysis",
           "More alerts catch more frauds but raise more false alarms. Alerts are the N highest-scoring test transactions.")
    ops = list(D["operating"].keys())
    with st.container(border=True):
        c1, c2 = st.columns([1, 2], gap="large")
        method = c1.selectbox("Method", ops, index=ops.index("PCA recon. error") if "PCA recon. error" in ops else 0)
        chosen = D["operating"][method]["chosen_alerts"]
        key = f"n_{method}"
        if key not in st.session_state:
            st.session_state[key] = chosen
        if c1.button("Jump to validation-chosen threshold"):
            st.session_state[key] = chosen
        n = c2.slider("Number of alerts", 10, 300, key=key)
        c2.caption(f"Validation-chosen threshold: {chosen} alerts. "
                   + ("You are at the validation-chosen setting." if n == chosen else "Move the slider or use the button to return to it."))
    j = int(np.searchsorted(GRID, n))
    tp = int(D["operating"][method]["tp"][j]); fp = n - tp; fn = N_POS - tp; tn = N_NEG - fp
    prec, rec = tp / n, tp / N_POS
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    st.write("")
    cards([
        ("Alerts raised", n, f"validation-chosen: {chosen}", "primary", "bell"),
        ("Frauds caught", tp, f"of {N_POS}", "good", "check"),
        ("False alarms", fp, f"{fp / N_NEG * 100:.3f}% of normal", "warn", "alert"),
        ("Precision", f"{prec * 100:.1f}%", "alerts that are real", "primary", "target"),
        ("Recall", f"{rec * 100:.1f}%", "frauds caught", "primary", "trend"),
        ("F1 Score", f"{f1:.3f}", "balance of both", "muted", "scale"),
    ])
    section("Confusion matrix at this threshold")
    left, right = st.columns([3, 2], gap="large")
    cm_cols = ["#0F4C5C", "#8A5A0B", "#8C2A2A", "#0B6B4F"]  # TN, FP, FN, TP
    cm = go.Figure(go.Heatmap(
        z=[[0, 1], [2, 3]], zmin=-0.5, zmax=3.5,
        x=["Predicted normal", "Predicted fraud"], y=["Actually normal", "Actually fraud"],
        text=[[f"TN<br>{tn:,}", f"FP<br>{fp:,}"], [f"FN<br>{fn:,}", f"TP<br>{tp:,}"]], texttemplate="%{text}",
        textfont=dict(size=20, color="#F8FAFC"), xgap=4, ygap=4, showscale=False, hoverinfo="skip",
        colorscale=[[0, cm_cols[0]], [.25, cm_cols[0]], [.25, cm_cols[1]], [.5, cm_cols[1]],
                    [.5, cm_cols[2]], [.75, cm_cols[2]], [.75, cm_cols[3]], [1, cm_cols[3]]]))
    cm.update_yaxes(autorange="reversed")
    cm.update_layout(title=f"{method} with {n} alerts")
    left.plotly_chart(style(cm, 400), use_container_width=True)
    defs = [("True positive (TP)", "A fraud that was flagged. The goal.", tp, cm_cols[3]),
            ("True negative (TN)", "A normal transaction correctly left alone.", tn, cm_cols[0]),
            ("False positive (FP)", "A normal transaction flagged by mistake. This is a false alarm.", fp, cm_cols[1]),
            ("False negative (FN)", "A fraud the detector missed.", fn, cm_cols[2])]
    terms = "".join(f"<div class='term'><div class='sw' style='background:{c}'></div><div style='flex:1'>"
                    f"<b>{a}</b><em>{v:,}</em><span>{b}</span></div></div>" for a, b, v, c in defs)
    right.markdown(f'<div class="sec first">How to read it</div>{terms}', unsafe_allow_html=True)

# ------------------------------------------------------------------ Live Transaction Scoring
elif page == "Live Transaction Scoring":
    header("Live transaction scoring",
           "Upload transactions as a CSV and rank them by anomaly score. Same preprocessing, models and thresholds as the notebook.")
    B = load_bundle()
    need = B["feature_names"]
    keys = list(B["models"].keys())
    with st.container(border=True):
        c1, c2 = st.columns([1, 2], gap="large")
        method = c1.selectbox("Detector", keys, index=keys.index("PCA recon. error"))
        use_demo = c1.button("Load demo file")
        up = c2.file_uploader("Upload a CSV file", type="csv")
        with st.expander("Supported format and required columns"):
            st.markdown(
                "- A CSV with a header row and one transaction per row, up to 100,000 rows.\n"
                f"- **Required:** `{need[0]}` to `{need[-1]}` and `Amount`. All numeric, no missing values, `Amount` not negative.\n"
                "- **Optional:** `Time` (ignored) and `Class` (0 = normal, 1 = fraud). With `Class`, the results show how many "
                "labelled frauds were caught.")

    raw, is_demo = None, False
    if use_demo:
        st.session_state["demo"] = True
    if up is not None:
        st.session_state["demo"] = False
        try:
            raw = pd.read_csv(up)
        except Exception:
            st.error("The file could not be read as a CSV. Check that it is comma-separated and has a header row.")
    elif st.session_state.get("demo"):
        raw, is_demo = pd.read_csv(BASE / "data" / "demo_transactions.csv"), True

    if raw is None and up is None:
        st.info("Upload a CSV or load the demo file to start scoring.")

    if is_demo:
        callout("<b>Demo sample, not a real-world rate.</b> 4,000 normal and 60 labelled fraud rows from the test set "
                "(1.5% fraud, about 9x the real rate). Precision here looks better than it would on real traffic. "
                "Real-rate results are on the Dashboard page.")

    if raw is not None:
        miss = [c for c in need if c not in raw.columns]
        if len(raw) == 0:
            st.error("The file has no rows.")
        elif miss:
            st.error(f"Missing required columns: {miss[:6]}{' ...' if len(miss) > 6 else ''}")
        elif not all(pd.api.types.is_numeric_dtype(raw[c]) for c in need):
            st.error("All required columns must be numeric. Remove text values and re-upload.")
        elif raw[need].isnull().any().any():
            st.error("The file has missing values in the required columns. Please clean it and re-upload.")
        elif (raw["Amount"] < 0).any():
            st.error("Amount must be non-negative.")
        elif len(raw) > 100000:
            st.error("Please upload at most 100,000 rows.")
        else:
            st.success(f"File checked: {len(raw):,} transactions, all {len(need)} required columns found.")
            prog = st.progress(10, text="Scoring transactions...")
            s = scored(raw, method)
            prog.progress(80, text="Ranking results...")
            thr = B["thresholds"][method]
            out = raw.copy()
            out.insert(0, "txn_id", ["TXN-%05d" % i for i in range(1, len(out) + 1)])
            out["anomaly_score"] = s
            out["threshold_utilization_pct"] = s / thr * 100
            out["risk_level"] = np.where(s >= thr, "High", np.where(s >= 0.5 * thr, "Medium", "Low"))
            out["detection_status"] = np.where(s >= thr, "Flagged", "Not flagged")
            out = out.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
            lead = ["txn_id", "anomaly_score", "threshold_utilization_pct", "risk_level", "detection_status"]
            out = out[lead + [c for c in out.columns if c not in lead]]
            prog.progress(100, text="Done")
            prog.empty()

            n_f = int((out["detection_status"] == "Flagged").sum())
            n_m = int((out["risk_level"] == "Medium").sum())
            section("Results summary", first=True)
            sub_lab = "enriched demo sample" if is_demo else "on this file"
            c4 = (("Labelled frauds caught", f"{int(((out['detection_status'] == 'Flagged') & (out['Class'] == 1)).sum())} / "
                   f"{int((out['Class'] == 1).sum())}",
                   f"precision {int(((out['detection_status'] == 'Flagged') & (out['Class'] == 1)).sum()) / max(n_f, 1) * 100:.1f}%, {sub_lab}",
                   "good", "check")
                  if "Class" in out.columns else
                  ("Detection threshold", f"{thr:.2f}", "anomaly score, chosen on validation data", "primary", "target"))
            cards([("Transactions scored", f"{len(out):,}", f"detector: {method}", "primary", "layers"),
                   ("Flagged", f"{n_f}", f"{n_f / len(out) * 100:.2f}% of file", "bad", "alert"),
                   ("Medium risk", f"{n_m}", "display tier only", "warn", "bell"), c4])

            section("Risk feed (top 25 by anomaly score)")
            lvl = {"High": ("#EF4444", "rgba(239,68,68,.14)"), "Medium": ("#F59E0B", "rgba(245,158,11,.14)"),
                   "Low": ("#10B981", "rgba(16,185,129,.14)")}
            stt = {"Flagged": ("#EF4444", "rgba(239,68,68,.14)"), "Not flagged": ("#CBD5E1", "rgba(148,163,184,.10)")}
            rows = ("<table class='lb'><tr><th>Transaction</th><th>Amount</th><th>Anomaly score</th>"
                    "<th>Threshold utilization</th><th>Risk level</th><th>Detection status</th></tr>")
            for _, r in out.head(25).iterrows():
                lf, lb = lvl[r["risk_level"]]; sf, sb = stt[r["detection_status"]]
                util = r["threshold_utilization_pct"]
                rows += (f"<tr><td><b>{r['txn_id']}</b></td><td>${r['Amount']:,.2f}</td><td>{r['anomaly_score']:,.2f}</td>"
                         f"<td><span class='bw'><span class='bar' style='width:{min(util, 100):.0f}%;background:{lf}'></span></span>"
                         f"{util:,.0f}%</td>"
                         f"<td><span class='badge' style='color:{lf};border-color:{lf};background:{lb}'>{r['risk_level']}</span></td>"
                         f"<td><span class='badge' style='color:{sf};border-color:{sf};background:{sb}'>{r['detection_status']}</span></td></tr>")
            st.markdown(rows + "</table>", unsafe_allow_html=True)
            callout("<b>The anomaly score is not a fraud probability.</b> It is the detector's raw output, and higher means "
                    "more unusual. Threshold utilization is the anomaly score as a percentage of the validation-chosen "
                    "threshold (the bar stops at 100%). Medium risk (50-100% of the threshold) is a display tier, "
                    "not a validated cut-off.", kind="info")

            section("Download results")
            with st.container(border=True):
                d1, d2 = st.columns(2, gap="large")
                d1.download_button("Download all scored results (CSV)", out.to_csv(index=False).encode(),
                                   "scored_transactions.csv", "text/csv", use_container_width=True)
                d2.download_button("Download flagged transactions only (CSV)",
                                   out[out["detection_status"] == "Flagged"].to_csv(index=False).encode(),
                                   "flagged_transactions.csv", "text/csv", use_container_width=True)

            with st.expander("Full results with all features"):
                flt = st.multiselect("Filter by risk level", ["High", "Medium", "Low"], default=["High"])
                st.dataframe(out[out["risk_level"].isin(flt)].head(500), use_container_width=True)

# ------------------------------------------------------------------ Methodology
elif page == "Methodology":
    header("Methodology and limitations", "How the results were produced, and what they do and do not show.")
    cards([("Raw rows", f"{META['raw_rows']:,}", f"{META['raw_fraud']} frauds", "primary", "database"),
           ("After de-duplication", f"{META['clean_rows']:,}", f"{META['clean_fraud']} frauds (0.167%)", "primary", "layers"),
           ("Training rows", f"{META['train_rows']:,}", "detectors fit on normal only", "good", "check"),
           ("Test rows", f"{META['n_test']:,}", f"{N_POS} frauds, labels used only to evaluate", "warn", "target")])
    st.write("")
    with st.expander("Dataset information", expanded=True):
        st.markdown(f"ULB credit-card fraud data: **{META['raw_rows']:,}** rows and **{META['raw_fraud']}** frauds. "
                    f"After removing duplicate rows: **{META['clean_rows']:,}** rows and **{META['clean_fraud']}** frauds (0.167%). "
                    "The features V1 to V28 are anonymized PCA components.")
    with st.expander("Data preprocessing"):
        st.markdown("- **Features:** V1 to V28 and log(1+Amount). `Time` was dropped.\n"
                    "- **Split:** stratified 70/30 before any scaling.\n"
                    "- **Scaling:** the scaler is fit on training data only.\n"
                    f"- **Validation:** a validation split is taken from the training data ({META['val_frauds']} frauds in it).")
    with st.expander("Statistical and ML methods"):
        a, b = st.columns(2, gap="large")
        a.markdown("**Statistical baselines**\n\n- Z-score\n- IQR\n- Mahalanobis distance")
        b.markdown(f"**ML detectors**\n\n- PCA reconstruction error (k = {META['K_PCA']})\n- Gaussian Mixture (k = {META['K_GMM']})\n"
                   "- Local Outlier Factor\n- One-Class SVM\n- Isolation Forest\n\nThe last two used default settings and were not tuned.")
        st.caption("Detectors are fit on normal training transactions only (one-class setting).")
    with st.expander("Evaluation methodology"):
        st.markdown("- **Headline metric:** AUPRC. ROC-AUC is above 0.90 for every method and does not separate them at a 0.17% fraud rate.\n"
                    "- **Uncertainty:** intervals are 95% bootstrap CIs over the test set.\n"
                    f"- **Leakage controls:** hyper-parameters (PCA k = {META['K_PCA']}, GMM k = {META['K_GMM']}) and alert thresholds "
                    "were chosen on the validation split; test labels are used only for evaluation.")
    with st.expander("Limitations and future improvements"):
        a, b = st.columns(2, gap="large")
        a.markdown("**Limitations**\n\n"
                   "- **Selection was not fully blind.** Test AUPRC was inspected during early model exploration; final hyper-parameters were then chosen on validation data.\n"
                   "- **Small sample.** 142 test frauds (83 in validation), so metrics are noisy. Validation F1 exceeded test F1 for every method.\n"
                   "- **Fit-to-fit variance.** GMM and LOF moved by about 0.04 AUPRC when refit on 75% of the training data; do not read a firm ordering between them.\n"
                   "- **Features are already PCA outputs**, which may favor a PCA-based detector. Results may not transfer to raw financial data.\n"
                   "- **Random split.** The data spans two days.\n"
                   "- **Missed frauds** tend to be small amounts (median 1.79 vs 33.59 for caught ones); this is descriptive, not causal.\n"
                   "- **The demo file is enriched** (1.5% fraud), so its precision is not representative.")
        b.markdown("**Future improvements**\n\n"
                   "- Train on the past and test on the future, as a real system would.\n"
                   "- Test on raw financial features instead of PCA components.\n"
                   "- Tune One-Class SVM and Isolation Forest.\n"
                   "- Collect more labelled frauds to tighten the intervals.")
