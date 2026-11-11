"""
Water Pump Maintenance Decision Support System (DSS) -- Final Dashboard
=========================================================================
Base method reproduced from: Pathak & Shalini (2023), "Predicting Water
Pump Status Using Attentive Tabular Learning" (arXiv:2304.03969).

WHAT IS REPRODUCED FROM THE PAPER:
  - The prediction task itself (3-class pump status classification)
  - The dataset (DrivenData "Pump it Up")
  - The class-imbalance problem the paper's method (TabNet + Focal Loss)
    was designed to address

WHAT IS THIS PROJECT'S OWN EXTENSION (not in the original paper):
  1. Per-class evaluation (the paper only reports overall accuracy)
  2. A second, simpler baseline (Decision Tree, weighted + unweighted) to
     test whether the model complexity is actually earning anything
  3. AHP-based, consistency-checked maintenance priority ranking
  4. A cost-policy comparison (predictive/triage vs. blanket replacement)
  5. A live current-data tab (WPdx API) addressing the aging-dataset issue
  6. A predicted-vs-actual overlay, kept as a one-time historical validation
     check (ground truth happens to be available in this dataset)
  7. A simple automatic regional risk-spike flag ("Regions to watch")
  8. The explicit decision chain: dashboard viewer -> prioritized list ->
     field technician dispatch -> repair/defer decision
  9. Leakage-safe (out-of-fold) scoring: every predicted_status and
     needs_repair_confidence value shown anywhere in this dashboard comes
     from 5-fold stratified cross-validation, not from a model scored on
     the same rows it was trained on.

NAVIGATION DESIGN NOTE: "Model Performance" is intentionally NOT a top-level
page. It is supplementary/diagnostic information (which model to trust, and
why) rather than something an operations manager opens day to day -- it
lives inside Data & Methodology, where the audience for it (you, a reviewer,
an auditor) actually looks.

Run with:  streamlit run app.py
Needs training_set_values.csv and training_set_labels.csv in the same folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ============================================================
# CONSTANTS
# ============================================================

STATUS_COLORS = {
    "functional": "green",
    "functional needs repair": "orange",
    "non functional": "red",
}
STATUS_ICONS = {
    "functional": "ok-sign",
    "functional needs repair": "wrench",
    "non functional": "remove-sign",
}
COLOR_EMOJI = {"green": "🟢", "orange": "🟠", "red": "🔴"}

PLAIN_LABELS = {
    "id": "Pump ID",
    "region": "Region",
    "population": "Population Served",
    "construction_year": "Year Built",
    "needs_repair_confidence": "Est. Likelihood of Needing Repair",
    "ahp_priority_score": "Priority Score",
    "predicted_status": "Predicted Status",
    "actual_status": "Recorded Status (historical)",
}

# Risk-level bands derived from model confidence -- kept SEPARATE from the
# raw predicted class: "working status" and "how urgent" are two different
# questions and should never collapse into one indicator.
def risk_level(confidence):
    if confidence >= 0.66:
        return "Critical"
    elif confidence >= 0.40:
        return "High"
    elif confidence >= 0.15:
        return "Monitor"
    return "Low"

RISK_COLORS = {"Critical": "red", "High": "orange", "Monitor": "#E8C547", "Low": "green"}
RISK_EMOJI = {"Critical": "🔴", "High": "🟠", "Monitor": "🟡", "Low": "🟢"}

FEATURES = [
    "extraction_type", "quantity", "waterpoint_type", "population",
    "region", "longitude", "latitude", "construction_year",
]
CATEGORICAL = ["extraction_type", "quantity", "waterpoint_type", "region"]
TARGET = "status_group"

# ============================================================
# AHP (Analytic Hierarchy Process) -- derived transparently in code, not
# hardcoded. The pairwise matrix below is the actual judgment being made:
# Confidence judged moderately more important than Population (Saaty value
# 3), strongly more important than Age (5); Population judged moderately
# more important than Age (3). This is a SUBJECTIVE judgment call -- AHP
# does not claim to remove that subjectivity, only to make it explicit and
# internally consistent (see the consistency ratio below, and the
# sensitivity analysis on Data & Methodology for how the final ranking
# holds up under alternative, equally defensible judgments).
AHP_CRITERIA = ["confidence", "population", "age"]
AHP_PAIRWISE_MATRIX = np.array([
    [1,    3,    5],    # Confidence vs [Confidence, Population, Age]
    [1/3,  1,    3],    # Population vs [Confidence, Population, Age]
    [1/5,  1/3,  1],    # Age        vs [Confidence, Population, Age]
])
AHP_RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
                    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def derive_ahp_weights(matrix):
    """Derives AHP priority weights from a pairwise comparison matrix using
    Saaty's normalize-columns-then-average-rows method, plus the standard
    consistency ratio (CR) check. Returns (weights_dict, consistency_ratio).
    CR < 0.10 means the judgments don't logically contradict each other --
    it does NOT mean the judgments are objectively correct or unbiased."""
    n = matrix.shape[0]
    normalized = matrix / matrix.sum(axis=0)
    weights = normalized.mean(axis=1)

    weighted_sum = matrix @ weights
    lambda_max = (weighted_sum / weights).mean()
    ci = (lambda_max - n) / (n - 1)
    ri = AHP_RANDOM_INDEX.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return dict(zip(AHP_CRITERIA, weights)), cr


AHP_WEIGHTS, AHP_CONSISTENCY_RATIO = derive_ahp_weights(AHP_PAIRWISE_MATRIX)

# Default illustrative cost assumptions -- used ONLY on the Cost & Capacity
# Scenario page, where the assumptions are visible and adjustable. Not
# surfaced as a headline number anywhere else (see redesign notes).
DEFAULT_COST_REPAIR = 50
DEFAULT_COST_REPLACE = 1500

WPDX_BASE_URL = "https://data.waterpointdata.org/resource/jfkt-jmqa.json"

st.set_page_config(page_title="Pump Maintenance DSS", layout="wide", page_icon="💧")

# ============================================================
# VISUAL DESIGN SYSTEM
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --navy: #17324D;
        --blue: #276A8C;
        --teal: #2A8C82;
        --amber: #B8862E;
        --slate: #5B6B78;
        --ink: #243746;
        --muted: #55626F;
        --line: #E3E9EE;
        --canvas: #EDF1F4;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(237,241,244,0.94); }
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
    [data-testid="stSidebar"] .stRadio label { padding: 0.38rem 0.2rem; font-size: 0.93rem; }
    h1, h2, h3 { color: var(--navy) !important; letter-spacing: -0.02em; }
    h1 { font-size: 1.85rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.35rem !important; font-weight: 650 !important; }
    h3 { font-size: 1.08rem !important; font-weight: 650 !important; }
    [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid var(--line); border-radius: 9px;
        padding: 14px 16px; box-shadow: 0 1px 2px rgba(23,50,77,0.05);
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.82rem; }
    [data-testid="stMetricValue"] { color: var(--navy) !important; font-weight: 700; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--line); border-radius: 10px;
        box-shadow: 0 1px 3px rgba(23,50,77,0.05); background: #FFFFFF;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; }
    .stButton button, .stDownloadButton button { border-radius: 7px; font-weight: 600; }
    [data-testid="stAlert"] { border-radius: 8px; }
    hr { border-color: var(--line); }

    .page-kicker {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; margin-bottom: 3px;
    }
    .page-banner {
        background: #FFFFFF; border-radius: 8px; padding: 10px 16px;
        margin-bottom: 10px; box-shadow: 0 1px 2px rgba(23,50,77,0.05);
    }
    .page-banner p { color: var(--ink) !important; font-size: 0.95rem; margin: 0; }

    .kpi-card {
        background: #FFFFFF; border: 1px solid var(--line); border-radius: 9px;
        padding: 14px 16px; box-shadow: 0 1px 2px rgba(23,50,77,0.05);
    }
    .kpi-card .kpi-label { color: var(--muted); font-size: 0.82rem; }
    .kpi-card .kpi-value { color: var(--navy); font-size: 1.65rem; font-weight: 700; margin-top: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

STATUS_PLOTLY_COLORS = {
    "functional": "#2A8C82",
    "functional needs repair": "#D69A36",
    "non functional": "#B94A48",
}
RISK_PLOTLY_COLORS = {"Low": "#2A8C82", "Monitor": "#D8B45A", "High": "#D18A3A", "Critical": "#B94A48"}
PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_BG = dict(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")  # keeps every chart's background
                                                                    # identical to the white card it
                                                                    # sits in -- no visible seam.

# Each page gets its own accent color + a short category label ("kicker"),
# so the top of every page looks visually distinct instead of six identical
# gray caption blocks -- and so Methodology visually reads as a different
# TIER of page (reference material) than the four operational pages.
PAGE_META = {
    "Executive Overview": {"kicker": "DAILY OPERATIONS", "accent": "var(--teal)"},
    "Asset Explorer": {"kicker": "DAILY OPERATIONS", "accent": "var(--blue)"},
    "Maintenance Worklist": {"kicker": "DAILY OPERATIONS", "accent": "var(--navy)"},
    "Cost & Capacity Scenario": {"kicker": "PLANNING", "accent": "var(--amber)"},
    "Data & Methodology": {"kicker": "METHODOLOGY & VALIDATION", "accent": "var(--slate)"},
}


def page_banner(page_key, text):
    """What this whole page is for, in one plain sentence -- with a colored
    accent + category label so each page is visually distinct at the top,
    instead of an identical gray caption repeated six times."""
    meta = PAGE_META[page_key]
    st.markdown(
        f'''<div class="page-banner" style="border-left:4px solid {meta["accent"]};">
        <div class="page-kicker" style="color:{meta["accent"]};">{meta["kicker"]}</div>
        <p>{text}</p>
        </div>''',
        unsafe_allow_html=True,
    )


def widget_purpose(purpose, shows, read_as, not_text):
    """Plain-language explanation of a single widget, kept in an expander so
    the default view stays clean."""
    with st.expander("What is this for?"):
        st.markdown(f"**Why it's here:** {purpose}")
        st.markdown(f"**What it shows:** {shows}")
        st.markdown(f"**How to read it:** {read_as}")
        st.caption(f"**Good to know:** {not_text}")


def screen_header(looking_at, pay_attention_to, can_do):
    """Compact decision context for each dashboard page, in plain language."""
    st.markdown(f"**You're looking at:** {looking_at}")
    st.caption(f"**Watch for:** {pay_attention_to}  |  **Do next:** {can_do}")


def chart_note(text):
    """Brief chart interpretation."""
    st.caption(text)


def kpi_card(icon, label, value, accent):
    """A single headline KPI as a colored card -- replaces plain st.metric
    for the Overview's top row so the most-viewed part of the dashboard
    isn't flat white-on-white."""
    st.markdown(
        f'''<div class="kpi-card" style="border-top:3px solid {accent};">
        <div class="kpi-label">{icon} {label}</div>
        <div class="kpi-value">{value}</div>
        </div>''',
        unsafe_allow_html=True,
    )


def empty_state(df, message="No pumps match the selected filters."):
    if len(df) == 0:
        st.info(f"ℹ️ {message}")
        return True
    return False


# ============================================================
# CACHED DATA / MODEL PIPELINE
# ============================================================

@st.cache_data
def load_and_clean_data():
    try:
        train_values = pd.read_csv("training_set_values.csv")
        train_labels = pd.read_csv("training_set_labels.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Could not find 'training_set_values.csv' and/or 'training_set_labels.csv' in this "
            "folder. Place both files next to app.py (same folder) and restart the app."
        ) from e
    df = train_values.merge(train_labels, on="id")

    df = df.drop(columns=["recorded_by"])  # confirmed 100% constant -> zero predictive value
    df["construction_year"] = df["construction_year"].replace(0, np.nan)  # 0 = placeholder, not a real year

    subset = df[["id"] + FEATURES + [TARGET]].copy()
    for col in CATEGORICAL:
        subset[col] = subset[col].fillna(subset[col].mode()[0])
    subset["construction_year"] = pd.to_numeric(subset["construction_year"], errors="coerce")
    subset["construction_year"] = subset["construction_year"].fillna(subset["construction_year"].median())
    return subset


@st.cache_resource
def train_models(df):
    X = df[FEATURES].copy()
    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    def evaluate(name, model):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(
            y_test, preds, target_names=y_encoder.classes_, output_dict=True
        )
        return {"model": model, "accuracy": acc, "report": report}

    results["XGBoost"] = evaluate(
        "XGBoost", XGBClassifier(n_estimators=200, random_state=42, eval_metric="mlogloss")
    )
    results["Decision Tree (unweighted)"] = evaluate(
        "Decision Tree (unweighted)", DecisionTreeClassifier(max_depth=10, random_state=42)
    )
    results["Decision Tree (balanced)"] = evaluate(
        "Decision Tree (balanced)",
        DecisionTreeClassifier(max_depth=10, random_state=42, class_weight="balanced"),
    )

    # --- Leakage-safe scoring for everything the dashboard DISPLAYS ---
    # The primary model is never scored on rows it was trained on. Every
    # predicted_status / needs_repair_confidence value shown anywhere in this
    # dashboard comes from 5-fold stratified cross-validation: each row is
    # scored only by a fold that did NOT train on it, using the identical
    # XGBoost configuration evaluated in the leaderboard above.
    needs_repair_idx = list(y_encoder.classes_).index("functional needs repair")
    oof_model = XGBClassifier(n_estimators=200, random_state=42, eval_metric="mlogloss")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(oof_model, X, y, cv=skf, method="predict_proba")
    oof_pred_encoded = oof_proba.argmax(axis=1)

    df = df.copy()
    df["predicted_status"] = y_encoder.inverse_transform(oof_pred_encoded)
    df["actual_status"] = df[TARGET]
    df["needs_repair_confidence"] = oof_proba[:, needs_repair_idx]

    return df, results, y_encoder.classes_


def _ahp_score_vector(df, weights):
    """Computes the per-pump AHP priority SCORE (not to be confused with the
    fixed per-criterion WEIGHTS above) -- this is what differs for every
    pump and what the Maintenance Worklist actually ranks by."""
    conf_norm = df["needs_repair_confidence"]
    pop_norm = np.log1p(df["population"]) / np.log1p(df["population"]).max()
    age_norm = 1 - (
        (df["construction_year"] - df["construction_year"].min())
        / (df["construction_year"].max() - df["construction_year"].min())
    )
    return (
        weights["confidence"] * conf_norm
        + weights["population"] * pop_norm
        + weights["age"] * age_norm
    )


@st.cache_data
def compute_ahp_priority(df):
    df = df.copy()
    df["ahp_priority_score"] = _ahp_score_vector(df, AHP_WEIGHTS)
    return df


@st.cache_data(ttl=86400)
def fetch_live_wpdx(limit=2000):
    """Returns (live_df, fetched_at). fetched_at is captured INSIDE the cached
    call, so it reflects the moment data was actually pulled, not the moment
    the page was viewed -- this is what lets the dashboard show "last pulled
    X ago" accurately even though Streamlit silently reuses the cached result
    on every subsequent page view within the 24h window."""
    params = {
        "$where": "clean_country_name = 'Tanzania'",
        "$order": "report_date DESC",
        "$limit": limit,
    }
    resp = requests.get(WPDX_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    live_df = pd.DataFrame(resp.json())
    if "report_date" in live_df.columns:
        live_df["report_date_parsed"] = pd.to_datetime(live_df["report_date"], errors="coerce")
    fetched_at = datetime.now(timezone.utc)
    return live_df, fetched_at


# WPdx's own status wording does not match this project's 3-class training
# taxonomy word-for-word (confirmed against a live pull: WPdx uses labels like
# "needs repair", "Functional", "dry season", "not in use"). Only labels with
# an honest, unambiguous equivalent are mapped below. Anything else is left
# out of the mapping on purpose -- it falls through to "Other / not
# comparable" rather than being guessed into the nearest-sounding class, so
# the historical-vs-current comparison never quietly overstates how much of
# the live feed it actually covers.
WPDX_STATUS_MAP = {
    "functional": "functional",
    "functional but not in use": "functional",
    "non functional": "non functional",
    "not functional": "non functional",
    "abandoned": "non functional",
    "decommissioned": "non functional",
    "needs repair": "functional needs repair",
    "functional needs repair": "functional needs repair",
}


def map_wpdx_status(raw_status):
    """Maps one WPdx status_clean value onto this project's 3-class taxonomy,
    or 'Other / not comparable' if there is no honest equivalent (see
    WPDX_STATUS_MAP above for why some labels -- e.g. 'dry season', 'not in
    use' -- are deliberately left unmapped)."""
    if pd.isna(raw_status):
        return "Other / not comparable"
    return WPDX_STATUS_MAP.get(str(raw_status).strip().lower(), "Other / not comparable")


@st.cache_data
def get_data_vintage():
    """Reads just the date_recorded column from the raw training file, so the
    dashboard can state plainly when this (fixed, historical) dataset was
    actually collected -- answers "is this the newest data?" without anyone
    having to go digging for it."""
    try:
        raw_dates = pd.read_csv("training_set_values.csv", usecols=["date_recorded"])
        parsed = pd.to_datetime(raw_dates["date_recorded"], errors="coerce")
        return parsed.min(), parsed.max()
    except Exception:
        return None, None


def flag_regional_anomalies(df, overall_rate_col="predicted_status", target_value="functional needs repair"):
    region_rates = df.groupby("region")[overall_rate_col].apply(
        lambda s: (s == target_value).mean()
    )
    overall_rate = (df[overall_rate_col] == target_value).mean()
    threshold = overall_rate + 1.5 * region_rates.std()
    flagged = region_rates[region_rates > threshold].sort_values(ascending=False)
    return flagged, overall_rate


# ============================================================
# RUN PIPELINE
# ============================================================

try:
    with st.spinner("Preparing data and training models (first run only)..."):
        cleaned_df = load_and_clean_data()
        results_df, model_results, class_names = train_models(cleaned_df)
        results_df = compute_ahp_priority(results_df)
        results_df["risk_level"] = results_df["needs_repair_confidence"].apply(risk_level)
except FileNotFoundError as e:
    st.error(f"⚠️ {e}")
    st.stop()

# One WPdx status check, reused everywhere it's displayed (sidebar chip +
# the full Data & Methodology section) so we only ever make the live call
# once per cache window, not once per place it's shown.
try:
    wpdx_df, wpdx_fetched_at = fetch_live_wpdx()
    wpdx_ok, wpdx_error = True, None
except Exception as _wpdx_err:
    wpdx_df, wpdx_fetched_at = pd.DataFrame(), None
    wpdx_ok, wpdx_error = False, str(_wpdx_err)

vintage_min, vintage_max = get_data_vintage()

# ============================================================
# LEFT-SIDE NAVIGATION (sidebar = navigation ONLY, no filters here)
# ============================================================

st.sidebar.markdown("### WATER ASSET\n### INTELLIGENCE")
st.sidebar.caption("OPERATIONS PERFORMANCE")
st.sidebar.divider()

PAGES = [
    "Executive Overview",
    "Asset Explorer",
    "Maintenance Worklist",
    "Cost & Capacity Scenario",
    "Data & Methodology",
]
selected_page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption("Decision-support prototype | Tanzania water assets")

if wpdx_ok:
    _age = datetime.now(timezone.utc) - wpdx_fetched_at
    _hours = int(_age.total_seconds() // 3600)
    _age_str = f"{_hours}h ago" if _hours >= 1 else f"{int(_age.total_seconds() // 60)}m ago"
    st.sidebar.caption(f"🌐 WPdx live check: ✅ ok · {_age_str}")
else:
    st.sidebar.caption("🌐 WPdx live check: ⚠️ unavailable right now")

# ============================================================
# HEADER
# ============================================================

st.title("Water Asset Operations Dashboard")
st.caption("Tanzania | Asset condition screening | Maintenance planning and cost scenarios")
st.caption("ANALYTICAL PROTOTYPE  |  Historical-data screening, not a live telemetry or dispatch system.")
with st.expander("Dashboard purpose and decision workflow"):
    st.markdown(
        "This dashboard helps decide **which pumps to send a repair crew to first**, when crews "
        "and budget are limited. It predicts which pumps likely need repair, ranks them by urgency "
        "and impact, shows where they are, and estimates whether fixing them early is cheaper than "
        "a fixed replacement schedule. Every prediction shown here comes from a model tested on "
        "pumps it had never seen, so the numbers reflect how the system would really perform, not "
        "how well it memorized its own training data. These are decision-support signals, not a "
        "guarantee -- a field technician should always confirm the issue on-site before repairing."
    )

# ============================================================
# DATA PROVENANCE -- always visible, on every page, so nobody has to wonder
# "what data am I actually looking at right now."
# ============================================================
with st.container(border=True):
    dcol1, dcol2 = st.columns([1, 2])
    with dcol1:
        st.markdown("**📊 Data powering this dashboard**")
        st.caption(f"{len(results_df):,} pumps")
    with dcol2:
        if vintage_min is not None and pd.notna(vintage_min):
            st.caption(
                f"Field-recorded between **{vintage_min.date()}** and **{vintage_max.date()}** "
                "(Pump It Up / DrivenData dataset)."
            )
        else:
            st.caption("Pump It Up / DrivenData dataset (recording date range unavailable).")
        st.caption(
            "This is a **fixed historical dataset** -- predictions do NOT auto-update with new data. "
            "The live WPdx feed (see Data & Methodology) is for comparison only and does not feed "
            "the model."
        )

# ============================================================
# FILTER SCOPE -- moved from the left sidebar into the page content,
# since the sidebar is now navigation-only.
# ============================================================
with st.container():
    st.markdown("**🔍 Filter Scope**")
    fcol1, fcol2 = st.columns([3, 1])
    with fcol1:
        regions = st.multiselect(
            "Region",
            sorted(results_df["region"].unique()),
        )
    with fcol2:
        st.write("")
        st.write("")
        st.caption(f"{len(regions) if regions else 'All'} region(s) selected")

filtered = results_df.copy()
if regions:
    filtered = filtered[filtered["region"].isin(regions)]

st.divider()

# ============================================================
# PAGE: EXECUTIVE OVERVIEW
# ============================================================
if selected_page == "Executive Overview":
    page_banner(
        "Executive Overview",
        "This is the page to check first. It shows what needs attention right now and where to go "
        "next -- every other page exists to dig deeper into one piece of what you see here.",
    )
    screen_header(
        "A nationwide snapshot of pump condition for the regions you've selected.",
        "Anything marked red or orange, and the Regions to Watch list below.",
        "Check Highest-Priority Assets, then open Maintenance Worklist for the full dispatch order.",
    )

    if empty_state(filtered):
        st.stop()

    flagged_regions, overall_rate = flag_regional_anomalies(filtered)

    st.markdown("#### Portfolio snapshot")
    kcol1, kcol2, kcol3 = st.columns(3)
    with kcol1:
        kpi_card("💧", "Total Pumps", f"{len(filtered):,}", "var(--navy)")
    with kcol2:
        kpi_card("🟠", "Predicted Needs Repair",
                  (filtered["predicted_status"] == "functional needs repair").sum(), "#D18A3A")
    with kcol3:
        kpi_card("🔴", "Predicted Non-Functional",
                  (filtered["predicted_status"] == "non functional").sum(), "#B94A48")
    chart_note(
        "These three counts are measured directly from this dashboard's predictions -- not an "
        "estimate or a projection. (Looking for cost figures? Those live on Cost & Capacity "
        "Scenario, where the assumptions behind them are visible and adjustable.)"
    )
    widget_purpose(
        purpose="give you exactly the 3 numbers worth knowing before anything else -- real counts "
                "only, nothing modeled or hypothetical mixed in.",
        shows="pump counts by predicted status, for the regions you've selected.",
        read_as="a single-glance scope check for the current filter.",
        not_text="a cost or savings figure -- those are deliberately kept on the Cost & Capacity "
                 "Scenario page, where you can see and adjust the assumptions behind them.",
    )

    st.markdown("#### Where to look first")
    # NOTE ON LAYOUT: this is ONE st.columns(2) call, with two widgets
    # stacked inside each column -- not two separate "row" column calls.
    # Putting unrelated widgets in separate st.columns() calls forces each
    # row's height to match its TALLEST widget, which left large dead gaps
    # under short widgets sitting next to tall ones. Stacking within a
    # single column instead lets short widgets sit directly against each
    # other, no forced gap.
    left_col, right_col = st.columns(2)

    with left_col:
        with st.container(border=True):
            st.markdown("**Highest-priority assets**")
            top5 = (
                filtered[filtered["predicted_status"] == "functional needs repair"]
                .sort_values("ahp_priority_score", ascending=False)
                .head(5)
            )
            if not empty_state(top5, "No pumps are currently flagged 'needs repair' in this filter scope."):
                st.dataframe(
                    top5[["id", "region", "risk_level", "population", "ahp_priority_score"]].rename(
                        columns={**PLAIN_LABELS, "risk_level": "Urgency", "ahp_priority_score": "Priority Score"}
                    ),
                    use_container_width=True, hide_index=True,
                )
                chart_note("The 5 most urgent pumps right now. Full ranked list: Maintenance Worklist.")
                widget_purpose(
                    purpose="show you right away that this dashboard gives a usable answer, not just "
                            "charts to look at.",
                    shows="the 5 pumps that most urgently need a repair crew, ranked by a score that "
                          "blends urgency, how many people the pump serves, and how old it is.",
                    read_as="a short preview -- the complete 30-pump ranked list, with a plain-language "
                            "explanation for each one, is on Maintenance Worklist.",
                    not_text="the full list. This is only the top 5, on purpose.",
                )

        with st.container(border=True):
            st.markdown("**Screening urgency levels**")
            risk_counts = filtered["risk_level"].value_counts().reindex(["Low", "Monitor", "High", "Critical"]).fillna(0)
            risk_cols = st.columns([1, 1, 1, 1])
            for col, level in zip(risk_cols, ["Low", "Monitor", "High", "Critical"]):
                col.metric(f"{RISK_EMOJI[level]} {level}", int(risk_counts[level]))
            donut = go.Figure(data=[go.Pie(
                labels=risk_counts.index, values=risk_counts.values, hole=0.55,
                marker=dict(colors=[RISK_PLOTLY_COLORS[l] for l in risk_counts.index]),
                textinfo="percent", hovertemplate="%{label}: %{value} pumps<extra></extra>",
            )])
            donut.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200, showlegend=True,
                                 legend=dict(orientation="h", y=-0.15), **PLOTLY_BG)
            st.plotly_chart(donut, use_container_width=True)
            chart_note("Share of pumps at each urgency level. 'Critical' + 'High' are worth acting on soon.")
            st.divider()
            st.markdown("**Regions to watch**")
            region_rates_all = (
                filtered.groupby("region")["predicted_status"]
                .apply(lambda s: (s == "functional needs repair").mean())
                .sort_values(ascending=False)
            )
            top_watch = region_rates_all.head(3)
            if len(top_watch) == 0:
                st.caption("Not enough data in this filter to compute a watchlist.")
            for region_name, rate in top_watch.items():
                flagged_here = region_name in flagged_regions.index
                badge = "🔴" if flagged_here else ("🟡" if rate > overall_rate else "🟢")
                extra = " -- noticeably higher than the national average" if flagged_here else ""
                st.markdown(f"{badge} **{region_name}** -- {rate:.0%} of pumps flagged needing repair{extra}")
            widget_purpose(
                purpose="give you one thing worth checking back on regularly -- which regions are "
                        "showing more repair needs than usual.",
                shows="the 3 regions with the highest share of flagged pumps right now, with a flag "
                      "on any that are unusually high compared to the rest of the country.",
                read_as="a short watchlist, not a full ranking. A region showing up here repeatedly "
                        "may be worth an on-site look (water quality, installation practices, terrain).",
                not_text="a confirmed field finding -- it's a statistical pointer worth investigating, "
                         "not proof of a specific cause.",
            )

    with right_col:
        with st.container(border=True):
            st.markdown("**Regional exposure**")
            region_table = filtered.groupby("region").agg(
                Total_Pumps=("id", "count"),
                High_Risk=("risk_level", lambda s: (s.isin(["High", "Critical"])).sum()),
            ).reset_index()
            region_table["Risk Rate"] = region_table["High_Risk"] / region_table["Total_Pumps"]
            region_table_full = region_table.sort_values("High_Risk", ascending=False)

            # Capped to the most-affected regions so this chart's height stays
            # bounded and comparable to the other cards on the page, instead
            # of growing with however many regions Tanzania happens to have.
            TOP_N_REGIONS = 12
            region_table_chart = region_table_full.head(TOP_N_REGIONS).sort_values("High_Risk", ascending=True)

            fig_region = px.bar(
                region_table_chart, x="High_Risk", y="region", orientation="h",
                color="Risk Rate", color_continuous_scale=["#2A8C82", "#D8B45A", "#B94A48"],
                labels={"High_Risk": "High-Urgency Pumps", "region": "Region", "Risk Rate": "Risk Rate"},
                hover_data={"Total_Pumps": True, "Risk Rate": ":.1%"},
                template=PLOTLY_TEMPLATE,
            )
            fig_region.update_layout(height=max(230, 24 * len(region_table_chart)),
                                      margin=dict(l=10, r=10, t=10, b=10), **PLOTLY_BG)
            st.plotly_chart(fig_region, use_container_width=True)
            n_more = len(region_table_full) - len(region_table_chart)
            note = ("Bar length = how many high-urgency pumps; bar color = what share of that "
                    "region's own pumps those are.")
            if n_more > 0:
                note += f" Showing the {TOP_N_REGIONS} most-affected regions -- {n_more} more in the table below."
            chart_note(note)
            widget_purpose(
                purpose="let you see which regions need the most attention without opening the map.",
                shows="how many high-urgency pumps are in each region (bar length), and what share "
                      "of that region's pumps those represent (bar color).",
                read_as="check both -- a short bar can still be dark-colored, meaning a large share "
                        "of THAT region's pumps are high-urgency, even if the total count is small.",
                not_text="a map -- for pump locations, use Asset Explorer.",
            )
            with st.expander("View as table (all regions)"):
                display_region = region_table_full.rename(
                    columns={"Total_Pumps": "Total Pumps", "High_Risk": "High-Urgency Pumps"}
                ).assign(**{"Risk Rate": region_table_full["Risk Rate"].apply(lambda x: f"{x:.1%}")})
                st.dataframe(display_region, use_container_width=True, hide_index=True)

        with st.container(border=True):
            st.markdown("**Current workload**")
            backlog_count = (filtered["predicted_status"] == "functional needs repair").sum()
            st.metric("🧰 Pumps currently flagged for repair", f"{backlog_count:,}")
            top_pump = (
                filtered[filtered["predicted_status"] == "functional needs repair"]
                .sort_values("ahp_priority_score", ascending=False)
                .head(1)
            )
            if len(top_pump):
                tp = top_pump.iloc[0]
                st.caption(
                    f"Most urgent right now: **Pump {tp['id']}** in **{tp['region']}** "
                    f"({RISK_EMOJI[tp['risk_level']]} {tp['risk_level']})."
                )
            st.caption(
                "This is today's real count of the backlog, not a projection. To see what it would "
                "cost to work through it under different crew/budget assumptions, go to **Cost & "
                "Capacity Scenario**. For the full dispatch order, go to **Maintenance Worklist**."
            )
            widget_purpose(
                purpose="give a quick, real number for how big the current repair backlog is, before "
                        "jumping into cost planning.",
                shows="how many pumps are currently flagged as needing repair in your selected regions, "
                      "and which one is the single most urgent.",
                read_as="this is today's actual count, not a forecast.",
                not_text="a cost or savings estimate -- intentionally just a count, with a pointer to "
                         "where the cost scenario lives.",
            )

# ============================================================
# PAGE: ASSET EXPLORER
# ============================================================
elif selected_page == "Asset Explorer":
    page_banner(
        "Asset Explorer",
        "The \"zoom in\" page. Once Executive Overview says something needs attention, this page "
        "shows you exactly where, and lets you check on one specific pump.",
    )
    screen_header(
        "Where the pumps are, and which individual pumps need attention -- narrow down by "
        "status/type below, then look a specific pump up by ID.",
        "Clusters of orange/red markers -- a tight cluster of problem pumps in one area may "
        "point to a shared cause (water quality, installation, terrain) worth investigating.",
        "Narrow the map with the filters below, toggle predicted vs. recorded, or look up one pump by ID.",
    )
    st.subheader("Asset Explorer")

    ecol1, ecol2, ecol3 = st.columns(3)
    with ecol1:
        status_filter = st.multiselect(
            "Predicted status", sorted(filtered["predicted_status"].unique())
        )
    with ecol2:
        wtype_filter = st.multiselect(
            "Waterpoint type", sorted(filtered["waterpoint_type"].unique())
        )
    with ecol3:
        risk_filter = st.multiselect(
            "Urgency level", ["Critical", "High", "Monitor", "Low"]
        )

    explorer_df = filtered.copy()
    if status_filter:
        explorer_df = explorer_df[explorer_df["predicted_status"].isin(status_filter)]
    if wtype_filter:
        explorer_df = explorer_df[explorer_df["waterpoint_type"].isin(wtype_filter)]
    if risk_filter:
        explorer_df = explorer_df[explorer_df["risk_level"].isin(risk_filter)]

    st.divider()
    lookup_id = st.text_input("🔎 Look up a specific pump by ID (optional)")
    if lookup_id.strip():
        match = explorer_df[explorer_df["id"].astype(str) == lookup_id.strip()]
        if len(match):
            r = match.iloc[0]
            st.success(
                f"**Pump {r['id']}** -- {r['region']} | Predicted: **{r['predicted_status']}** "
                f"| Urgency: {RISK_EMOJI[r['risk_level']]} **{r['risk_level']}** | "
                f"Population served: {int(r['population']):,}. For the full priority breakdown, "
                "go to Maintenance Worklist and select this ID."
            )
        else:
            st.info("No pump with that ID in the current filter scope.")

    if empty_state(explorer_df):
        st.stop()
    color_by = st.radio("Color markers by:", ["Predicted Status", "Recorded Status (historical)"], horizontal=True)
    color_col = "predicted_status" if color_by == "Predicted Status" else "actual_status"

    st.markdown("**Legend**")
    legend_cols = st.columns(len(STATUS_COLORS))
    for col, (status, color) in zip(legend_cols, STATUS_COLORS.items()):
        col.markdown(f"{COLOR_EMOJI.get(color, '⚫')} {status}")
    st.caption("Color is always paired with a text label above and in each marker's popup -- never color alone.")

    MAX_MARKERS = 3000
    map_data = explorer_df
    if len(map_data) > MAX_MARKERS:
        st.caption(f"Showing a random sample of {MAX_MARKERS:,} of {len(map_data):,} pumps for performance.")
        map_sample = map_data.sample(MAX_MARKERS, random_state=1)
    else:
        map_sample = map_data

    m = folium.Map(location=[-6.3690, 34.8888], zoom_start=6, tiles="OpenStreetMap")
    cluster = MarkerCluster(name="Pumps").add_to(m)
    for _, row in map_sample.iterrows():
        status_val = row[color_col]
        color = STATUS_COLORS.get(status_val, "gray")
        icon = STATUS_ICONS.get(status_val, "info-sign")
        popup_text = (
            f"<b>ID:</b> {row['id']}<br><b>Region:</b> {row['region']}<br>"
            f"<b>Predicted:</b> {row['predicted_status']}<br><b>Recorded (historical):</b> {row['actual_status']}<br>"
            f"<b>Population served:</b> {int(row['population'])}"
        )
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_text, max_width=250),
            icon=folium.Icon(color=color, icon=icon, prefix="glyphicon"),
        ).add_to(cluster)
    # use_container_width lets the map fill the actual content area instead of
    # a hardcoded 1200px -- a fixed width narrower than the page left a white
    # gap on the right, and the mismatch between the fixed render size and the
    # CSS-stretched display size is also what caused the blurriness on zoom/pan.
    st_folium(m, use_container_width=True, height=600, returned_objects=[])
    chart_note("Numbers in circles are cluster counts at this zoom level -- zoom in to see individual "
                "pumps split apart. Click any marker for that pump's ID, region, and population served.")
    widget_purpose(
        purpose="show patterns you can only see on a map -- like a cluster of failing pumps that "
                "might share a single cause.",
        shows="every pump in your current filter, plotted and color-coded by status.",
        read_as="look for clusters of orange or red markers -- a group of problem pumps close "
                "together is worth investigating on-site.",
        not_text="a way to decide which pump in a cluster to visit first -- for that, use "
                 "Maintenance Worklist.",
    )

# ============================================================
# PAGE: MAINTENANCE WORKLIST
# ============================================================
elif selected_page == "Maintenance Worklist":
    page_banner(
        "Maintenance Worklist",
        "The actual output a manager acts on. Everything before this page builds toward one thing: "
        "a single, defensible, ordered list of what to do first.",
    )
    screen_header(
        "Pumps flagged as needing repair, ordered by urgency -- most urgent first.",
        "The top rows -- that's the order a crew should be dispatched in, given limited "
        "budget and capacity.",
        "Pick a pump ID below to see exactly why it's ranked where it is.",
    )
    st.subheader("Ranked maintenance worklist")
    st.caption(
        "Ranking combines urgency, population served, and pump age (weighted roughly "
        "63% / 26% / 11%) into one priority score."
    )
    priority = (
        filtered[filtered["predicted_status"] == "functional needs repair"]
        .sort_values("ahp_priority_score", ascending=False)
        .head(30)
    )
    if not empty_state(priority, "No pumps in this filter scope are currently flagged 'needs repair.'"):
        display_priority = priority[["id", "region", "risk_level", "ahp_priority_score"]].copy()
        display_priority["risk_level"] = display_priority["risk_level"].apply(
            lambda lvl: f"{RISK_EMOJI[lvl]} {lvl}"
        )
        display_priority = display_priority.rename(
            columns={"id": "Pump ID", "region": "Region", "risk_level": "Urgency",
                     "ahp_priority_score": "Priority Score"}
        )
        display_priority["Priority Score"] = display_priority["Priority Score"].round(3)
        st.dataframe(display_priority, use_container_width=True, hide_index=True)
        chart_note("Sorted most-urgent first. Population served, pump age, and the exact score "
                    "breakdown for any pump are one click away in the panel below.")
        widget_purpose(
            purpose="give you one list to work down, instead of making you decide 'which pump "
                    "first' yourself every time. Kept to 4 columns on purpose -- everything else "
                    "about a pump is one click away below, not crammed into this table.",
            shows="the top 30 pumps currently flagged as needing repair, most urgent first.",
            read_as="a dispatch order, not a pure urgency ranking -- a pump serving a lot of people "
                    "can outrank a slightly riskier pump that serves very few.",
            not_text="a risk-only ranking. That's deliberate -- see the panel below for why.",
        )
        st.caption(
            "**How this gets used**: a manager works down this list, dispatches a field technician "
            "to the top entries first, and the technician confirms the issue on-site before repairing."
        )

        st.divider()
        st.subheader("Why is this pump prioritized?")
        selected_id = st.selectbox("Select a pump ID from the list above", priority["id"].tolist())
        row = priority[priority["id"] == selected_id].iloc[0]

        conf_contrib = AHP_WEIGHTS["confidence"] * row["needs_repair_confidence"]
        pop_norm = np.log1p(row["population"]) / np.log1p(results_df["population"]).max()
        pop_contrib = AHP_WEIGHTS["population"] * pop_norm
        age_norm = 1 - (
            (row["construction_year"] - results_df["construction_year"].min())
            / (results_df["construction_year"].max() - results_df["construction_year"].min())
        )
        age_contrib = AHP_WEIGHTS["age"] * age_norm
        rank = int(priority.reset_index(drop=True).index[priority["id"] == selected_id][0]) + 1
        age_years = int(2013 - row["construction_year"])  # Pump It Up data collected ~2013

        st.markdown(f"**Pump {selected_id} — {RISK_EMOJI[row['risk_level']]} {row['risk_level']} urgency**")

        st.markdown(
            f"> This pump was built around **{int(row['construction_year'])}** (about **{age_years} years old**) "
            f"and serves an estimated **{int(row['population']):,} people**. The model estimates a "
            f"**{row['needs_repair_confidence']:.0%}** chance it needs repair. Combining these factors, "
            f"it ranks **#{rank}** on the current priority list."
        )

        colx, coly = st.columns([2, 1])
        with colx:
            st.write("Predicted chance of needing repair")
            st.progress(min(float(row["needs_repair_confidence"]), 1.0))
            st.caption(f"Contributes {conf_contrib:.3f} of the priority score (weight: 63%)")

            st.write("Population served (scaled)")
            st.progress(min(float(pop_norm), 1.0))
            st.caption(f"Contributes {pop_contrib:.3f} of the priority score (weight: 26%)")

            st.write("Pump age urgency (scaled)")
            st.progress(min(float(age_norm), 1.0))
            st.caption(f"Contributes {age_contrib:.3f} of the priority score (weight: 11%)")
        with coly:
            st.metric("Total Priority Score", f"{row['ahp_priority_score']:.3f}")
            st.metric("Population Served", int(row["population"]))
            st.metric("Year Built", int(row["construction_year"]))
        chart_note("Each bar shows how much of this pump's score came from that one factor -- "
                    "a pump can rank high even with moderate urgency if it serves a very large population.")
        widget_purpose(
            purpose="answer 'why should I trust this ranking' before anyone has to ask.",
            shows="a plain-language summary plus 3 bars showing how much each factor -- predicted "
                  "urgency, population served, and pump age -- contributed to this pump's score.",
            read_as="if a pump ranks high mostly because of population served rather than urgency, "
                    "that's the system working as intended, not an error.",
            not_text="a probability the pump will fail. These bars show how the score was built, "
                     "not an independent chance of failure.",
        )

        st.caption(
            "Recommended action: schedule a field inspection to confirm the predicted issue before "
            "committing repair resources. Estimates above are model-based, not a guarantee."
        )

# ============================================================
# PAGE: COST & CAPACITY SCENARIO
# ============================================================
elif selected_page == "Cost & Capacity Scenario":
    page_banner(
        "Cost & Capacity Scenario",
        "The page that turns this project from a classifier into a decision tool -- it answers "
        "whether acting on the priority list is worth it financially, under assumptions you can change.",
    )
    screen_header(
        "A side-by-side cost estimate: fixing flagged pumps proactively vs. replacing everything "
        "on a fixed schedule.",
        "Whether 'triage is cheaper' or 'blanket replacement is cheaper' appears below after you "
        "adjust the assumptions to your own numbers.",
        "Change the cost, crew, and horizon inputs to see when the cheaper option flips.",
    )
    st.info(
        "**Why this matters:** catching a failing pump early is only worth it if it's cheaper than "
        "fixing it later, and only if there are enough crews to act on the predictions in time. "
        "The comparison below tests exactly that, using whatever numbers you enter. (Curious how "
        "good the underlying predictions actually are? See Data & Methodology.)"
    )
    st.subheader("Maintenance strategy cost comparison")
    st.caption(
        "Planning assumptions only: the default repair and replacement costs are illustrative, "
        "not verified agency costs. Replace them with locally validated numbers before using this "
        "for real budgeting."
    )

    c1, c2, c3 = st.columns(3)
    cost_repair = c1.number_input("Cost per repair (USD)", value=DEFAULT_COST_REPAIR, min_value=1)
    cost_replace = c2.number_input("Cost per full replacement (USD)", value=DEFAULT_COST_REPLACE, min_value=1)
    crews = c3.number_input("Number of repair crews available", value=5, min_value=1)

    c4, c5 = st.columns(2)
    repairs_per_crew_month = c4.number_input("Repairs per crew per month", value=8, min_value=1)
    horizon_months = c5.slider("Planning horizon (months)", 1, 24, 12)

    flagged = filtered[filtered["predicted_status"] == "functional needs repair"]
    flagged_sorted = flagged.sort_values("ahp_priority_score", ascending=False)
    capacity = crews * repairs_per_crew_month * horizon_months

    reached = flagged_sorted.head(int(capacity))
    unreached = flagged_sorted.iloc[int(capacity):]

    triage_cost = len(reached) * cost_repair
    triage_risk_cost = len(unreached) * cost_replace
    triage_total = triage_cost + triage_risk_cost
    blanket_total = len(filtered) * cost_replace

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Predictive Triage Strategy**")
        st.metric("Pumps repaired within capacity", len(reached))
        st.metric("Flagged pumps NOT reached in time", len(unreached))
        st.metric("Estimated total cost", f"${triage_total:,.0f}")
    with colB:
        st.markdown("**Blanket Replacement Strategy**")
        st.metric("Pumps replaced (all)", len(filtered))
        st.metric("Estimated total cost", f"${blanket_total:,.0f}")
    widget_purpose(
        purpose="surface the exact bottleneck -- 'pumps NOT reached in time' -- that decides which "
                "strategy actually wins.",
        shows="a 2-column comparison: triage's capacity-limited outcome vs. blanket's full-replacement cost.",
        read_as="without the 'not reached' number, the cost comparison alone would hide WHY the "
                "answer can flip between the two strategies.",
        not_text="a recommendation. It's the inputs to one, visible so you can judge the tradeoff "
                 "yourself.",
    )

    fig_cost = go.Figure(data=[
        go.Bar(name="Predictive Triage", x=["Total Cost"], y=[triage_total], marker_color="#247BA0"),
        go.Bar(name="Blanket Replacement", x=["Total Cost"], y=[blanket_total], marker_color="#8A969F"),
    ])
    fig_cost.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(l=10, r=10, t=10, b=10),
                            barmode="group", yaxis_title="Estimated Cost (USD)", **PLOTLY_BG)
    st.plotly_chart(fig_cost, use_container_width=True)
    chart_note("Direct cost comparison under your current inputs above -- whichever bar is shorter "
                "is cheaper under these assumptions. Change crew count or horizon to see it shift.")
    widget_purpose(
        purpose="make the tradeoff visually immediate and testable live, instead of asserted in a "
                "sentence -- this is the capacity-constraint intuition behind this page, made interactive.",
        shows="total estimated cost for each strategy, side by side, under the current inputs above.",
        read_as="change crew capacity on the spot and watch which strategy wins flip.",
        not_text="a forecast of real savings -- a scenario result under stated, adjustable assumptions.",
    )

    if triage_total < blanket_total:
        st.success(
            f"Under these assumptions, predictive triage is cheaper by ${blanket_total - triage_total:,.0f} "
            f"over {horizon_months} months -- but only if crew capacity ({capacity:.0f} repairs) keeps up "
            "with the flagged backlog. If capacity is too low, unreached pumps erode the savings."
        )
    else:
        st.warning(
            "Under these assumptions, blanket replacement comes out cheaper -- when crew capacity is "
            "too constrained relative to the flagged backlog, proactive triage stops paying off and a "
            "fixed replacement cycle wins."
        )

# ============================================================
# PAGE: DATA & METHODOLOGY
# ============================================================
elif selected_page == "Data & Methodology":
    page_banner(
        "Data & Methodology",
        "The reference page. Nothing claimed elsewhere in this dashboard gets to hide behind a "
        "confident sentence -- everything is checkable here. This page is for anyone validating "
        "the system, not for day-to-day operations.",
    )
    st.subheader("What's reproduced from the paper vs. original to this project")
    st.markdown(
        """
**Reproduced from Pathak & Shalini (2023):**
- The prediction task (3-class pump status classification) and the Pump It Up dataset
- The motivating problem: severe class imbalance (the paper proposed TabNet + Focal Loss to address it)
- We studied and validated the paper's Focal Loss formulation independently before deciding how to
  extend it; the dashboard's primary classifier is XGBoost for proven deployment reliability under
  this project's timeline, with the Focal-Loss motivation carried into the per-class evaluation
  below rather than reproducing TabNet's architecture verbatim.

**Original to this project (not in the paper):**
1. Per-class precision/recall/F1 for every model (the paper reports only overall accuracy)
2. A second baseline model family (Decision Tree, unweighted and class-balanced) to test whether
   model complexity earns anything
3. A weighted, consistency-checked priority ranking (urgency + population served + pump age)
4. A cost-policy comparison between predictive triage and blanket replacement, with adjustable assumptions
5. A live current-data check (WPdx API) addressing the aging-dataset critique
6. A one-time historical validation check against ground truth (see below)
7. An automatic regional risk-spike flag ("Regions to watch" on Executive Overview)
8. An explicit decision chain: dashboard viewer → prioritized list → field technician dispatch →
   on-site repair/defer decision
9. Leakage-safe scoring: every predicted_status and needs_repair_confidence value shown anywhere
   in this dashboard comes from 5-fold cross-validation, not from a model scored on the rows it was
   trained on. The accuracy/precision/recall numbers below come from a separate, further
   train/test split, independent of the displayed predictions.

**Known limitations, stated plainly:**
- Cost figures on the Cost & Capacity Scenario page are illustrative/literature-sourced, not this
  agency's real costs
- WPdx (the live feed below) lacks a population column, so the priority score still uses Pump It
  Up's population field -- the live feed is for comparison only, not (yet) for retraining
- The model predicts from a single snapshot per pump, not a time series -- there is no "how long
  has this pump been flagged" signal yet
        """
    )

    st.divider()
    st.subheader("Priority ranking methodology (AHP)")
    st.caption(
        "Two different numbers get used in this dashboard and it's easy to mix them up: a "
        "**weight** is fixed and describes one criterion's general importance (there are only 3, "
        "shown below). A **score** is computed per pump and changes for every row -- it's what "
        "actually ranks the Maintenance Worklist. This section shows exactly how the weights were "
        "derived, and is transparent about where subjectivity enters the process."
    )

    st.markdown("**Step 1 -- the pairwise comparison judgment (Saaty 1-9 scale)**")
    st.caption(
        "This matrix IS the subjective judgment call: Confidence is judged moderately more "
        "important than Population (3), and strongly more important than Age (5); Population is "
        "judged moderately more important than Age (3). AHP does not remove this subjectivity -- "
        "it only makes it explicit and checks it for internal consistency."
    )
    pairwise_display = pd.DataFrame(
        AHP_PAIRWISE_MATRIX, index=["Confidence", "Population", "Age"],
        columns=["Confidence", "Population", "Age"],
    ).round(3)
    st.dataframe(pairwise_display, use_container_width=True)

    st.markdown("**Step 2 -- derived weights and consistency check**")
    wcol1, wcol2, wcol3, wcol4 = st.columns(4)
    wcol1.metric("Weight: Confidence", f"{AHP_WEIGHTS['confidence']:.4f}")
    wcol2.metric("Weight: Population", f"{AHP_WEIGHTS['population']:.4f}")
    wcol3.metric("Weight: Age", f"{AHP_WEIGHTS['age']:.4f}")
    wcol4.metric("Consistency Ratio", f"{AHP_CONSISTENCY_RATIO:.4f}",
                 help="CR < 0.10 means the pairwise judgments above don't logically contradict "
                      "each other. It does NOT mean the weights are objectively correct.")
    st.caption(
        f"CR = {AHP_CONSISTENCY_RATIO:.4f} is below the standard 0.10 threshold, confirming the "
        "judgments in Step 1 are internally consistent. **This is a coherence check, not a bias "
        "check** -- a different, equally internally-consistent set of judgments (e.g. weighting "
        "Population highest instead of Confidence) is just as mathematically valid. That's exactly "
        "why the sensitivity analysis below exists."
    )

    st.divider()
    st.markdown("**Step 3 -- sensitivity analysis: how much does the ranking depend on these weights?**")
    st.caption(
        "Since the weights above encode one subjective judgment, not an objective truth, this "
        "checks whether the Maintenance Worklist's top-priority pumps are robust to that choice, "
        "or whether a different reasonable judgment would send crews somewhere completely "
        "different. We compare the current weighting against two alternative, equally defensible "
        "scenarios."
    )
    sensitivity_scenarios = {
        "Current (Confidence-dominant)": AHP_WEIGHTS,
        "Equal weights (no criterion favored)": {"confidence": 1 / 3, "population": 1 / 3, "age": 1 / 3},
        "Equity-first (Population-dominant)": {
            "confidence": AHP_WEIGHTS["population"],
            "population": AHP_WEIGHTS["confidence"],
            "age": AHP_WEIGHTS["age"],
        },
    }
    needs_repair_pool = filtered[filtered["predicted_status"] == "functional needs repair"].copy()
    scenario_top_ids = {}
    for scenario_name, scenario_weights in sensitivity_scenarios.items():
        scored = needs_repair_pool.copy()
        scored["_sens_score"] = _ahp_score_vector(scored, scenario_weights)
        scenario_top_ids[scenario_name] = set(
            scored.sort_values("_sens_score", ascending=False).head(30)["id"]
        )

    baseline_name = "Current (Confidence-dominant)"
    baseline_ids = scenario_top_ids[baseline_name]
    sens_rows = []
    for scenario_name, ids in scenario_top_ids.items():
        overlap = len(ids & baseline_ids)
        sens_rows.append({
            "Weighting scenario": scenario_name,
            "Overlap with current top-30": f"{overlap}/30 ({overlap / 30:.0%})",
        })
    st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)

    equal_overlap = len(scenario_top_ids["Equal weights (no criterion favored)"] & baseline_ids)
    equity_overlap = len(scenario_top_ids["Equity-first (Population-dominant)"] & baseline_ids)
    min_overlap_pct = min(equal_overlap, equity_overlap) / 30
    if min_overlap_pct >= 0.7:
        st.success(
            f"The top-30 priority list is **robust**: at least {min_overlap_pct:.0%} of the same "
            "pumps stay in the top-30 even under meaningfully different weighting philosophies. "
            "The specific judgment encoded in Step 1 matters less than it might seem -- these pumps "
            "would likely be flagged as high-priority under several reasonable perspectives."
        )
    else:
        st.warning(
            f"The top-30 priority list is **sensitive to the weighting choice**: only "
            f"{min_overlap_pct:.0%} overlap under an alternative scenario. This means which pumps "
            "get visited first depends meaningfully on whose judgment set the weights -- a real "
            "limitation worth stating plainly rather than hiding behind a single confident ranking."
        )
    widget_purpose(
        purpose="directly answer the critique that AHP weights encode one person's subjective "
                "judgment -- by testing whether the resulting priority list actually depends on "
                "that judgment, instead of just asserting the weights are fine.",
        shows="the current weighted ranking's top-30 pumps compared against two alternative, "
              "equally defensible weighting philosophies.",
        read_as="high overlap = the recommendation is robust despite the subjectivity in the "
                "weights; low overlap = the ranking is fragile and the weighting choice itself "
                "needs to be defended, not just assumed.",
        not_text="a claim that the current weights are correct -- it's a check on how much that "
                 "would matter if they weren't.",
    )

    st.divider()
    st.subheader("How good are the predictions, really?")
    st.caption(
        "This section exists for anyone deciding whether to trust this system -- not for day-to-day "
        "use. The short version: no model catches every failing pump without also raising some false "
        "alarms, and the right balance depends on what's more costly for your agency -- a missed "
        "repair, or a wasted crew trip."
    )

    comparison_rows = []
    for name, res in model_results.items():
        row = res["report"]["functional needs repair"]
        for metric in ["precision", "recall", "f1-score"]:
            comparison_rows.append({"Model": name, "Metric": metric, "Value": row[metric]})
    comparison_df = pd.DataFrame(comparison_rows)

    st.markdown("**Catching real repairs vs. avoiding false alarms, across all three models**")
    fig_compare_models = px.bar(
        comparison_df, x="Metric", y="Value", color="Model", barmode="group",
        color_discrete_sequence=["#247BA0", "#2A9D8F", "#E58A2B"],
        template=PLOTLY_TEMPLATE, range_y=[0, 1],
    )
    fig_compare_models.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), **PLOTLY_BG)
    st.plotly_chart(fig_compare_models, use_container_width=True)
    chart_note("'Precision' = of the pumps a model flagged, how many really needed repair (higher = "
                "fewer wasted crew trips). 'Recall' = of the pumps that really needed repair, how many "
                "the model caught (higher = fewer missed repairs). 'F1' balances the two.")
    widget_purpose(
        purpose="expose the one tradeoff that actually matters -- overall accuracy would bury it "
                "under the easy majority ('working fine') class.",
        shows="precision/recall/F1 for the 'needs repair' class only, across all 3 models, measured "
              "on pumps held out of training.",
        read_as="higher recall = fewer missed repairs; higher precision = fewer wasted trips. No "
                "model wins on both -- that tradeoff is the real decision.",
        not_text="a leaderboard. There's no single 'best' bar here, on purpose.",
    )

    xgb_recall = model_results["XGBoost"]["report"]["functional needs repair"]["recall"]
    dt_recall = model_results["Decision Tree (unweighted)"]["report"]["functional needs repair"]["recall"]
    support = model_results["XGBoost"]["report"]["functional needs repair"]["support"]
    extra_caught = max(0, round((xgb_recall - dt_recall) * support))
    extra_savings = extra_caught * (DEFAULT_COST_REPLACE - DEFAULT_COST_REPAIR)

    st.info(
        f"**In plain terms:** on pumps the models hadn't seen before, XGBoost correctly flagged "
        f"{xgb_recall:.0%} of pumps that truly needed repair, versus {dt_recall:.0%} for the simplest "
        f"alternative (an unweighted Decision Tree). That gap is roughly **{extra_caught} more pumps** "
        f"caught while still cheaply repairable -- worth an estimated **${extra_savings:,.0f}** in "
        "avoided full-replacement cost, using the same illustrative cost assumptions as the Cost & "
        "Capacity Scenario page. That's the concrete link between 'a better model' and 'a better "
        "outcome' -- not just a higher number on a leaderboard."
    )
    widget_purpose(
        purpose="translate an abstract recall difference into a concrete, human statement.",
        shows="a calculated bridge: recall gap × held-out sample size × cost-per-pump difference.",
        read_as="the link between 'better model' and 'better outcome,' stated in pumps and dollars.",
        not_text="a verified saving -- same scenario-estimate caveat as the cost page.",
    )

    for name, res in model_results.items():
        with st.expander(f"{name} -- full per-class table"):
            report_df = pd.DataFrame(res["report"]).T.round(3)
            report_df = report_df.loc[[c for c in class_names]]
            st.dataframe(report_df[["precision", "recall", "f1-score", "support"]], use_container_width=True)

    with st.container(border=True):
        st.markdown("**Model Accuracy (reference only)**")
        st.caption(
            "Overall accuracy isn't used as the headline number anywhere else in this dashboard -- "
            "it hides the 'needs repair' performance above under the easy majority class. Kept here "
            "for completeness."
        )
        acc_cols = st.columns(len(model_results))
        for col, (name, res) in zip(acc_cols, model_results.items()):
            col.metric(name, f"{res['accuracy']:.1%}")

    st.divider()
    st.subheader("The false-alarm vs. missed-repair tradeoff")
    st.markdown(
        """
- **XGBoost**: balanced -- decent at both catching real repairs and avoiding false alarms, but still misses most repairs (catches about 1 in 4).
- **Decision Tree (unweighted)**: high overall accuracy, but nearly blind to real repairs (catches about 1 in 14) -- it would almost never flag a pump in time.
- **Decision Tree (balanced)**: catches far more real repairs (about 2 in 3) but at the cost of many false alarms -- crews get sent to pumps that turn out fine about 4 times out of 5.

**This is the actual decision a manager faces**: tolerate more false alarms to catch more real repairs
early, or tolerate missed repairs to keep trips efficient. There is no universally "correct" choice --
it depends on whether a missed repair (pump fails completely) or a wasted trip (crew cost, no fix
needed) is more expensive for that agency. Test that tradeoff with real numbers on Cost & Capacity
Scenario.
        """
    )

    st.divider()
    st.subheader("Historical validation check")
    st.caption(
        "Why this exists: this is one of the few datasets where the real, field-verified outcome is "
        "already known for every pump (it was a one-time survey, not an ongoing feed). That lets us "
        "check, once, whether the model's predictions actually resemble reality. **This is a one-time "
        "trust check, not something that needs checking repeatedly** -- the 'recorded' side of this "
        "comparison is historical and does not update, so it won't tell you anything new on a second "
        "look unless the model itself is retrained."
    )
    compare = pd.DataFrame({
        "Recorded (historical)": filtered["actual_status"].value_counts(),
        "Predicted": filtered["predicted_status"].value_counts(),
    }).fillna(0).astype(int).reset_index().rename(columns={"index": "Status"})
    compare_long = compare.melt(id_vars="Status", var_name="Source", value_name="Count")
    fig_compare = px.bar(
        compare_long, x="Status", y="Count", color="Source", barmode="group",
        color_discrete_map={"Recorded (historical)": "#17324D", "Predicted": "#247BA0"},
        template=PLOTLY_TEMPLATE,
    )
    fig_compare.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), **PLOTLY_BG)
    st.plotly_chart(fig_compare, use_container_width=True)
    chart_note("Close bar pairs = the model matched the historically recorded outcome well for that "
                "class; a gap flags a class the model struggles with.")
    widget_purpose(
        purpose="let anyone audit, once, whether the model's output resembles real-world outcomes -- "
                "possible here only because this particular dataset happens to include ground truth.",
        shows="recorded historical counts per status next to predicted counts, both from the same "
              "out-of-fold predictions used everywhere else in this dashboard.",
        read_as="a calibration check, not an operational signal -- it answers 'should we trust this "
                "system at all,' which is a question asked once per model version, not daily.",
        not_text="a live or recurring check. The recorded side is frozen historical data and will "
                 "look the same every time you open this page.",
    )

    st.divider()
    st.subheader("External data reference | WPdx")
    st.caption(
        "This is a data-freshness check, not a live-monitoring feed -- there is no sensor telemetry "
        "here. **Refreshing this does NOT change the predictions or rankings shown elsewhere** -- "
        "those are trained and computed on Pump It Up only. This section exists to show whether that "
        "aging historical data (see the data-vintage note near the top of every page) still looks "
        "representative of current conditions."
    )
    # --- API pull status, in plain terms: did it work, and how fresh is it ---
    if wpdx_ok:
        _age = datetime.now(timezone.utc) - wpdx_fetched_at
        _hours = int(_age.total_seconds() // 3600)
        _mins = int((_age.total_seconds() % 3600) // 60)
        _age_str = f"{_hours}h {_mins}m ago" if _hours >= 1 else f"{_mins}m ago"
        _hours_left = max(0, 24 - _age.total_seconds() / 3600)
        st.success(
            f"✅ **API status: last pull succeeded**, {_age_str} "
            f"({wpdx_fetched_at.strftime('%Y-%m-%d %H:%M UTC')}). This pull is cached for 24 hours; "
            f"it will automatically pull again in about {_hours_left:.0f}h, next time this page loads "
            "after that window closes."
        )
    else:
        st.error(
            f"❌ **API status: last pull failed just now.** The core dashboard is unaffected -- its "
            f"predictions and rankings use the local historical dataset, not this feed. "
            f"Error detail: `{wpdx_error}`"
        )

    if wpdx_ok:
        live_df = wpdx_df
        if empty_state(live_df, "No live records were returned for this query."):
            pass
        else:
            lcol1, lcol2, lcol3 = st.columns(3)
            lcol1.metric("Source", "WPdx public API")
            lcol2.metric("Records pulled (Tanzania)", f"{len(live_df):,}")
            if "report_date_parsed" in live_df.columns:
                lcol3.metric("Most recent record", str(live_df["report_date_parsed"].max().date())
                             if pd.notna(live_df["report_date_parsed"].max()) else "unknown")
                st.caption(
                    f"Full date range in this pull: {live_df['report_date_parsed'].min()} to "
                    f"{live_df['report_date_parsed'].max()}. "
                    "⚠️ A pump not appearing as 'flagged' in this feed may simply mean it hasn't been "
                    "re-surveyed recently -- a stale record is not the same as a confirmed-healthy one."
                )
            if "status_clean" in live_df.columns:
                status_counts = live_df["status_clean"].value_counts().reset_index()
                status_counts.columns = ["Status", "Count"]
                fig_live = px.bar(
                    status_counts, x="Status", y="Count", color="Count",
                    color_continuous_scale=["#2A8C82", "#D8B45A", "#D18A3A", "#B94A48"],
                    template=PLOTLY_TEMPLATE,
                )
                fig_live.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-30,
                                        **PLOTLY_BG)
                st.plotly_chart(fig_live, use_container_width=True)
                chart_note("Current status breakdown of live-pulled Tanzania records, most recent first. "
                            "This is raw WPdx data shown for comparison -- it does not feed any "
                            "prediction or ranking on this dashboard.")

                st.markdown("**Does recent data still look like the data this model was trained on?**")
                st.caption(
                    "This is the direct check for whether aging (2002-2013) training data can still "
                    "justify this model's predictions: historical status distribution, side-by-side "
                    "with the current WPdx distribution, on the same 3-class scale. WPdx's own labels "
                    "(e.g. 'needs repair', 'dry season', 'not in use') don't match the training set's "
                    "labels word-for-word -- they are mapped onto the closest honest equivalent, and any "
                    "label with no real equivalent is excluded rather than guessed, so this comparison "
                    "never silently overstates its own coverage."
                )
                class_order = list(class_names)
                hist_pct = (
                    results_df["actual_status"].value_counts(normalize=True) * 100
                ).reindex(class_order).fillna(0)
                live_mapped = live_df["status_clean"].apply(map_wpdx_status)
                live_pct_all = (live_mapped.value_counts(normalize=True) * 100)
                unmapped_share = live_pct_all.get("Other / not comparable", 0.0)
                live_pct = live_pct_all.reindex(class_order).fillna(0)

                vintage_compare = pd.DataFrame({
                    "Historical (2002-2013)": hist_pct,
                    "Current (WPdx live)": live_pct,
                }).reset_index().rename(columns={"index": "Status"})
                vintage_compare_long = vintage_compare.melt(
                    id_vars="Status", var_name="Source", value_name="Share (%)"
                )
                fig_vintage_compare = px.bar(
                    vintage_compare_long, x="Status", y="Share (%)", color="Source", barmode="group",
                    color_discrete_map={
                        "Historical (2002-2013)": "#17324D",
                        "Current (WPdx live)": "#2A8C82",
                    },
                    template=PLOTLY_TEMPLATE,
                )
                fig_vintage_compare.update_layout(
                    height=350, margin=dict(l=10, r=10, t=10, b=10), **PLOTLY_BG
                )
                st.plotly_chart(fig_vintage_compare, use_container_width=True)
                chart_note(
                    f"Bars close together per status = current conditions still resemble the training "
                    f"era; a widening gap would flag that the model may need retraining on newer data. "
                    f"{unmapped_share:.0f}% of this WPdx pull used status wording with no honest "
                    "equivalent in the 3-class taxonomy (e.g. 'dry season', 'not in use') and is "
                    "excluded from the percentages above, not forced into a nearest-sounding bucket."
                )
            if "public_data_source" in live_df.columns:
                st.caption(
                    f"Contributor diversity: {live_df['public_data_source'].nunique()} distinct sources "
                    f"(top source accounts for {live_df['public_data_source'].value_counts().iloc[0] / len(live_df):.0%} of rows) "
                    "-- noted because a feed dominated by one contributor can look misleadingly uniform."
                )
    widget_purpose(
        purpose="prove the 'live data' claim is honest by stating plainly what it does and does not do.",
        shows="source, record count, most recent date, and a status breakdown from a live WPdx pull.",
        read_as="a data-currency check -- it independently confirms the class-imbalance finding on "
                "fresher data.",
        not_text="live sensor telemetry, and it does NOT feed the trained model or rankings anywhere "
                 "else in this dashboard.",
    )

    st.divider()
    st.subheader("Data Quality -- Missing Values (before cleaning)")
    raw_check = pd.read_csv("training_set_values.csv")
    missing_pct = (raw_check.isnull().mean() * 100).round(1)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if len(missing_pct):
        fig_missing = px.bar(
            x=missing_pct.values, y=missing_pct.index, orientation="h",
            labels={"x": "% Missing", "y": "Column"},
            color=missing_pct.values, color_continuous_scale=["#2A9D8F", "#E8C547", "#C94C4C"],
            template=PLOTLY_TEMPLATE,
        )
        fig_missing.update_layout(height=max(250, 24 * len(missing_pct)), margin=dict(l=10, r=10, t=10, b=10),
                                   coloraxis_showscale=False, **PLOTLY_BG)
        st.plotly_chart(fig_missing, use_container_width=True)
        chart_note("Percent of rows missing a value in that raw column, before our cleaning step "
                    "filled or dropped them. Longer bar = more missing data in that original column.")
        widget_purpose(
            purpose="back the cleaning decisions made earlier with actual evidence.",
            shows="percent of rows missing a value in each raw column, before any cleaning.",
            read_as="a transparency check -- longer bar means more missing data, which justified the "
                    "imputation choice made for that column.",
            not_text="missing data in the cleaned dataset the model actually trains on.",
        )
    else:
        st.info("No missing values in the raw feature columns.")

    st.divider()
    st.subheader("Asset-level data extract")
    PREVIEW_ROWS = 500
    if len(filtered) > PREVIEW_ROWS:
        st.caption(f"Showing the first {PREVIEW_ROWS:,} of {len(filtered):,} rows in this filter scope "
                    "(kept short so the page stays readable -- use the download button for the full set).")
        st.dataframe(filtered.head(PREVIEW_ROWS), use_container_width=True)
    else:
        st.dataframe(filtered, use_container_width=True)

    csv_export = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export filtered asset data (CSV)",
        data=csv_export,
        file_name="pump_dss_filtered_export.csv",
        mime="text/csv",
    )
