import os

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Wildfire Hazard Map | Portfolio", page_icon="🗺️", layout="wide")

MAP_PATH = "data/ca_risk_map_full_blue_red.html"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@700&family=IBM+Plex+Sans&family=IBM+Plex+Mono&display=swap');
    h1, h2, h3 { font-family: 'Source Serif 4', serif; }
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #f7f5f0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.page_link("Home.py", label="← Back to portfolio", icon="🏠")
st.title("California Wildfire Hazard Model")

st.markdown(
    """
    A Random Forest hazard classifier scored across an H3 resolution-7 hex grid
    of California, trained on public FPA-FOD fire records with SRTM terrain,
    NLCD land cover, and MODIS NDVI covariates. Validation uses spatially-blocked
    train/test splits so nearby hexes never leak across the split, and SHAP
    values explain which covariates drive the predicted hazard at each hex.
    """
)



if os.path.exists(MAP_PATH):
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        map_html = f.read()
    components.html(map_html, height=700, scrolling=False)
else:
    st.info(
        "This page is a placeholder. Once the public-data rebuild of the hex-grid "
        "Random Forest model is exported, drop the Folium HTML or the model "
        "artifacts in here and this page will render the interactive map.",
        icon="🚧",
    )

# Drop this in pages/1_Wildfire_Hazard_Map.py, after the model description
# paragraph and before (or after) the embedded map — your call on ordering.

st.markdown("---")
st.subheader("What the model found")

st.markdown(
    """
Running SHAP across the full grid of at-risk hexes surfaced two distinct,
roughly equal-sized risk archetypes — not one dominant story, but two
genuinely different pathways to high wildfire hazard. The beeswarm below
shows every feature's SHAP contribution across a sample of hexes: `pct_developed`
and `ndvi` clearly separate out as the two strongest drivers, each with a wide
spread of both positive and negative impact.
"""
)

st.image("data/shap_beeswarm_v2.png", use_container_width=True)
st.caption(
    "SHAP value distribution by feature. Red = high feature value, blue = low. "
    "Position left/right shows whether that value pushed the predicted risk "
    "down or up for that hex."
)

st.write("")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
        <div class="metric-box">
        <div style="font-size:0.75rem;color:#445168;text-transform:uppercase;letter-spacing:0.06em;">Archetype 1 · ~38% of high-risk hexes</div>
        <div style="font-family:'Source Serif 4',serif;font-size:1.15rem;font-weight:700;margin:0.4rem 0;">Development-adjacent ignition risk</div>
        <div style="color:#445168;font-size:0.92rem;line-height:1.5;">
        Percent-developed land is the single strongest driver. These hexes sit near human
        activity, where most California wildfires actually start — human-caused fire
        categories show substantially higher average development percentage than
        lightning-caused fires in the underlying data, which is consistent with this
        pattern rather than coincidental to it.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="metric-box">
        <div style="font-size:0.75rem;color:#445168;text-transform:uppercase;letter-spacing:0.06em;">Archetype 2 · ~32% of high-risk hexes</div>
        <div style="font-family:'Source Serif 4',serif;font-size:1.15rem;font-weight:700;margin:0.4rem 0;">Vegetation-dryness risk</div>
        <div style="color:#445168;font-size:0.92rem;line-height:1.5;">
        Low NDVI (a satellite-based vegetation greenness index) is the dominant driver
        instead — dry, sparse vegetation independent of how close a hex is to development.
        The relationship holds in both directions: healthy, dense vegetation (high NDVI)
        measurably pulls risk scores down, confirming the effect isn't an artifact.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
The remaining ~30% of high-risk hexes are explained by a mix of grassland
cover, elevation, and forest characteristics — meaning the model isn't
leaning on a single dominant feature everywhere, it's genuinely picking up
different local risk profiles across the state.
"""
)

st.write("")

m1, m2, m3 = st.columns(3)
m1.markdown(
    '<div class="metric-box"><div style="font-size:0.75rem;color:#445168;">ROC-AUC</div>'
    '<div style="font-size:1.4rem;font-weight:600;">0.787</div>'
    '<div style="font-size:0.75rem;color:#445168;">with NDVI added (from 0.779)</div></div>',
    unsafe_allow_html=True,
)
m2.markdown(
    '<div class="metric-box"><div style="font-size:0.75rem;color:#445168;">OPERATING THRESHOLD</div>'
    '<div style="font-size:1.4rem;font-weight:600;">0.25</div>'
    '<div style="font-size:0.75rem;color:#445168;">92% recall, chosen for underwriting use</div></div>',
    unsafe_allow_html=True,
)
m3.markdown(
    '<div class="metric-box"><div style="font-size:0.75rem;color:#445168;">GRID COVERAGE</div>'
    '<div style="font-size:1.4rem;font-weight:600;">50.3%</div>'
    '<div style="font-size:0.75rem;color:#445168;">of hexes flagged above threshold</div></div>',
    unsafe_allow_html=True,
)

st.write("")
st.subheader("Threshold selection and validation curves")
st.markdown(
    "Three views of the same underlying tradeoff, used together to pick the "
    "0.25 operating threshold above."
)

st.markdown("**Precision and recall vs. threshold** — where the 0.25 cutoff sits")
st.image("data/precision_recall_tradeoff.png", use_container_width=True)
st.caption(
    "At low thresholds, recall stays high while precision is low — the model "
    "flags most true positives but also a lot of false ones. Recall was "
    "weighted more heavily than precision here since missing a genuinely "
    "high-risk hex is the worse failure mode for an underwriting screening tool."
)

curve_col1, curve_col2 = st.columns(2, gap="large")
with curve_col1:
    st.markdown("**Precision-recall curve**")
    st.image("data/precision_recall_curve.png", use_container_width=True)
    st.caption("PR-AUC of 0.393 against a 0.155 no-skill baseline — roughly 2.5x better than chance.")
with curve_col2:
    st.markdown("**ROC curve**")
    st.image("data/roc_curve.png", use_container_width=True)

with st.expander("Validation methodology — why this isn't overfit"):
    st.markdown(
        """
A published benchmark model on similar data reported near-perfect accuracy
(~0.99 AUC) under a naive train/test split — but that number collapsed to
0.52–0.66 once evaluated with spatial cross-validation instead. Nearby hexes
share almost all their environmental covariates, so a naive split leaks
information between train and test sets and produces an inflated, misleading
accuracy figure.

This model was validated with a **spatially-blocked hex-level split** from
the start — no hex's neighbors are allowed to appear on the opposite side of
the train/test boundary — specifically to avoid that trap. The 0.787 AUC
reported above is the honest number under that stricter validation, not the
inflated one a naive split would have produced.
        """
    )