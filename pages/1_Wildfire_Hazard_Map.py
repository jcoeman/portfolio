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
    A Random Forest hazard classifier scored across an **H3 resolution-7 hex grid**
    of California, trained on public FPA-FOD fire records with SRTM terrain,
    NLCD land cover, and MODIS NDVI covariates. Validation uses **spatially-blocked**
    train/test splits so nearby hexes never leak across the split, and **SHAP**
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
