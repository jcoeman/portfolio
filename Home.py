import streamlit as st

st.set_page_config(
    page_title="Joshua Briscoe | Actuarial & Data Science Portfolio",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Global style ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"]  {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Source Serif 4', serif;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    code, .mono {
        font-family: 'IBM Plex Mono', monospace;
    }

    :root {
        --ink: #1b2430;
        --paper: #f7f5f0;
        --slate: #445168;
        --rust: #a1462f;
        --moss: #4c6555;
        --line: #d8d3c6;
    }

    .stApp {
        background-color: var(--paper);
    }

    .hero-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.75rem;
        color: var(--rust);
        margin-bottom: 0.25rem;
    }

    .hero-title {
        font-size: 3rem;
        line-height: 1.05;
        color: var(--ink);
        margin-bottom: 0.5rem;
    }

    .hero-sub {
        font-size: 1.15rem;
        color: var(--slate);
        max-width: 640px;
        line-height: 1.5;
    }

    .divider {
        border: none;
        border-top: 1px solid var(--line);
        margin: 2rem 0;
    }

    .project-card {
        background: #fffdf9;
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 1.75rem;
        height: 100%;
        transition: border-color 0.15s ease;
    }
    .project-card:hover {
        border-color: var(--rust);
    }

    .project-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--moss);
        border: 1px solid var(--moss);
        border-radius: 100px;
        padding: 0.15rem 0.6rem;
        display: inline-block;
        margin-bottom: 0.75rem;
    }

    .project-title {
        font-family: 'Source Serif 4', serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 0.5rem;
    }

    .project-desc {
        color: var(--slate);
        font-size: 0.95rem;
        line-height: 1.55;
    }

    .status-live {
        color: var(--moss);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
    }
    .status-soon {
        color: #9a8f6f;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Hero ----------
st.markdown('<div class="hero-eyebrow">Actuarial &amp; Data Science — Portfolio</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Risk, made<br>understandable.</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-sub">
    Projects at the intersection of actuarial science and data science, 
    built with public data and open methodology.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ---------- Project cards ----------
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown(
        """
        <div class="project-card">
            <span class="project-tag">Geospatial · ML</span>
            <div class="project-title">California Wildfire Hazard Model</div>
            <div class="project-desc">
                A Random Forest hazard model over a hex grid of California,
                trained on public FPA-FOD fire records with terrain, land cover,
                and vegetation covariates.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Wildfire_Hazard_Map.py", label="Open project", icon="🗺️")

with col2:
    st.markdown(
        """
        <div class="project-card">
            <span class="project-tag">Actuarial · Credibility</span>
            <div class="project-title">Bühlmann–Straub Credibility Calculator</div>
            <div class="project-desc">
                Empirical Bayes credibility weighting for classes with
                non-uniform exposures. Edit or upload an exposure / loss-cost
                matrix and get exposure-weighted, credibility-blended loss
                cost estimates — with the full EPV/VHM decomposition shown.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_Buhlmann_Credibility_Calculator.py", label="Open project", icon="📊")

with col3:
    st.markdown(
        """
        <div class="project-card">
            <span class="project-tag">Actuarial · Simulation</span>
            <div class="project-title">Loss Simulator</div>
            <div class="project-desc">
                Monte Carlo aggregate loss model with configurable frequency
                and severity distributions. Computes ILFs, LER curves, and
                VaR/TVaR from the simulated draws, and lets you stack
                deductibles, XOL reinsurance layers, aggregate stop-loss,
                and quota share to see retained vs. ceded loss cost.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/3_Loss_Simulator.py", label="Open project", icon="🎲")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

st.markdown(
    """
    <div style="color: var(--slate); font-size: 0.85rem; font-family: 'IBM Plex Mono', monospace;">
    Built with Streamlit · Joshua Briscoe, ACAS candidate
    </div>
    """,
    unsafe_allow_html=True,
)
