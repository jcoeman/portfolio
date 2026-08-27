import streamlit as st

from lib.theme import GLOBAL_CSS

st.set_page_config(
    page_title="Joshua Briscoe | Actuarial & Data Science Portfolio",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown('<div class="hero-eyebrow">Actuarial Analytics Portfolio</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Exploring<br>Insurance Risk</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-sub">
   How do actuaries and underwriters make decisions when the outcome is uncertain? This interactive portfolio uses visual explanations and practical examples to explore wildfire risk, loss-cost estimation, insurance limits, and reinsurance.
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
            <span class="project-tag">Wildfire · Underwriting</span>
            <div class="project-title">California Wildfire Risk</div>
            <div class="project-desc">
                Using public data on historical fires, terrain, land cover,
                development, and satellite vegetation, this project estimates
                wildfire hazard across California. Explore how an underwriter
                might use this information to identify properties that warrant
                closer review before quoting a homeowners policy.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(
        "pages/1_Wildfire_Hazard_Map.py",
        label="Explore wildfire risk",
        icon="🗺️",
    )

with col2:
    st.markdown(
        """
        <div class="project-card">
            <span class="project-tag">Insurance Pricing · Credibility</span>
            <div class="project-title">Estimating Loss Costs</div>
            <div class="project-desc">
                An insurer's recent experience can provide valuable information,
                but limited data may produce unstable results. Explore how
                actuaries combine a group's own loss experience with information
                from a broader portfolio to produce more reliable estimates for
                pricing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(
        "pages/2_Buhlmann_Credibility_Calculator.py",
        label="Explore credibility",
        icon="📊",
    )

with col3:
    st.markdown(
        """
        <div class="project-card">
            <span class="project-tag">Claims · Coverage · Reinsurance</span>
            <div class="project-title">Understanding Insurance Losses</div>
            <div class="project-desc">
                Insurers do not know how many claims will occur or how costly
                they will be. Explore simulated years of claims to see how
                deductibles, policy limits, and reinsurance change the losses
                paid by the insurer, policyholders, and reinsurers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(
        "pages/3_Loss_Simulator.py",
        label="Explore claim uncertainty",
        icon="🎲",
    )

st.markdown('<hr class="divider">', unsafe_allow_html=True)

st.markdown(
    """
    <div style="color: var(--slate); font-size: 0.85rem; font-family: 'IBM Plex Mono', monospace;">
    Built with Streamlit · Joshua Briscoe, ACAS candidate
    </div>
    """,
    unsafe_allow_html=True,
)
