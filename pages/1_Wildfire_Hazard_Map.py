import os

import streamlit as st

from lib.theme import (
    apply_theme,
    back_link,
    metric_box_html,
    question_placard,
    section_heading,
    so_what_box,
)

MAP_PATH = "data/ca_risk_map_full_blue_red.html"

apply_theme("Wildfire Hazard Map", "🗺️")
back_link()

# =========================================================================
# INTRODUCTION
# =========================================================================
st.title("Mapping Wildfire Hazard Across California")

st.markdown(
    """
This project combines historical fire records with information about terrain,
vegetation, land cover, and nearby development to examine geographic patterns
associated with wildfire occurrence across California. A model trained on historical wildfire locations assigns each geographic area
a relative hazard score. Additional techniques help explain which local
characteristics raised or lowered that score.
    """
)

question_placard(
    "Where is increased fire risk and, and which local "
    "features contribute to higher risk?"
)

# =========================================================================
# 1. RESULTS AT A GLANCE
# =========================================================================
section_heading(
    "01",
    "Results at a Glance",
    "A summary of the model's performance",
)

m1, m2, m3 = st.columns(3)

m1.markdown(
    metric_box_html(
        "Model discrimination",
        "0.787 ROC-AUC",
        "using geographic validation",
    ),
    unsafe_allow_html=True,
)

m2.markdown(
    metric_box_html(
        "Detection rate",
        "92% recall",
        "at the selected cutoff",
    ),
    unsafe_allow_html=True,
)

m3.markdown(
    metric_box_html(
        "Screening coverage",
        "50.3%",
        "of evaluated hexagons flagged",
    ),
    unsafe_allow_html=True,
)

# =========================================================================
# 2. INTERACTIVE MAP
# =========================================================================
section_heading(
    "02",
    "Explore the Map",
    "Compare modeled wildfire hazard across California",
)

st.markdown(
    """
The map divides California into hexagonal geographic areas. Each area was
evaluated using the same environmental and development-related information. Select a hexagon to view its modeled hazard score and learn which local
characteristics contributed to the prediction.
    """
)

if os.path.exists(MAP_PATH):
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        map_html = f.read()

    st.iframe(map_html, height=700)
else:
    st.info(
        "The interactive map is currently unavailable because the exported "
        "map file could not be found.",
        icon="🚧",
    )

st.markdown("---")

# =========================================================================
# 3. KEY FINDINGS
# =========================================================================
section_heading(
    "03",
    "What Influenced the Scores?",
    "Similar hazard scores can reflect different local conditions",
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
<div class="metric-box">
<div style="font-size:0.75rem;color:#445168;text-transform:uppercase;letter-spacing:0.06em;">Pattern 1 · Approximately 38% of higher-scoring hexagons</div>
<div style="font-family:'Source Serif 4',serif;font-size:1.15rem;font-weight:700;margin:0.4rem 0;">Proximity to developed areas</div>
<div style="color:#445168;font-size:0.92rem;line-height:1.5;">The amount of developed land had the greatest influence on the model's predictions for these locations. This indicates that the model found a meaningful association between wildfire occurrence and areas where human development meets surrounding landscapes. Potentially humans could be a significant contributor to wildfires?</div>
</div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
<div class="metric-box">
<div style="font-size:0.75rem;color:#445168;text-transform:uppercase;letter-spacing:0.06em;">Pattern 2 · Approximately 32% of higher-scoring hexagons</div>
<div style="font-family:'Source Serif 4',serif;font-size:1.15rem;font-weight:700;margin:0.4rem 0;">Drier or sparser vegetation</div>
<div style="color:#445168;font-size:0.92rem;line-height:1.5;">Vegetation conditions had the greatest influence on the model's predictions for these locations. Areas with less green, less actively growing vegetation generally received higher hazard scores, while greener areas tended to receive lower scores. This was measured using satellite imagery and a vegetation index called NDVI. Although lower values can indicate dry or sparse vegetation, NDVI does not measure vegetation moisture directly.</div>
</div>
        """,
        unsafe_allow_html=True,
    )
st.markdown(
    """
For the remaining higher-scoring locations, no single characteristic dominated
the prediction. Their scores reflected combinations of grassland cover,
elevation, forest characteristics, development, and vegetation conditions. This suggests that wildfire hazard does not follow one uniform geographic
pattern across the state.
    """
)

st.markdown("---")

# =========================================================================
# 4. SUPPORTING GRAPHS
# =========================================================================
section_heading(
    "04",
    "Supporting Graphs",
    "A closer look at model interpretation and validation",
)

st.markdown(
    """
These graphs provide additional detail about what influenced the predictions,
how the screening cutoff was selected, and how well the model separates
locations with different observed outcomes.
    """
)

graph_col1, graph_col2 = st.columns(2, gap="large")

with graph_col1:
    st.markdown("### What influenced the predictions?")

    st.image(
        "data/shap_beeswarm_v2.png",
        width="stretch",
    )

    st.markdown(
        """
This graph shows how each characteristic influenced predictions across a sample
of locations. Points to the right increased the modeled hazard score, while
points to the left decreased it. The color indicates whether the underlying
characteristic had a relatively high or low value.

The graph uses SHAP feature attribution, a method for explaining how a model
arrived at its predictions. It identifies model influences, not causal effects.
        """
    )

with graph_col2:
    st.markdown("### How was the cutoff selected?")

    st.image(
        "data/precision_recall_tradeoff.png",
        width="stretch",
    )

    st.markdown(
        """
This graph shows how the model's behavior changes as the screening cutoff moves.
A lower cutoff identifies more positive cases but also produces more false
alerts. A higher cutoff reduces false alerts but misses more positive cases.

The selected cutoff of 0.25 favors detection because the project treats a
missed hazardous location as more consequential than an additional review.
        """
    )

graph_col3, graph_col4 = st.columns(2, gap="large")

with graph_col3:
    st.markdown("### How reliable are positive flags?")

    st.image(
        "data/precision_recall_curve.png",
        width="stretch",
    )

    st.markdown(
        """
This curve summarizes the tradeoff between detecting positive cases and ensuring
that flagged locations are truly positive. The model achieved a
precision-recall area of 0.393, compared with a positive-case prevalence of
0.155 in the validation data.

This indicates that the model provides useful separation, although a positive
flag should still be interpreted as a reason for further review rather than a
confirmed outcome.
        """
    )

with graph_col4:
    st.markdown("### How well does the model separate locations?")

    st.image(
        "data/roc_curve.png",
        width="stretch",
    )

    st.markdown(
        """
The ROC curve measures how well the model ranks positive cases above negative
cases across every possible cutoff. A model with no useful separation would
follow the diagonal reference line.

The model's ROC-AUC of 0.787 indicates meaningful, but imperfect,
discrimination on geographically separated validation data.
        """
    )

st.markdown("---")

# =========================================================================
# 5. METHODOLOGY
# =========================================================================
section_heading(
    "05",
    "How the Model Was Built",
    "Data, modeling, and geographic validation",
)

st.markdown(
    """
A Random Forest model was trained using public historical fire records together
with terrain, land-cover, vegetation, and development-related information.
California was divided into resolution-7 H3 hexagons so that data from several
sources could be evaluated on a common geographic grid.

Random Forest was selected because it can identify nonlinear relationships and
interactions among geographic characteristics without requiring every
relationship to follow a predetermined shape.
    """
)

with st.expander("Why geographic validation matters"):
    st.markdown(
        """
Nearby locations often share similar terrain, vegetation, and development
patterns. If neighboring hexagons were randomly divided between the training
and testing data, the model could appear successful partly because it had
already observed nearly identical areas.

To reduce this problem, the model was evaluated using geographically separated
training and testing areas. This creates a more demanding and realistic test of
whether the model can recognize patterns in locations it did not observe during
training.

The reported ROC-AUC of 0.787 reflects performance under this stricter
validation approach.
        """
    )

with st.expander("What the model does—and does not—measure"):
    st.markdown(
        """
The model estimates associations between geographic characteristics and
historical wildfire occurrence. Its score is a relative screening measure, not
a forecast that a specific location will experience a wildfire within a stated
period.

The model also does not estimate property damage, insured loss, vulnerability,
or the financial consequences of a fire. Those questions would require
additional information about structures, insurance values, mitigation,
coverage, and potential fire severity.
        """
    )

st.markdown("---")
