import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.loss_engine import FrequencySpec, SeveritySpec, aggregate_statistics, simulate_aggregate_losses
from lib.theme import COLORS, apply_theme, back_link, question_placard, section_heading

apply_theme("Loss Distribution & Deductible Simulator", "🎲")
back_link()

st.title("Loss Distribution & Deductible Simulator")
st.markdown(
    "Claim costs vary from year to year, even when an insurer has a reliable estimate "
    "of the long-run average. This simulator generates a range of possible annual "
    "outcomes and shows how a per-claim deductible divides those costs between the "
    "insured and the insurer."
)
DIVIDER = '<hr style="border-top:1px solid #d8d3c6; margin: 1.5rem 0;">'

# =========================================================================
# 1. THE QUESTION
# =========================================================================
question_placard(
    "How does the deductible affect what the insured and insurer each pay?"
)

# =========================================================================
# 2. WHY MODEL LOSSES?
# =========================================================================
section_heading("02", "Why Model Losses?", "An average does not show the full range of outcomes")
st.markdown(
    "An insurer can estimate its average claim cost, but actual results will vary. "
    "Some years may have few or no claims; others may have several claims or one "
    "especially costly claim. Looking only at the average hides that uncertainty."
)
st.markdown(
    "A **loss distribution** describes the range of outcomes that could occur under a "
    "set of assumptions. By simulating many hypothetical years, we can examine both "
    "typical results and unusually costly ones. The **expected loss** is the average "
    "across all simulated years."
)

st.markdown(DIVIDER, unsafe_allow_html=True)

# =========================================================================
# 3. HOW THE SIMULATION WORKS
# =========================================================================
section_heading("03", "How the Simulation Works", "Build annual losses from individual claims")
st.markdown(
    """
For each hypothetical year, the model follows three steps:

1. Simulate **claim frequency:** how many claims occur.
2. Simulate **claim severity:** the dollar amount of each claim.
3. Add the claims to obtain **aggregate annual loss:** the total claim cost for that year.

Repeating these steps builds a distribution of possible annual losses. The model
keeps each claim separate because the deductible applies to every claim individually,
not once to the year's combined losses.
"""
)
st.markdown("**Available modeling choices**")
st.markdown(
    "Actuaries choose these distributions using historical data and professional judgment. "
    "The options here are illustrative and have not been fitted to a specific portfolio."
)

frequency_col, severity_col = st.columns(2, gap="large")
with frequency_col:
    st.markdown(
        """
**Frequency: How often claims occur**

- **Poisson:** Claims occur independently at a relatively stable average rate.
- **Negative Binomial:** Allows claim counts to vary more widely around the average than the Poisson model."""
    )
with severity_col:
    st.markdown(
        """
**Severity: How large claims become**

- **Lognormal:** Many moderate claims and some much larger claims.
- **Gamma:** Positive and right-skewed, generally with a more controlled tail.
- **Weibull:** Flexible enough to represent several claim-size shapes.
- **Pareto:** A heavy upper tail and greater possibility of extremely large claims.
"""
    )

with st.expander("Technical formulas and parameter notes"):
    st.markdown(
        r"""
**Collective-risk model**

If $N$ is the number of claims, each $X_i$ is an individual claim amount, and
$S$ is total annual loss, then

$$S = X_1 + X_2 + \cdots + X_N = \sum_{i=1}^{N} X_i$$

**Per-claim deductible**

For claim amount $X$ and deductible $d$:

$$\text{paid by insured}=\min(X,d)$$
$$\text{paid by insurer}=\max(X-d,0)$$

The two portions always add back to the original claim amount. The Mean & CV input
mode translates an average claim size and coefficient of variation into the raw
parameters required by the selected distribution. CV measures claim-size variability
relative to the average claim size.
"""
    )

st.markdown(DIVIDER, unsafe_allow_html=True)

# =========================================================================
# 4. EXPLORE THE SIMULATOR
# =========================================================================
section_heading("04", "Explore the Simulator", "Choose assumptions and generate possible years")
st.markdown(
    "Use the sidebar to choose how frequently claims occur and how large they may be. Coefficient of Variation (CV) adjusts the spread of a distribution."
)

with st.sidebar:
    st.header("Portfolio assumptions")
    st.caption("Choose a claim-count model and a claim-size model.")

    st.subheader("1. How often claims occur")
    freq_dist_label = st.selectbox(
        "Claim count model", ["Poisson", "Negative Binomial"], key="freq_dist_label",
        help="Poisson has variance equal to its mean. Negative Binomial permits greater claim-count variation.",
    )
    freq_dist = "poisson" if freq_dist_label == "Poisson" else "negative_binomial"
    freq_mean = st.number_input(
        "Expected claims per year", min_value=0.01, value=5.0, step=0.5,
        format="%.2f", key="freq_mean",
        help="The average annual claim count assumed by either model.",
    )
    freq_variance = None
    if freq_dist == "negative_binomial":
        freq_variance = st.number_input(
            "Annual claim-count variance", min_value=freq_mean * 1.0001,
            value=max(freq_mean * 2.0, freq_mean * 1.0001), step=0.5,
            format="%.2f", key="freq_variance",
            help="This must exceed the expected count and controls the additional year-to-year variation.",
        )

    st.subheader("2. How large claims are")
    sev_dist_label = st.selectbox(
        "Claim size model", ["Lognormal", "Pareto", "Gamma", "Weibull"],
        key="sev_dist_label",
        help="Each option makes a different assumption about the shape of positive claim amounts.",
    )
    sev_dist = sev_dist_label.lower()
    param_mode = st.segmented_control(
        "Claim-size input method", ["Mean & CV", "Raw parameters"],
        default="Mean & CV", required=True, width="stretch", key="sev_param_mode",
        help="Mean & CV is recommended for educational scenarios. Use raw parameters when working from a fitted distribution.",
    )

    sev_mean = sev_cv = None
    sev_mu = sev_sigma = sev_alpha = sev_x_m = sev_beta = sev_k = sev_lam = None
    if param_mode == "Mean & CV":
        sev_mean = st.number_input(
            "Average claim size ($)", min_value=1.0, value=10_000.0,
            step=500.0, format="%.2f", key="sev_mean",
        )
        sev_cv = st.number_input(
            "Claim-size coefficient of variation (CV)", min_value=0.05,
            value=1.5, step=0.1, format="%.2f", key="sev_cv",
            help="CV is the claim-size standard deviation divided by the average claim size. Higher values produce more variable claims.",
        )
    elif sev_dist == "lognormal":
        sev_mu = st.number_input(
            "Lognormal location (μ)", value=8.5, format="%.4f", key="sev_mu",
            help="The mean of the natural logarithm of claim size—not the average claim size in dollars.",
        )
        sev_sigma = st.number_input(
            "Lognormal spread (σ)", min_value=0.0001, value=1.0,
            format="%.4f", key="sev_sigma", help="Larger values create more right-skewed claim amounts.",
        )
    elif sev_dist == "pareto":
        sev_alpha = st.number_input(
            "Pareto shape (α)", min_value=2.0001, value=3.0,
            format="%.4f", key="sev_alpha",
            help="Lower values create a heavier upper tail. This simulator requires α above 2 so variance is finite.",
        )
        sev_x_m = st.number_input(
            "Pareto minimum claim (xₘ)", min_value=0.01, value=5_000.0,
            format="%.2f", key="sev_x_m",
        )
    elif sev_dist == "gamma":
        sev_alpha = st.number_input(
            "Gamma shape (α)", min_value=0.01, value=2.0,
            format="%.4f", key="sev_alpha_gamma",
        )
        sev_beta = st.number_input(
            "Gamma scale (β)", min_value=0.01, value=5_000.0,
            format="%.2f", key="sev_beta",
        )
    elif sev_dist == "weibull":
        sev_k = st.number_input(
            "Weibull shape (k)", min_value=0.05, value=1.2,
            format="%.4f", key="sev_k",
        )
        sev_lam = st.number_input(
            "Weibull scale (λ)", min_value=0.01, value=10_000.0,
            format="%.2f", key="sev_lam",
        )

    with st.expander("Technical simulation settings"):
        n_years = st.number_input(
            "Number of hypothetical years", min_value=1_000, max_value=200_000,
            value=25_000, step=1_000, key="n_years",
            help="More years reduce random simulation noise but do not resolve uncertainty in the assumptions.",
        )
        seed = st.number_input(
            "Random seed", min_value=0, value=42, step=1, key="seed",
            help="The same seed and assumptions reproduce the same simulation results.",
        )

FREQUENCY_GUIDANCE = {
    "Poisson": "Claim counts fluctuate around the selected average with variance equal to the mean.",
    "Negative Binomial": "Claim counts can vary more from year to year than under Poisson.",
}
SEVERITY_GUIDANCE = {
    "Lognormal": "Claim sizes are right-skewed, with many moderate claims and some much larger claims.",
    "Gamma": "Claim sizes are positive and right-skewed, generally with a more controlled tail.",
    "Weibull": "The flexible shape can represent several claim-size patterns.",
    "Pareto": "The heavy upper tail creates a greater possibility of extremely large claims.",
}

@st.cache_data(show_spinner="Generating hypothetical years...", max_entries=20)
def run_simulation(
    freq_dist, freq_mean, freq_variance, sev_dist, sev_mean, sev_cv,
    sev_mu, sev_sigma, sev_alpha, sev_x_m, sev_beta, sev_k, sev_lam,
    n_years, seed,
):
    freq_spec = FrequencySpec(dist=freq_dist, mean=freq_mean, variance=freq_variance)
    sev_spec = SeveritySpec(
        dist=sev_dist, mean=sev_mean, cv=sev_cv, mu=sev_mu, sigma=sev_sigma,
        alpha=sev_alpha, x_m=sev_x_m, beta=sev_beta, k=sev_k, lam=sev_lam,
    )
    return simulate_aggregate_losses(freq_spec, sev_spec, int(n_years), int(seed))


try:
    sim = run_simulation(
        freq_dist, freq_mean, freq_variance, sev_dist, sev_mean, sev_cv,
        sev_mu, sev_sigma, sev_alpha, sev_x_m, sev_beta, sev_k, sev_lam,
        n_years, seed,
    )
except ValueError as exc:
    st.error(f"Couldn't generate that scenario: {exc}")
    st.stop()

freq_spec_display = FrequencySpec(dist=freq_dist, mean=freq_mean, variance=freq_variance)
sev_spec_display = SeveritySpec(
    dist=sev_dist, mean=sev_mean, cv=sev_cv, mu=sev_mu, sigma=sev_sigma,
    alpha=sev_alpha, x_m=sev_x_m, beta=sev_beta, k=sev_k, lam=sev_lam,
)


st.subheader("Range of total annual claim costs")
st.markdown(
    "Each bar counts hypothetical years with a similar **aggregate annual loss**, or "
    "total claim cost before any deductible. A longer right tail means that uncommon "
    "years can be much more expensive than the average."
)

aggregate_stats = aggregate_statistics(sim.aggregate_losses)
annual_loss_p95 = float(np.quantile(sim.aggregate_losses, 0.95))
annual_loss_p99 = float(np.quantile(sim.aggregate_losses, 0.99))
histogram = go.Figure()
histogram.add_trace(go.Histogram(
    x=sim.aggregate_losses, nbinsx=80, marker_color=COLORS["moss"], opacity=0.78,
    name="Annual claim cost",
    hovertemplate="Annual loss: $%{x:,.0f}<br>Hypothetical years: %{y:,.0f}<extra></extra>",
))
histogram.add_vline(
    x=aggregate_stats["mean"], line_dash="dot", line_color=COLORS["rust"],
    annotation_text="Expected annual loss", annotation_position="top",
)
histogram.update_layout(
    height=390, template="plotly_white", bargap=0.02, showlegend=False,
    xaxis_title="Total annual claim cost ($)", yaxis_title="Hypothetical years",
    margin=dict(t=35, b=10, l=10, r=10), font=dict(family="IBM Plex Sans"),
)
st.plotly_chart(histogram, width="stretch")

st.caption(
    f"Current run: **{freq_dist_label}** with an expected "
    f"{freq_spec_display.analytic_mean():,.2f} claims per year · **{sev_dist_label}** "
    f"with an expected ${sev_spec_display.analytic_mean():,.0f} per claim · "
    f"{int(n_years):,} hypothetical years · {len(sim.occurrence_losses):,} claims generated"
)

with st.container(horizontal=True):
    st.metric("Expected annual loss", f"${aggregate_stats['mean']:,.0f}", border=True)
    st.metric("Annual loss standard deviation", f"${aggregate_stats['std']:,.0f}", border=True)
    st.metric("95th percentile annual loss", f"${annual_loss_p95:,.0f}", border=True)
    st.metric("99th percentile annual loss", f"${annual_loss_p99:,.0f}", border=True)


st.caption(
    "A percentile is a threshold: 95% or 99% of the modeled annual outcomes fall at "
    "or below the corresponding value. These estimates depend on the assumptions "
    "and the number of hypothetical years."
)

st.markdown(DIVIDER, unsafe_allow_html=True)

# =========================================================================
# 5. DEDUCTIBLE ANALYSIS
# =========================================================================
section_heading("05", "How a Deductible Changes Who Pays", "Split each claim between insured and insurer")
st.markdown(
    r"""A **deductible** is the portion of each covered claim the insured pays before
    insurance contributes. With a \$1,000 deductible, for example, the insured pays
    the first \$1,000 of a claim and the insurer pays the remaining covered amount."""
)
occurrence_losses = sim.occurrence_losses
occurrence_year = sim.occurrence_year
n_years_int = int(sim.n_years)
suggested_curve_max = (
    max(1_000.0, float(round(np.quantile(occurrence_losses, 0.95), -2)))
    if len(occurrence_losses) else 10_000.0
)
max_deductible = st.number_input(
    "Maximum deductible shown", min_value=100.0, value=suggested_curve_max,
    step=max(100.0, float(round(suggested_curve_max / 20, -2))),
    format="%.0f", key="deductible_curve_max",
    help="The suggested value is based on the 95th percentile of simulated individual claim amounts.",
)
selected_deductible = st.slider(
    "Deductible applied separately to each claim ($)", min_value=0.0,
    max_value=float(max_deductible), value=min(1_000.0, float(max_deductible)),
    step=max(100.0, float(round(max_deductible / 100, -2))),
    key="selected_deductible",
    help="Move the slider to see how a per-claim deductible changes expected payments.",
)

insured_occurrence = np.minimum(occurrence_losses, selected_deductible)
insurer_occurrence = np.maximum(occurrence_losses - selected_deductible, 0.0)
if not np.allclose(insured_occurrence + insurer_occurrence, occurrence_losses):
    st.error("The deductible payment split did not reconcile to the original claims.")
    st.stop()

insured_annual = (
    np.bincount(occurrence_year, weights=insured_occurrence, minlength=n_years_int)
    if len(occurrence_losses) else np.zeros(n_years_int)
)
insurer_annual = (
    np.bincount(occurrence_year, weights=insurer_occurrence, minlength=n_years_int)
    if len(occurrence_losses) else np.zeros(n_years_int)
)
insurer_payment_counts = (
    np.bincount(
        occurrence_year, weights=(insurer_occurrence > 0).astype(float),
        minlength=n_years_int,
    ) if len(occurrence_losses) else np.zeros(n_years_int)
)

expected_total = float(np.mean(sim.aggregate_losses))
expected_insured = float(np.mean(insured_annual))
expected_insurer = float(np.mean(insurer_annual))
insured_share = expected_insured / expected_total if expected_total else 0.0
insurer_share = expected_insurer / expected_total if expected_total else 0.0
expected_claims_with_insurer_payment = float(np.mean(insurer_payment_counts))
if not np.isclose(expected_insured + expected_insurer, expected_total):
    st.error("Expected insured and insurer payments did not reconcile to expected total claim cost.")
    st.stop()

with st.container(horizontal=True):
    st.metric("Total claim cost", f"${expected_total:,.0f}", border=True)
    st.metric(
        "Paid by the insured", f"${expected_insured:,.0f}", border=True,
    )
    st.metric(
        "Paid by the insurer", f"${expected_insurer:,.0f}", border=True,
    )
    st.metric(
        "Claims with an insurer payment",
        f"{expected_claims_with_insurer_payment:,.2f} per year", border=True,
    )
st.caption(
    "The insured payment shown here is claim cost paid through the deductible. It is "
    "not the insured's premium or total cost of insurance."
)

payment_split = go.Figure()
payment_split.add_trace(go.Bar(
    y=["Expected annual claim cost"], x=[expected_insured], orientation="h",
    name="Paid by the insured", marker_color=COLORS["rust"],
    hovertemplate="Paid by the insured: $%{x:,.0f}<extra></extra>",
))
payment_split.add_trace(go.Bar(
    y=["Expected annual claim cost"], x=[expected_insurer], orientation="h",
    name="Paid by the insurer", marker_color=COLORS["moss"],
    hovertemplate="Paid by the insurer: $%{x:,.0f}<extra></extra>",
))
payment_split.update_layout(
    barmode="stack", height=210, template="plotly_white",
    xaxis_title="Expected annual claim cost ($)", yaxis_title=None,
    legend=dict(orientation="h", y=1.28), margin=dict(t=45, b=10, l=10, r=10),
    font=dict(family="IBM Plex Sans"),
)
st.plotly_chart(payment_split, width="stretch")

deductible_levels = np.linspace(0.0, float(max_deductible), 41)
curve_rows = []
for deductible in deductible_levels:
    paid_by_insured = np.minimum(occurrence_losses, deductible)
    paid_by_insurer = np.maximum(occurrence_losses - deductible, 0.0)
    curve_rows.append({
        "Deductible": deductible,
        "Paid by the insured": float(paid_by_insured.sum() / n_years_int),
        "Paid by the insurer": float(paid_by_insurer.sum() / n_years_int),
        "Total claim cost": expected_total,
    })

deductible_curve = pd.DataFrame(curve_rows)
curve_insured = deductible_curve["Paid by the insured"].to_numpy()
curve_insurer = deductible_curve["Paid by the insurer"].to_numpy()
if np.any(np.diff(curve_insured) < -1e-8):
    st.error("Expected insured payments should not decrease as the deductible rises.")
    st.stop()
if np.any(np.diff(curve_insurer) > 1e-8):
    st.error("Expected insurer payments should not increase as the deductible rises.")
    st.stop()
if not np.allclose(curve_insured + curve_insurer, expected_total):
    st.error("The deductible curve does not reconcile to expected total claim cost.")
    st.stop()

st.subheader("Expected annual payments across deductible choices")
st.markdown(
    "Each deductible is applied to the same set of simulated claims. The total expected "
    "claim cost remains the same; what changes is how much each party pays."
)
deductible_chart = go.Figure()
for column, color in [
    ("Paid by the insured", COLORS["rust"]),
    ("Paid by the insurer", COLORS["moss"]),
    ("Total claim cost", COLORS["slate"]),
]:
    deductible_chart.add_trace(go.Scatter(
        x=deductible_curve["Deductible"], y=deductible_curve[column], mode="lines",
        name=column,
        line=dict(color=color, width=2.5, dash="dot" if column == "Total claim cost" else "solid"),
        hovertemplate=f"Deductible: $%{{x:,.0f}}<br>{column}: $%{{y:,.0f}}<extra></extra>",
    ))
deductible_chart.add_vline(
    x=selected_deductible, line_dash="dash", line_color=COLORS["slate"],
    annotation_text="Selected deductible", annotation_position="top",
)
deductible_chart.update_layout(
    height=420, template="plotly_white",
    xaxis_title="Deductible applied to each claim ($)",
    yaxis_title="Expected annual claim cost ($)",
    legend=dict(orientation="h", y=1.12), hovermode="x unified",
    margin=dict(t=50, b=10, l=10, r=10), font=dict(family="IBM Plex Sans"),
)
st.plotly_chart(deductible_chart, width="stretch")
st.markdown(
    "As the deductible rises, the insured pays more of the expected claim cost and the "
    "insurer pays less. The reduction in the insurer's expected payments will not usually "
    "produce an equal reduction in premium because premiums also account for expenses, "
    "risk, profit, and other factors."
)