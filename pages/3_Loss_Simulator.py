import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.loss_engine import (
    CoverageLayer,
    FrequencySpec,
    SeveritySpec,
    aggregate_statistics,
    apply_alae_ulae,
    apply_coverage_layers,
    credibility_weighted_loss_cost,
    ilf_curve,
    ler_curve,
    simulate_aggregate_losses,
    var_tvar_table,
)

st.set_page_config(page_title="Loss Simulator | Portfolio", page_icon="🎲", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Source Serif 4', serif; font-weight: 700; }
    .stApp { background-color: #f7f5f0; }
    code, .mono { font-family: 'IBM Plex Mono', monospace; }

    .metric-box {
        background: #fffdf9;
        border: 1px solid #d8d3c6;
        border-radius: 4px;
        padding: 1rem 1.25rem;
    }
    .formula-box {
        background: #fffdf9;
        border-left: 3px solid #a1462f;
        padding: 0.9rem 1.1rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #1b2430;
        margin: 0.5rem 0 1rem 0;
    }
    .note-box {
        background: #fbf3ec;
        border-left: 3px solid #a1462f;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #445168;
        margin: 0.75rem 0;
        border-radius: 0 4px 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.page_link("Home.py", label="← Back to portfolio", icon="🏠")

st.title("Loss Simulator")
st.markdown(
    "A Monte Carlo collective-risk model: draw an annual claim count from a "
    "**frequency** distribution, draw that many i.i.d. claim sizes from a "
    "**severity** distribution, sum to an aggregate loss per year, and repeat "
    "for thousands of simulated years. The same simulated draws drive the "
    "Increased Limits Factors, Loss Elimination Ratios, VaR/TVaR, and "
    "coverage-layer results below — nothing is re-simulated between sections."
)

DIVIDER = '<hr style="border-top:1px solid #d8d3c6; margin: 1.5rem 0;">'

# =============================================================================
# Sidebar — distribution pickers, simulation size, seed
# =============================================================================
with st.sidebar:
    st.header("Simulation setup")

    st.subheader("Frequency")
    freq_dist_label = st.selectbox("Frequency distribution", ["Poisson", "Negative Binomial"], key="freq_dist_label")
    freq_dist = "poisson" if freq_dist_label == "Poisson" else "negative_binomial"

    freq_mean = st.number_input(
        "Expected annual claim count", min_value=0.01, value=5.0, step=0.5, format="%.2f", key="freq_mean",
        help="λ for Poisson; mean claim count for Negative Binomial.",
    )

    freq_variance = None
    if freq_dist == "negative_binomial":
        freq_variance = st.number_input(
            "Variance of annual claim count",
            min_value=freq_mean * 1.0001,
            value=max(freq_mean * 2.0, freq_mean * 1.0001),
            step=0.5, format="%.2f", key="freq_variance",
            help="Must exceed the mean — that overdispersion (variance > mean) is what "
                 "distinguishes Negative Binomial from Poisson (variance == mean).",
        )

    st.subheader("Severity")
    sev_dist_label = st.selectbox("Severity distribution", ["Lognormal", "Pareto", "Gamma", "Weibull"], key="sev_dist_label")
    sev_dist = sev_dist_label.lower()

    param_mode = st.radio("Parameterize by", ["Mean & CV", "Raw parameters"], horizontal=True, key="sev_param_mode")

    sev_mean = sev_cv = None
    sev_mu = sev_sigma = sev_alpha = sev_x_m = sev_beta = sev_k = sev_lam = None

    if param_mode == "Mean & CV":
        sev_mean = st.number_input("Mean severity ($)", min_value=1.0, value=10_000.0, step=500.0, format="%.2f", key="sev_mean")
        sev_cv = st.number_input(
            "Coefficient of variation (CV)", min_value=0.05, value=1.5, step=0.1, format="%.2f", key="sev_cv",
            help="CV = std / mean. Higher CV → heavier tail. Pareto requires CV such that "
                 "shape α > 2 is solvable (any CV > 0 works). Weibull CV is solved numerically "
                 "and is valid roughly in (0.03, 8).",
        )
    else:
        if sev_dist == "lognormal":
            sev_mu = st.number_input("μ (log-scale mean)", value=8.5, format="%.4f", key="sev_mu")
            sev_sigma = st.number_input("σ (log-scale std)", min_value=0.0001, value=1.0, format="%.4f", key="sev_sigma")
        elif sev_dist == "pareto":
            sev_alpha = st.number_input("α (shape)", min_value=2.0001, value=3.0, format="%.4f", key="sev_alpha")
            sev_x_m = st.number_input("x_m (minimum / scale)", min_value=0.01, value=5_000.0, format="%.2f", key="sev_x_m")
        elif sev_dist == "gamma":
            sev_alpha = st.number_input("α (shape)", min_value=0.01, value=2.0, format="%.4f", key="sev_alpha_gamma")
            sev_beta = st.number_input("β (scale)", min_value=0.01, value=5_000.0, format="%.2f", key="sev_beta")
        elif sev_dist == "weibull":
            sev_k = st.number_input("k (shape)", min_value=0.05, value=1.2, format="%.4f", key="sev_k")
            sev_lam = st.number_input("λ (scale)", min_value=0.01, value=10_000.0, format="%.2f", key="sev_lam")

    st.subheader("Simulation size")
    n_years = st.number_input(
        "Number of simulated years", min_value=1_000, max_value=200_000, value=25_000, step=1_000, key="n_years",
        help="More years = less simulation noise, slower run. 10,000–100,000 is a good range.",
    )
    seed = st.number_input("Random seed", min_value=0, value=42, step=1, key="seed")

# =============================================================================
# Run (cached) simulation
# =============================================================================
@st.cache_data(show_spinner="Running Monte Carlo simulation...")
def run_simulation(
    freq_dist, freq_mean, freq_variance,
    sev_dist, sev_mean, sev_cv, sev_mu, sev_sigma, sev_alpha, sev_x_m, sev_beta, sev_k, sev_lam,
    n_years, seed,
):
    freq_spec = FrequencySpec(dist=freq_dist, mean=freq_mean, variance=freq_variance)
    sev_spec = SeveritySpec(
        dist=sev_dist, mean=sev_mean, cv=sev_cv,
        mu=sev_mu, sigma=sev_sigma, alpha=sev_alpha, x_m=sev_x_m,
        beta=sev_beta, k=sev_k, lam=sev_lam,
    )
    return simulate_aggregate_losses(freq_spec, sev_spec, int(n_years), int(seed))


try:
    sim = run_simulation(
        freq_dist, freq_mean, freq_variance,
        sev_dist, sev_mean, sev_cv, sev_mu, sev_sigma, sev_alpha, sev_x_m, sev_beta, sev_k, sev_lam,
        n_years, seed,
    )
except ValueError as e:
    st.error(f"Couldn't build that simulation: {e}")
    st.stop()

freq_spec_display = FrequencySpec(dist=freq_dist, mean=freq_mean, variance=freq_variance)
sev_spec_display = SeveritySpec(
    dist=sev_dist, mean=sev_mean, cv=sev_cv,
    mu=sev_mu, sigma=sev_sigma, alpha=sev_alpha, x_m=sev_x_m,
    beta=sev_beta, k=sev_k, lam=sev_lam,
)

st.caption(
    f"**{freq_dist_label}** frequency (E[N]={freq_spec_display.analytic_mean():,.2f}) · "
    f"**{sev_dist_label}** severity (E[X]={sev_spec_display.analytic_mean():,.0f}) · "
    f"{int(n_years):,} simulated years · seed {int(seed)} · "
    f"{len(sim.occurrence_losses):,} total claims simulated"
)

st.markdown(DIVIDER, unsafe_allow_html=True)

# =============================================================================
# 1. Aggregate loss distribution histogram
# =============================================================================
st.subheader("1. Aggregate loss distribution")

stats = aggregate_statistics(sim.aggregate_losses)
var_df = var_tvar_table(sim.aggregate_losses, [0.95, 0.99, 0.995])

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=sim.aggregate_losses, nbinsx=80, marker_color="#4c6555", opacity=0.75, name="Aggregate loss",
))
fig_hist.add_vline(x=stats["mean"], line_dash="dot", line_color="#a1462f",
                    annotation_text="Mean", annotation_position="top")
fig_hist.add_vline(x=var_df.loc[var_df["confidence"] == 0.99, "VaR"].iloc[0], line_dash="dash", line_color="#445168",
                    annotation_text="VaR 99%", annotation_position="top")
fig_hist.update_layout(
    height=400, template="plotly_white", bargap=0.02,
    xaxis_title="Aggregate annual loss ($)", yaxis_title="Simulated years",
    margin=dict(t=30, b=10, l=10, r=10), font=dict(family="IBM Plex Sans"),
)
st.plotly_chart(fig_hist, use_container_width=True)

# =============================================================================
# 2. Summary statistics
# =============================================================================
st.subheader("2. Summary statistics")

load_col, _ = st.columns([1, 2])
with load_col:
    apply_loads = st.checkbox("Apply ALAE / ULAE loading to the loss cost figure", key="apply_loads")
    alae_pct = ulae_pct = 0.0
    if apply_loads:
        alae_pct = st.number_input("ALAE load (%)", min_value=0.0, value=5.0, step=0.5, format="%.2f", key="alae_pct") / 100
        ulae_pct = st.number_input("ULAE load (%)", min_value=0.0, value=8.0, step=0.5, format="%.2f", key="ulae_pct") / 100

loaded_losses = apply_alae_ulae(sim.aggregate_losses, alae_pct, ulae_pct) if apply_loads else sim.aggregate_losses
loaded_stats = aggregate_statistics(loaded_losses)

m1, m2, m3, m4 = st.columns(4)
m1.markdown(
    f'<div class="metric-box"><div style="font-size:0.75rem;color:#445168;">EXPECTED ANNUAL LOSS COST</div>'
    f'<div style="font-size:1.4rem;font-weight:600;">${loaded_stats["mean"]:,.0f}</div></div>',
    unsafe_allow_html=True,
)
m2.markdown(
    f'<div class="metric-box"><div style="font-size:0.75rem;color:#445168;">STD DEV</div>'
    f'<div style="font-size:1.4rem;font-weight:600;">${loaded_stats["std"]:,.0f}</div></div>',
    unsafe_allow_html=True,
)
m3.markdown(
    f'<div class="metric-box"><div style="font-size:0.75rem;color:#445168;">CV</div>'
    f'<div style="font-size:1.4rem;font-weight:600;">{loaded_stats["cv"]:.3f}</div></div>',
    unsafe_allow_html=True,
)
m4.markdown(
    f'<div class="metric-box"><div style="font-size:0.75rem;color:#445168;">SKEWNESS</div>'
    f'<div style="font-size:1.4rem;font-weight:600;">{loaded_stats["skewness"]:.3f}</div></div>',
    unsafe_allow_html=True,
)

if apply_loads:
    st.caption(f"Loaded: raw simulated mean ${stats['mean']:,.0f} × (1 + ALAE {alae_pct:.1%}) × (1 + ULAE {ulae_pct:.1%}).")

st.write("")
st.markdown("**VaR / TVaR (CTE) at selected confidence levels**")
loaded_var_df = var_tvar_table(loaded_losses, [0.95, 0.99, 0.995])
show_var = loaded_var_df.rename(columns={"confidence": "Confidence", "VaR": "VaR ($)", "TVaR": "TVaR / CTE ($)"})
show_var["Confidence"] = show_var["Confidence"].map(lambda q: f"{q:.1%}")
st.dataframe(
    show_var.style.format({"VaR ($)": "{:,.0f}", "TVaR / CTE ($)": "{:,.0f}"}),
    use_container_width=True, hide_index=True,
)

# =============================================================================
# 3. ILF curve
# =============================================================================
st.subheader("3. Increased Limits Factors (ILF)")
st.caption("ILF(x) = LEV(x) / LEV(base limit), where LEV(x) = E[min(Severity, x)] estimated from the simulated severity draws.")

sev_sample = sim.occurrence_losses
mean_sev = float(np.mean(sev_sample)) if len(sev_sample) else 0.0

ilf_col1, ilf_col2 = st.columns(2)
with ilf_col1:
    base_limit = st.number_input("Base limit ($)", min_value=1.0, value=round(mean_sev * 2, -2) or 10_000.0, step=1_000.0, key="ilf_base_limit")
with ilf_col2:
    max_limit = st.number_input("Max limit to chart ($)", min_value=base_limit * 2, value=round(mean_sev * 50, -3) or 500_000.0, step=10_000.0, key="ilf_max_limit")

ilf_limits = np.unique(np.concatenate([
    np.geomspace(max(base_limit / 10, 1.0), max_limit, 25),
    [base_limit],
]))
ilf_df = ilf_curve(sev_sample, ilf_limits, base_limit)

fig_ilf = go.Figure()
fig_ilf.add_trace(go.Scatter(x=ilf_df["limit"], y=ilf_df["ILF"], mode="lines+markers",
                              line=dict(color="#a1462f", width=2.5), marker=dict(size=5), name="ILF"))
fig_ilf.add_vline(x=base_limit, line_dash="dot", line_color="#445168", annotation_text="Base limit")
fig_ilf.update_layout(
    height=380, template="plotly_white", xaxis_title="Limit ($)", yaxis_title="ILF",
    margin=dict(t=30, b=10, l=10, r=10), font=dict(family="IBM Plex Sans"),
)
st.plotly_chart(fig_ilf, use_container_width=True)

# =============================================================================
# 4. LER curve
# =============================================================================
st.subheader("4. Loss Elimination Ratio (LER)")
st.caption("LER(d) = LEV(d) / E[Severity] — the share of expected severity eliminated by a deductible of d.")

ler_max_ded = st.number_input(
    "Max deductible to chart ($)", min_value=1.0, value=round(mean_sev * 5, -2) or 50_000.0, step=1_000.0, key="ler_max_ded",
)
ler_deds = np.unique(np.concatenate([[0.0], np.geomspace(max(mean_sev / 50, 1.0), ler_max_ded, 25)]))
ler_df = ler_curve(sev_sample, ler_deds)

fig_ler = go.Figure()
fig_ler.add_trace(go.Scatter(x=ler_df["deductible"], y=ler_df["LER"], mode="lines+markers",
                              line=dict(color="#4c6555", width=2.5), marker=dict(size=5), name="LER"))
fig_ler.update_layout(
    height=380, template="plotly_white", xaxis_title="Deductible ($)", yaxis_title="LER",
    yaxis_tickformat=".0%", margin=dict(t=30, b=10, l=10, r=10), font=dict(family="IBM Plex Sans"),
)
st.plotly_chart(fig_ler, use_container_width=True)

# =============================================================================
# 5. Coverage layer builder
# =============================================================================
st.markdown(DIVIDER, unsafe_allow_html=True)
st.subheader("5. Coverage layer builder")
st.markdown(
    "Stack per-occurrence deductibles/limits, per-occurrence XOL reinsurance layers, "
    "an aggregate stop-loss, and/or quota share. Each layer transforms the loss "
    "**retained after the prior layer** — order matters."
)

st.markdown(
    '<div class="note-box">⚠️ These are <strong>expected ceded loss costs from simulation</strong> — '
    'plain averages of simulated ceded amounts. They carry <strong>no risk load, '
    "profit margin, expense loading, or market pricing adjustment</strong>. This is not "
    "a reinsurance premium quote.</div>",
    unsafe_allow_html=True,
)

KIND_LABELS = {
    "Per-occurrence (deductible/limit or XOL)": "occurrence",
    "Aggregate stop-loss": "aggregate",
    "Quota share": "quota_share",
}
KIND_LABELS_INV = {v: k for k, v in KIND_LABELS.items()}

if "layer_table" not in st.session_state:
    st.session_state.layer_table = pd.DataFrame([
        {"name": "Primary", "kind": "Per-occurrence (deductible/limit or XOL)",
         "attachment": 5_000.0, "limit": 95_000.0, "no_limit": False, "quota_share_pct": 0.0},
    ])

layer_edited = st.data_editor(
    st.session_state.layer_table,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "name": st.column_config.TextColumn("Layer name", required=True),
        "kind": st.column_config.SelectboxColumn("Type", options=list(KIND_LABELS.keys()), required=True),
        "attachment": st.column_config.NumberColumn("Attachment ($)", min_value=0.0, format="%.0f"),
        "limit": st.column_config.NumberColumn("Limit ($, ignored if unlimited)", min_value=0.0, format="%.0f"),
        "no_limit": st.column_config.CheckboxColumn("Unlimited?"),
        "quota_share_pct": st.column_config.NumberColumn("Quota share % (0-1, occurrence/agg layers ignore this)", min_value=0.0, max_value=1.0, format="%.2f"),
    },
    key="layer_editor",
)
if st.button("Reset layers"):
    del st.session_state.layer_table
    st.rerun()

layer_edited = layer_edited.dropna(subset=["name", "kind"])
st.session_state.layer_table = layer_edited

layers = []
for _, row in layer_edited.iterrows():
    kind = KIND_LABELS[row["kind"]]
    limit = float("inf") if bool(row.get("no_limit", False)) else float(row.get("limit", 0.0) or 0.0)
    layers.append(CoverageLayer(
        name=str(row["name"]),
        kind=kind,
        attachment=float(row.get("attachment", 0.0) or 0.0),
        limit=limit,
        quota_share_pct=float(row.get("quota_share_pct", 0.0) or 0.0),
    ))

if layers:
    stages, final_occ, final_annual = apply_coverage_layers(sim, layers)

    stage_rows = [{
        "Layer": s.layer.name,
        "Type": KIND_LABELS_INV[s.layer.kind],
        "Ceded loss cost ($/yr)": s.ceded_loss_cost,
        "Retained loss cost ($/yr)": s.retained_loss_cost,
    } for s in stages]
    gross_loss_cost = sim.occurrence_losses.sum() / sim.n_years
    net_loss_cost = final_occ.sum() / sim.n_years

    st.dataframe(
        pd.DataFrame(stage_rows).style.format({
            "Ceded loss cost ($/yr)": "{:,.0f}", "Retained loss cost ($/yr)": "{:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    c1, c2 = st.columns(2)
    c1.markdown(
        f'<div class="metric-box"><div style="font-size:0.75rem;color:#445168;">GROSS EXPECTED LOSS COST ($/yr)</div>'
        f'<div style="font-size:1.4rem;font-weight:600;">${gross_loss_cost:,.0f}</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="metric-box"><div style="font-size:0.75rem;color:#445168;">NET RETAINED LOSS COST ($/yr)</div>'
        f'<div style="font-size:1.4rem;font-weight:600;">${net_loss_cost:,.0f}</div></div>',
        unsafe_allow_html=True,
    )

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(stages) + ["total"],
        x=["Gross"] + [s.layer.name for s in stages] + ["Net retained"],
        y=[gross_loss_cost] + [-s.ceded_loss_cost for s in stages] + [0],
        decreasing=dict(marker=dict(color="#a1462f")),
        increasing=dict(marker=dict(color="#a1462f")),
        totals=dict(marker=dict(color="#4c6555")),
        connector=dict(line=dict(color="#d8d3c6")),
    ))
    fig_wf.update_layout(
        height=420, template="plotly_white", showlegend=False,
        yaxis_title="Loss cost ($/yr)", margin=dict(t=30, b=10, l=10, r=10), font=dict(family="IBM Plex Sans"),
    )
    st.plotly_chart(fig_wf, use_container_width=True)
else:
    st.info("Add at least one layer to see the retained/ceded breakdown.")
    net_loss_cost = sim.occurrence_losses.sum() / sim.n_years

# =============================================================================
# 6. Credibility blend (optional tie-in)
# =============================================================================
with st.expander("6. Credibility-weighted loss cost (optional)"):
    st.markdown(
        "Blend this simulation's expected annual loss cost against a broader/prior "
        "mean using the same $Z = m/(m+k)$ credibility pattern as the "
        "[Bühlmann–Straub calculator](Buhlmann_Credibility_Calculator) — useful when this "
        "simulated scenario represents one class and you want to credibility-weight it "
        "against a wider portfolio mean."
    )
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        class_exposure = st.number_input(
            "Real exposure/years this class was observed", min_value=0.01, value=5.0, key="cred_exposure",
            help="Actual historical experience behind this class — not the simulated year count.",
        )
    with cc2:
        prior_mean = st.number_input("Broader/prior mean loss cost ($)", min_value=0.0, value=float(round(net_loss_cost, 0)), key="cred_prior_mean")
    with cc3:
        k_param = st.number_input("Credibility parameter k̂ = v̂/â", min_value=0.0001, value=10.0, key="cred_k")

    z, blended = credibility_weighted_loss_cost(net_loss_cost, class_exposure, prior_mean, k_param)
    st.markdown(
        f'<div class="formula-box">Z = {class_exposure:.2f} / ({class_exposure:.2f} + {k_param:.2f}) = {z:.3f}<br>'
        f'Blended estimate = {z:.3f} × ${net_loss_cost:,.0f} + {1-z:.3f} × ${prior_mean:,.0f} '
        f'= <strong>${blended:,.0f}</strong></div>',
        unsafe_allow_html=True,
    )

# =============================================================================
# Methodology
# =============================================================================
st.markdown(DIVIDER, unsafe_allow_html=True)
with st.expander("Methodology — how this is calculated"):
    st.markdown(
        r"""
**Collective risk / aggregate loss model**

For each simulated year, draw a claim count $N$ from the frequency distribution,
then draw $N$ i.i.d. severities $X_1,\dots,X_N$ from the severity distribution.
The aggregate loss for that year is
$$S = \sum_{i=1}^{N} X_i$$

This is repeated for every simulated year, producing an empirical distribution
of $S$ without assuming a closed-form for the compound distribution.

**Coverage layer transform**

A layer with attachment $a$ and limit $l$ splits a loss $x$ (either a single
occurrence, or a year's aggregate for a stop-loss) into

$$\text{ceded} = \min\big(\max(x - a,\ 0),\ l\big), \qquad \text{retained} = x - \text{ceded}$$

A per-occurrence deductible/limit and a per-occurrence XOL reinsurance layer
(e.g. \$500K xs \$500K) apply this identically to each simulated claim. An
aggregate stop-loss applies it once to each year's total. Quota share is the
degenerate case $\text{ceded} = q \cdot x$ for cession rate $q$, no attachment/limit.
Layers stack: each layer transforms the loss retained after the prior layer.

**Increased Limits Factor**

$$\text{LEV}(x) = E[\min(X, x)], \qquad \text{ILF}(x) = \frac{\text{LEV}(x)}{\text{LEV}(\text{base limit})}$$

estimated from the simulated severity draws. ILF is monotone non-decreasing
and concave in $x$ — each additional dollar of limit buys progressively less
expected recovery, since $\min(X,x)$ is itself concave in $x$.

**Loss Elimination Ratio**

$$\text{LER}(d) = \frac{\text{LEV}(d)}{E[X]}$$

the share of expected severity eliminated by a deductible of $d$.

**VaR / TVaR (CTE)**

$$\text{VaR}_q = \inf\{s : P(S \le s) \ge q\}, \qquad \text{TVaR}_q = E[S \mid S \ge \text{VaR}_q]$$

VaR is the $q$-quantile of simulated aggregate losses; TVaR (also called CTE,
Conditional Tail Expectation) is the average loss in the tail beyond that quantile —
always $\ge$ VaR, and more sensitive to tail thickness.

**Credibility blend**

$$Z = \frac{m}{m+\hat k}, \qquad \hat X = Z\,\bar X_{\text{sim}} + (1-Z)\,\bar X_{\text{prior}}$$

the same Bühlmann–Straub-style shrinkage used in the credibility calculator,
applied here to blend a simulated scenario's loss cost against a broader prior mean.

**ALAE / ULAE loading**

Simple proportional loads on the aggregate loss, applied in sequence:
$$\text{Ultimate} = S \times (1+\text{ALAE\%}) \times (1+\text{ULAE\%})$$
        """
    )
