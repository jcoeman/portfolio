import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.theme import (
    COLORS,
    apply_theme,
    back_link,
    metric_box_html,
    question_placard,
    section_heading,
    so_what_box,
)

apply_theme("Experience-Based Loss Cost Estimation", "📊")
back_link()

st.title("Estimating Loss Costs with Limited Data")
st.markdown(
    """
An insurer may want to use an insured's own claim experience when estimating its
future losses. However, a result based on limited experience can be heavily
influenced by random variation. One unusually bad year should not, by itself,
cause a large increase in an insured's premium.

This project demonstrates how actuaries balance an individual indured's experience
with the broader experience of a portfolio. Insureds with more experience generally
receive more weight on their own results, while insureds with less experience remain
closer to the portfolio average.

The final estimate is a weighted average of the insured's own experience and the
portfolio average:
"""
)

st.latex(
    r"""
    \text{Final Estimate}
    =
    Z(\text{Group Average})
    +
    (1-Z)(\text{Portfolio Average})
    """
)

st.markdown(
    """
Here, **Z** represents the credibility assigned to the individual insured's own experience.
It ranges from 0 to 1:

- When **Z is close to 1**, the estimate relies mostly on the insured's own experience.
- When **Z is close to 0**, the estimate relies mostly on the portfolio average.
"""
)
# =========================================================================
# 1. THE QUESTION
# =========================================================================
question_placard(
    "How much should an insurer rely on a insured's own experience when some "
    "insureds have much more data than others?"
)

# =========================================================================
# 2. THE CONCEPT — interactive intuition builder
# =========================================================================
section_heading(
    "02",
    "See Credibility in Action",
    "Explore how additional experience changes the estimate",
)
st.markdown(
    """
Imagine an insurer is estimating the future **loss cost** for a particular group
of policyholders. Loss cost is the average amount the insurer expects to pay in
claims for each policy or unit of exposure. It represents the portion of
the premium paid to claims before expenses, profit, and other adjustments are added.

An insured or group of insureds has its own observed claim experience, but the insurer also has an
average loss cost calculated from a much larger portfolio.

When an insured or group of insureds has only a small amount of experience, its observed loss cost may
be strongly affected by chance. A few unusually large claims or an unusually
claim-free year could make the group appear riskier or safer than it truly is.
For this reason, the estimate initially remains close to the portfolio average.

As more experience is collected, results become more reliable and
receive greater credibility. The loss cost estimate then moves closer to the insured's own
observed loss cost.

Move the slider to increase the amount of available experience. Notice how the
credibility assigned to the insured changes and how the final estimate shifts from
the portfolio average toward the insured's own experience.
"""
)

CONCEPT_K = 8.0
CONCEPT_CLASS_MEAN = 500.0
CONCEPT_OVERALL_MEAN = 750.0

years = st.slider(
    "Amount of experience available",
    min_value=0,
    max_value=40,
    value=3,
    step=1,
    help=(
        "An exposure unit represents one unit of insured activity, such as one "
        "vehicle insured for one year. More exposure provides more information "
        "about the group's expected losses."
    ),
)

Z = years / (years + CONCEPT_K) if (years + CONCEPT_K) > 0 else 0.0
blended = Z * CONCEPT_CLASS_MEAN + (1 - Z) * CONCEPT_OVERALL_MEAN

concept_fig = go.Figure()
concept_fig.add_trace(go.Bar(
    y=["Credibility weight"], x=[Z], orientation="h",
    marker_color=COLORS["rust"], name="Weight on this class's own experience",
    text=[f"{Z:.0%} own experience"], textposition="inside", insidetextanchor="middle",
))
concept_fig.add_trace(go.Bar(
    y=["Credibility weight"], x=[1 - Z], orientation="h",
    marker_color=COLORS["slate"], opacity=0.55, name="Weight on collective average",
    text=[f"{1-Z:.0%} collective average"], textposition="inside", insidetextanchor="middle",
))
concept_fig.update_layout(
    barmode="stack", height=140, template="plotly_white",
    showlegend=False, xaxis=dict(range=[0, 1], tickformat=".0%", title=None),
    yaxis=dict(title=None),
    margin=dict(t=10, b=30, l=10, r=10), font=dict(family="IBM Plex Sans"),
)
st.plotly_chart(concept_fig, width="stretch")

st.markdown(
    f"""
<div class="takeaway-line">
With {years} exposure unit{"s" if years != 1 else ""}, the calculation places
<strong>{Z:.0%}</strong> of the weight on the insured's own experience and
<strong>{1-Z:.0%}</strong> on the broader portfolio.

The insured's observed loss cost is {CONCEPT_CLASS_MEAN:,.0f} per exposure,
compared with a portfolio average of {CONCEPT_OVERALL_MEAN:,.0f}. Combining
those two sources produces an estimated loss cost of
<strong>{blended:,.0f} per exposure</strong>.
</div>
    """,
    unsafe_allow_html=True,
)
show_formula = st.checkbox(
    "Show how the weight was calculated",
    key="concept_formula_toggle",
)
if show_formula:
    st.markdown(
        f'<div class="formula-box">Z = m / (m + k) &nbsp;&nbsp;→&nbsp;&nbsp; '
        f'{years} / ({years} + {CONCEPT_K:.0f}) = {Z:.3f}<br>'
        f'Estimate = Z · (class mean) + (1 − Z) · (overall mean)</div>',
        unsafe_allow_html=True,
    )
    st.caption(
    "Here, m represents the amount of experience available. "
    "K determines how quickly the weight on a group's own experience "
    "increases."
)

st.markdown("---")

# =========================================================================
# 3. MY APPROACH
# =========================================================================
section_heading(
    "02",
    "How the Weight Is Determined",
    "Distinguishing meaningful differences from random variation",
)

st.markdown(
    """
Our credibility calculation depends on two kinds of variation to determine how much credibility
a group's own experience should receive:

- **Random variation within a group:** A group's loss cost will naturally change
  from year to year, even if its underlying level of risk has not changed. Large
  year-to-year fluctuations make it harder to determine the group's true
  expected loss cost from its observed results.

- **Differences between groups:** Groups may have genuinely different underlying
  levels of risk. When their results remain meaningfully different as more
  experience is collected, those differences provide evidence that each group
  has its own expected loss cost rather than simply fluctuating around the same
  portfolio average.

Credibility depends on the strength of this signal relative to the random
variation. If groups are clearly differentiated, the portfolio average may not
represent every group equally well. A group's own experience therefore provides
more useful information about its underlying loss cost and receives greater
weight.

If year-to-year variation is large while differences between groups are small,
it is difficult to tell whether a group's observed result reflects a real
difference or simply chance. More experience is then required before the
group's result receives substantial weight.

This approach is called **Bühlmann–Straub credibility**. It also accounts for
differences in exposure, allowing groups with more experience to receive greater
credibility than groups with less experience.
"""
)

with st.expander("Technical derivation and formulas"):
    st.markdown(
        r"""
The plain-language process above is implemented with the formulas below. For
class $i$ and period $j$, $m_{ij}$ is exposure and $X_{ij}$ is loss cost per
exposure unit.

**Exposure-weighted class mean**
$$\bar{X}_i = \frac{\sum_j m_{ij} X_{ij}}{m_i}, \qquad m_i = \sum_j m_{ij}$$

**Overall mean**
$$\bar{X} = \frac{\sum_i m_i \bar{X}_i}{m}, \qquad m = \sum_i m_i$$

**Expected process variance (within-class variance)**
$$\hat{v} = \frac{\sum_i \sum_j m_{ij}(X_{ij}-\bar{X}_i)^2}{\sum_i (n_i - 1)}$$

**Variance of hypothetical means (between-class variance)**
$$\hat{a} = \frac{\sum_i m_i(\bar{X}_i - \bar{X})^2 - (r-1)\hat{v}}{m - \sum_i m_i^2 / m}$$

(floored at 0 if negative, per standard practice — a negative raw estimate implies no detectable between-class variance beyond noise)

**Credibility factor and estimate**
$$Z_i = \frac{m_i}{m_i + \hat{v}/\hat{a}}, \qquad \hat{X}_i = Z_i \bar{X}_i + (1-Z_i)\bar{X}$$

The resulting $Z_i$ is the share of the indication assigned to the class's own
experience. Values near 1 place more weight on the class; values near 0 place
more weight on the collective mean. Bühlmann–Straub extends the classic
Bühlmann model by allowing exposure to vary across classes and periods.
            """
    )

st.markdown("---")

# =========================================================================
# 4. TRY IT YOURSELF — the existing calculator
# =========================================================================
section_heading(
    "04",
    "Explore the Calculator",
    "See how credibility changes the estimated loss cost",
)

st.markdown(
    """
Start with the sample data, adjust the values directly, or upload your own CSV.
The recommended option calculates credibility from observations for multiple
classes and time periods. If the credibility assumptions have already been
calculated elsewhere, you can instead enter summarized values.
"""
)
# ---- Sample data ----
SAMPLE_LONG = pd.DataFrame(
    {
        "class": (
            ["Class A"] * 4
            + ["Class B"] * 4
            + ["Class C"] * 3
            + ["Class D"] * 5
        ),
        "period": (
            [1, 2, 3, 4]
            + [1, 2, 3, 4]
            + [1, 2, 3]
            + [1, 2, 3, 4, 5]
        ),
        "exposure": [
            120, 135, 128, 140,
            45, 50, 48, 52,
            800, 820, 810,
            10, 12, 9, 11, 13,
        ],
        "loss_cost": [
            412.50, 388.20, 455.10, 401.75,
            610.00, 590.30, 725.40, 580.20,
            301.20, 295.80, 310.50,
            980.00, 1120.50, 750.30, 1050.75, 890.40,
        ],
    }
)

if "credibility_matrix" not in st.session_state:
    st.session_state.credibility_matrix = SAMPLE_LONG.copy()

mode = st.radio(
    "Choose how to provide the data",
    [
        "Class-period data (recommended)",
        "Previously summarized inputs",
    ],
    horizontal=True,
    help=(
        "Choose class-period data to calculate credibility directly from observations "
        "for multiple classes and periods. Choose previously summarized inputs if you "
        "already have a selected credibility parameter (K) and only want to calculate "
        "the indicated loss cost."
    ),
)

st.markdown('<hr style="border-top:1px solid #d8d3c6; margin: 1.5rem 0;">', unsafe_allow_html=True)

# =========================================================================
# MODE 1 — full matrix
# =========================================================================
if mode.startswith("Class-period"):

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Class-period experience")

        edit_tab, upload_tab = st.tabs(
            ["✏️ Edit matrix", "📤 Upload CSV"]
        )

        with edit_tab:
            st.caption(
                "Each row represents one group in one observation period. "
                "Add, remove, or edit rows to see how the estimates respond."
            )

            edited = st.data_editor(
                st.session_state.credibility_matrix,
                num_rows="dynamic",
                width="stretch",
                column_config={
                    "class": st.column_config.TextColumn(
                        "Class",
                        required=True,
                    ),
                    "period": st.column_config.NumberColumn(
                        "Period",
                        required=True,
                    ),
                    "exposure": st.column_config.NumberColumn(
                        "Exposure",
                        required=True,
                        min_value=0.0001,
                        format="%.2f",
                    ),
                    "loss_cost": st.column_config.NumberColumn(
                        "Loss cost (per exposure)",
                        required=True,
                        format="%.2f",
                    ),
                },
                key="matrix_editor",
            )

            if st.button("Reset to sample data"):
                st.session_state.credibility_matrix = SAMPLE_LONG.copy()
                st.rerun()

            st.session_state.credibility_matrix = edited

        with upload_tab:
            st.markdown(
                "Provide one row per group and year. The CSV must include "
                "`class`, `period`, `exposure`, and `loss_cost`, where loss "
                "cost is the loss amount per exposure unit for that observation."
            )

            uploaded = st.file_uploader(
                "Upload class-period CSV",
                type=["csv"],
            )

            if uploaded is not None:
                try:
                    df_up = pd.read_csv(uploaded)
                    df_up.columns = [
                        column.lower() for column in df_up.columns
                    ]

                    required_cols = {
                        "class",
                        "period",
                        "exposure",
                        "loss_cost",
                    }

                    if not required_cols.issubset(df_up.columns):
                        st.error(
                            f"CSV must contain columns: "
                            f"{sorted(required_cols)}"
                        )
                    else:
                        st.session_state.credibility_matrix = df_up[
                            [
                                "class",
                                "period",
                                "exposure",
                                "loss_cost",
                            ]
                        ].copy()

                        st.success(
                            f"Loaded {len(df_up)} rows across "
                            f"{df_up['class'].nunique()} classes."
                        )

                except Exception as e:
                    st.error(f"Couldn't parse that file: {e}")

            st.download_button(
                "Download sample CSV template",
                data=SAMPLE_LONG.to_csv(index=False).encode("utf-8"),
                file_name="credibility_sample_template.csv",
                mime="text/csv",
            )

    df = st.session_state.credibility_matrix.dropna()
    df = df[df["exposure"] > 0]

    if df["class"].nunique() < 2:
        st.warning(
            "At least two classes are required to estimate differences "
            "between class means. Add another class to continue."
        )
        st.stop()

    if (df.groupby("class")["period"].count() < 2).any():
        st.warning(
            "At least one class has only one observation period. Two or more "
            "periods are needed to estimate variation within that class. "
            "The class will still receive an indication, but it cannot "
            "contribute to the within-class variance estimate."
        )

    # ---------------------------------------------------------------
    # Bühlmann–Straub math
    # ---------------------------------------------------------------
    def buhlmann_straub(df: pd.DataFrame):
        classes = df["class"].unique()
        r = len(classes)

        class_stats = []
        for c in classes:
            sub = df[df["class"] == c]
            m_i = sub["exposure"].sum()
            xbar_i = (sub["exposure"] * sub["loss_cost"]).sum() / m_i
            n_i = len(sub)
            class_stats.append({"class": c, "m_i": m_i, "xbar_i": xbar_i, "n_i": n_i})
        stats_df = pd.DataFrame(class_stats)

        m = stats_df["m_i"].sum()
        xbar = (stats_df["m_i"] * stats_df["xbar_i"]).sum() / m

        # EPV: v_hat = sum_i sum_j m_ij (X_ij - Xbar_i)^2 / sum_i (n_i - 1)
        epv_num = 0.0
        epv_den = 0.0
        for c in classes:
            sub = df[df["class"] == c]
            xbar_i = stats_df.loc[stats_df["class"] == c, "xbar_i"].values[0]
            n_i = len(sub)
            epv_num += (sub["exposure"] * (sub["loss_cost"] - xbar_i) ** 2).sum()
            epv_den += (n_i - 1)

        v_hat = epv_num / epv_den if epv_den > 0 else np.nan

        # VHM: a_hat = [sum_i m_i (Xbar_i - Xbar)^2 - (r-1) v_hat] / [m - sum_i m_i^2 / m]
        vhm_num = (stats_df["m_i"] * (stats_df["xbar_i"] - xbar) ** 2).sum() - (r - 1) * v_hat
        vhm_den = m - (stats_df["m_i"] ** 2).sum() / m
        a_hat_raw = vhm_num / vhm_den if vhm_den > 0 else np.nan
        a_hat = max(a_hat_raw, 0.0) if not np.isnan(a_hat_raw) else np.nan

        k_hat = v_hat / a_hat if (a_hat and a_hat > 0) else np.inf

        stats_df["Z_i"] = stats_df["m_i"] / (stats_df["m_i"] + k_hat) if np.isfinite(k_hat) else 0.0
        stats_df["credibility_estimate"] = stats_df["Z_i"] * stats_df["xbar_i"] + (1 - stats_df["Z_i"]) * xbar

        return stats_df, {
            "m": m, "xbar": xbar, "v_hat": v_hat, "a_hat_raw": a_hat_raw,
            "a_hat": a_hat, "k_hat": k_hat, "r": r,
        }

    stats_df, params = buhlmann_straub(df)

    with right:
        st.subheader("Credibility estimates")

        m1, m2, m3 = st.columns(3)
        m1.markdown(metric_box_html("Collective mean", f'{params["xbar"]:,.2f}', "exposure-weighted across all classes"), unsafe_allow_html=True)
        m2.markdown(metric_box_html("Within-class variation", f'{params["v_hat"]:,.2f}', "expected process variance (EPV)"), unsafe_allow_html=True)
        m3.markdown(metric_box_html("Between-class variation", f'{params["a_hat"]:,.4f}', "variance of hypothetical means (VHM)"), unsafe_allow_html=True)

        st.write("")
        if params["a_hat_raw"] < 0:
            st.caption(
                "The raw between-class variance estimate is negative. This can occur "
                "when the data do not identify class differences beyond estimated "
                "process variation. Following the calculator's stated convention, the "
                "estimate is set to zero, so every indication equals the collective mean."
            )

        display_df = stats_df.copy()
        display_df.columns = ["Class", "Total exposure", "Observed class loss cost", "Periods observed", "Weight on class experience", "Indicated loss cost"]
        st.dataframe(
            display_df.style.format({
                "Total exposure": "{:,.1f}",
                "Observed class loss cost": "{:,.2f}",
                "Weight on class experience": "{:.1%}",
                "Indicated loss cost": "{:,.2f}",
            }),
            width="stretch",
            hide_index=True,
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stats_df["class"], y=stats_df["xbar_i"], name="Observed class loss cost",
            marker_color="#a1462f", opacity=0.55,
        ))
        fig.add_trace(go.Bar(
            x=stats_df["class"], y=stats_df["credibility_estimate"], name="Credibility-weighted indication",
            marker_color="#4c6555",
        ))
        fig.add_hline(y=params["xbar"], line_dash="dot", line_color="#445168",
                       annotation_text="Collective mean", annotation_position="top left")
        fig.update_layout(
            barmode="group", height=380, template="plotly_white",
            legend=dict(orientation="h", y=1.15),
            margin=dict(t=30, b=10, l=10, r=10),
            font=dict(family="IBM Plex Sans"),
        )
        st.plotly_chart(fig, width="stretch")

# =========================================================================
# MODE 2 — direct inputs
# =========================================================================
else:
    st.subheader("Previously summarized inputs")
    st.caption(
        "Use this mode when the class loss cost, collective mean, and credibility "
        "parameter k have already been estimated in another study. The calculator "
        "will apply those inputs to one or more classes."
    )

    n_classes = st.number_input("Number of classes to evaluate", min_value=1, max_value=20, value=3, step=1)

    default_rows = pd.DataFrame({
        "class": [f"Class {chr(65+i)}" for i in range(n_classes)],
        "exposure_m_i": [100.0] * n_classes,
        "sample_mean_xbar_i": [500.0] * n_classes,
    })

    col_a, col_b = st.columns([2, 1], gap="large")
    with col_a:
        direct_df = st.data_editor(
            default_rows,
            num_rows="fixed",
            width="stretch",
            column_config={
                "class": st.column_config.TextColumn("Class"),
                "exposure_m_i": st.column_config.NumberColumn("Total exposure", min_value=0.0001, format="%.2f"),
                "sample_mean_xbar_i": st.column_config.NumberColumn("Observed class loss cost", format="%.2f"),
            },
            key="direct_editor",
        )
    with col_b:
        overall_mean = st.number_input("Collective mean loss cost", value=450.0, format="%.2f")
        k_hat = st.number_input(
            "Credibility parameter (k)",
            value=50.0, min_value=0.0, format="%.4f",
            help="k compares within-class process variation with estimated differences between class means. A smaller value gives class experience more weight at a given exposure.",
        )

    if len(direct_df) > 0 and k_hat is not None:
        direct_df = direct_df.copy()
        direct_df["Z_i"] = direct_df["exposure_m_i"] / (direct_df["exposure_m_i"] + k_hat)
        direct_df["credibility_estimate"] = (
            direct_df["Z_i"] * direct_df["sample_mean_xbar_i"] + (1 - direct_df["Z_i"]) * overall_mean
        )

        st.markdown('<hr style="border-top:1px solid #d8d3c6; margin: 1.5rem 0;">', unsafe_allow_html=True)
        st.subheader("Credibility estimates")

        show_df = direct_df.rename(columns={
            "class": "Class", "exposure_m_i": "Total exposure",
            "sample_mean_xbar_i": "Observed class loss cost",
            "Z_i": "Weight on class experience", "credibility_estimate": "Indicated loss cost",
        })
        st.dataframe(
            show_df.style.format({
                "Total exposure": "{:,.1f}",
                "Observed class loss cost": "{:,.2f}",
                "Weight on class experience": "{:.1%}",
                "Indicated loss cost": "{:,.2f}",
            }),
            width="stretch", hide_index=True,
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=direct_df["class"], y=direct_df["sample_mean_xbar_i"],
                               name="Observed class loss cost", marker_color="#a1462f", opacity=0.55))
        fig2.add_trace(go.Bar(x=direct_df["class"], y=direct_df["credibility_estimate"],
                               name="Credibility-weighted indication", marker_color="#4c6555"))
        fig2.add_hline(y=overall_mean, line_dash="dot", line_color="#445168",
                        annotation_text="Collective mean", annotation_position="top left")
        fig2.update_layout(barmode="group", height=380, template="plotly_white",
                            legend=dict(orientation="h", y=1.15),
                            margin=dict(t=30, b=10, l=10, r=10),
                            font=dict(family="IBM Plex Sans"))
        st.plotly_chart(fig2, width="stretch")

        st.markdown(
            f'<div class="formula-box">Z_i = m_i / (m_i + k̂), &nbsp; k̂ = {k_hat:,.3f}<br>'
            f'X̂_i = Z_i · X̄_i + (1 − Z_i) · X̄</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# =========================================================================
# 5. SO WHAT?
# =========================================================================
