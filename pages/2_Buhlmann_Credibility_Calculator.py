import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Bühlmann–Straub Calculator | Portfolio", page_icon="📊", layout="wide")

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
    </style>
    """,
    unsafe_allow_html=True,
)

st.page_link("Home.py", label="← Back to portfolio", icon="🏠")

st.title("Bühlmann–Straub Credibility Calculator")
st.markdown(
    "Empirical Bayes credibility weighting for classes observed with "
    "**non-uniform exposures** — the Bühlmann–Straub extension of the "
    "classic Bühlmann model. Edit the matrix below, upload your own, or "
    "switch to the direct-input mode if you already have summarized "
    "class means."
)

# =========================================================================
# Sample data
# =========================================================================
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
        "aggregate_loss": [
            49500, 52407, 58253, 56245,
            27450, 29515, 34819, 30170,
            240960, 242556, 251505,
            9800, 13446, 6753, 11558, 11575,
        ],
    }
)
SAMPLE_LONG["loss_cost"] = SAMPLE_LONG["aggregate_loss"] / SAMPLE_LONG["exposure"]

if "credibility_matrix" not in st.session_state:
    st.session_state.credibility_matrix = SAMPLE_LONG.copy()

# =========================================================================
# Mode selector
# =========================================================================
mode = st.radio(
    "Input mode",
    ["Exposure / loss-cost matrix (recommended)", "Direct summary inputs"],
    horizontal=True,
    help=(
        "Matrix mode estimates the credibility parameter (k = v̂/â) from your "
        "data via the full EPV/VHM decomposition. Direct mode lets you type "
        "in a sample mean, overall mean, exposure, and k directly if you've "
        "already computed those elsewhere."
    ),
)

st.markdown('<hr style="border-top:1px solid #d8d3c6; margin: 1.5rem 0;">', unsafe_allow_html=True)

# =========================================================================
# MODE 1 — full matrix
# =========================================================================
if mode.startswith("Exposure"):

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("1. Exposure & aggregate loss data")

        upload_tab, edit_tab = st.tabs(["📤 Upload CSV", "✏️ Edit matrix"])

        with upload_tab:
            st.markdown(
                "CSV needs columns: `class`, `period`, `exposure`, `aggregate_loss` "
                "(the **total** losses for that class-period — the app divides by "
                "exposure automatically to get the per-unit rate Bühlmann–Straub needs)."
            )
            uploaded = st.file_uploader("Upload exposure/loss CSV", type=["csv"])
            if uploaded is not None:
                try:
                    df_up = pd.read_csv(uploaded)
                    required_cols = {"class", "period", "exposure", "aggregate_loss"}
                    if not required_cols.issubset(set(c.lower() for c in df_up.columns)):
                        st.error(f"CSV must contain columns: {sorted(required_cols)}")
                    else:
                        df_up.columns = [c.lower() for c in df_up.columns]
                        df_up = df_up[["class", "period", "exposure", "aggregate_loss"]].copy()
                        df_up["loss_cost"] = df_up["aggregate_loss"] / df_up["exposure"]
                        st.session_state.credibility_matrix = df_up
                        st.success(f"Loaded {len(df_up)} rows across {df_up['class'].nunique()} classes.")
                except Exception as e:
                    st.error(f"Couldn't parse that file: {e}")

            st.download_button(
                "Download sample CSV template",
                data=SAMPLE_LONG[["class", "period", "exposure", "aggregate_loss"]].to_csv(index=False).encode("utf-8"),
                file_name="credibility_sample_template.csv",
                mime="text/csv",
            )

        with edit_tab:
            st.caption(
                "Add, remove, or edit rows directly. Each row = one class-period "
                "observation. Enter **total** losses for the period — the per-unit "
                "rate is calculated automatically."
            )
            editor_input = st.session_state.credibility_matrix[["class", "period", "exposure", "aggregate_loss"]]
            edited = st.data_editor(
                editor_input,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "class": st.column_config.TextColumn("Class", required=True),
                    "period": st.column_config.NumberColumn("Period", required=True),
                    "exposure": st.column_config.NumberColumn("Exposure", required=True, min_value=0.0001, format="%.2f"),
                    "aggregate_loss": st.column_config.NumberColumn("Aggregate loss (period total)", required=True, format="%.2f"),
                },
                key="matrix_editor",
            )
            if st.button("Reset to sample data"):
                st.session_state.credibility_matrix = SAMPLE_LONG.copy()
                st.rerun()
            edited = edited.dropna(subset=["class", "period", "exposure", "aggregate_loss"])
            edited = edited[edited["exposure"] > 0]
            edited["loss_cost"] = edited["aggregate_loss"] / edited["exposure"]
            st.session_state.credibility_matrix = edited

    df = st.session_state.credibility_matrix.dropna()
    df = df[df["exposure"] > 0]

    if df["class"].nunique() < 2:
        st.warning("Need at least 2 classes with data to estimate credibility. Add more rows.")
        st.stop()
    if (df.groupby("class")["period"].count() < 2).any():
        st.warning(
            "At least one class has only 1 period of data. Bühlmann–Straub's "
            "EPV estimate needs 2+ periods per class to compute within-class "
            "variance. That class will still get a credibility score, but the "
            "overall v̂ estimate excludes classes with only 1 observation."
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
        st.subheader("2. Results")

        m1, m2, m3 = st.columns(3)
        m1.markdown(
            f'<div class="metric-box"><div style="font-size:0.75rem;color:#445168;">OVERALL MEAN (X̄)</div>'
            f'<div style="font-size:1.4rem;font-weight:600;">{params["xbar"]:,.2f}</div></div>',
            unsafe_allow_html=True,
        )
        m2.markdown(
            f'<div class="metric-box"><div style="font-size:0.75rem;color:#445168;">EPV (v̂)</div>'
            f'<div style="font-size:1.4rem;font-weight:600;">{params["v_hat"]:,.2f}</div></div>',
            unsafe_allow_html=True,
        )
        m3.markdown(
            f'<div class="metric-box"><div style="font-size:0.75rem;color:#445168;">VHM (â)</div>'
            f'<div style="font-size:1.4rem;font-weight:600;">{params["a_hat"]:,.4f}</div></div>',
            unsafe_allow_html=True,
        )

        st.write("")
        if params["a_hat_raw"] < 0:
            st.caption(
                "⚠️ Raw â estimate was negative (common with few classes / noisy data) "
                "and was floored at 0 — meaning k̂ = v̂/â is effectively infinite, "
                "so credibility collapses toward 0 for every class (all weight on the overall mean)."
            )

        st.markdown(
            f'<div class="formula-box">k̂ = v̂ / â = {params["k_hat"]:,.3f}<br>'
            f'Z_i = m_i / (m_i + k̂)</div>',
            unsafe_allow_html=True,
        )

        display_df = stats_df.copy()
        display_df.columns = ["Class", "Total exposure (m_i)", "Class mean (X̄_i)", "# periods", "Credibility (Z_i)", "Credibility-weighted estimate"]
        st.dataframe(
            display_df.style.format({
                "Total exposure (m_i)": "{:,.1f}",
                "Class mean (X̄_i)": "{:,.2f}",
                "Credibility (Z_i)": "{:.3f}",
                "Credibility-weighted estimate": "{:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stats_df["class"], y=stats_df["xbar_i"], name="Class mean (X̄_i)",
            marker_color="#a1462f", opacity=0.55,
        ))
        fig.add_trace(go.Bar(
            x=stats_df["class"], y=stats_df["credibility_estimate"], name="Credibility-weighted",
            marker_color="#4c6555",
        ))
        fig.add_hline(y=params["xbar"], line_dash="dot", line_color="#445168",
                       annotation_text="Overall mean", annotation_position="top left")
        fig.update_layout(
            barmode="group", height=380, template="plotly_white",
            legend=dict(orientation="h", y=1.15),
            margin=dict(t=30, b=10, l=10, r=10),
            font=dict(family="IBM Plex Sans"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr style="border-top:1px solid #d8d3c6; margin: 1.5rem 0;">', unsafe_allow_html=True)
    with st.expander("Methodology — how this is calculated"):
        st.markdown(
            r"""
For each class $i$ with periods $j = 1,\dots,n_i$, exposures $m_{ij}$, and per-exposure loss costs $X_{ij}$:

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

This is the Bühlmann–Straub model — the extension of the classic Bühlmann
model that allows exposures to vary by period and by class, weighting each
period's contribution to the class mean and to EPV/VHM by its exposure
rather than treating all periods as equally sized.
            """
        )

# =========================================================================
# MODE 2 — direct inputs
# =========================================================================
else:
    st.subheader("Direct summary inputs")
    st.caption(
        "For when you've already computed the class mean, overall mean, "
        "and credibility parameter k = v̂/â elsewhere (e.g. from a prior "
        "study) and just want the credibility-weighted result for one or "
        "more classes."
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
            use_container_width=True,
            column_config={
                "class": st.column_config.TextColumn("Class"),
                "exposure_m_i": st.column_config.NumberColumn("Exposure (m_i)", min_value=0.0001, format="%.2f"),
                "sample_mean_xbar_i": st.column_config.NumberColumn("Sample mean (X̄_i)", format="%.2f"),
            },
            key="direct_editor",
        )
    with col_b:
        overall_mean = st.number_input("Overall mean (X̄)", value=450.0, format="%.2f")
        k_hat = st.number_input(
            "Credibility parameter k̂ = v̂ / â",
            value=50.0, min_value=0.0, format="%.4f",
            help="Ratio of expected process variance to variance of hypothetical means. "
                 "Smaller k → credibility rises faster with exposure.",
        )

    if len(direct_df) > 0 and k_hat is not None:
        direct_df = direct_df.copy()
        direct_df["Z_i"] = direct_df["exposure_m_i"] / (direct_df["exposure_m_i"] + k_hat)
        direct_df["credibility_estimate"] = (
            direct_df["Z_i"] * direct_df["sample_mean_xbar_i"] + (1 - direct_df["Z_i"]) * overall_mean
        )

        st.markdown('<hr style="border-top:1px solid #d8d3c6; margin: 1.5rem 0;">', unsafe_allow_html=True)
        st.subheader("Results")

        show_df = direct_df.rename(columns={
            "class": "Class", "exposure_m_i": "Exposure (m_i)",
            "sample_mean_xbar_i": "Sample mean (X̄_i)",
            "Z_i": "Credibility (Z_i)", "credibility_estimate": "Credibility-weighted estimate",
        })
        st.dataframe(
            show_df.style.format({
                "Exposure (m_i)": "{:,.1f}",
                "Sample mean (X̄_i)": "{:,.2f}",
                "Credibility (Z_i)": "{:.3f}",
                "Credibility-weighted estimate": "{:,.2f}",
            }),
            use_container_width=True, hide_index=True,
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=direct_df["class"], y=direct_df["sample_mean_xbar_i"],
                               name="Sample mean", marker_color="#a1462f", opacity=0.55))
        fig2.add_trace(go.Bar(x=direct_df["class"], y=direct_df["credibility_estimate"],
                               name="Credibility-weighted", marker_color="#4c6555"))
        fig2.add_hline(y=overall_mean, line_dash="dot", line_color="#445168",
                        annotation_text="Overall mean", annotation_position="top left")
        fig2.update_layout(barmode="group", height=380, template="plotly_white",
                            legend=dict(orientation="h", y=1.15),
                            margin=dict(t=30, b=10, l=10, r=10),
                            font=dict(family="IBM Plex Sans"))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            f'<div class="formula-box">Z_i = m_i / (m_i + k̂), &nbsp; k̂ = {k_hat:,.3f}<br>'
            f'X̂_i = Z_i · X̄_i + (1 − Z_i) · X̄</div>',
            unsafe_allow_html=True,
        )