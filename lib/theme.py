"""
Shared visual language for the portfolio: colors, global CSS, and small
render helpers for the "exploratorium" page structure (Question / Concept /
Approach / Try it / So what) used consistently across Home.py and every
pages/*.py file.
"""

import streamlit as st

COLORS = {
    "ink": "#1b2430",
    "paper": "#f7f5f0",
    "card": "#fffdf9",
    "slate": "#445168",
    "rust": "#a1462f",
    "moss": "#4c6555",
    "line": "#d8d3c6",
    "moss_bg": "#eef2ee",
}

GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,600;0,700;1,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
}}
h1, h2, h3 {{
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    letter-spacing: -0.01em;
}}
code, .mono {{
    font-family: 'IBM Plex Mono', monospace;
}}

:root {{
    --ink: {COLORS["ink"]};
    --paper: {COLORS["paper"]};
    --card: {COLORS["card"]};
    --slate: {COLORS["slate"]};
    --rust: {COLORS["rust"]};
    --moss: {COLORS["moss"]};
    --line: {COLORS["line"]};
    --moss-bg: {COLORS["moss_bg"]};
}}

.stApp {{ background-color: var(--paper); }}

/* ---------- landing page ---------- */
.hero-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.75rem;
    color: var(--rust);
    margin-bottom: 0.25rem;
}}
.hero-title {{ font-size: 3rem; line-height: 1.05; color: var(--ink); margin-bottom: 0.5rem; }}
.hero-sub {{ font-size: 1.15rem; color: var(--slate); max-width: 640px; line-height: 1.5; }}
.divider {{ border: none; border-top: 1px solid var(--line); margin: 2rem 0; }}

.project-card {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 1.75rem;
    height: 100%;
    transition: border-color 0.15s ease;
}}
.project-card:hover {{ border-color: var(--rust); }}
.project-tag {{
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
}}
.project-question {{
    font-family: 'Source Serif 4', serif;
    font-style: italic;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.35;
    margin-bottom: 0.6rem;
}}
.project-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--slate);
    margin-bottom: 0.9rem;
}}
.project-desc {{
    color: var(--slate);
    font-size: 0.9rem;
    line-height: 1.55;
}}

/* ---------- shared page components ---------- */
.metric-box {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 1rem 1.25rem;
}}
.metric-box .metric-label {{ font-size: 0.75rem; color: var(--slate); text-transform: uppercase; letter-spacing: 0.06em; }}
.metric-box .metric-value {{ font-family: 'Source Serif 4', serif; font-size: 1.4rem; font-weight: 700; margin-top: 0.15rem; }}
.metric-box .metric-caption {{ font-size: 0.75rem; color: var(--slate); margin-top: 0.15rem; }}

.formula-box {{
    background: var(--card);
    border-left: 3px solid var(--rust);
    padding: 0.9rem 1.1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--ink);
    margin: 0.5rem 0 1rem 0;
}}
.note-box {{
    background: #fbf3ec;
    border-left: 3px solid var(--rust);
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: var(--slate);
    margin: 0.75rem 0;
    border-radius: 0 4px 4px 0;
}}

.section-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.72rem;
    color: var(--rust);
    margin: 0 0 0.2rem 0;
}}
.section-title {{
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    font-size: 1.6rem;
    color: var(--ink);
    margin: 0 0 0.75rem 0;
}}

.question-placard {{
    background: var(--card);
    border: 1px solid var(--line);
    border-left: 4px solid var(--rust);
    border-radius: 0 4px 4px 0;
    padding: 1.5rem 1.75rem;
    margin: 0.25rem 0 1.5rem 0;
}}
.question-placard .placard-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.72rem;
    color: var(--rust);
    margin-bottom: 0.5rem;
}}
.question-placard .placard-text {{
    font-family: 'Source Serif 4', serif;
    font-style: italic;
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.45;
}}

.takeaway-line {{
    font-size: 0.92rem;
    color: var(--slate);
    font-style: italic;
    margin: 0.6rem 0 0 0;
}}

.so-what-box {{
    background: var(--moss-bg);
    border: 1px solid var(--moss);
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    margin: 0.5rem 0 1rem 0;
}}
.so-what-box .placard-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.72rem;
    color: var(--moss);
    margin-bottom: 0.5rem;
}}
.so-what-box .so-what-text {{
    color: var(--ink);
    font-size: 1.02rem;
    line-height: 1.55;
}}

.status-live {{ color: var(--moss); font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; }}
.status-soon {{ color: #9a8f6f; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; }}
</style>
"""


def apply_theme(page_title: str, page_icon: str):
    """Must be the first Streamlit call on the page (sets page config)."""
    st.set_page_config(page_title=f"{page_title} | Portfolio", page_icon=page_icon, layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def back_link():
    st.page_link("Home.py", label="← Back to portfolio", icon="🏠")


def divider():
    st.markdown('<hr class="divider">', unsafe_allow_html=True)


def section_heading(number: str, label: str, title: str):
    """Small numbered eyebrow + section title, e.g. 01 · THE QUESTION / <title>."""
    st.markdown(
        f'<div class="section-eyebrow">{number} · {label}</div>'
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def question_placard(text: str):
    st.markdown(
        f"""
        <div class="question-placard">
            <div class="placard-eyebrow">01 · The Question</div>
            <div class="placard-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def so_what_box(text: str, number: str = "05", label: str = "Decision Relevance"):
    st.markdown(
        f"""
        <div class="so-what-box">
            <div class="placard-eyebrow">{number} · {label}</div>
            <div class="so-what-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_box_html(label: str, value: str, caption: str = "") -> str:
    caption_html = f'<div class="metric-caption">{caption}</div>' if caption else ""
    return (
        f'<div class="metric-box"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>{caption_html}</div>'
    )
