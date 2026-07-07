"""
app.py  --  Streamlit web app for the AI Resume Screening & Candidate Ranking System

Run it from the project folder with:

    streamlit run app.py

It reuses the exact same engine as the notebook (screening_utils.py).
"""

import html

import altair as alt
import pandas as pd
import streamlit as st

from screening_utils import (
    extract_text,
    rank_candidates,
    extract_candidate_name,
    get_top_keywords,
    match_skills,
)

# ---------------------------------------------------------------- Page setup
st.set_page_config(page_title="Resume Screening", page_icon="◈", layout="wide")

# ---------------------------------------------------------------- Look & feel
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        #MainMenu, header[data-testid="stHeader"], footer {visibility: hidden;}
        .stApp {
            background: radial-gradient(circle at 15% 0%, #f4f2ee 0%, #eef0f3 45%, #e9ecf2 100%);
        }
        .block-container {
            max-width: 1080px;
            padding-top: 2.6rem;
            padding-bottom: 4rem;
        }
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #1c1f26;
        }

        /* ---------- Masthead ---------- */
        .masthead { margin-bottom: 2.4rem; }
        .masthead .eyebrow {
            font-family: 'Inter', sans-serif;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #8a6d3b;
            margin-bottom: 0.5rem;
        }
        .masthead h1 {
            font-family: 'Fraunces', serif;
            font-weight: 600;
            font-size: 2.6rem;
            line-height: 1.15;
            color: #171a21;
            margin: 0 0 0.6rem 0;
        }
        .masthead p {
            font-size: 1.02rem;
            color: #565c6b;
            max-width: 640px;
            line-height: 1.55;
            margin: 0;
        }
        .divider {
            height: 1px;
            background: linear-gradient(90deg, rgba(23,26,33,0.18), rgba(23,26,33,0));
            margin: 2.2rem 0 2rem 0;
        }

        /* ---------- Step labels ---------- */
        .step-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #a8863f;
            margin-bottom: 0.35rem;
        }
        .step-title {
            font-family: 'Fraunces', serif;
            font-size: 1.35rem;
            font-weight: 600;
            color: #171a21;
            margin-bottom: 0.9rem;
        }

        /* ---------- Buttons ---------- */
        .stButton > button {
            background: #171a21;
            color: #f6f4ef;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.6rem;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.01em;
            transition: background 0.15s ease;
        }
        .stButton > button:hover { background: #34384a; color: #ffffff; }

        /* ---------- Stat cards ---------- */
        .stat-row { display: flex; gap: 14px; margin-bottom: 2rem; flex-wrap: wrap; }
        .stat-card {
            flex: 1;
            min-width: 160px;
            background: #ffffff;
            border: 1px solid #e6e3db;
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 1px 2px rgba(23,26,33,0.04);
        }
        .stat-card .label {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #8b8f9c;
            margin-bottom: 0.4rem;
        }
        .stat-card .value {
            font-family: 'Fraunces', serif;
            font-size: 1.55rem;
            font-weight: 600;
            color: #171a21;
        }

        /* ---------- Candidate cards ---------- */
        .candidate-card {
            background: #ffffff;
            border: 1px solid #e6e3db;
            border-radius: 14px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(23,26,33,0.05);
        }
        .candidate-card.is-top {
            border: 1px solid #cbb46b;
            box-shadow: 0 4px 14px rgba(180,140,50,0.14);
        }
        .candidate-head { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.2rem; }
        .rank-badge {
            flex-shrink: 0;
            width: 34px; height: 34px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700;
            font-size: 0.95rem;
            background: #eef0f3;
            color: #565c6b;
        }
        .rank-badge.top1 { background: #171a21; color: #e9c46a; }
        .rank-badge.top2 { background: #3b3f4d; color: #f2f2f2; }
        .rank-badge.top3 { background: #56422a; color: #f2e6cf; }
        .candidate-name { font-size: 1.15rem; font-weight: 700; color: #171a21; }
        .candidate-source { font-size: 0.78rem; color: #9a9ea8; margin-left: 0.2rem; }
        .match-line { display: flex; align-items: center; gap: 0.8rem; margin: 0.7rem 0 0.9rem 0; }
        .match-pct { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.05rem; color: #171a21; min-width: 58px; }
        .bar-track { flex: 1; height: 8px; background: #eceae3; border-radius: 6px; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 6px; }
        .tier-badge {
            font-size: 0.68rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
            padding: 0.2rem 0.55rem; border-radius: 20px;
        }

        .pill-group { margin-top: 0.5rem; }
        .pill-label { font-size: 0.72rem; font-weight: 600; color: #9a9ea8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem; }
        .pill {
            display: inline-block;
            font-size: 0.8rem;
            padding: 0.22rem 0.65rem;
            border-radius: 20px;
            margin: 0 0.35rem 0.35rem 0;
            font-weight: 500;
        }
        .pill-match { background: #eaf3ea; color: #3c6e47; border: 1px solid #cfe3cf; }
        .pill-missing { background: #f4f1ec; color: #9a9488; border: 1px solid #e5e0d6; }

        /* ---------- Skill matrix ---------- */
        .matrix-wrap { overflow-x: auto; border: 1px solid #e6e3db; border-radius: 12px; background: #fff; }
        table.skill-matrix { border-collapse: collapse; width: 100%; font-size: 0.86rem; }
        table.skill-matrix th, table.skill-matrix td {
            padding: 0.55rem 0.9rem; text-align: center; border-bottom: 1px solid #efece5;
        }
        table.skill-matrix th { background: #f7f5f0; font-weight: 600; color: #565c6b; white-space: nowrap; }
        table.skill-matrix td.skill-name { text-align: left; font-weight: 500; color: #171a21; white-space: nowrap; }
        table.skill-matrix td.cell-yes { color: #3c6e47; font-weight: 700; }
        table.skill-matrix td.cell-no { color: #c9c4b8; }
        table.skill-matrix tr:last-child td { border-bottom: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- Masthead
st.markdown(
    """
    <div class="masthead">
        <div class="eyebrow">Candidate Intelligence</div>
        <h1>Resume Screening &amp; Ranking</h1>
        <p>Drop in a job description and a stack of resumes. Every candidate is scored
        against the role, matched proficiency by proficiency, and ranked so you can
        see who fits best at a glance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- Inputs
DEFAULT_JD = (
    "We are looking for a Machine Learning Engineer with strong Python skills "
    "and experience in deep learning frameworks such as TensorFlow or PyTorch. "
    "Responsibilities include data preprocessing, training machine learning and "
    "deep learning models, natural language processing, and deploying models via "
    "REST APIs. Required skills: Python, scikit-learn, NLP, neural networks, SQL."
)

col_jd, col_upload = st.columns([1.3, 1], gap="large")

with col_jd:
    st.markdown('<div class="step-label">Step 1</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Job description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Job description",
        value=DEFAULT_JD,
        height=220,
        label_visibility="collapsed",
    )

with col_upload:
    st.markdown('<div class="step-label">Step 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Upload resumes</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.caption("PDF, DOCX, or TXT · multiple files supported")

st.write("")
run_clicked = st.button("Rank Candidates", type="primary")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- Helpers
def tier_for(pct):
    """Return (label, color) for a match percentage."""
    if pct >= 75:
        return "Strong match", "#2f7a3e"
    if pct >= 50:
        return "Good match", "#a8863f"
    if pct >= 30:
        return "Partial match", "#b4762e"
    return "Weak match", "#a34c3f"


def render_pills(items, css_class):
    if not items:
        return ""
    return "".join(f'<span class="pill {css_class}">{html.escape(item)}</span>' for item in items)


def build_matrix_html(jd_keywords, ordered_names, matched_lookup):
    header_cells = "".join(f"<th>{html.escape(name)}</th>" for name in ordered_names)
    body_rows = []
    for kw in jd_keywords:
        cells = []
        for name in ordered_names:
            hit = kw in matched_lookup[name]
            css = "cell-yes" if hit else "cell-no"
            mark = "&#10003;" if hit else "&#8212;"
            cells.append(f'<td class="{css}">{mark}</td>')
        body_rows.append(f'<tr><td class="skill-name">{html.escape(kw)}</td>{"".join(cells)}</tr>')
    return (
        '<div class="matrix-wrap"><table class="skill-matrix">'
        f'<thead><tr><th>Proficiency</th>{header_cells}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        "</table></div>"
    )


# ---------------------------------------------------------------- Action
if run_clicked:

    if not job_description.strip():
        st.error("Please enter a job description first.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        resumes = {}          # unique key (filename) -> raw text
        display_names = {}    # unique key -> extracted candidate name
        for file in uploaded_files:
            try:
                text = extract_text(file, file.name)
                fallback_label = file.name.rsplit(".", 1)[0]
                resumes[file.name] = text
                display_names[file.name] = extract_candidate_name(text, fallback_label)
            except Exception as e:
                st.warning(f"Could not read {file.name}: {e}")

        if not resumes:
            st.error("None of the uploaded files could be read.")
        else:
            ranked = rank_candidates(job_description, resumes)
            jd_keywords = get_top_keywords(job_description, top_n=12)

            rows = []
            matched_lookup = {}
            for key, score in ranked:
                name = display_names[key]
                matched, missing = match_skills(jd_keywords, resumes[key])
                matched_lookup[name] = set(matched)
                rows.append(
                    {
                        "Candidate": name,
                        "Source file": key,
                        "Score": round(float(score), 4),
                        "Match %": round(float(score) * 100, 1),
                        "Matched": matched,
                        "Missing": missing,
                    }
                )

            df = pd.DataFrame(rows)
            df.index = range(1, len(df) + 1)
            df.index.name = "Rank"

            # ---------------------------------------------------- Summary
            top = df.iloc[0]
            avg_pct = round(df["Match %"].mean(), 1)

            st.markdown(
                f"""
                <div class="stat-row">
                    <div class="stat-card">
                        <div class="label">Candidates evaluated</div>
                        <div class="value">{len(df)}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Best match</div>
                        <div class="value">{html.escape(top['Candidate'])}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Top score</div>
                        <div class="value">{top['Match %']}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Average score</div>
                        <div class="value">{avg_pct}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---------------------------------------------------- Leaderboard
            st.markdown('<div class="step-label">Results</div>', unsafe_allow_html=True)
            st.markdown('<div class="step-title">Candidate ranking</div>', unsafe_allow_html=True)

            rank_classes = {1: "top1", 2: "top2", 3: "top3"}

            for i, (_, row) in enumerate(df.iterrows(), start=1):
                pct = row["Match %"]
                tier_label, tier_color = tier_for(pct)
                badge_class = rank_classes.get(i, "")
                card_class = "candidate-card is-top" if i == 1 else "candidate-card"

                matched_html = render_pills(row["Matched"], "pill-match")
                missing_html = render_pills(row["Missing"], "pill-missing")

                pill_section = ""
                if matched_html:
                    pill_section += (
                        '<div class="pill-group"><div class="pill-label">Proficiencies matched</div>'
                        f'{matched_html}</div>'
                    )
                if missing_html:
                    pill_section += (
                        '<div class="pill-group"><div class="pill-label">Not evidenced</div>'
                        f'{missing_html}</div>'
                    )

                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div class="candidate-head">
                            <div class="rank-badge {badge_class}">{i}</div>
                            <div>
                                <span class="candidate-name">{html.escape(row['Candidate'])}</span>
                                <span class="candidate-source">from {html.escape(row['Source file'])}</span>
                            </div>
                        </div>
                        <div class="match-line">
                            <div class="match-pct">{pct}%</div>
                            <div class="bar-track">
                                <div class="bar-fill" style="width:{pct}%; background:{tier_color};"></div>
                            </div>
                            <span class="tier-badge" style="background:{tier_color}1a; color:{tier_color};">{tier_label}</span>
                        </div>
                        {pill_section}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ---------------------------------------------------- Comparison chart
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="step-label">Overall comparison</div>', unsafe_allow_html=True)
            st.markdown('<div class="step-title">Score comparison</div>', unsafe_allow_html=True)

            chart = (
                alt.Chart(df[["Candidate", "Match %"]])
                .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, size=22)
                .encode(
                    x=alt.X("Match %:Q", scale=alt.Scale(domain=[0, 100]), title="Match %"),
                    y=alt.Y("Candidate:N", sort="-x", title=None),
                    color=alt.Color(
                        "Match %:Q",
                        scale=alt.Scale(scheme="goldgreen", domain=[0, 100]),
                        legend=None,
                    ),
                    tooltip=["Candidate", "Match %"],
                )
                .properties(height=alt.Step(38))
                .configure_axis(grid=False)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, use_container_width=True)

            # ---------------------------------------------------- Proficiency matrix
            if jd_keywords:
                st.markdown('<div class="step-label">Overall comparison</div>', unsafe_allow_html=True)
                st.markdown('<div class="step-title">Proficiency matrix</div>', unsafe_allow_html=True)
                ordered_names = list(df["Candidate"])
                st.markdown(
                    build_matrix_html(jd_keywords, ordered_names, matched_lookup),
                    unsafe_allow_html=True,
                )

            # ---------------------------------------------------- Raw scores
            with st.expander("View detailed scores"):
                st.dataframe(
                    df[["Candidate", "Source file", "Score", "Match %"]],
                    use_container_width=True,
                )
