"""FinLingo Research Studio — a focused five-tab entrypoint for the existing app.

Run: python3 -m streamlit run finlingo_studio.py
This file leaves the original app.py and every existing imported response intact.
"""

from __future__ import annotations

from collections import defaultdict
import json
import urllib.error
import urllib.request

import altair as alt
import pandas as pd
import streamlit as st

import finlingo_metrics as fm


st.set_page_config(page_title="FinLingo | Research Studio", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
  /* This single-page studio already has its own navigation tabs. Preserve the
     original Streamlit pages on disk, but hide their duplicate sidebar links. */
  [data-testid="stSidebar"], [data-testid="stSidebarNav"],
  [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"]
      {display:none !important;}
  .block-container {max-width: 1500px; padding-top: 4.5rem !important;}
  [data-testid="stMetric"] {border:1px solid rgba(127,152,185,.19); border-radius: 11px; padding: 14px 16px;}
  div[data-testid="stAlert"] {border-radius: 10px;}
  div[data-testid="stAlert"] p {line-height:1.65;}
  .research-eyebrow {display:block; letter-spacing:.10em; text-transform:uppercase;
                     color:#a3c8ff; font-weight:700; font-size:.86rem;
                     line-height:1.8; padding:.4rem 0 .25rem 0; margin:0;
                     white-space:normal; overflow:visible;}
  .research-subtitle {color:#b5c0cf; max-width: 1100px; margin-top:.25rem; margin-bottom:1.1rem; line-height:1.6;}
  div[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:14px;}
  /* The default secondary text becomes near-invisible on some dark themes. */
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
      color:#bac9db !important; opacity:1 !important;}
  [data-testid="stDataFrame"] {border-radius:12px; overflow:hidden;}
  .studio-insight {border:1px solid #354d6b; border-left:4px solid #5faaff;
      padding:18px 22px; margin:12px 0 20px 0; border-radius:13px;
      background:linear-gradient(130deg, rgba(29,52,83,.96),rgba(19,31,48,.94));}
  .studio-insight .label {color:#98d2ff; text-transform:uppercase;
      font-size:.75rem; font-weight:750; letter-spacing:.12em; margin-bottom:8px;}
  .studio-insight .copy {font-size:1.02rem; color:#ecf5ff; line-height:1.65;}
  .studio-chart-note {font-size:.9rem; color:#c1d2e3; line-height:1.5;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="research-eyebrow">Finance · LLM evaluation · reproducible research</div>', unsafe_allow_html=True)
st.title("FinLingo  /  Research Studio")
st.markdown('<div class="research-subtitle">Collect answers · run an independent local evaluator · compare models question by question · export defensible results.</div>', unsafe_allow_html=True)

try:
    gold, prompts, _ = fm.sources()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"Cannot locate the existing FinLingo datasets: {exc}")
    st.info("Extract this small patch into the SAME directory as your existing app.py, data/ and model_answers/.")
    st.stop()

try:
    responses = fm.all_answers(prompts, gold)
    saved_scores = fm.score_index()
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Cannot load your saved responses or score CSV: {exc}")
    st.stop()

model_set = sorted({key[0] for key in responses})
supported_models = list(dict.fromkeys(["ChatGPT", "Gemini", "Claude", "Qwen", "Grok", "DeepSeek"] + model_set))
judge_default = "qwen2.5:3b"
count_pilot = sum(r["phase"] == "pilot" for r in responses.values())
count_final = sum(r["phase"] == "final" for r in responses.values())

header_a, header_b, header_c = st.columns(3)
header_a.metric("Answers stored", f"{len(responses)}", help="Unique model + prompt + trial records, after taking the latest import")
header_b.metric("Pilot responses", count_pilot)
header_c.metric("Final-study responses", count_final)

tab_import, tab_evaluate, tab_results, tab_explore, tab_method = st.tabs([
    "01  Import", "02  Automatic evaluation", "03  Results dashboard",
    "04  Question explorer", "05  Methodology & instructions",
])


def main_scores(phase: str, trial: int, judge: str):
    return fm.scored_view(responses, saved_scores, phase, trial, judge)


def show_status(phase: str, trial: int, judge: str):
    matching = [r for r in responses.values() if r["phase"] == phase and r["trial"] == trial]
    judged = sum((fm.current_score(r, saved_scores, judge) or {}).get("status") == "judged" for r in matching)
    sim = sum((fm.current_score(r, saved_scores, judge) or {}).get("status") == "text_similarity_only" for r in matching)
    st.progress((judged + sim) / len(matching) if matching else 0,
                text=f"{judged} full judgments · {sim} text-only scores · {len(matching)} stored responses")
    return judged, len(matching)


REPORT_COLUMN_LABELS = {
    "model": "AI model",
    "condition": "Prompt condition",
    "category": "Question category",
    "collected_n": "Answers collected (n)",
    "auto_graded_n": "Fully judged (n)",
    "factual_full_pct": "Fully factually correct (%)",
    "factual_credit_pct": "Mean factual credit (%)",
    "completeness_pct": "Essential-point coverage (%)",
    "numerical_correct_pct": "Correct numeric results (%)",
    "numeric_n": "Measurable numeric answers (n)",
    "risk_quality_pct": "Risk reasoning credit (%)",
    "risk_n": "Applicable risk answers (n)",
    "flagged_claim_pct": "Potential contradictions (%)",
    "hallucination_decided_n": "Decidable evidence checks (n)",
    "hallucination_uncertain_n": "Uncertain evidence checks (n)",
    "rouge_l_f1": "Mean ROUGE-L F1",
}


def display_research_summary(raw_rows):
    """Human-readable headers; never modifies the saved/exported research data."""
    return pd.DataFrame(raw_rows).rename(columns=REPORT_COLUMN_LABELS)


def local_research_commentary(model: str, summary: list[dict],
                              paired: list[dict], overall: dict) -> str:
    """Optional interpretation of aggregates only; never grades or edits source data."""
    data = {
        "model_summaries": summary,
        "matched_context": paired,
        "evaluation_coverage": overall,
        "limitations": [
            "The reference answers are provisional.",
            "The small local judge may be biased or mistaken.",
            "A high proportion of 2/2 scores may reflect a ceiling effect.",
            "Uncertain evidence flags are neither proof nor absence of hallucination.",
            "Context comparisons are associations, not clean causal effects.",
        ],
    }
    payload = {
        "model": model, "stream": False,
        "options": {"temperature": 0, "num_predict": 400},
        "messages": [
            {"role": "system", "content": (
                "You are a cautious financial NLP research analyst. Interpret ONLY the "
                "provided aggregate data. Write three short numbered findings with numbers "
                "and denominators, then one research limitation. Explicitly separate "
                "observations from interpretation. Never invent statistics, sources, "
                "confidence intervals, significance, model versions or causal claims. "
                "Never call uncertain evidence flags proven hallucinations. "
                "Never rank a model from unpaired conditions or unequal denominators."
            )},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False, default=str)},
        ],
    }
    request = urllib.request.Request(
        fm.OLLAMA_URL + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as reply:
        result = json.load(reply)
    return str(result.get("message", {}).get("content", "")).strip()


with tab_import:
    st.subheader("Import model responses")
    st.caption("Your original importer and existing responses.csv remain untouched. Import only missing booklets here.")
    booklet_list = fm.manifest()
    if not booklet_list:
        st.warning("No booklet_manifest.json found. Your already imported responses still work in the other tabs.")
    else:
        one, two = st.columns(2)
        with one:
            chosen_booklet_id = st.selectbox("Booklet", [r["booklet_id"] for r in booklet_list], key="imp_booklet")
            raw_model = st.selectbox("Model", ["GPT", "Gemini", "Claude", "Qwen", "Grok", "DeepSeek"], key="imp_model",
                                     help="GPT and ChatGPT are shown together in the results.")
            version = st.text_input("Model version (as displayed)", placeholder="e.g. exact product model name", key="imp_version")
        with two:
            trial = st.selectbox("Independent trial", [1, 2, 3], key="imp_trial")
            mode = st.text_input("Chat settings", value="Manual chat; tools disabled if available", key="imp_mode")
        booklet = next(r for r in booklet_list if r["booklet_id"] == chosen_booklet_id)
        st.info(f"**{chosen_booklet_id}**  ·  {len(booklet['run_ids'])} expected answers  ·  "
                f"`{booklet.get('pdf', 'PDF booklet')}`")
        uploaded = st.file_uploader("Upload a complete .txt / .md response (optional)", type=["txt", "md"], key="imp_file")
        typed = st.text_area("Or paste the entire model response", height=220,
                             placeholder="[[A01-NOCTX]]\nFull model answer...\n[[/A01-NOCTX]]", key="imp_text")
        raw = uploaded.getvalue().decode("utf-8-sig", errors="replace") if uploaded else typed
        if raw.strip():
            preview = fm.parse_booklet(raw, booklet["run_ids"])
            st.write(f"**Recognized:** {len(preview['answers'])}/{len(booklet['run_ids'])} answers")
            for issue in ("missing", "duplicates", "unexpected", "empty", "malformed"):
                if preview[issue]:
                    st.warning(f"{issue.title()}: {', '.join(preview[issue][:15])}")
            with st.expander("Preview parsed answer IDs"):
                st.write(list(preview["answers"].keys()))
        if st.button("Import recognized answers", type="primary", key="imp_save"):
            if not raw.strip():
                st.error("Paste or upload answers first.")
            else:
                try:
                    result = fm.import_answers(booklet, raw_model, version, mode, trial, raw)
                    st.success(f"Saved {result['saved']} new answers. "
                               f"Skipped {len(result['skipped'])} existing answers without replacing anything.")
                    if result["missing"]:
                        st.warning("Missing IDs: " + ", ".join(result["missing"]))
                    st.rerun()
                except (OSError, ValueError) as exc:
                    st.error(str(exc))
    st.divider()
    st.markdown("**Existing imports at a glance**")
    tally = defaultdict(int)
    for row in responses.values():
        tally[(row["model"], row["phase"], row["condition"])] += 1
    if tally:
        st.dataframe([{"model": k[0], "phase": k[1], "condition": k[2], "responses": v}
                      for k, v in sorted(tally.items())], hide_index=True, use_container_width=True)
    st.caption(f"Source: `{fm.response_path().relative_to(fm.ROOT)}`. Appending new answers never changes old rows.")


with tab_evaluate:
    st.subheader("Automatic grading, with a saved evidence trail")
    st.caption("The expensive step runs once per answer. Every evaluated answer is immediately saved, so a stopped batch can resume.")
    e1, e2, e3 = st.columns(3)
    with e1:
        phase = st.selectbox("Study phase", ["final", "pilot"], key="eval_phase")
    with e2:
        trial = st.selectbox("Trial", [1, 2, 3], key="eval_trial")
    with e3:
        eligible_models = sorted({r["model"] for r in responses.values() if r["phase"] == phase and r["trial"] == trial})
        chosen_models = st.multiselect("Models to grade", eligible_models, default=eligible_models, key="eval_models")
    mode = st.radio("Evaluation method", ["Full local AI judge", "Quick ROUGE-L only"], horizontal=True,
                    help="ROUGE-L measures text overlap, not correctness or hallucination detection.")
    use_judge = mode == "Full local AI judge"
    judge = st.text_input("Local judge model", judge_default, disabled=not use_judge, key="eval_judge") if use_judge else "reference_only"
    st.info("**Full grading:** factual correctness (0–2), completeness (0–2), verifiable numeric answer "
            "(0/2 or unmeasurable), risk reasoning (0–2 when applicable), suspected hallucinations "
            "(yes/no/uncertain) and ROUGE-L. No commercial API keys.")
    connected_models = fm.installed_ollama_models() if use_judge else []
    ready = not use_judge or judge in connected_models
    if use_judge and not ready:
        st.error(f"Cannot connect to local judge `{judge}`. Install and open Ollama on your Mac, "
                 f"then in VS Code Terminal run `ollama pull {judge}`. Return here and rerun the app.")
    elif use_judge:
        st.success(f"Local judge ready: {judge}")
    selected = [r for r in responses.values() if r["phase"] == phase and r["trial"] == trial
                and r["model"] in chosen_models]
    pending = [r for r in selected if not (fm.current_score(r, saved_scores, judge) or {}).get("status")
               in ("judged", "text_similarity_only")]
    completed = len(selected) - len(pending)
    st.progress(completed / len(selected) if selected else 0,
                text=f"{completed}/{len(selected)} already scored  ·  {len(pending)} remaining")
    col_lim, col_run = st.columns([2, 1])
    with col_lim:
        limit = st.selectbox("Batch size (resume anytime)", [10, 20, 50, 100, "All remaining"], index=2, key="eval_batch")
    take = len(pending) if limit == "All remaining" else min(int(limit), len(pending))
    with col_run:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        begin = st.button(f"Grade next {take}", type="primary", disabled=not ready or take == 0, key="eval_run")
    if begin:
        progress = st.progress(0, text=f"Starting {take} cases...")
        errors = 0
        for i, answer in enumerate(pending[:take], 1):
            row = fm.evaluate_one(answer, judge)
            fm.append_score(row)
            errors += row.get("status") == "judge_error"
            progress.progress(i / take, text=f"{i}/{take}  ·  {answer['model']}  ·  {answer['run_id']}")
        if errors:
            st.warning(f"{errors} judge errors were logged, not incorrectly marked wrong. You can retry them.")
        st.success(f"Saved {take} results in `evaluation/auto_scores.csv`. Resumable.")
        st.rerun()
    st.divider()
    st.markdown("#### Evaluation checkpoint")
    current = main_scores(phase, trial, judge)
    if current:
        st.dataframe([{"Model": x["model"], "ID": x["run_id"], "Condition": x["condition"],
                       "Status": x["status"], "Factual / 2": x["factual_score"],
                       "Completeness / 2": x["completeness_score"],
                       "ROUGE-L": x["rouge_l_f1"]} for x in current], hide_index=True,
                     use_container_width=True, height=330)
        st.download_button("Download checkpoint CSV", fm.to_csv(current), "FinLingo_Evaluation_Checkpoint.csv",
                           "text/csv", key="eval_download")
    st.warning("An Ollama grade is a **provisional LLM-as-judge estimate**, not independently verified "
               "financial truth. Check disputed claims and important calculations against primary sources "
               "before reporting your findings.")


with tab_results:
    st.subheader("Research Results · Evidence Lab")
    st.caption("Explore the benchmark from the big picture down to individual answers. "
               "Every visual uses your saved evaluation data, not invented scores.")
    r1, r2, r3 = st.columns(3)
    with r1:
        report_phase = st.selectbox("Dataset", ["final", "pilot"], key="results_phase")
    with r2:
        report_trial = st.selectbox("Trial", [1, 2, 3], key="results_trial")
    with r3:
        report_judge = st.selectbox(
            "Scoring source", [judge_default, "reference_only"],
            format_func=lambda x: "Local AI judge · " + x if x != "reference_only"
            else "ROUGE-L only · text overlap", key="results_judge")

    rows = main_scores(report_phase, report_trial, report_judge)
    judged = [r for r in rows if r["status"] == "judged"]
    text_only = [r for r in rows if r["status"] == "text_similarity_only"]
    table = fm.summaries(responses, saved_scores, report_phase, report_trial, report_judge)
    by_category = fm.summaries(responses, saved_scores, report_phase, report_trial, report_judge, category=True)
    context = (fm.paired_context(responses, saved_scores, report_trial, report_judge)
               if report_phase == "final" and report_judge != "reference_only" else [])

    def points(row, key):
        raw = row.get(key)
        return int(raw) if str(raw) in ("0", "1", "2") else None

    factual_grades = [points(r, "factual_score") for r in judged]
    factual_grades = [v for v in factual_grades if v is not None]
    coverage_grades = [points(r, "completeness_score") for r in judged]
    coverage_grades = [v for v in coverage_grades if v is not None]
    numeric_grades = [points(r, "numerical_score") for r in judged]
    numeric_grades = [v for v in numeric_grades if v is not None]
    full_count = sum(v == 2 for v in factual_grades)
    full_pct = 100 * full_count / len(factual_grades) if factual_grades else None
    coverage_pct = 50 * sum(coverage_grades) / len(coverage_grades) if coverage_grades else None
    numeric_pct = 100 * sum(v == 2 for v in numeric_grades) / len(numeric_grades) if numeric_grades else None
    completed = len(judged) if report_judge != "reference_only" else len(text_only)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Evaluation coverage", f"{completed}/{len(rows)}" if rows else "0/0",
              help="Successfully processed answers out of collected answers, not out of all possible models.")
    k2.metric("Fully correct · judge estimate", f"{full_pct:.1f}%" if full_pct is not None else "—",
              help=f"{full_count} full-credit results among {len(factual_grades)} evaluated answers.")
    k3.metric("Essential-point coverage", f"{coverage_pct:.1f}%" if coverage_pct is not None else "—",
              help="Mean completeness rubric score divided by 2; judged answers only.")
    k4.metric("Numerical match rate", f"{numeric_pct:.1f}%" if numeric_pct is not None else "—",
              help=f"Based on {len(numeric_grades)} measurable answers; not all answers involve calculations.")
    if rows and len(judged) < len(rows) and report_judge != "reference_only":
        st.caption(f"{len(rows) - len(judged)} answer(s) are pending or had evaluator errors. "
                   "They are excluded from the correctness percentages.")

    if not rows:
        st.info("No answers have been collected for this dataset and trial yet.")
    else:
        panel_overview, panel_categories, panel_context, panel_evidence = st.tabs([
            "01  Detailed research insights", "02  Model & category analysis",
            "03  Context comparison", "04  Detailed results & exports"
        ])
        condition_labels = {
            "no_added_context": "No added context",
            "verified_context": "Supplied context",
            "pilot_no_added_context": "Pilot · no context",
        }
        palette = {"No added context": "#5D9FFF", "Supplied context": "#AB91FF",
                   "Pilot · no context": "#4ED8BC"}
        evaluated_summaries = [r for r in table if r.get("auto_graded_n", 0) > 0]

        with panel_overview:
            st.markdown("### The research story in one view")
            if evaluated_summaries:
                baseline = [r for r in evaluated_summaries if r["condition"] == "no_added_context"
                            and r["factual_full_pct"] is not None]
                baseline.sort(key=lambda r: r["factual_full_pct"], reverse=True)
                if len(baseline) >= 2:
                    a, b = baseline[:2]
                    observation = (
                        f"<b>{a['model']}</b> received full factual credit on "
                        f"<b>{a['factual_full_pct']:.1f}%</b> of its "
                        f"<b>{a['auto_graded_n']}</b> assessed no-context answers, compared with "
                        f"<b>{b['factual_full_pct']:.1f}%</b> across "
                        f"<b>{b['auto_graded_n']}</b> for <b>{b['model']}</b>. "
                        "This is a local-judge comparison against provisional reference answers."
                    )
                else:
                    observation = ("The current selection has graded responses. "
                                   "Add more assessed models for a direct model comparison.")
                st.markdown(
                    '<div class="studio-insight"><div class="label">Observed finding</div>'
                    f'<div class="copy">{observation}</div></div>', unsafe_allow_html=True)

                bar_rows = [{
                    "Model": r["model"],
                    "Condition": condition_labels.get(r["condition"], r["condition"]),
                    "Full factual credit (%)": r["factual_full_pct"],
                    "Judged answers (n)": r["auto_graded_n"],
                } for r in evaluated_summaries if r["factual_full_pct"] is not None]
                st.markdown("#### Full factual-credit rate by model and context")
                chart = (alt.Chart(pd.DataFrame(bar_rows))
                         .mark_bar(cornerRadiusEnd=6, size=30)
                         .encode(
                             x=alt.X("Model:N", title=None),
                             xOffset=alt.XOffset("Condition:N"),
                             y=alt.Y("Full factual credit (%):Q",
                                     scale=alt.Scale(domain=[0, 100]),
                                     axis=alt.Axis(values=[0, 25, 50, 75, 100]),
                                     title="Fully correct among judged (%)"),
                             color=alt.Color("Condition:N", title="Prompt condition",
                                             scale=alt.Scale(
                                                 domain=list(palette), range=list(palette.values()))),
                             tooltip=["Model:N", "Condition:N",
                                      alt.Tooltip("Full factual credit (%):Q", format=".1f"),
                                      "Judged answers (n):Q"])
                         .properties(height=315)
                         .configure_view(strokeOpacity=0)
                         .configure_axis(gridColor="#35445B", labelColor="#D6E5F6",
                                         titleColor="#E7F2FF", labelFontSize=12)
                         .configure_legend(labelColor="#E6F3FF", titleColor="#E6F3FF"))
                st.altair_chart(chart, use_container_width=True, theme=None)
                st.caption("Conditions have different question sets. Use the **Context comparison** "
                           "tab for matched-question changes.")
            else:
                st.info("Run the full local AI judge to unlock factual correctness charts. "
                        "ROUGE-L-only values are not accuracy scores.")

            if factual_grades:
                dist_left, dist_right = st.columns([1.3, 1], gap="large")
                distribution = [
                    {"Verdict": "Fully correct (2/2)", "Count": sum(s == 2 for s in factual_grades)},
                    {"Verdict": "Partly correct (1/2)", "Count": sum(s == 1 for s in factual_grades)},
                    {"Verdict": "Incorrect (0/2)", "Count": sum(s == 0 for s in factual_grades)},
                ]
                verdict_palette = {
                    "Fully correct (2/2)": "#35C7A5",
                    "Partly correct (1/2)": "#F3C16A",
                    "Incorrect (0/2)": "#F07985",
                }
                with dist_left:
                    st.markdown("#### Distribution of factual scores")
                    verdict_chart = (alt.Chart(pd.DataFrame(distribution))
                                     .mark_bar(cornerRadiusEnd=6, size=34)
                                     .encode(
                                         y=alt.Y("Verdict:N", sort=list(verdict_palette), title=None),
                                         x=alt.X("Count:Q", title="Assessed model answers (n)"),
                                         color=alt.Color("Verdict:N", title=None,
                                                         scale=alt.Scale(
                                                             domain=list(verdict_palette),
                                                             range=list(verdict_palette.values())),
                                                         legend=None),
                                         tooltip=["Verdict:N", "Count:Q"])
                                     .properties(height=260)
                                     .configure_view(strokeOpacity=0)
                                     .configure_axis(gridColor="#35445B",
                                                     labelColor="#DCEAF8", titleColor="#E9F3FF",
                                                     labelFontSize=12))
                    st.altair_chart(verdict_chart, use_container_width=True, theme=None)
                with dist_right:
                    st.markdown("#### Share of evaluated answers")
                    donut = (alt.Chart(pd.DataFrame([d for d in distribution if d["Count"]]))
                             .mark_arc(innerRadius=72, outerRadius=110, padAngle=0.016,
                                       stroke="#172433", strokeWidth=2)
                             .encode(
                                 theta=alt.Theta("Count:Q"),
                                 color=alt.Color("Verdict:N", title=None,
                                                 scale=alt.Scale(domain=list(verdict_palette),
                                                                 range=list(verdict_palette.values()))),
                                 tooltip=["Verdict:N", "Count:Q"])
                             .properties(height=260)
                             .configure_view(strokeOpacity=0)
                             .configure_legend(labelColor="#E8F2FE", labelFontSize=12))
                    st.altair_chart(donut, use_container_width=True, theme=None)
                st.caption(f"Denominator: {len(factual_grades)} fully evaluated responses. "
                           "Unprocessed answers are excluded, never placed in the 'incorrect' category.")

            with st.expander("Important data-quality checks"):
                st.write(f"**Evaluation status:** {len(judged)} full judgments, "
                         f"{len(text_only)} text-only results and "
                         f"{len(rows) - len(judged) - len(text_only)} pending/error records.")
                st.write(f"**Quantitative coverage:** {len(numeric_grades)} measurable numerical "
                         f"results among {len(judged)} full judgments.")
                risk_cases = [r for r in judged if r["category"] == "Risk and uncertainty"]
                risk_scored = sum(points(r, "risk_score") is not None for r in risk_cases)
                if risk_cases:
                    st.write(f"**Risk rubric coverage:** {risk_scored}/{len(risk_cases)} "
                             "risk-category responses have a usable risk score. Missing values are not zero.")
                uncertain_flags = sum(r.get("hallucination") == "uncertain" for r in judged)
                st.write(f"**Unresolved evidence flags:** {uncertain_flags}/{len(judged)} "
                         "full judgments. These cannot be called confirmed hallucinations.")
                if full_pct is not None and full_pct >= 90:
                    st.write("**Possible ceiling effect:** most responses receive full factual "
                             "credit. The rubric or the local evaluator may not distinguish "
                             "strong responses as finely as you want.")
                st.caption("Reference answers are still provisional unless individually validated "
                           "against authoritative financial statements.")

            st.divider()
            st.markdown("### Local AI · Research interpretation")
            st.caption("Optional. Ask your existing Ollama judge to interpret **aggregated results only**. "
                       "This does not re-evaluate responses, modify scores, or contact a paid API.")
            if report_judge == "reference_only":
                st.info("Select the full local AI judge to enable the research interpretation.")
            else:
                insight_key = f"results_insight_{report_phase}_{report_trial}_{report_judge}_{len(judged)}"
                if st.button("Generate an AI research summary", key="results_ai_button",
                             disabled=not judged, type="primary"):
                    try:
                        with st.spinner("Your local AI is reading the aggregate research metrics..."):
                            st.session_state[insight_key] = local_research_commentary(
                                report_judge, table, context,
                                {"collected": len(rows), "fully_judged": len(judged),
                                 "full_factual_credit": full_count,
                                 "measurable_numeric_n": len(numeric_grades)})
                    except (OSError, ValueError, KeyError, TimeoutError) as exc:
                        st.error(f"The local model could not create the summary: {exc}")
                if st.session_state.get(insight_key):
                    st.markdown(st.session_state[insight_key])
                    st.caption("AI-generated interpretation, not an independent statistical test. "
                               "Check every number against the tables before citing it.")

        with panel_categories:
            st.markdown("### Where does each model perform well or poorly?")
            st.caption("Category heatmaps make weaknesses easier to spot. The sample size for each "
                       "cell appears on hover; cells without valid measurements are left blank.")
            measure_options = {
                "Full factual-credit rate": ("factual_full_pct", "auto_graded_n"),
                "Essential-point coverage": ("completeness_pct", "auto_graded_n"),
                "Numerical accuracy": ("numerical_correct_pct", "numeric_n"),
                "Risk reasoning credit": ("risk_quality_pct", "risk_n"),
            }
            selected_metric = st.selectbox("Analyze this metric", list(measure_options),
                                           key="results_category_measure")
            metric_field, sample_field = measure_options[selected_metric]
            heatmap_rows = []
            for item in by_category:
                value = item.get(metric_field)
                n = item.get(sample_field) or 0
                if value is None or not n:
                    continue
                heatmap_rows.append({
                    "Category": item["category"],
                    "Model & condition": item["model"] + " · " +
                                         condition_labels.get(item["condition"], item["condition"]),
                    "Percent": float(value),
                    "Label": f"{float(value):.0f}%",
                    "Sample (n)": int(n),
                })
            if heatmap_rows:
                heat = pd.DataFrame(heatmap_rows)
                heat_base = alt.Chart(heat).encode(
                    x=alt.X("Category:N", title=None,
                            axis=alt.Axis(labelAngle=-25, labelLimit=220)),
                    y=alt.Y("Model & condition:N", title=None),
                    tooltip=["Model & condition:N", "Category:N",
                             alt.Tooltip("Percent:Q", format=".1f"),
                             "Sample (n):Q"])
                rectangles = heat_base.mark_rect(cornerRadius=6, stroke="#141D2B", strokeWidth=3).encode(
                    color=alt.Color("Percent:Q", title="Score (%)",
                                    scale=alt.Scale(domain=[0, 50, 100],
                                                    range=["#B85162", "#EDB45C", "#22A68B"])))
                labels = heat_base.mark_text(color="#FFFFFF", fontSize=13,
                                             fontWeight="bold").encode(text="Label:N")
                st.altair_chart(
                    (rectangles + labels).properties(height=max(190, heat["Model & condition"].nunique() * 65))
                    .configure_view(strokeOpacity=0)
                    .configure_axis(labelColor="#E3EFFB", titleColor="#E3EFFB", labelFontSize=12)
                    .configure_legend(labelColor="#E3EFFB", titleColor="#E3EFFB"),
                    use_container_width=True, theme=None)
                if selected_metric == "Risk reasoning credit":
                    st.warning("Few risk scores may be available because the local judge sometimes "
                               "returns N/A. A blank cell is not a zero-risk score.")
                elif selected_metric == "Numerical accuracy":
                    st.caption("Only gold-comparable final numbers and units are included; "
                               "non-numerical questions are excluded.")
            else:
                st.info("No measurable scores are available for this criterion yet. "
                        "Choose another metric to explore.")

            difficulty_groups = defaultdict(list)
            for assessed in judged:
                gold_row = gold.get(assessed["question_id"], {})
                difficulty = str(gold_row.get("difficulty") or
                                 gold_row.get("difficulty_level") or "").strip().title()
                value = points(assessed, "factual_score")
                if difficulty in ("Easy", "Medium", "Hard") and value is not None:
                    difficulty_groups[(assessed["model"], assessed["condition"], difficulty)].append(value)
            if difficulty_groups:
                st.markdown("### How does performance vary with question difficulty?")
                difficulty_rows = [
                    {"Difficulty": difficulty,
                     "Model & condition": model + " · " +
                                          condition_labels.get(condition, condition),
                     "Fully correct (%)": round(100 * sum(s == 2 for s in scores) / len(scores), 1),
                     "Judged (n)": len(scores)}
                    for (model, condition, difficulty), scores in difficulty_groups.items()
                ]
                difficulty_chart = (
                    alt.Chart(pd.DataFrame(difficulty_rows))
                    .mark_line(point=alt.OverlayMarkDef(filled=True, size=95),
                               strokeWidth=3)
                    .encode(
                        x=alt.X("Difficulty:N", sort=["Easy", "Medium", "Hard"],
                                title="Question difficulty"),
                        y=alt.Y("Fully correct (%):Q", scale=alt.Scale(domain=[0, 100]),
                                title="Fully correct among judged (%)"),
                        color=alt.Color("Model & condition:N", title="Model and condition"),
                        tooltip=["Model & condition:N", "Difficulty:N",
                                 alt.Tooltip("Fully correct (%):Q", format=".1f"),
                                 "Judged (n):Q"])
                    .properties(height=300)
                    .configure_view(strokeOpacity=0)
                    .configure_axis(gridColor="#31425B", labelColor="#DDEBF8",
                                    titleColor="#E7F2FF", labelFontSize=12)
                    .configure_legend(labelColor="#E7F2FF", titleColor="#E7F2FF")
                )
                st.altair_chart(difficulty_chart, use_container_width=True, theme=None)
                st.caption("The line follows the ordered easy → medium → hard difficulty levels. "
                           "It is not a time trend; each point uses its own judged-answer denominator.")

            st.markdown("### Compare models within one condition")
            conditions_available = list(dict.fromkeys(item["condition"] for item in table))
            chosen_condition = st.selectbox(
                "Prompt condition", conditions_available,
                format_func=lambda value: condition_labels.get(value, value),
                key="results_condition_table")
            focused = [item for item in table if item["condition"] == chosen_condition]
            display_columns = [
                "AI model", "Answers collected (n)", "Fully judged (n)",
                "Fully factually correct (%)", "Mean factual credit (%)",
                "Essential-point coverage (%)", "Correct numeric results (%)",
                "Measurable numeric answers (n)",
            ]
            frame = display_research_summary(focused)
            st.dataframe(frame[[c for c in display_columns if c in frame]],
                         hide_index=True, use_container_width=True)
            st.caption("The sample counts (n) are part of the result. A difference of a few "
                       "percentage points is not automatically statistically significant.")
            with st.expander("All category-level measures, denominators and evidence flags"):
                st.dataframe(display_research_summary(by_category), hide_index=True,
                             use_container_width=True, height=330)

        with panel_context:
            st.markdown("### What changed when financial context was supplied?")
            st.caption("For each model, this analysis pairs the **same original question** with "
                       "its no-context and supplied-context versions. No unmatched questions enter the chart.")
            if context:
                paired_rows = []
                for entry in context:
                    paired_rows.extend([
                        {"Model": entry["model"], "Condition": "No added context",
                         "Full credit (%)": entry["no_context_full_pct"],
                         "Pairs (n)": entry["matched_questions_n"]},
                        {"Model": entry["model"], "Condition": "Supplied context",
                         "Full credit (%)": entry["with_context_full_pct"],
                         "Pairs (n)": entry["matched_questions_n"]},
                    ])
                line = (alt.Chart(pd.DataFrame(paired_rows))
                        .mark_line(point=alt.OverlayMarkDef(filled=True, size=145),
                                   strokeWidth=4)
                        .encode(
                            x=alt.X("Condition:N",
                                    sort=["No added context", "Supplied context"], title=None),
                            y=alt.Y("Full credit (%):Q", scale=alt.Scale(domain=[0, 100]),
                                    title="Fully correct on matched questions (%)"),
                            color=alt.Color("Model:N", title="Candidate model",
                                            scale=alt.Scale(scheme="tableau10")),
                            tooltip=["Model:N", "Condition:N",
                                     alt.Tooltip("Full credit (%):Q", format=".1f"),
                                     "Pairs (n):Q"])
                        .properties(height=340)
                        .configure_view(strokeOpacity=0)
                        .configure_axis(gridColor="#31425B",
                                        labelColor="#DDEBF8", titleColor="#E7F2FF", labelFontSize=13)
                        .configure_legend(labelColor="#E4F0FF"))
                st.altair_chart(line, use_container_width=True, theme=None)
                st.caption("The connecting lines show a change between two test conditions, "
                           "**not** a time trend or a causal effect.")

                change_rows = []
                change_colors = {"Improved": "#3DD1AD", "Unchanged": "#6B89B4",
                                 "Worsened": "#F17986"}
                for entry in context:
                    for label, field in (("Improved", "improved_n"),
                                         ("Unchanged", "unchanged_n"),
                                         ("Worsened", "worsened_n")):
                        change_rows.append({"Model": entry["model"], "Outcome": label,
                                            "Questions": entry[field]})
                st.markdown("#### Direction of change for individual questions")
                movement = (alt.Chart(pd.DataFrame(change_rows))
                            .mark_bar(cornerRadiusEnd=3, size=35)
                            .encode(
                                y=alt.Y("Model:N", title=None),
                                x=alt.X("Questions:Q", stack="normalize",
                                        axis=alt.Axis(format=".0%"),
                                        title="Share of matched questions"),
                                color=alt.Color("Outcome:N", title=None,
                                                scale=alt.Scale(domain=list(change_colors),
                                                                range=list(change_colors.values()))),
                                tooltip=["Model:N", "Outcome:N", "Questions:Q"])
                            .properties(height=max(145, len(context) * 78))
                            .configure_view(strokeOpacity=0)
                            .configure_axis(gridColor="#31425B", labelColor="#DDEBF8",
                                            titleColor="#E7F2FF", labelFontSize=12)
                            .configure_legend(labelColor="#E7F2FF"))
                st.altair_chart(movement, use_container_width=True, theme=None)
                st.caption("Improved/worsened counts refer to the **0–2 factual rubric score**. "
                           "They need not match changes in the binary full-credit percentage.")

                paired_table = [{
                    "AI model": p["model"], "Matched questions (n)": p["matched_questions_n"],
                    "No-context full credit (%)": p["no_context_full_pct"],
                    "With-context full credit (%)": p["with_context_full_pct"],
                    "Difference (percentage points)": p["context_change_pp"],
                    "Improved (n)": p["improved_n"], "Unchanged (n)": p["unchanged_n"],
                    "Worsened (n)": p["worsened_n"],
                } for p in context]
                st.dataframe(paired_table, hide_index=True, use_container_width=True)
                st.info("**Interpretation limit:** If context and no-context questions were "
                        "answered in the same conversation, earlier prompts may have influenced "
                        "later answers. Differences are associations, not experimentally isolated effects.")
            else:
                st.info("A matched context comparison requires both versions of each original "
                        "question to be fully evaluated. Pilot and text-overlap-only runs do not qualify.")

        with panel_evidence:
            st.markdown("### Detailed results · filter, inspect, export")
            st.caption("All stored answers in this selection are available here. "
                       "You can locate a low-scoring question and read the judge's actual rationale.")
            available_models = sorted({r["model"] for r in rows})
            available_conditions = sorted({r["condition"] for r in rows})
            available_categories = sorted({r["category"] for r in rows})
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                model_filter = st.selectbox("Model", ["All models"] + available_models,
                                            key="result_detail_model")
            with f2:
                condition_filter = st.selectbox(
                    "Condition", ["All conditions"] + available_conditions,
                    format_func=lambda value: condition_labels.get(value, value),
                    key="result_detail_condition")
            with f3:
                category_filter = st.selectbox("Category", ["All categories"] + available_categories,
                                               key="result_detail_category")
            with f4:
                status_filter = st.selectbox(
                    "Evaluation status", ["All statuses"] + sorted({r["status"] for r in rows}),
                    key="result_detail_status")
            displayed = [
                r for r in rows
                if (model_filter == "All models" or r["model"] == model_filter)
                and (condition_filter == "All conditions" or r["condition"] == condition_filter)
                and (category_filter == "All categories" or r["category"] == category_filter)
                and (status_filter == "All statuses" or r["status"] == status_filter)
            ]
            st.markdown(f"**{len(displayed)} matching responses**")
            display_keys = [
                "model", "run_id", "category", "condition", "status",
                "factual_score", "completeness_score", "numerical_score",
                "risk_score", "hallucination", "rouge_l_f1",
            ]
            rename_keys = {
                "model": "AI model", "run_id": "Prompt ID", "category": "Category",
                "condition": "Condition", "status": "Evaluation status",
                "factual_score": "Factual / 2", "completeness_score": "Coverage / 2",
                "numerical_score": "Numeric / 2", "risk_score": "Risk / 2",
                "hallucination": "Evidence flag", "rouge_l_f1": "ROUGE-L F1",
            }
            if displayed:
                st.dataframe(pd.DataFrame(displayed)[display_keys].rename(columns=rename_keys),
                             hide_index=True, use_container_width=True, height=375)
                choices = list(range(len(displayed)))
                chosen_idx = st.selectbox(
                    "Inspect a specific evaluated response", choices,
                    format_func=lambda i: f"{displayed[i]['model']} · {displayed[i]['run_id']}"
                                          f" · {displayed[i]['status']}",
                    key="result_inspect_answer")
                inspected = displayed[chosen_idx]
                st.markdown(f"#### {inspected['model']} · {inspected['run_id']}")
                matching_gold = gold.get(inspected["question_id"], {})
                st.write("**Question:**", matching_gold.get("question_text") or "Not found in reference bank")
                st.write("**Evaluator rationale:**",
                         inspected.get("judge_notes") or "No full judge rationale is available.")
                if inspected.get("hallucination") == "yes":
                    st.warning("**Potential evidence contradiction:** " +
                               str(inspected.get("unsupported_claim") or "See evaluator rationale."))
                elif inspected.get("hallucination") == "uncertain":
                    st.caption("The evidence flag is uncertain. This is not a verified hallucination.")
                saved_original = responses.get((
                    inspected["model"], inspected["run_id"], report_trial))
                if saved_original:
                    with st.expander("Read the complete original answer"):
                        st.write(saved_original["answer"])
            else:
                st.info("No responses match these filters.")

            st.markdown("#### Export your research data")
            x1, x2, x3 = st.columns(3)
            x1.download_button(
                "Detailed answer-level scores", fm.to_csv(rows),
                f"FinLingo_{report_phase}_detailed_metrics.csv", "text/csv", key="dl_detail")
            x2.download_button(
                "Model comparison summary", fm.to_csv(table),
                f"FinLingo_{report_phase}_model_summary.csv", "text/csv", key="dl_summary")
            x3.download_button(
                "Category-level summary", fm.to_csv(by_category),
                f"FinLingo_{report_phase}_category_results.csv", "text/csv", key="dl_cat")
            st.download_button(
                "Download integrated dataset · prompts + answers + scores + explanations",
                fm.to_csv(fm.integrated_dataset(
                    responses, saved_scores, report_phase, report_trial, report_judge)),
                f"FinLingo_{report_phase}_integrated_research_dataset.csv",
                "text/csv", type="primary", key="dl_integrated")
            st.caption("Downloads preserve the raw research fields for reproducibility. "
                       "The visual labels above are for readability only.")


with tab_explore:
    st.subheader("Question Explorer")
    st.caption("Choose one original question. All five AI responses appear first; their results are compared underneath.")
    bench_models = ("ChatGPT", "Gemini", "Claude", "Grok", "DeepSeek")

    with st.container(border=True):
        st.markdown("##### 1. Choose a question")
        control_a, control_b, control_c = st.columns([1, 1.5, 1])
        with control_a:
            phase = st.selectbox("Dataset", ["final", "pilot"], key="explore_phase")
        phase_cases = [p for p in prompts.values()
                       if ("PILOT" in p.get("run_id", "").upper()) == (phase == "pilot")]
        categories = sorted({p.get("category") or gold.get(p.get("question_id"), {}).get("category", "")
                             for p in phase_cases})
        with control_b:
            category = st.selectbox("Financial category", ["All categories"] + categories,
                                    key="explore_category")
        with control_c:
            trial = st.selectbox("Response trial", [1, 2, 3], key="explore_trial")

        filtered_cases = [p for p in phase_cases
                          if category == "All categories" or
                          (p.get("category") or gold.get(p.get("question_id"), {}).get("category")) == category]
        question_ids = sorted({p["question_id"] for p in filtered_cases if p.get("question_id") in gold})
        if question_ids:
            qid = st.selectbox("Select a question from the original dataset", question_ids,
                               format_func=lambda value: f"{value} · {gold[value].get('question_text', '')[:115]}",
                               key="explore_qid")
            variants = [p["run_id"] for p in filtered_cases if p["question_id"] == qid]
            controls_left, controls_right = st.columns([1.6, 1])
            with controls_left:
                rid = st.selectbox("Question condition", variants,
                                   format_func=lambda value: (
                                       "With supplied financial context" if prompts[value].get("condition") == "verified_context"
                                       else "Without added context" if prompts[value].get("condition") == "no_added_context"
                                       else "Pilot question") + f" · {value}",
                                   key="explore_run")
            with controls_right:
                judge = st.selectbox("Scoring method", [judge_default, "reference_only"],
                                     format_func=lambda value: "Full local AI evaluation" if value != "reference_only"
                                     else "ROUGE-L only", key="explore_judge")
        else:
            qid = rid = None

    if qid and rid:
        question = gold[qid]
        prompt = prompts.get(rid, {})
        st.markdown(f"### {qid} · {question.get('question_text', '')}")
        st.caption(f"Prompt ID: `{rid}`  ·  Category: {question.get('category', '')}  ·  "
                   f"Condition: {prompt.get('condition', 'not recorded')}")
        if prompt.get("condition") == "verified_context":
            with st.expander("Financial context supplied with this question"):
                st.write(prompt.get("user_prompt", ""))

        model_records = {}
        for model in bench_models:
            answer = responses.get((model, rid, trial))
            score = fm.current_score(answer, saved_scores, judge) if answer else None
            model_records[model] = (answer, score)

        imported_n = sum(answer is not None for answer, _ in model_records.values())
        graded_n = sum(score is not None and score.get("status") == "judged"
                       for _, score in model_records.values())
        s1, s2, s3 = st.columns(3)
        s1.metric("AI models in benchmark", 5)
        s2.metric("Answers collected for this question", f"{imported_n}/5")
        s3.metric("Answers fully evaluated", f"{graded_n}/5")

        judged_records = [(model, score) for model, (_, score) in model_records.items()
                          if score is not None and score.get("status") == "judged"]

        def determine_leaders(graded):
            """Predeclared rubric ordering; an incomplete comparison remains provisional."""
            contenders = graded[:]
            decision = ""
            for field, label in (("factual_score", "factual correctness"),
                                 ("numerical_score", "verifiable numerical correctness"),
                                 ("completeness_score", "coverage of required points"),
                                 ("risk_score", "risk interpretation")):
                if len(contenders) < 2:
                    break
                if not all(str(score.get(field, "")) in ("0", "1", "2")
                           for _, score in contenders):
                    continue
                top_value = max(int(score[field]) for _, score in contenders)
                leading = [(model, score) for model, score in contenders
                           if int(score[field]) == top_value]
                if len(leading) < len(contenders):
                    contenders = leading
                    decision = f"It scored highest on **{label}** ({top_value}/2)."
            if len(contenders) > 1 and all(score.get("hallucination") in ("yes", "no")
                                               for _, score in contenders):
                supported = [(model, score) for model, score in contenders
                             if score["hallucination"] == "no"]
                if supported and len(supported) < len(contenders):
                    contenders = supported
                    decision = "The local judge found a specific contradiction in a competing answer."
            return [model for model, _ in contenders], decision

        leading, deciding_reason = determine_leaders(judged_records) if judged_records else ([], "")
        featured = (leading[0] if leading else
                    next((model for model in bench_models if model_records[model][0] is not None),
                         bench_models[0]))

        st.markdown("### 2. Best available response and explanation")
        if graded_n == 0:
            st.info("No responses have a full factual evaluation yet. "
                    "The first collected response is shown here without a ranking.")
        elif graded_n == 1:
            st.info(f"**Currently evaluated: {featured} (1/5).** "
                    "Its response is featured below, but there is not enough evidence to call it the best yet.")
        elif len(leading) > 1:
            st.info(f"**Joint leaders under the current rubric:** {', '.join(leading)}. "
                    "No unique winner is justified.")
        else:
            heading = "Top-rated among all five" if graded_n == 5 else "Provisional leader"
            st.success(f"**{heading}: {featured}.** "
                       f"{deciding_reason or 'It leads under the available rubric scores.'}")
        if 0 < graded_n < 5:
            st.caption(f"Only {graded_n}/5 models have a completed local-judge evaluation. "
                       "The ranking may change when more models are evaluated.")

        def render_model_answer(model_name, *, featured_card=False):
            answer, score = model_records[model_name]
            with st.container(border=True):
                label = f"{model_name}  ·  Featured response" if featured_card else model_name
                st.markdown(f"#### {label}")
                if answer is None:
                    st.caption("NOT COLLECTED")
                    st.info("No response has been imported for this question.")
                    return
                if score and score.get("status") == "judged":
                    status = "Full evaluation available"
                elif score and score.get("status") == "text_similarity_only":
                    status = "ROUGE-L only"
                elif score and score.get("status") == "judge_error":
                    status = "Evaluation error"
                else:
                    status = "Awaiting automatic evaluation"
                st.caption(f"{status} · Version: {answer.get('model_version') or 'Not recorded'}")
                if featured_card and score and score.get("status") == "judged":
                    st.markdown("**Evaluator's explanation for this score**")
                    rationale = str(score.get("judge_notes") or "").strip()
                    if rationale and rationale.lower() not in ("null", "none"):
                        st.write(rationale)
                    else:
                        st.caption("No detailed explanation was returned. Check the rubric scores below.")
                    st.caption(f"Factual: {score.get('factual_score', '—')}/2 · "
                               f"Completeness: {score.get('completeness_score', '—')}/2 · "
                               f"Contradiction: {score.get('hallucination') or 'unassessed'}")
                st.markdown("**Original model answer**")
                st.write(answer["answer"])

        render_model_answer(featured, featured_card=True)
        st.markdown("#### Other model responses")
        st.caption("The remaining models are listed one below another. Missing responses stay visible.")
        for model in bench_models:
            if model != featured:
                render_model_answer(model)

        st.divider()
        st.markdown("### 3. Results for this specific question")
        st.caption("Judge-assigned 0–2 rubric scores, not calibrated probabilities. "
                   "Uncollected and unassessed answers do not count as incorrect.")

        def valid_rubric(value):
            return int(value) if str(value) in ("0", "1", "2") else None

        def rubric_label(value, names):
            score_value = valid_rubric(value)
            return f"{score_value}/2 · {names[score_value]}" if score_value is not None else "Not assessed"

        def evidence_label(value):
            return {"yes": "Potential contradiction",
                    "no": "No contradiction identified",
                    "uncertain": "Not resolved"}.get(str(value).lower(), "Not assessed")

        correct_n = sum(valid_rubric(score.get("factual_score")) == 2
                        for _, score in judged_records)
        covered_n = [valid_rubric(score.get("completeness_score"))
                     for _, score in judged_records]
        covered_n = [s for s in covered_n if s is not None]
        measurable_numerical = [valid_rubric(score.get("numerical_score"))
                                for _, score in judged_records]
        measurable_numerical = [s for s in measurable_numerical if s is not None]
        quality_a, quality_b, quality_c = st.columns(3)
        quality_a.metric("Fully factually correct", f"{correct_n}/{graded_n}" if graded_n else "—",
                         help="Only the models given a full local-judge score enter the denominator.")
        quality_b.metric("Essential-point coverage",
                         f"{(100 * sum(covered_n) / (2 * len(covered_n))):.0f}%"
                         if covered_n else "—",
                         help="Average completeness score ÷ 2, over assessed answers; this is rubric coverage.")
        quality_c.metric("Numeric answers within tolerance",
                         f"{sum(s == 2 for s in measurable_numerical)}/{len(measurable_numerical)}"
                         if measurable_numerical else "Not measurable",
                         help="Only calculations with an extractable result, compatible unit, and gold tolerance.")

        comparison_rows = []
        for model in bench_models:
            answer, score = model_records[model]
            status = ("Not collected" if answer is None else
                      "Awaiting judge" if score is None else
                      "Evaluated" if score.get("status") == "judged" else
                      "Text overlap only" if score.get("status") == "text_similarity_only" else
                      "Evaluation error")
            is_graded = score is not None and score.get("status") == "judged"
            num = valid_rubric(score.get("numerical_score")) if is_graded else None
            risk = valid_rubric(score.get("risk_score")) if is_graded else None
            rouge = score.get("rouge_l_f1") if score is not None else None
            try:
                rouge_display = f"{float(rouge):.3f}" if rouge not in ("", None) else "—"
            except (TypeError, ValueError):
                rouge_display = "—"
            comparison_rows.append({
                "AI model": model,
                "Assessment": status,
                "Factual correctness": (
                    rubric_label(score.get("factual_score"),
                                 ("Incorrect", "Partly correct", "Correct"))
                    if is_graded else "—"),
                "Key-point coverage": (
                    rubric_label(score.get("completeness_score"),
                                 ("Missing", "Partial", "Complete"))
                    if is_graded else "—"),
                "Numeric result": (
                    ("Within tolerance" if num == 2 else "Outside tolerance")
                    if num is not None else "Not measurable") if is_graded else "—",
                "Risk reasoning": (
                    ("Strong" if risk == 2 else "Partial" if risk == 1 else "Weak")
                    if risk is not None else "Not applicable") if is_graded else "—",
                "Evidence check": evidence_label(score.get("hallucination")) if is_graded else "—",
                "ROUGE-L overlap": rouge_display,
            })
        st.dataframe(comparison_rows, hide_index=True, use_container_width=True)
        st.caption("**Evidence check:** ‘Not resolved’ means the evaluator could not verify the "
                   "claim, not that a hallucination occurred. **ROUGE-L:** wording overlap with the "
                   "reference (0–1); it does not determine whether a finance answer is true.")

        st.markdown("### 4. Visual comparison")
        st.caption("These charts describe answer quality among assessed models. "
                   "Missing models and pending evaluations are excluded from quality denominators.")
        colors = {"Factual accuracy": "#73ADFF", "Required-point coverage": "#49D8BC"}
        if judged_records:
            bars_data = [{"Model": model, "Metric": label, "Score": int(score[field])}
                         for model, score in judged_records
                         for field, label in (("factual_score", "Factual accuracy"),
                                              ("completeness_score", "Required-point coverage"))
                         if valid_rubric(score.get(field)) is not None]
            bar_chart = (alt.Chart(pd.DataFrame(bars_data))
                         .mark_bar(cornerRadiusEnd=6, size=28)
                         .encode(y=alt.Y("Model:N", sort=list(bench_models), title=None),
                                 yOffset=alt.YOffset("Metric:N"),
                                 x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 2]),
                                         axis=alt.Axis(values=[0, 1, 2]),
                                         title="Rubric points out of 2"),
                                 color=alt.Color("Metric:N",
                                                 scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                                                 title=None),
                                 tooltip=["Model:N", "Metric:N", "Score:Q"])
                         .properties(height=max(170, len(judged_records) * 66),
                                     title="Quality by model · factual accuracy and key-point coverage")
                         .configure_view(strokeOpacity=0)
                         .configure_axis(gridColor="#2D3D50", labelColor="#D5E0EE",
                                         titleColor="#E8F2FE", labelFontSize=13)
                         .configure_legend(labelColor="#E8F2FE", titleColor="#E8F2FE",
                                           labelFontSize=12)
                         .configure_title(color="#EFF7FF", fontSize=17, anchor="start"))
            st.altair_chart(bar_chart, use_container_width=True, theme=None)
        else:
            st.info("Quality charts will appear after at least one full local evaluation.")

        pie_col, overlap_col = st.columns([1, 1.45], gap="large")
        with pie_col:
            st.markdown("#### Factual verdict distribution")
            if judged_records:
                verdicts = [
                    ("Fully correct", "#39CBA2", 2),
                    ("Partly correct", "#F3BC68", 1),
                    ("Incorrect", "#F17485", 0),
                ]
                pie_rows = [
                    {"Verdict": label, "Answers": sum(
                        valid_rubric(score.get("factual_score")) == value
                        for _, score in judged_records)}
                    for label, _, value in verdicts
                ]
                pie_rows = [row for row in pie_rows if row["Answers"] > 0]
                palette = {label: color for label, color, _ in verdicts}
                pie_base = alt.Chart(pd.DataFrame(pie_rows)).encode(
                    theta=alt.Theta("Answers:Q", stack=True),
                    color=alt.Color("Verdict:N", title=None,
                                    scale=alt.Scale(domain=list(palette),
                                                    range=list(palette.values()))),
                    tooltip=[alt.Tooltip("Verdict:N"),
                             alt.Tooltip("Answers:Q", title="Model answers")]
                )
                donut = pie_base.mark_arc(innerRadius=71, outerRadius=109,
                                          padAngle=0.025, stroke="#121820", strokeWidth=2)
                st.altair_chart(
                    donut.properties(height=280)
                         .configure_view(strokeOpacity=0)
                         .configure_legend(labelColor="#E8F2FE", labelFontSize=12),
                    use_container_width=True, theme=None)
                st.caption(f"Only {graded_n} fully evaluated answer(s) included. "
                           "Each slice represents a factual rubric outcome, not collection status.")
            else:
                st.info("Factual verdicts appear after full evaluations.")

        with overlap_col:
            st.markdown("#### ROUGE-L · similarity to reference wording")
            overlap_rows = []
            for model, (_, score) in model_records.items():
                if not score:
                    continue
                try:
                    raw_overlap = float(score.get("rouge_l_f1"))
                except (ValueError, TypeError):
                    continue
                if 0 <= raw_overlap <= 1:
                    overlap_rows.append({"Model": model, "Overlap": raw_overlap})
            if overlap_rows:
                overlap_chart = (
                    alt.Chart(pd.DataFrame(overlap_rows))
                    .mark_bar(cornerRadiusEnd=6, size=24, color="#AB91FF")
                    .encode(
                        y=alt.Y("Model:N", sort=list(bench_models), title=None),
                        x=alt.X("Overlap:Q", scale=alt.Scale(domain=[0, 1]),
                                axis=alt.Axis(format=".0%"),
                                title="Textual overlap · 0–100%"),
                        tooltip=[alt.Tooltip("Model:N"),
                                 alt.Tooltip("Overlap:Q", format=".3f")]
                    )
                    .properties(height=280)
                    .configure_view(strokeOpacity=0)
                    .configure_axis(gridColor="#2D3D50", labelColor="#D5E0EE",
                                    titleColor="#E8F2FE", labelFontSize=13)
                )
                st.altair_chart(overlap_chart, use_container_width=True, theme=None)
                st.caption("Diagnostic only. Correct paraphrases can have low ROUGE-L; "
                           "use factual and completeness scores for quality comparisons.")
            else:
                st.info("Text overlap appears once an answer has been processed.")

        st.download_button("Download this question's collected answers and evaluation results",
                           fm.to_csv([r for r in fm.integrated_dataset(responses, saved_scores,
                                                                       phase, trial, judge)
                                      if r["run_id"] == rid and r["model"] in bench_models]),
                           f"FinLingo_{rid}_comparison.csv", mime="text/csv", key="explore_download",
                           disabled=imported_n == 0)
        with st.expander("Private reference answer, required points and evidence source"):
            st.warning("Main-dataset reference answers are provisional until checked against primary sources.")
            st.write("**Reference answer:**", question.get("reference_answer") or "Not available")
            st.write("**Required points:**", question.get("required_answer_points") or question.get("rubric") or "N/A")
            if question.get("source_url"):
                st.write("**Original source:**", question["source_url"])


with tab_method:
    st.subheader("Methodology & instructions")
    st.caption("A practical guide to collecting, scoring and interpreting the finance LLM benchmark.")

    st.markdown("### What this study tests")
    st.write(
        "FinLingo compares how language models answer the **same finance questions**. "
        "The analysis examines factual correctness, coverage of essential points, calculations, "
        "risk reasoning and claims that may contradict available evidence. "
        "For questions with two versions, it also compares answers **with and without supplied financial context**."
    )
    design_a, design_b, design_c = st.columns(3)
    design_a.metric("Original finance questions", 120)
    design_b.metric("Final prompt cases", 200, help="120 no-context + 80 context variants of existing questions")
    design_c.metric("Separate pilot cases", 20)
    st.caption("The 200 final prompt cases are **not** 200 different questions. "
               "The extra 80 are context-supplied variants, making paired comparisons possible.")

    st.markdown("### How to use FinLingo")
    st.markdown("""
1. **Import:** Save answers from each model, recording its model version, prompt ID and trial.
   Your existing ChatGPT/Gemini answers remain saved. Add other models as they become available.
2. **Automatic evaluation:** Run the **Full local AI judge** in batches. Results are written
   after each answer, so you can stop and resume. The text-overlap-only option is a quick diagnostic,
   **not** a substitute for correctness evaluation.
3. **Results dashboard:** Compare models *within the same condition and category*.
   Check how many answers were actually evaluated before drawing conclusions from percentages.
4. **Question Explorer:** Pick any question to read all five model responses. The top response is
   ranked only among completed evaluations; ties and incomplete comparisons remain visible.
   Read its explanation, the comparison table and answer-quality charts.
5. **Export:** Download detailed scores or the integrated question–response–evaluation CSV
   from the Results tab for your report and further analysis.
""")

    st.markdown("### Evaluation criteria and what the numbers mean")
    st.caption("The full judge uses a fixed 0–2 rubric against the saved reference answer and required points.")
    st.dataframe([
        {"Measure": "Factual correctness",
         "How it is calculated": "2 = materially correct; 1 = partly correct; 0 = materially wrong or missing.",
         "How to interpret it": "Full-credit rate = answers scoring 2 ÷ fully judged answers."},
        {"Measure": "Required-point coverage",
         "How it is calculated": "2 = all essential points; 1 = some; 0 = none.",
         "How to interpret it": "Mean coverage = average score ÷ 2. A complete answer can still contain errors."},
        {"Measure": "Numerical accuracy",
         "How it is calculated": "Extracted final number + units checked against reference tolerance.",
         "How to interpret it": "Report correct / measurable. Unmeasurable answers are excluded."},
        {"Measure": "Risk reasoning",
         "How it is calculated": "0–2 rubric on risk/uncertainty questions only.",
         "How to interpret it": "Compare within applicable risk questions; otherwise N/A."},
        {"Measure": "Potential contradiction",
         "How it is calculated": "Judge flags yes, no or uncertain, with a named disputed claim for yes.",
         "How to interpret it": "A flag is not proof of hallucination. Uncertain is a separate outcome."},
        {"Measure": "ROUGE-L F1",
         "How it is calculated": "Longest-common-subsequence text overlap with the reference (0–1).",
         "How to interpret it": "A lexical diagnostic; low overlap is not the same as being wrong."},
    ], hide_index=True, use_container_width=True)
    st.info("**Research rule:** No assessment is not a score of zero. A model that has not been "
            "imported or judged must never be treated as a worse performer.")

    st.markdown("### How rankings and context comparisons work")
    st.markdown("""
- **Question-level leader:** Factual score takes priority, followed by measurable numerical
  accuracy when available for every remaining contender, essential-point coverage,
  applicable risk reasoning and an explicitly evidenced contradiction flag. **Ties are allowed.**
- **Across models:** Use the same question IDs and conditions and report the assessed
  sample size alongside every result. Do not call an incomplete leaderboard a definitive winner.
- **Effect of supplied context:** Compare only matched question IDs for which
  *both* context versions were evaluated. Treat the difference as an **association**,
  especially where questions were answered in a shared chat.
""")

    st.markdown("### Research limitations")
    st.write(
        "The local Ollama model is a separate, smaller **LLM-as-judge** and can make scoring mistakes. "
        "Reference answers may still be provisional; therefore the results are estimates against "
        "the current rubric, not certified financial facts. Automatic contradiction detection does "
        "not independently check every assertion against external filings. Single manual runs "
        "do not establish statistical significance, cost, latency or run-to-run consistency."
    )
    st.markdown("**Research foundations** · "
                "[ROUGE (Lin, 2004)](https://aclanthology.org/W04-1013/) · "
                "[G-Eval (Liu et al., 2023)](https://aclanthology.org/2023.emnlp-main.153/) · "
                "[FinQA (Chen et al., 2021)](https://aclanthology.org/2021.emnlp-main.300/) · "
                "[FActScore (Min et al., 2023)](https://aclanthology.org/2023.emnlp-main.741/)")
    st.caption("These works motivate the metric design; this app does **not** reproduce G-Eval, "
               "FinQA evaluation or FActScore. ROUGE-L is implemented as a lexical overlap measure.")

