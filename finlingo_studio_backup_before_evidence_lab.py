"""FinLingo Research Studio — a focused five-tab entrypoint for the existing app.

Run: python3 -m streamlit run finlingo_studio.py
This file leaves the original app.py and every existing imported response intact.
"""

from __future__ import annotations

from collections import defaultdict

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
    st.subheader("Research results dashboard")
    r1, r2, r3 = st.columns(3)
    with r1:
        report_phase = st.selectbox("Dataset", ["final", "pilot"], key="results_phase")
    with r2:
        report_trial = st.selectbox("Trial", [1, 2, 3], key="results_trial")
    with r3:
        report_judge = st.selectbox("Scoring source", [judge_default, "reference_only"], key="results_judge")

    rows = main_scores(report_phase, report_trial, report_judge)
    judged = [r for r in rows if r["status"] == "judged"]
    ready_count = len(judged) if report_judge != "reference_only" else sum(r["status"] == "text_similarity_only" for r in rows)
    total_models = len({r["model"] for r in rows})
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Responses collected", len(rows))
    d2.metric("Evaluations completed", ready_count)
    d3.metric("Models represented", total_models)
    d4.metric("Completion of collected", f"{100 * ready_count / len(rows):.1f}%" if rows else "0%")
    if not rows:
        st.info("No responses found for this selection. Import them first.")
    else:
        table = fm.summaries(responses, saved_scores, report_phase, report_trial, report_judge)
        st.markdown("#### 1. Model × test condition")
        st.caption("Percentages use fully evaluated answers, except where the header specifies another "
                   "denominator. Missing metrics are **not measured**, not zero. "
                   "Read the sample sizes (n) before comparing models.")
        st.dataframe(display_research_summary(table), hide_index=True, use_container_width=True)
        scored_rows = [r for r in table if r["auto_graded_n"]]
        if scored_rows:
            chart = [{"Model / condition": r["model"] + " · " + r["condition"],
                      "Fully correct (%)": r["factual_full_pct"],
                      "Completeness (%)": r["completeness_pct"]} for r in scored_rows]
            st.bar_chart(chart, x="Model / condition", y=["Fully correct (%)", "Completeness (%)"],
                         use_container_width=True)
        else:
            st.info("Factual-correctness charts appear after **Full local AI judge** evaluations finish. "
                    "ROUGE-L-only results cannot be treated as accuracy.")
        st.markdown("#### 2. Category-level strengths and weaknesses")
        by_category = fm.summaries(responses, saved_scores, report_phase, report_trial, report_judge,
                                   category=True)
        st.dataframe(display_research_summary(by_category), hide_index=True,
                     use_container_width=True, height=335)
        category_chart_rows = [r for r in by_category if r["auto_graded_n"]]
        if category_chart_rows:
            focus_model = st.selectbox("Category chart · model", sorted({r["model"] for r in category_chart_rows}),
                                       key="results_focus_model")
            focus_condition = st.selectbox(
                "Category chart · test condition",
                sorted({r["condition"] for r in category_chart_rows if r["model"] == focus_model}),
                key="results_focus_condition")
            focus = [r for r in category_chart_rows
                     if r["model"] == focus_model and r["condition"] == focus_condition]
            st.bar_chart([{"Category": row["category"], "Fully correct (%)": row["factual_full_pct"]}
                          for row in focus], x="Category", y="Fully correct (%)", use_container_width=True)
        if report_phase == "final" and report_judge != "reference_only":
            st.markdown("#### 3. Matched context analysis (same question, same model)")
            context = fm.paired_context(responses, saved_scores, report_trial, report_judge)
            if context:
                st.dataframe(context, use_container_width=True, hide_index=True)
                st.bar_chart([{"Model": row["model"], "Context change (percentage points)": row["context_change_pp"]}
                              for row in context], x="Model", y="Context change (percentage points)",
                             use_container_width=True)
                st.caption("Percentage-point difference uses ONLY questions with both a no-context answer "
                           "and a verified-context answer fully graded. Not a causal estimate if chat history was shared.")
            else:
                st.info("Context comparison will appear when matching no-context and with-context "
                        "answers have both been fully graded.")
        st.markdown("#### 4. Question-level outcomes and evidence")
        st.dataframe(rows, hide_index=True, use_container_width=True, height=370)
        x1, x2, x3 = st.columns(3)
        x1.download_button("Download detailed metrics", fm.to_csv(rows),
                           f"FinLingo_{report_phase}_detailed_metrics.csv", "text/csv", key="dl_detail")
        x2.download_button("Download model summary", fm.to_csv(table),
                           f"FinLingo_{report_phase}_model_summary.csv", "text/csv", key="dl_summary")
        x3.download_button("Download category results", fm.to_csv(by_category),
                           f"FinLingo_{report_phase}_category_results.csv", "text/csv", key="dl_cat")
        st.download_button("Download INTEGRATED dataset: questions + original answers + scores + judge reasons",
                           fm.to_csv(fm.integrated_dataset(responses, saved_scores, report_phase,
                                                           report_trial, report_judge)),
                           f"FinLingo_{report_phase}_integrated_research_dataset.csv", "text/csv",
                           type="primary", key="dl_integrated")


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

