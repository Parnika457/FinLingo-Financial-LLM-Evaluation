"""FinLingo Research Studio — an optional five-tab entrypoint for the existing app.

Run: python3 -m streamlit run finlingo_studio.py
This file leaves the original app.py and every existing imported response intact.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

import altair as alt
import pandas as pd
import streamlit as st

import finlingo_metrics as fm


st.set_page_config(page_title="FinLingo | Research Studio", page_icon="📊", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1500px; padding-top: 2.2rem;}
  [data-testid="stMetric"] {border:1px solid rgba(127,152,185,.19); border-radius: 11px; padding: 14px 16px;}
  div[data-testid="stAlert"] {border-radius: 10px;}
  .research-eyebrow {letter-spacing:.16em; text-transform:uppercase; color:#84a9d3;
                      font-weight:650; font-size:.76rem; margin-bottom:.4rem;}
  .research-subtitle {color:#a4afbc; max-width: 1000px; margin-top:-.35rem; margin-bottom:1.1rem;}
  div[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:14px;}
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
    st.warning("An Ollama grade is a **provisional LLM-as-judge estimate**. It is not verified financial truth. "
               "Cross-check official sources and manually audit a small stratified sample before publication.")


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
        st.caption("An empty metric is **not measured**, not zero. Columns explicitly show evaluation sample sizes.")
        st.dataframe(table, hide_index=True, use_container_width=True)
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
        st.dataframe(by_category, hide_index=True, use_container_width=True, height=335)
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
        st.caption("Factual and completeness grades use a 0–2 rubric. Missing evaluations are shown as pending, not zero.")
        def shown(value, fallback="—"):
            return fallback if value is None or str(value).strip().lower() in ("", "null", "none") else value

        comparison_rows = []
        for model in bench_models:
            answer, score = model_records[model]
            status = ("Not collected" if answer is None else
                      "Not graded" if score is None else
                      "Graded" if score.get("status") == "judged" else
                      "ROUGE-L only" if score.get("status") == "text_similarity_only" else
                      "Evaluation error")
            is_graded = score is not None and score.get("status") == "judged"
            comparison_rows.append({
                "Model": model,
                "Status": status,
                "Factual / 2": score.get("factual_score", "") if is_graded else "—",
                "Completeness / 2": score.get("completeness_score", "") if is_graded else "—",
                "Numerical / 2": shown(score.get("numerical_score"), "N/A") if is_graded else "—",
                "Risk / 2": shown(score.get("risk_score"), "N/A") if is_graded else "—",
                "Contradiction flag": shown(score.get("hallucination"), "Unassessed") if is_graded else "—",
                "ROUGE-L F1": shown(score.get("rouge_l_f1")) if score else "—",
            })
        st.dataframe(comparison_rows, hide_index=True, use_container_width=True)

        st.markdown("### 4. Visual comparison")
        st.caption("Charts use the five models' actual statuses and available grades. "
                   "Missing scores are never plotted as zero.")
        colors = {"Factual correctness": "#5D9CFF", "Completeness": "#30C4B1"}
        if judged_records:
            bars_data = [{"Model": model, "Metric": label, "Score": int(score[field])}
                         for model, score in judged_records
                         for field, label in (("factual_score", "Factual correctness"),
                                              ("completeness_score", "Completeness"))]
            bar_chart = (alt.Chart(pd.DataFrame(bars_data))
                         .mark_bar(cornerRadiusEnd=5)
                         .encode(x=alt.X("Model:N", sort=list(bench_models), title=None),
                                 xOffset=alt.XOffset("Metric:N"),
                                 y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 2]),
                                         title="Rubric score (0–2)"),
                                 color=alt.Color("Metric:N",
                                                 scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                                                 title=None),
                                 tooltip=["Model:N", "Metric:N", "Score:Q"])
                         .properties(height=270, title="Factual correctness and completeness")
                         .configure_view(strokeOpacity=0)
                         .configure_axis(gridColor="#364252", labelColor="#B9C5D5", titleColor="#DDE7F0")
                         .configure_legend(labelColor="#DDE7F0", titleColor="#DDE7F0"))
            st.altair_chart(bar_chart, use_container_width=True, theme=None)
        else:
            st.info("The rubric comparison chart will appear after the first full evaluation.")

        profile_col, status_col = st.columns([1.5, 1])
        with profile_col:
            st.markdown("**Score profile by evaluation criterion**")
            profile_rows = []
            for model, score in judged_records:
                for field, label in (("factual_score", "Factual"),
                                     ("completeness_score", "Completeness"),
                                     ("numerical_score", "Numerical"),
                                     ("risk_score", "Risk")):
                    if str(score.get(field, "")) in ("0", "1", "2"):
                        profile_rows.append({"Model": model, "Criterion": label,
                                             "Score": int(score[field])})
            if profile_rows:
                line_chart = (alt.Chart(pd.DataFrame(profile_rows))
                              .mark_line(point=alt.OverlayMarkDef(filled=True, size=85), strokeWidth=3)
                              .encode(x=alt.X("Criterion:N",
                                              sort=["Factual", "Completeness", "Numerical", "Risk"],
                                              title=None),
                                      y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 2]),
                                              title="Rubric score (0–2)"),
                                      color=alt.Color("Model:N", title=None),
                                      tooltip=["Model:N", "Criterion:N", "Score:Q"])
                              .properties(height=270)
                              .configure_view(strokeOpacity=0)
                              .configure_axis(gridColor="#364252", labelColor="#B9C5D5",
                                              titleColor="#DDE7F0")
                              .configure_legend(labelColor="#DDE7F0", titleColor="#DDE7F0"))
                st.altair_chart(line_chart, use_container_width=True, theme=None)
                st.caption("This line shows a rubric profile across criteria, not a trend over time.")
            else:
                st.info("No graded score profiles are available yet.")

        with status_col:
            st.markdown("**Response and evaluation coverage**")
            status_counts = defaultdict(int)
            for row in comparison_rows:
                status_counts[row["Status"]] += 1
            pie_rows = [{"Status": label, "Count": count}
                        for label, count in status_counts.items()]
            status_palette = {
                "Graded": "#30C4B1", "Not graded": "#F3B562",
                "Not collected": "#718096", "ROUGE-L only": "#A78BFA",
                "Evaluation error": "#FB7185",
            }
            pie_base = alt.Chart(pd.DataFrame(pie_rows)).encode(
                theta=alt.Theta("Count:Q", stack=True),
                color=alt.Color("Status:N", title=None,
                                scale=alt.Scale(domain=list(status_palette),
                                                range=list(status_palette.values()))),
                tooltip=["Status:N", "Count:Q"])
            donut_chart = (pie_base.mark_arc(innerRadius=66, outerRadius=104, padAngle=0.025)
                           .properties(height=270)
                           .configure_view(strokeOpacity=0)
                           .configure_legend(labelColor="#DDE7F0", titleColor="#DDE7F0"))
            st.altair_chart(donut_chart, use_container_width=True, theme=None)
            st.caption(f"{graded_n} of five model answers are fully evaluated for this question.")

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
    st.caption("Scoring definitions, steps to use the app, and research limitations.")
    metrics_tab, use_tab, limits_tab = st.tabs([
        "Metrics & scoring rules", "How to use the app", "Validity & references",
    ])

    with metrics_tab:
        st.markdown("""
#### Automatic evaluation protocol

- **Independent local judge:** Ollama running `qwen2.5:3b` by default,
  called with a structured JSON output schema, temperature **0** and seed **17**.
- **Reference and rubric:** The judge receives the question, reference answer,
  required answer points and available supporting evidence.
- **Ordinal ratings:** Factual correctness and completeness each use **0 =
  incorrect/missing, 1 = partly correct, 2 = materially correct/complete**.
- **Numerical verification:** The extracted final number is compared against
  `numerical_gold`, its units and the dataset's declared tolerance. If the
  final value or units cannot be established, the result is **unmeasurable**.
- **Hallucination/contradiction check:** `yes` requires a specific factual
  contradiction to supplied evidence, `no` means none found, and `uncertain`
  means available evidence is insufficient. These are **judge flags**, not
  verified hallucination labels.
- **Text similarity:** Basic token-level ROUGE-L F1 (longest common subsequence).
  A different but correct wording can receive a low similarity score.

#### How the percentages are calculated

| Dashboard measure | Formula / denominator | Interpretation |
|---|---|---|
| **Factual fully correct (%)** | `100 × count(factual = 2) / judged_n` | Fraction rated fully correct by the local judge |
| **Factual partial credit (%)** | `100 × mean(factual / 2)` over judged | Mean rubric correctness, including partial credit |
| **Completeness (%)** | `100 × mean(completeness / 2)` over judged | Required-point coverage estimate |
| **Numerical correct (%)** | `100 × exact_with_tolerance / numeric_n` | Only answers with a measurable numerical grade |
| **Risk reasoning (%)** | `100 × mean(risk / 2)` over applicable cases | Excludes non-risk cases and missing values |
| **Contradicted-claim flag (%)** | `100 × yes / (yes + no)` | Only determinate flags; report uncertain cases separately |
| **ROUGE-L F1** | Mean token LCS F1 (0–1) | Lexical overlap, **not** a correctness percentage |

**Reporting rule:** Display the corresponding **n** beside every percentage.
Missing answers, judge errors, uncertain contradiction assessments, and
unmeasurable numerical results must **not** be silently counted as zeros.

#### Question-level comparison rule

The explorer compares factual scores first, then verifiable numerical scores
(if both are measurable), completeness, applicable risk scores and finally
specifically evidenced contradiction flags. It permits **tie** or
**not determined**. Judge notes explain its *provisional* assessment;
this is rubric transparency, not SHAP, causal XAI or a proof of correctness.
        """)

    with use_tab:
        st.markdown("""
#### Using FinLingo, step by step

1. **Check imports.** Open **01 Import** and verify the number of ChatGPT
   and Gemini responses for pilot, no-context and supplied-context conditions.
   Keep the original model outputs and ID markers unchanged.
2. **Check the local evaluator.** Open **02 Automatic evaluation**, select
   **Full local AI judge**, and make sure Ollama recognizes the installed model.
   First run **10 cases** to check compatibility, then choose larger batches.
3. **Resume when needed.** Every graded response is saved immediately.
   Restarting the app will skip completed, unchanged evaluations. The fast
   ROUGE-L option is useful as a baseline, **not** as factual grading.
4. **Read results carefully.** In **03 Results dashboard**, choose the
   study phase and judge, then inspect the **graded sample size** before
   comparing any percentage. Use category tables and matched-context
   results, and download the integrated research CSV.
5. **Inspect individual disagreements.** In **04 Question explorer**, select
   a question and compare the full answers, metrics and judge explanations.
   Treat a winner as a rubric-based judgment, not an independently proven fact.
6. **Perform quality control.** Check reference answers against the cited
   original financial statements or educational sources, especially numbers.
   Once **both** models are substantially graded, audit a stratified sample,
   record human corrections and calculate agreement with the judge.
7. **Write your research findings.** Report model/condition sample sizes,
   the scoring method, category patterns, paired-context findings, audit
   agreement and limitations together. Don't report latency, cost, stability
   or statistical significance unless actually measured.

**Data locations:** `model_answers/responses.csv` retains original responses;
`evaluation/auto_scores.csv` stores append-only judge outputs and rationale;
the Results tab exports the integrated question–answer–score dataset.
No commercial LLM API subscription is required for this local workflow.
        """)

    with limits_tab:
        st.markdown("""
#### Validity threats and how to disclose them

| Issue | Consequence | Reporting safeguard |
|---|---|---|
| **Provisional reference answers** | A flawed reference can cause a wrong grade | Independently check official sources and note unresolved items |
| **AI-assisted reference drafting** | Possible wording/source bias toward the model used to draft gold | Audit reference quality; do not equate a model's similarity to truth |
| **Small local judge** | Scores and explanations may be inconsistent or inaccurate | Report judge identity, use a stratified human audit and record agreement |
| **Shared chat history during collection** | Earlier questions may influence later answers | State that context effects are descriptive, **not causal** |
| **One trial / incomplete model coverage** | Rankings may be sample- and run-specific | Report sample sizes and avoid claims about stability |
| **Uncertain hallucination flags** | Determinate rate covers only a subset | Report both `yes/(yes+no)` and the count of `uncertain` results |
| **Qwen-family candidate judged by Qwen** | Potential same-family evaluator preference | Disclose and consider a different independent judge |

#### Research sources

- [Lin (2004), ROUGE, ACL](https://aclanthology.org/W04-1013/): reference-based overlap.
- [Liu et al. (2023), G-Eval, EMNLP](https://aclanthology.org/2023.emnlp-main.153/): structured LLM evaluation inspiration.
- [Chen et al. (2021), FinQA, EMNLP](https://aclanthology.org/2021.emnlp-main.300/): financial numerical QA benchmark motivation.
- [Min et al. (2023), FActScore, EMNLP](https://aclanthology.org/2023.emnlp-main.741/): evidence-grounded factual precision motivation.

**Citation boundary:** This project does **not** implement published G-Eval or
FActScore procedures in full. It implements its own rubric-based local judge,
numeric tolerance checks and a simple ROUGE-L F1 calculation inspired by
those evaluation traditions.
        """)

    st.divider()
    st.markdown("#### Stratified human quality-control sample")
    st.caption("This is a validation step for the automatic evaluator, not a request to grade all imported answers.")
    m1, m2 = st.columns(2)
    with m1:
        audit_phase = st.selectbox("Phase to audit", ["final", "pilot"], key="audit_phase")
    with m2:
        audit_judge = st.text_input("Judge identity", judge_default, key="audit_judge")
    audit_rows = [r for r in fm.scored_view(responses, saved_scores, audit_phase, 1, audit_judge)
                  if r["status"] == "judged"]
    strata = defaultdict(list)
    for row in audit_rows:
        strata[(row["model"], row["condition"], row["category"])].append(row)
    rng = random.Random(20261114)
    sample = []
    for key, items in sorted(strata.items()):
        sample.extend(rng.sample(sorted(items, key=lambda row: row["run_id"]),
                                 max(1, math.ceil(.10 * len(items)))))
    if sample:
        imported_phase = [r for r in responses.values() if r["phase"] == audit_phase and r["trial"] == 1]
        sample_models = {r["model"] for r in sample}
        all_models = {r["model"] for r in imported_phase}
        st.write(f"**Current sampling frame:** {len(audit_rows)} fully judged / "
                 f"{len(imported_phase)} imported responses. Suggested audit: **{len(sample)}** answers.")
        st.caption("Deterministic stratification by model × condition × category; minimum one case per "
                   "observed stratum, so the fraction can exceed 10%.")
        if all_models - sample_models:
            st.warning("The sample does not yet represent all imported models. Finish grading "
                       + ", ".join(sorted(all_models - sample_models))
                       + " before treating this as a GPT-vs-Gemini validation sample.")
        uncertain_count = sum(r.get("hallucination") == "uncertain" for r in sample)
        if uncertain_count:
            st.info(f"{uncertain_count}/{len(sample)} sampled responses have an **uncertain** "
                    "contradiction flag. This is not evidence of zero hallucinations; "
                    "review them against primary sources.")
        st.dataframe([{"model": r["model"], "run_id": r["run_id"],
                       "category": r["category"], "condition": r["condition"],
                       "factual / 2": r["factual_score"], "hallucination": r["hallucination"]} for r in sample],
                     hide_index=True, use_container_width=True)
        indexed_full = {(r["model"], r["run_id"]): r
                        for r in fm.integrated_dataset(responses, saved_scores, audit_phase, 1, audit_judge)}
        audit_export = []
        for row in sample:
            record = dict(indexed_full[(row["model"], row["run_id"])])
            record.update(human_factual_score="", human_completeness_score="",
                          human_contradiction_flag="", human_review_notes="", reviewed_by="")
            audit_export.append(record)
        st.download_button("Download review-ready audit checklist (with answers + blank human fields)",
                           fm.to_csv(audit_export),
                           "FinLingo_Stratified_Audit.csv", "text/csv", key="dl_audit")
    else:
        st.info("The audit sample will appear once full local grades exist.")
