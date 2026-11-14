"""Manual-only FinLingo answer importer and reviewer. No external AI integrations."""

from __future__ import annotations

import streamlit as st

from export_results import generate
from research import (ANSWER_FILE, BOOKLETS, MODELS, GRADE_FILE, RESULTS,
                      cases_by_id, compute_metrics, import_booklet, latest_grades,
                      latest_responses, load_gold, load_manifest, parse_answers,
                      save_grade)

st.set_page_config(page_title="FinLingo | Manual Research", page_icon="📊", layout="wide")
st.title("FinLingo · Manual Financial LLM Evaluation")
st.caption("GPT · Claude · Gemini · Qwen. No API keys, API calls or billing.")
st.info("Use the same blind PDF/TXT booklets for every model. Their reference answers are kept private in the grading tab.")

tab_import, tab_review, tab_report, tab_help = st.tabs(
    ["1. Import model answers", "2. Grade answers", "3. Results", "Instructions"])
manifest = load_manifest()
cases = cases_by_id()

with tab_import:
    st.subheader("Paste an entire model response")
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Model", MODELS, key="import_model")
        model_version = st.text_input("Exact version displayed in chat", placeholder="Name shown in the model menu")
        trial = st.selectbox("Independent trial", [1, 2, 3], key="import_trial")
    with col2:
        booklet_id = st.selectbox("Which booklet did you send?", [b["booklet_id"] for b in manifest])
        mode = st.text_input("Chat settings", value="Fresh chat; browsing/tools disabled if possible")
        allow_replace = st.checkbox("This replaces answers already imported for these IDs", value=False)

    booklet = next(b for b in manifest if b["booklet_id"] == booklet_id)
    st.write(f"**Booklet:** `{booklet['pdf']}` · **Expected answers:** {len(booklet['run_ids'])}")
    st.caption("Required IDs: " + ", ".join(booklet["run_ids"]))
    uploaded = st.file_uploader("Optional: upload the response as a .txt file", type=["txt", "md"])
    raw_input = st.text_area("Or paste the model's complete response", height=300,
                             placeholder="[[A01-NOCTX]]\n...\n[[/A01-NOCTX]]")
    raw = uploaded.getvalue().decode("utf-8-sig", errors="replace") if uploaded else raw_input

    if raw.strip():
        preview = parse_answers(raw, booklet["run_ids"])
        st.write(f"Detected **{len(preview['answers'])} / {len(booklet['run_ids'])}** valid paired answers.")
        if preview["missing"]:
            st.warning("Missing IDs: " + ", ".join(preview["missing"]))
        for name in ("unexpected", "duplicate", "empty", "possibly_malformed"):
            if preview[name]:
                st.error(f"{name}: {preview[name]}")
        with st.expander("Preview detected answers"):
            for rid, answer in preview["answers"].items():
                st.markdown(f"**{rid}**")
                st.text(answer[:600] + ("…" if len(answer) > 600 else ""))

    if st.button("Import parsed responses", type="primary", key="import_button"):
        if not raw.strip():
            st.error("Paste a response or attach a .txt file first.")
        else:
            try:
                report = import_booklet(model, model_version, mode, trial, booklet, raw,
                                        allow_replace=allow_replace)
                st.success(f"Saved {report['imported']} answers into {ANSWER_FILE.name}.")
                if report["missing"]:
                    st.warning("Still missing: " + ", ".join(report["missing"]))
                if report["skipped_existing"]:
                    st.info("Already recorded, not overwritten: " + ", ".join(report["skipped_existing"]))
            except ValueError as error:
                st.error(str(error))

with tab_review:
    st.subheader("Grade each saved response against the reference answer")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        phase = st.selectbox("Study phase", ["pilot", "final"], key="grade_phase")
    with c2:
        model_to_grade = st.selectbox("Model", MODELS, key="grade_model")
    with c3:
        trial_to_grade = st.selectbox("Trial", [1, 2, 3], key="grade_trial")
    with c4:
        reviewer = st.text_input("Reviewer name", value="primary")

    all_resp = latest_responses(phase)
    selected = {rid: r for (m, rid, t), r in all_resp.items()
                if m == model_to_grade and t == trial_to_grade}
    done = latest_grades(reviewer.strip() or "primary")
    graded = {rid for rid in selected if (model_to_grade, rid, trial_to_grade, reviewer) in done}

    if not selected:
        st.warning("No imported answers for this model and phase yet. Go to Import model answers.")
    else:
        ordered = sorted(selected, key=lambda rid: (rid in graded, rid))
        st.progress(len(graded)/len(selected), text=f"{len(graded)} of {len(selected)} imported answers graded")
        rid = st.selectbox("Select a response to grade", ordered,
                           format_func=lambda value: f"{'✓ ' if value in graded else ''}{value}")
        item = cases[rid]
        gold = load_gold(phase)[item["question_id"]]
        left, right = st.columns(2)
        with left:
            st.markdown("#### Question given to the model")
            st.write(item["user_prompt"])
            st.markdown("#### Model answer")
            st.write(selected[rid]["answer"])
        with right:
            st.markdown("#### Private grading reference (never sent in PDF)")
            st.info(gold.get("reference_answer", "Reference answer unavailable"))
            st.markdown("**Required points / rubric**")
            st.write(gold.get("required_answer_points") or gold.get("rubric") or "Review required")
            st.markdown("**Scoring notes / numerical tolerance**")
            st.write(gold.get("units_and_tolerance") or "Not specified")
            if gold.get("validation_status"):
                st.warning(gold["validation_status"])
            if gold.get("source_url"):
                st.write("Evidence source (verify independently):", gold["source_url"])

        st.markdown("#### Your scoring")
        st.caption("0 = incorrect/missing; 1 = partially correct; 2 = fully correct. "
                   "N/A is excluded from numerical/risk averages.")
        with st.form(key=f"grade_{phase}_{model_to_grade}_{trial_to_grade}_{rid}", clear_on_submit=False):
            g1, g2, g3 = st.columns(3)
            with g1:
                factual = st.selectbox("Factual correctness", [0, 1, 2], index=1)
                numeric = st.selectbox("Numerical correctness", ["N/A", "0", "1", "2"],
                                       index=0 if item["category"] != "Numerical reasoning" else 1)
            with g2:
                complete = st.selectbox("Answer completeness", [0, 1, 2], index=1)
                risk = st.selectbox("Risk interpretation", ["N/A", "0", "1", "2"],
                                    index=0 if item["category"] != "Risk and uncertainty" else 1)
            with g3:
                hallucinated = st.selectbox("Unsupported invented factual claim?", [0, 1],
                                            format_func=lambda v: "No" if v == 0 else "Yes")
            notes = st.text_area("Evidence and reasons for your grading", height=110,
                                 placeholder="Mention missing required points, incorrect numbers, source conflicts, invented claims…")
            if st.form_submit_button("Save grade", type="primary"):
                if hallucinated and not notes.strip():
                    st.error("Please explain the unsupported claim before flagging hallucination.")
                else:
                    try:
                        save_grade(reviewer, model_to_grade, rid, trial_to_grade,
                                   factual, numeric, complete, hallucinated, risk, notes)
                        st.success(f"Saved evaluation of {rid}.")
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))

with tab_report:
    st.subheader("Research metrics")
    ra, rb, rc = st.columns(3)
    with ra:
        report_phase = st.selectbox("Phase", ["pilot", "final"])
    with rb:
        report_reviewer = st.text_input("Reviewer", value="primary", key="report_reviewer")
    with rc:
        report_trial = st.selectbox("Trial", [1, 2, 3], key="report_trial")
    by_model, by_cat = compute_metrics(report_phase, report_reviewer, report_trial)
    st.markdown("**By model and test condition**")
    st.dataframe(by_model, hide_index=True, use_container_width=True)
    st.markdown("**By question category**")
    st.dataframe(by_cat, hide_index=True, use_container_width=True)
    if st.button("Generate and save CSV reports + charts"):
        generate(report_phase, report_reviewer, report_trial)
        st.success(f"Reports saved inside {RESULTS.name}/. Only graded answers affect accuracy metrics.")
    st.caption("Never compare results based on different denominators without showing sample sizes. "
               "Manual response time is not measured. Model self-scoring is not used.")

with tab_help:
    st.markdown("""
    1. Send `question_booklets/pilot/pilot_01.pdf` to GPT, Claude, Gemini and Qwen separately.
    2. Use fresh chats and keep the model's `[[ID]]` markers in its reply.
    3. Paste the **entire reply** into **Import model answers**. Select that booklet and model.
    4. Repeat for all pilot PDFs. Compare imports, review any missing IDs, and manually grade them.
    5. Independently verify master reference answers before conducting the final 200-case evaluation.
    6. Run final `no_context` and `with_context` batches separately, then grade and generate results.

    **No API keys are needed anywhere.** Data are stored in local CSVs that you can back up.
    """)
