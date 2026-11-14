# FinLingo Research Studio — PATCH ONLY

This ZIP **does not contain a new project structure**. It contains only:

- `finlingo_studio.py` — a clean five-tab interface.
- `finlingo_metrics.py` — safe CSV readers, answer importer, automatic local evaluator and reporting.
- This README.

**Your original `app.py`, imported model responses, datasets, question PDFs,
existing scores and old manual grades are neither overwritten nor deleted.**

## Install in your existing VS Code project

1. Open **the existing `FinLingo_Manual_Research`** folder in VS Code. The folder
   must contain the current `app.py` and your `data/` directory.
2. Download `FinLingo_Research_Studio_PATCH_ONLY.zip` to your Mac's Downloads folder.
3. Open **Terminal → New Terminal** in VS Code. Stop your previously running
   Streamlit app with `Control + C` if necessary.
4. Paste these commands **one at a time**, without an `if/else` wrapper:

   ```bash
   unzip -n "$HOME/Downloads/FinLingo_Research_Studio_PATCH_ONLY.zip" -d .
   source .venv/bin/activate
   python3 -m streamlit run finlingo_studio.py
   ```

   If `source .venv/bin/activate` says the environment does not exist, skip it
   and use the Python environment in which your current Streamlit app works.

5. Streamlit opens the Research Studio. Your original app still works using
   `python3 -m streamlit run app.py` if you ever need it.

## New tabs

1. **Import:** Existing answers remain visible. You can append missing booklet
   answers; repeated keys are skipped without overwriting.
2. **Automatic evaluation:** ROUGE-L-only mode runs quickly. Full rubric-based
   mode uses **local Ollama** (`qwen2.5:3b` by default) for judge estimates.
   Scores are saved after every case in `evaluation/auto_scores.csv`; resume
   whenever convenient. No paid model API.
3. **Results dashboard:** Counts, model/condition and category tables, charts,
   matched context comparison and CSV downloads (including a single **integrated
   dataset** with original questions, model answers, scores and judge reasons). Blank or ungraded values are
   never treated as correct or incorrect.
4. **Question explorer:** Pick any question and see the original model responses,
   each metric, the judge's evidence notes and a clearly defined better/tie/
   not-yet-determined comparison. Numerical grades require verifiable gold.
5. **Methodology & instructions:** Research definitions, limitations, prior
   publications and an optional stratified human-audit checklist.

## Important research quality notes

- Both ROUGE-L and the local judge are **proxies**. They do not certify truth.
- Main-dataset reference answers are marked **provisional**; cross-check primary
  source filings before publishing a factual-accuracy claim.
- Hallucination flags are judge proposals. An `uncertain` flag is not silently
  counted as a clean or hallucinated answer.
- The 80 context-question pairs must be compared by the same question IDs.
- Reusing one chat for many questions risks carry-over effects. Mention this
  in limitations, rather than interpreting a context difference as causal.
- A 10% stratified manual audit of the *automatic judge*, not every answer,
  gives the study a useful quality-control check.

This patch requires your **original data/** and **pilot/** files. The new
interface was tested with the known 120-question master, 200 final prompt cases,
20 pilot prompt cases and 22 booklet manifest entries. It has not been run
against the imported answers currently stored on your Mac.
