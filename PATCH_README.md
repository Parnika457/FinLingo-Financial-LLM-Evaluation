# FinLingo automatic-evaluation PATCH (no new project)

This ZIP contains **one new Streamlit page only**. It does not replace `app.py`,
your `data/`, `pilot/`, question PDFs, existing imported answers, or grades.

## Install in 30 seconds

1. **Keep your current project folder open in Finder or VS Code.**
2. Extract the ZIP. Copy its `pages` folder **into the same folder that contains
   your existing `app.py`**, merging if macOS asks. Do not replace/delete any
   existing files. Leave `PATCH_README.md` wherever you like.
3. Start your usual Streamlit application:

   ```bash
   source .venv/bin/activate
   python -m streamlit run app.py
   ```

4. The existing Streamlit sidebar now shows **Automatic Evaluation** as another
   page. Your current four tabs remain exactly as they are.

5. It reads your existing `model_answers/responses.csv` and original master
   dataset. It writes **new scores** to `evaluation/auto_scores.csv`.

## Two methods

- **Fast ROUGE-L only:** click **Fast ROUGE-L only** then **Evaluate automatically**.
  All 400 final responses should score quickly. It measures answer/reference
  text overlap only. It cannot establish whether financial facts are correct.
- **Full local judge:** Install Ollama from https://ollama.com/download/mac,
  open it, and run `ollama pull qwen2.5:3b` once (~1.9 GB model). Select
  **Full local judge**, then **Evaluate automatically**, and choose all remaining.
  This estimates factual accuracy, completeness, risk interpretation, and
  potential hallucinations and checks numerical answers against explicit gold
  tolerances. No paid API, no model calls to the cloud. Processing 400 answers
  may take time on a MacBook Air. Each answer is saved immediately; rerun to resume.

## What to put in the research paper

- Full judge: rubric-based **LLM-as-a-judge** inspired by *G-Eval* (Liu et al.,
  EMNLP 2023), with a separate small local judge model and explicit schema.
- Numerical: final-answer agreement within the task's stated units/tolerances,
  inspired by financial numerical benchmarks such as *FinQA*.
- ROUGE-L F1: published text-overlap metric (not truthfulness).
- Hallucination: judge-proposed flags with an `uncertain` category. **Not** the
  published *FActScore* implementation.
- To support research conclusions, independently verify dataset gold and
  spot-check a stratified 10% of the judge results. Report this limitation.

Sources: https://aclanthology.org/2023.emnlp-main.153/ and
https://arxiv.org/abs/2109.00122
