# FinLingo: Manual-Only LLM Research Benchmark

This is the replacement project for **GPT, Claude, Gemini, and Qwen tested manually**. It has **no API calls, no API keys, no `.env` file, no paid services, and no model-training requirement**.

## The folder structure

```text
FinLingo_Manual_Research/
  data/                    Your original 120-question dataset in CSV, JSONL and Excel,
                           plus the original 200 blind model prompts in JSONL.
  pilot/                   Separate 20-question practice set. Not final study data.
  question_booklets/
    pilot/                 2 PDFs and 2 TXT files, 10 questions in each.
    no_context/            12 PDFs and 12 TXT files, 120 questions total.
    with_context/          8 PDFs and 8 TXT files, 80 context-enhanced cases.
    booklet_manifest.json  Maps each booklet to its exact case IDs.
  model_answers/           Your pasted/imported responses appear here automatically.
  evaluation/              Your manual grading appears here automatically.
  results/                 Metrics and plots appear here automatically.
  research.py              Shared parsing, saving, grading and analysis logic.
  app.py                   The one-screen local collection and grading interface.
  make_booklets.py         Recreate the PDF/TXT booklets (optional).
  export_results.py        Compute summary metrics from saved grades.
  requirements.txt         Streamlit, ReportLab and Matplotlib only.
```

We kept the original downloaded dataset and renamed **none of its content**. The booklets use only the blind prompts; **reference answers never appear in PDFs or TXT files sent to models**.

## Getting started on your Mac

1. Download and unzip this project. In VS Code, choose **File > Open Folder** and open `FinLingo_Manual_Research`.
2. Your previous `.venv` might still work, but the standalone setup is:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python3 -m unittest discover -s tests -v
   streamlit run app.py
   ```

3. Open `question_booklets/pilot/`. Start with `pilot_01.pdf` and the **same file for all four models**.
4. Upload the PDF to ChatGPT, Claude, Gemini and Qwen, or paste its matching `pilot_01.txt` if a service does not support PDF uploads. Request **all questions in the file** and insist on exact `[[ID]]` and `[[/ID]]` answer delimiters.
5. Copy each model's *complete* response to the app's **Import answers** tab. Select the exact booklet/model and click Import. Repeat for all batches.
6. Use **Grade answers** in the app. Review each answer against the verified gold answer and rubric. The app saves factual correctness, numerical accuracy, completeness, hallucinations, risk handling and notes.
7. Open **Results** or run `python3 export_results.py --phase pilot` to compute preliminary results. The same tool works for final data after source verification.

## Important: why multiple PDFs?

There are **120 unique final questions**, but **200 question cases** because 80 of those questions have a second version with a supplied verified context packet. To keep the test fair and reduce skipped/truncated answers, PDF booklets have only 10 cases each. Sending 200 questions in one massive PDF and requesting all responses at once often risks omissions, context overload or inconsistent formatting. There is one question per document page. Do not mix the context and no-context booklets in one chat.

The pilot's 20 questions are **not** counted in the main research results.

### How to submit a booklet to any model

The PDF/TXT already contains these instructions and the 10 question IDs. You can say:

> Please answer every question in the attached FinLingo booklet. Return all 10 answers using the exact `[[RUN_ID]]` and `[[/RUN_ID]]` markers shown. No introduction or summary. Do not browse or use external tools.

Use a **fresh chat for each booklet** where possible. Use the same booklets in the same order for each model. Note the displayed model/version and settings; disable search/tools where available. If a model skips a question, ask it to complete only the missing ID, or retry that batch. Avoid asking models for their reference answers or grading standards.

### Example response format

```text
[[A01-NOCTX]]
The company's common shares represent equity ownership, while its bonds represent debt ...
[[/A01-NOCTX]]

[[A02-NOCTX]]
The bond's price generally falls when the market yield rises ...
[[/A02-NOCTX]]
```

The delimiters are essential. They let the app map each response to the correct question without needing an API or manually renaming hundreds of files.

## How the scoring works

The interface exposes the **reference answer and required points only to you, the grader**. Assign scores per case:

| Metric | Human input | Report calculation |
| --- | --- | --- |
| Factual accuracy | Incorrect=0, partially correct=1, correct=2 | Fully correct rate = `count(2) / graded` |
| Factual partial-credit score | Same 0/1/2 score | `sum(score) / (2 * graded)` |
| Numerical reasoning | 0/1/2, or N/A if not applicable | Fully correct among numerical cases |
| Answer completeness | Missing=0, partial=1, complete=2 | `sum(score) / (2 * graded)` |
| Hallucination rate | 1 if any unsupported invented factual claim, otherwise 0 | `sum(flags) / graded` |
| Risk interpretation | 0/1/2, or N/A | Mean normalized risk score |
| Response consistency | Repeat the same case in new chats | Exact-text agreement reported as a narrow proxy; semantic consistency needs separate review |
| Response time | Not collected for manual web chats | **Not comparable or reported** |

Report denominators and sample sizes for each metric. **N/A items are excluded**, not treated as zero. Error-free answers can still receive partial factual/completeness scores. The automatic code *aggregates your judgments*: it does not falsely claim that keyword matching can verify nuanced financial facts.

The saved files are:

- `model_answers/responses.csv`: model/version, question ID, condition, trial, response and import time. Appends preserve previous data.
- `evaluation/grades.csv`: rubric scores, hallucination flag, reviewer name, justification. Appends preserve audit history; the latest grade per reviewer/case is used in that reviewer's report.
- `results/metrics_by_model.csv` and `results/metrics_by_category.csv`: per-condition metrics and coverage.
- `results/figures/*.png`: factual accuracy and hallucination comparisons.

`export_results.py` also warns about cases with responses but no grades. Do not report pilot results as final conclusions.

## Academic integrity and verification

- All 120 main reference answers in the supplied master dataset currently carry **PROVISIONAL** source-verification status. Independently check the financial figures against the linked filings **before final grading**.
- The baseline and evidence-provided conditions must be analyzed separately. A model can look better only because additional context was supplied.
- Web chat sessions have hidden system settings, browsing options, usage caps and model routing, so the paper compares **manual website usage modes**, not pure model architecture. Document the test dates and model labels.
- Record why you marked an answer as hallucinated, and ideally have a second reviewer independently score a subset. Use the app's Reviewer field to record separate evaluations.
- Do not infer overall investment safety from this small benchmark. It evaluates answer quality for historical financial questions.

## Migrating from your previous folders (safe steps)

Your existing Mac folder currently contains `data/` and `FinLingo_Hybrid_Benchmark/`. **Keep the original `data/` untouched**. Extract this new ZIP in a different location first. Open and validate this project, and copy any answers you already collected if there are any.

Only after confirming the new project works, you may move the old **`FinLingo_Hybrid_Benchmark` folder** into macOS Trash using Finder. Do **not** delete the parent `FinLingo-LLM-Benchmark`, its `data/`, or your original Excel/CSV files. Your old `.venv` can remain; it is not a dataset.

To verify the dataset's structure, run `python3 -m unittest discover -s tests -v`. The master CSV and blind prompts are unmodified copies from the earlier FinLingo dataset build.
