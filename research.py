"""Local-only data reading, manual response importing, rubrics and summaries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PILOT = ROOT / "pilot"
BOOKLETS = ROOT / "question_booklets"
ANSWER_FILE = ROOT / "model_answers" / "responses.csv"
GRADE_FILE = ROOT / "evaluation" / "grades.csv"
RESULTS = ROOT / "results"
MODELS = ("GPT", "Claude", "Gemini", "Qwen")

RESPONSE_COLUMNS = ["model", "model_version", "mode", "run_id", "question_id",
                    "condition", "phase", "trial", "answer", "booklet_id",
                    "prompt_sha256", "imported_utc"]
GRADE_COLUMNS = ["reviewer", "model", "run_id", "trial", "factual_score",
                 "numerical_score", "completeness_score", "hallucination_flag",
                 "risk_score", "notes", "graded_utc"]

# The opening and closing IDs must match exactly; answers can contain newlines.
ANSWER_RE = re.compile(r"\[\[(?P<id>[A-Za-z0-9-]+)\]\]\s*"
                       r"(?P<answer>.*?)\s*\[\[/(?P=id)\]\]", re.S)
MARKER_RE = re.compile(r"\[\[/?([A-Za-z0-9-]+)\]\]")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_cases(phase: str) -> list[dict]:
    if phase == "pilot":
        return load_jsonl(PILOT / "pilot_prompts.jsonl")
    if phase == "final":
        return load_jsonl(DATA / "FinLingo_200_Blind_Model_Prompts.jsonl")
    raise ValueError(f"Unknown phase: {phase}")


def cases_by_id() -> dict[str, dict]:
    return {r["run_id"]: {**r, "phase": phase}
            for phase in ("pilot", "final") for r in load_cases(phase)}


def load_gold(phase: str) -> dict[str, dict]:
    p = (PILOT / "pilot_questions.csv") if phase == "pilot" else (DATA / "FinLingo_120_Master_Dataset.csv")
    with p.open(newline="", encoding="utf-8-sig") as handle:
        return {r["question_id"]: r for r in csv.DictReader(handle)}


def prompt_hash(item: dict) -> str:
    prompt = item["system_prompt"].strip() + "\n\n" + item["user_prompt"].strip()
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_rows(path: Path, fieldnames: list[str], batch: list[dict]) -> int:
    if not batch:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows({k: entry.get(k, "") for k in fieldnames} for entry in batch)
    return len(batch)


def load_manifest() -> list[dict]:
    with (BOOKLETS / "booklet_manifest.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_answers(raw: str, expected_ids: list[str]) -> dict:
    """Return parsed answers and explicit warnings. Does NOT silently fill gaps."""
    matches = [(m.group("id").upper(), m.group("answer").strip()) for m in ANSWER_RE.finditer(raw)]
    expected = set(expected_ids)
    found: dict[str, str] = {}
    duplicate, unexpected, empty = [], [], []
    for rid, answer in matches:
        if rid not in expected:
            unexpected.append(rid)
        elif rid in found:
            duplicate.append(rid)
        elif not answer:
            empty.append(rid)
        else:
            found[rid] = answer
    malformed = sorted(set(m.upper() for m in MARKER_RE.findall(raw)) - {m[0] for m in matches})
    missing = [rid for rid in expected_ids if rid not in found]
    return {"answers": found, "missing": missing, "unexpected": unexpected,
            "duplicate": duplicate, "empty": empty, "possibly_malformed": malformed}


def latest_responses(phase: str | None = None) -> dict[tuple[str, str, int], dict]:
    index = {}
    for row in rows(ANSWER_FILE):
        if phase and row.get("phase") != phase:
            continue
        index[(row["model"], row["run_id"], int(row["trial"]))] = row
    return index


def import_booklet(model: str, model_version: str, mode: str, trial: int,
                   booklet: dict, raw: str, *, allow_replace=False) -> dict:
    if model not in MODELS:
        raise ValueError("Unknown model")
    if not model_version.strip() or not 1 <= trial <= 3:
        raise ValueError("Enter exact model version and trial 1-3")
    parsed = parse_answers(raw, booklet["run_ids"])
    if parsed["duplicate"] or parsed["unexpected"] or parsed["empty"] or parsed["possibly_malformed"]:
        raise ValueError("Ambiguous/malformed response markers. Fix the marked issues before saving: " +
                         str({k: v for k, v in parsed.items() if k != "answers"}))
    by_id = cases_by_id()
    existing = latest_responses()
    entries = []
    skipped = []
    for rid, answer in parsed["answers"].items():
        key = (model, rid, trial)
        if key in existing and not allow_replace:
            skipped.append(rid)
            continue
        item = by_id[rid]
        entries.append({"model": model, "model_version": model_version.strip(),
                        "mode": mode, "run_id": rid, "question_id": item["question_id"],
                        "condition": item["condition"], "phase": booklet["phase"],
                        "trial": trial, "answer": answer, "booklet_id": booklet["booklet_id"],
                        "prompt_sha256": prompt_hash(item), "imported_utc": now_utc()})
    append_rows(ANSWER_FILE, RESPONSE_COLUMNS, entries)
    return {"imported": len(entries), "skipped_existing": skipped, "missing": parsed["missing"]}


def latest_grades(reviewer: str | None = None) -> dict[tuple[str, str, int, str], dict]:
    index = {}
    for row in rows(GRADE_FILE):
        if reviewer and row["reviewer"] != reviewer:
            continue
        index[(row["model"], row["run_id"], int(row["trial"]), row["reviewer"])] = row
    return index


def save_grade(reviewer: str, model: str, run_id: str, trial: int, factual_score: int,
               numerical_score: str, completeness_score: int,
               hallucination_flag: int, risk_score: str, notes: str) -> None:
    if not reviewer.strip():
        raise ValueError("Reviewer name required")
    if (model, run_id, trial) not in latest_responses():
        raise ValueError("Cannot grade a missing response")
    if factual_score not in (0, 1, 2) or completeness_score not in (0, 1, 2):
        raise ValueError("Factual and completeness scores must be 0-2")
    if str(numerical_score) not in ("N/A", "0", "1", "2") or str(risk_score) not in ("N/A", "0", "1", "2"):
        raise ValueError("Numerical/risk must be N/A or 0-2")
    if hallucination_flag not in (0, 1):
        raise ValueError("Hallucination flag must be 0 or 1")
    append_rows(GRADE_FILE, GRADE_COLUMNS, [{
        "reviewer": reviewer.strip(), "model": model, "run_id": run_id,
        "trial": trial, "factual_score": factual_score,
        "numerical_score": numerical_score, "completeness_score": completeness_score,
        "hallucination_flag": hallucination_flag, "risk_score": risk_score,
        "notes": notes.strip(), "graded_utc": now_utc(),
    }])


def grade_metrics(items: list[dict]) -> dict:
    """Aggregate human rubric judgments with explicit denominators."""
    n = len(items)
    if not n:
        return {"graded": 0, "fully_correct_pct": None, "factual_partial_credit_pct": None,
                "numerical_correct_pct": None, "numerical_n": 0,
                "completeness_pct": None, "hallucination_rate_pct": None,
                "risk_quality_pct": None, "risk_n": 0}
    scores = [int(x["factual_score"]) for x in items]
    numerical = [int(x["numerical_score"]) for x in items if x["numerical_score"] != "N/A"]
    risk = [int(x["risk_score"]) for x in items if x["risk_score"] != "N/A"]
    return {
        "graded": n,
        "fully_correct_pct": round(100 * sum(x == 2 for x in scores) / n, 2),
        "factual_partial_credit_pct": round(100 * sum(scores) / (2 * n), 2),
        "numerical_correct_pct": round(100 * sum(x == 2 for x in numerical) / len(numerical), 2) if numerical else None,
        "numerical_n": len(numerical),
        "completeness_pct": round(100 * sum(int(x["completeness_score"]) for x in items) / (2 * n), 2),
        "hallucination_rate_pct": round(100 * sum(int(x["hallucination_flag"]) for x in items) / n, 2),
        "risk_quality_pct": round(100 * sum(risk) / (2 * len(risk)), 2) if risk else None,
        "risk_n": len(risk),
    }


def compute_metrics(phase: str, reviewer: str = "primary", trial: int = 1):
    """Per model and per category; case condition kept separate."""
    cases = {r["run_id"]: r for r in load_cases(phase)}
    responses = latest_responses(phase)
    grades = latest_grades(reviewer)
    groups = defaultdict(list)
    categories = defaultdict(list)
    matched = defaultdict(set)
    for (model, rid, t), response in responses.items():
        if t != trial or rid not in cases:
            continue
        condition = cases[rid]["condition"]
        matched[(model, condition)].add(rid)
        grade = grades.get((model, rid, t, reviewer))
        if grade:
            groups[(model, condition)].append(grade)
            categories[(model, condition, cases[rid]["category"])].append(grade)
    condition_counts = defaultdict(int)
    for case in cases.values():
        condition_counts[case["condition"]] += 1

    overall_rows = []
    for model in MODELS:
        for condition, total in condition_counts.items():
            scores = groups.get((model, condition), [])
            overall_rows.append({"phase": phase, "reviewer": reviewer, "trial": trial,
                                 "model": model, "condition": condition,
                                 "total_cases": total, "responses_collected": len(matched[(model, condition)]),
                                 **grade_metrics(scores)})
    category_rows = []
    cat_counts = defaultdict(int)
    for case in cases.values():
        cat_counts[(case["condition"], case["category"])] += 1
    for model in MODELS:
        for (condition, category), total in cat_counts.items():
            items = categories.get((model, condition, category), [])
            category_rows.append({"phase": phase, "reviewer": reviewer, "trial": trial,
                                  "model": model, "condition": condition, "category": category,
                                  "total_cases": total,
                                  "responses_collected": sum(1 for rid in matched[(model, condition)]
                                                             if cases[rid]["category"] == category),
                                  **grade_metrics(items)})
    return overall_rows, category_rows


def export_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def lexical_repeat_agreement(phase: str):
    """Very limited exploratory proxy: exact text after whitespace normalization."""
    paired = defaultdict(list)
    for (model, rid, trial), row in latest_responses(phase).items():
        paired[(model, rid)].append((trial, " ".join(row["answer"].lower().split())))
    records = []
    for (model, rid), values in paired.items():
        values.sort()
        if len(values) < 2:
            continue
        records.append({"model": model, "run_id": rid,
                        "repeat_count": len(values),
                        "exact_text_agreement": int(all(v == values[0][1] for _, v in values))})
    return records
