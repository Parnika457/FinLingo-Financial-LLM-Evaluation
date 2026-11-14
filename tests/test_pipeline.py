"""Offline reproducibility tests: no keys, no network, no external model calls."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research import (BOOKLETS, grade_metrics, load_cases, load_gold,
                      load_manifest, parse_answers, prompt_hash)
import research


class TestManualBenchmark(unittest.TestCase):
    def test_source_counts(self):
        self.assertEqual(len(load_gold("final")), 120)
        self.assertEqual(len(load_cases("final")), 200)
        self.assertEqual(len(load_cases("pilot")), 20)

    def test_independent_pilot(self):
        final_ids = {r["question_id"] for r in load_cases("final")}
        pilot_ids = {r["question_id"] for r in load_cases("pilot")}
        self.assertFalse(final_ids & pilot_ids)

    def test_all_cases_in_manifest_once(self):
        m = load_manifest()
        self.assertEqual(len(m), 22)
        ids = [rid for b in m for rid in b["run_ids"]]
        self.assertEqual(len(ids), 220)
        self.assertEqual(len(set(ids)), 220)
        for b in m:
            self.assertEqual(len(b["run_ids"]), 10)
            self.assertTrue((BOOKLETS / b["pdf"]).exists())
            self.assertTrue((BOOKLETS / b["txt"]).exists())
        self.assertEqual(set(ids), {r["run_id"] for r in load_cases("pilot") + load_cases("final")})

    def test_parser_and_missing_warning(self):
        text = "[[A01-NOCTX]]Owner equity.\nDetails.[[/A01-NOCTX]]\n"
        parsed = parse_answers(text, ["A01-NOCTX", "A02-NOCTX"])
        self.assertEqual(parsed["answers"]["A01-NOCTX"], "Owner equity.\nDetails.")
        self.assertEqual(parsed["missing"], ["A02-NOCTX"])

    def test_parser_rejects_inconsistent_marker(self):
        bad = parse_answers("[[A01-NOCTX]]Hello[[/A02-NOCTX]]", ["A01-NOCTX"])
        self.assertFalse(bad["answers"])
        self.assertTrue(bad["possibly_malformed"])

    def test_metrics_use_non_na_denominators(self):
        data = [{"factual_score": "2", "numerical_score": "N/A", "completeness_score": "2",
                 "hallucination_flag": "0", "risk_score": "N/A"},
                {"factual_score": "0", "numerical_score": "2", "completeness_score": "1",
                 "hallucination_flag": "1", "risk_score": "1"}]
        metric = grade_metrics(data)
        self.assertEqual(metric["fully_correct_pct"], 50)
        self.assertEqual(metric["numerical_n"], 1)
        self.assertEqual(metric["numerical_correct_pct"], 100)
        self.assertEqual(metric["completeness_pct"], 75)
        self.assertEqual(metric["hallucination_rate_pct"], 50)

    def test_gold_never_in_model_prompts(self):
        prompts = load_cases("final")
        gold = load_gold("final")
        for p in prompts:
            expected = gold[p["question_id"]]["reference_answer"].strip()
            self.assertNotIn(expected, p["user_prompt"])
            self.assertEqual(len(prompt_hash(p)), 64)

    def test_import_grade_and_report_end_to_end(self):
        """Use disposable files; never invent research answers in delivered data."""
        original_answer, original_grade = research.ANSWER_FILE, research.GRADE_FILE
        with tempfile.TemporaryDirectory() as temp:
            research.ANSWER_FILE = Path(temp) / "answers.csv"
            research.GRADE_FILE = Path(temp) / "grades.csv"
            try:
                booklet = next(x for x in load_manifest() if x["booklet_id"] == "pilot_01")
                first = booklet["run_ids"][0]
                body = f"[[{first}]]Sample answer, for test only.[[/{first}]]"
                report = research.import_booklet("GPT", "Test Model", "no tools", 1, booklet, body)
                self.assertEqual(report["imported"], 1)
                self.assertEqual(len(report["missing"]), 9)
                research.save_grade("primary", "GPT", first, 1, 2, "N/A", 1, 0, "N/A", "Test only")
                overall, _ = research.compute_metrics("pilot", "primary", 1)
                found = next(r for r in overall if r["model"] == "GPT")
                self.assertEqual(found["graded"], 1)
                self.assertEqual(found["fully_correct_pct"], 100)
                self.assertEqual(found["responses_collected"], 1)
            finally:
                research.ANSWER_FILE, research.GRADE_FILE = original_answer, original_grade


if __name__ == "__main__":
    unittest.main()
