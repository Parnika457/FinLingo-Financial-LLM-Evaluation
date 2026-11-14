"""Offline summary and plots of human-graded model answers."""

from __future__ import annotations

import argparse

from research import RESULTS, compute_metrics, export_csv, lexical_repeat_agreement


def generate(phase: str, reviewer: str = "primary", trial: int = 1):
    overall, categories = compute_metrics(phase, reviewer, trial)
    prefix = f"{phase}_{reviewer}_trial{trial}"
    path_overall = RESULTS / f"{prefix}_metrics_by_model.csv"
    path_categories = RESULTS / f"{prefix}_metrics_by_category.csv"
    export_csv(path_overall, overall)
    export_csv(path_categories, categories)
    repeats = lexical_repeat_agreement(phase)
    if repeats:
        export_csv(RESULTS / f"{phase}_repeat_exact_text_proxy.csv", repeats)

    scored = [row for row in overall if row["graded"]]
    if scored:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for condition in sorted(set(row["condition"] for row in scored)):
            part = [r for r in scored if r["condition"] == condition]
            names = [row["model"] for row in part]
            accuracy = [row["fully_correct_pct"] for row in part]
            halluc = [row["hallucination_rate_pct"] for row in part]
            n = [row["graded"] for row in part]
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            pos = list(range(len(names)))
            ax.bar([x - 0.2 for x in pos], accuracy, width=0.39, color="#4F46E5", label="Fully correct")
            ax.bar([x + 0.2 for x in pos], halluc, width=0.39, color="#F97373", label="Hallucination flagged")
            ax.set_xticks(pos, [f"{model}\nn={count}" for model, count in zip(names, n)])
            ax.set_ylim(0, 100)
            ax.set_ylabel("Percent of graded responses")
            ax.set_title(f"FinLingo: {phase} / {condition} (reviewer: {reviewer})")
            ax.legend(loc="upper right")
            ax.grid(axis="y", alpha=0.18)
            fig.tight_layout()
            folder = RESULTS / "figures"
            folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(folder / f"{prefix}_{condition}.png", dpi=170)
            plt.close(fig)

    print("Saved per-model metrics:", path_overall)
    print("Saved per-category metrics:", path_categories)
    if not scored:
        print("No graded cases yet. CSVs show 0 coverage, not model performance.")
    else:
        for r in scored:
            print(f"{r['model']} / {r['condition']}: graded {r['graded']}/{r['total_cases']}, "
                  f"fully correct {r['fully_correct_pct']}%, hallucination {r['hallucination_rate_pct']}%")
    return overall, categories


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--reviewer", default="primary")
    parser.add_argument("--trial", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    generate(args.phase, args.reviewer, args.trial)
