from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="outputs/synthetic/summary.json")
    parser.add_argument("--out", default="paper/tables/main_results.tex")
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))["summary"]
    rows = []
    for key, label in [
        ("oa", "OA"),
        ("aa", "AA"),
        ("kappa", "Kappa"),
        ("macro_f1", "Macro-F1"),
        ("worst_class_accuracy", "Worst-Class Acc."),
        ("rare_class_accuracy", "Rare-Class Acc."),
        ("ece", "ECE"),
    ]:
        item = summary[key]
        rows.append(f"{label} & {item['mean'] * 100:.2f} $\\pm$ {item['std'] * 100:.2f} \\\\")
    table = "\\begin{tabular}{lc}\n\\hline\nMetric & DR-GSMamba \\\\\n\\hline\n"
    table += "\n".join(rows)
    table += "\n\\hline\n\\end{tabular}\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
