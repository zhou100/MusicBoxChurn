from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _fmt_metric(name: str, val: float) -> str:
    if name in {"n"}:
        return f"| {name} | {int(val):,} |"
    return f"| {name} | {val:.4f} |"


def _table(metrics: dict[str, float]) -> str:
    rows = [_fmt_metric(k, v) for k, v in metrics.items()]
    return "| metric | value |\n|---|---|\n" + "\n".join(rows)


def _reliability_table(rel: dict[str, list[float]]) -> str:
    header = "| bin | mean_pred | empirical_rate | count |\n|---|---|---|---|"
    rows = []
    for i, (mp, er, c) in enumerate(zip(rel["mean_pred"], rel["empirical_rate"], rel["count"])):
        if c == 0:
            rows.append(f"| {i} | — | — | 0 |")
        else:
            rows.append(f"| {i} | {mp:.3f} | {er:.3f} | {int(c)} |")
    return header + "\n" + "\n".join(rows)


def render_report(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    val = json.loads((run_dir / "val_eval.json").read_text())
    test = json.loads((run_dir / "test_eval.json").read_text())
    threshold = json.loads((run_dir / "threshold.json").read_text())
    schema = json.loads((run_dir / "feature_schema.json").read_text())

    return f"""# Evaluation report — `{run_dir.name}`

Model type: **{schema.get("model_type", "?")}**

## Framing
Outputs are **ranking scores**, not calibrated probabilities. PR-AUC and
recall_at_top_k are the load-bearing metrics; ROC-AUC and Brier are reported
for context. See TODOS.md (P3) for the calibration unwind path if probability
semantics become required downstream.

## Validation set
{_table(val["metrics"])}

### Score distribution (validation)
{_table(val["scores_summary"])}

### Reliability (validation, 10 bins)
{_reliability_table(val["reliability"])}

## Test set
{_table(test["metrics"])}

### Score distribution (test)
{_table(test["scores_summary"])}

### Reliability (test, 10 bins)
{_reliability_table(test["reliability"])}

## Threshold
Policy: `{threshold.get("policy")}`
Selected: `{threshold.get("threshold"):.4f}`
"""


def write_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    text = render_report(run_dir)
    out = run_dir / "evaluation_report.md"
    out.write_text(text)
    logger.info("wrote %s", out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description="Render evaluation_report.md from a run dir.")
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()
    print(write_report(args.run_dir))


if __name__ == "__main__":
    main()
