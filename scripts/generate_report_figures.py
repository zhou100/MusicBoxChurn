"""Generate figures for REPORT.md from saved run artifacts.

Reads each artifacts/<model>_<ts>/ run, recomputes test-set raw and
calibrated scores, and writes:

  report/figures/model_comparison.png       bar chart of test PR-AUC + lift
  report/figures/pr_curves.png              PR curves, all models
  report/figures/reliability.png            reliability before AND after calibration
  report/figures/lift_curve.png             cumulative-gains curve
  report/figures/feature_importance.png     top 15 features (RF + GBM, side-by-side)
  report/figures/score_histogram.png        score distribution by true label
  report/figures/decision_curve.png         expected net value vs audience size

Run from repo root: PYTHONPATH=src python3 scripts/generate_report_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicbox_churn.data.load_data import load_csv
from musicbox_churn.data.schema import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COL,
)
from musicbox_churn.data.splits import stratified_split
from musicbox_churn.inference.batch_score import load_run

FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"lr": "#4C72B0", "rf": "#DD8452", "gbm": "#55A467", "mlp": "#C44E52"}
NICE = {"lr": "Logistic Regression", "rf": "Random Forest", "gbm": "LightGBM", "mlp": "MLP (PyTorch)"}

# Locked unit economics (configs/inference.yaml). Reframe as
# "ranked-audience economics under assumed unit costs", not real revenue.
COST_PER_INTERVENTION = 1.0
VALUE_PER_RETAINED = 10.0
DEFAULT_INTERVENTION_SUCCESS = 0.10  # fraction of true churners who stay if contacted

sns.set_theme(style="whitegrid", context="talk")

Scores = tuple[np.ndarray, np.ndarray, np.ndarray | None]  # y, raw, cal


def latest_runs() -> dict[str, Path]:
    """Pick the newest run for each model_type."""
    out: dict[str, Path] = {}
    for d in sorted((ROOT / "artifacts").glob("*_*")):
        if not d.is_dir():
            continue
        try:
            mtype = json.loads((d / "feature_schema.json").read_text()).get("model_type")
        except FileNotFoundError:
            mtype = d.name.split("_", 1)[0]
        if mtype is None:
            mtype = d.name.split("_", 1)[0]
        if mtype not in out or d.stat().st_mtime > out[mtype].stat().st_mtime:
            out[mtype] = d
    return out


def get_test_split_scores(runs: dict[str, Path]) -> dict[str, Scores]:
    df = load_csv(ROOT / "Processed_data" / "df_model_final.csv")
    splits = stratified_split(df, ratios=(0.70, 0.15, 0.15), seed=42)
    test_df = df.iloc[splits.test].reset_index(drop=True)
    y = test_df[TARGET_COL].to_numpy()

    out: dict[str, Scores] = {}
    for mtype, run_dir in runs.items():
        handle, pre, calibrator, _ = load_run(run_dir)
        X = np.asarray(pre.transform(test_df[FEATURE_COLUMNS]), dtype=np.float32)
        raw = handle.predict_proba(X)
        cal = calibrator.transform(raw) if calibrator is not None else None
        out[mtype] = (y, raw, cal)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_model_comparison(runs: dict[str, Path]) -> None:
    rows = []
    for mtype, d in runs.items():
        m = json.loads((d / "metrics.json").read_text())["test_metrics"]
        rows.append(
            {
                "model": NICE[mtype],
                "key": mtype,
                "PR-AUC": m["pr_auc"],
                "ROC-AUC": m["roc_auc"],
                "lift@10pct": m["lift_at_10pct"],
                "Brier (lower=better)": m["brier"],
            }
        )
    df = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for ax, metric in zip(axes, ["PR-AUC", "ROC-AUC", "lift@10pct", "Brier (lower=better)"]):
        colors = [PALETTE[k] for k in df["key"]]
        bars = ax.bar(df["model"], df[metric], color=colors, edgecolor="black", linewidth=0.8)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, df[metric]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=12)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
    fig.suptitle("Test-set metrics across models (n=8,692; prevalence=0.621)", y=1.02, fontsize=18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_pr_curves(scored: dict[str, Scores]) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for mtype, (y, s, _) in scored.items():
        precision, recall, _ = precision_recall_curve(y, s)
        ax.plot(recall, precision, label=NICE[mtype], color=PALETTE[mtype], linewidth=2.5)
    base = float(np.mean(next(iter(scored.values()))[0]))
    ax.axhline(base, ls="--", color="grey", label=f"Prevalence = {base:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curves (test set)")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _reliability_points(y: np.ndarray, s: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(s, bins[1:-1])
    xs, ys, ws = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        xs.append(s[mask].mean())
        ys.append(y[mask].mean())
        ws.append(int(mask.sum()))
    return np.array(xs), np.array(ys), np.array(ws)


def fig_reliability(scored: dict[str, Scores]) -> None:
    """One panel per model, showing raw + calibrated reliability overlaid."""
    n = len(scored)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    for ax, (mtype, (y, raw, cal)) in zip(axes.ravel(), scored.items()):
        ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
        x_r, y_r, w_r = _reliability_points(y, raw)
        ax.plot(x_r, y_r, marker="o", color=PALETTE[mtype], linewidth=2,
                label="Raw scores", markersize=8)
        if cal is not None:
            x_c, y_c, w_c = _reliability_points(y, cal)
            ax.plot(x_c, y_c, marker="s", color="black", linewidth=2, alpha=0.65,
                    label="Calibrated (isotonic)", markersize=7, linestyle=":")
        ax.set_title(NICE[mtype])
        ax.set_xlabel("Mean predicted score / probability")
        ax.set_ylabel("Empirical positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=11)
    fig.suptitle("Reliability — raw scores vs isotonic-calibrated probabilities",
                 y=1.0, fontsize=16)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reliability.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_lift_curve(scored: dict[str, Scores]) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for mtype, (y, s, _) in scored.items():
        order = np.argsort(-s)
        y_sorted = y[order]
        cum_pos = np.cumsum(y_sorted)
        cum_pct_audience = np.arange(1, len(y) + 1) / len(y)
        cum_pct_caught = cum_pos / cum_pos[-1]
        ax.plot(cum_pct_audience, cum_pct_caught, label=NICE[mtype], color=PALETTE[mtype], linewidth=2.5)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random")
    ax.set_xlabel("Fraction of audience contacted (sorted by score)")
    ax.set_ylabel("Fraction of churners caught")
    ax.set_title("Cumulative-gains curve — \"How much budget catches how much churn?\"")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    for marker in (0.10, 0.20, 0.50):
        ax.axvline(marker, color="lightgrey", linewidth=1, ls=":")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lift_curve.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_decision_curve(scored: dict[str, Scores]) -> None:
    """Expected net value vs fraction of audience contacted, under assumed
    unit economics. Three intervention-success scenarios on the left;
    optimal-cutoff sensitivity to success rate on the right (using the
    best-PR-AUC model, GBM by default).
    """
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    # --- Panel 1: model-vs-random net value across three success-rate regimes
    ax = axes[0]
    mtype = "gbm" if "gbm" in scored else next(iter(scored))
    y, raw, cal = scored[mtype]
    s = cal if cal is not None else raw
    order = np.argsort(-s)
    y_sorted = y[order].astype(float)
    cum_caught_model = np.cumsum(y_sorted)
    n = len(y)
    ks = np.arange(1, n + 1)
    # Random baseline: expected churners caught is prevalence × k.
    prevalence = float(y.mean())
    cum_caught_random = prevalence * ks

    scenarios = [
        ("low (10%)",   0.10, "#888888"),
        ("medium (20%)", 0.20, PALETTE["gbm"]),
        ("high (35%)",  0.35, PALETTE["rf"]),
    ]
    frac = ks / n
    for label, r, color in scenarios:
        net_model = cum_caught_model * r * VALUE_PER_RETAINED - ks * COST_PER_INTERVENTION
        net_rand = cum_caught_random * r * VALUE_PER_RETAINED - ks * COST_PER_INTERVENTION
        ax.plot(frac, net_model, color=color, linewidth=2.6, label=f"Model — {label} success")
        ax.plot(frac, net_rand, color=color, linewidth=1.4, linestyle=":", alpha=0.85,
                label=f"Random — {label}")
        argmax = int(np.argmax(net_model))
        ax.scatter([frac[argmax]], [net_model[argmax]], color=color, edgecolor="black",
                   s=140, zorder=5)
    ax.axhline(0, color="grey", linewidth=1)
    ax.set_xlabel("Fraction of audience contacted")
    ax.set_ylabel(f"Expected net value ($)\n(${VALUE_PER_RETAINED:.0f}/save, "
                  f"${COST_PER_INTERVENTION:.0f}/contact, n={n:,})")
    ax.set_title(
        f"Decision curve — {NICE[mtype]} vs random outreach\n"
        "(dot = optimal cutoff under each scenario)",
        fontsize=14,
    )
    ax.legend(loc="lower right", fontsize=10, ncol=1)

    # --- Panel 2: optimal-cutoff sensitivity to success rate -----------------
    ax = axes[1]
    success_rates = np.linspace(0.05, 0.40, 36)
    optimal_fracs = []
    optimal_nets = []
    rand_nets = []
    for r in success_rates:
        net_model = cum_caught_model * r * VALUE_PER_RETAINED - ks * COST_PER_INTERVENTION
        net_rand = cum_caught_random * r * VALUE_PER_RETAINED - ks * COST_PER_INTERVENTION
        i = int(np.argmax(net_model))
        j = int(np.argmax(net_rand))
        optimal_fracs.append(ks[i] / n)
        optimal_nets.append(net_model[i])
        rand_nets.append(max(0.0, net_rand[j]))
    ax2 = ax.twinx()
    ax.plot(success_rates * 100, np.array(optimal_fracs) * 100, color=PALETTE[mtype],
            linewidth=2.6, marker="o", markersize=5)
    ax.set_xlabel("Assumed intervention success rate (%)")
    ax.set_ylabel("Optimal audience contacted (%)", color=PALETTE[mtype])
    ax.tick_params(axis="y", colors=PALETTE[mtype])

    ax2.plot(success_rates * 100, optimal_nets, color="black", linewidth=2,
             linestyle="--", marker="s", markersize=5, label="Model")
    ax2.plot(success_rates * 100, rand_nets, color="grey", linewidth=1.5,
             linestyle=":", marker="x", markersize=5, label="Random outreach")
    ax2.set_ylabel("Best achievable net value ($)", color="black")
    ax2.legend(loc="upper left", fontsize=11)

    ax.set_title(f"Sensitivity: optimal cutoff and best net value vs assumed success\n({NICE[mtype]})",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "decision_curve.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _ohe_feature_names() -> list[str]:
    rf_run = next(iter(d for k, d in latest_runs().items() if k == "rf"), None)
    if rf_run is None:
        return list(NUMERIC_FEATURE_COLUMNS)
    pre = joblib.load(rf_run / "preprocessor.pkl")
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        return list(NUMERIC_FEATURE_COLUMNS) + list(CATEGORICAL_FEATURE_COLUMNS)


def fig_feature_importance(runs: dict[str, Path]) -> None:
    feature_names = _ohe_feature_names()
    panels: list[tuple[str, np.ndarray]] = []
    for mtype in ("rf", "gbm"):
        if mtype not in runs:
            continue
        blob = joblib.load(runs[mtype] / "model.pkl")
        est = blob["estimator"]
        importances = getattr(est, "feature_importances_", None)
        if importances is None:
            continue
        panels.append((mtype, importances))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 9), sharey=False)
    if len(panels) == 1:
        axes = [axes]
    for ax, (mtype, imp) in zip(axes, panels):
        names = feature_names[: len(imp)]
        order = np.argsort(imp)[-15:]
        ax.barh(np.array(names)[order], imp[order], color=PALETTE[mtype], edgecolor="black", linewidth=0.6)
        ax.set_title(f"{NICE[mtype]} — top 15 features")
        ax.set_xlabel("Importance" + ("" if mtype != "gbm" else " (split gain)"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_score_histogram(scored: dict[str, Scores]) -> None:
    mtype = "gbm" if "gbm" in scored else next(iter(scored))
    y, _raw, cal = scored[mtype]
    s = cal if cal is not None else _raw
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(s[y == 0], bins=40, alpha=0.65, label="Active (label=0)", color="#4C72B0", edgecolor="white")
    ax.hist(s[y == 1], bins=40, alpha=0.65, label="Churned (label=1)", color="#DD8452", edgecolor="white")
    ax.set_xlabel("Calibrated P(churn)")
    ax.set_ylabel("Users (test set)")
    ax.set_title(f"Calibrated probability distribution by true label — {NICE[mtype]}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "score_histogram.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    runs = latest_runs()
    print(f"Using runs: { {k: str(v.name) for k, v in runs.items()} }")
    scored = get_test_split_scores(runs)

    fig_model_comparison(runs)
    fig_pr_curves(scored)
    fig_reliability(scored)
    fig_lift_curve(scored)
    fig_decision_curve(scored)
    fig_feature_importance(runs)
    fig_score_histogram(scored)
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
