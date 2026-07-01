"""
Reporting -- turns a walk-forward result into a human-readable verdict, a saved
equity-curve plot, and the all-important ACCEPTANCE GATE check (applied to
out-of-sample numbers only).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def check_gate(oos_metrics: dict, gate: config.AcceptanceGate | None = None) -> dict:
    gate = gate or config.GATE
    ts_res = oos_metrics.get("topstep")
    checks = {
        "sharpe >= min": (oos_metrics["sharpe"], gate.min_sharpe,
                          oos_metrics["sharpe"] >= gate.min_sharpe),
        "max_dd_pct <= limit": (abs(oos_metrics["max_drawdown_pct"]), gate.max_drawdown_pct * 100,
                                abs(oos_metrics["max_drawdown_pct"]) <= gate.max_drawdown_pct * 100),
        "profit_factor >= min": (oos_metrics.get("profit_factor") or 0, gate.min_profit_factor,
                                 (oos_metrics.get("profit_factor") or 0) >= gate.min_profit_factor),
        "n_trades >= min": (oos_metrics["n_trades"], gate.min_trades,
                            oos_metrics["n_trades"] >= gate.min_trades),
        "pct_folds_profitable >= min": (oos_metrics.get("pct_folds_profitable", 0),
                                        gate.min_pct_oos_folds_profitable * 100,
                                        oos_metrics.get("pct_folds_profitable", 0) >= gate.min_pct_oos_folds_profitable * 100),
    }
    if gate.must_pass_topstep:
        passed_ts = bool(ts_res.passed) if ts_res is not None else False
        checks["passes_topstep_rules"] = ("-", "-", passed_ts)

    overall = all(v[2] for v in checks.values())
    return {"overall_pass": overall, "checks": checks}


def plot_equity(oos_equity: pd.Series, path, title="Out-of-sample equity"):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(oos_equity.index, oos_equity.values, lw=1.2)
    peak = oos_equity.cummax()
    ax.fill_between(oos_equity.index, oos_equity.values, peak.values,
                    where=(oos_equity < peak), alpha=0.2, color="red")
    ax.axhline(config.TOPSTEP.starting_balance, ls="--", c="grey", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("Account equity ($)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(wf_result, discovery_table, mt_report, instrument_key, out_dir=None):
    out_dir = out_dir or config.OUTPUT_DIR
    gate = check_gate(wf_result.oos_metrics)

    # equity plot
    plot_path = out_dir / f"equity_{instrument_key}.png"
    plot_equity(wf_result.oos_equity, plot_path,
                title=f"{instrument_key} -- stitched OUT-OF-SAMPLE equity")

    # fold table
    wf_result.fold_summaries.to_csv(out_dir / f"folds_{instrument_key}.csv", index=False)
    discovery_table.head(40).to_csv(out_dir / f"discovery_{instrument_key}.csv", index=False)

    m = wf_result.oos_metrics
    ts_res = m.get("topstep")

    lines = []
    P = lines.append
    P("=" * 70)
    P(f"  RESULT FOR {instrument_key}  (OUT-OF-SAMPLE, walk-forward)")
    P("=" * 70)
    P("")
    P("  Signal discovery (in-sample, descriptive only):")
    P(f"    features tested        : {mt_report['n_features_tested']}")
    P(f"    significant (raw p<.05): {mt_report['n_significant_raw']}")
    P(f"    expected by chance     : {mt_report['n_expected_by_chance']}")
    P(f"    survive FDR control    : {mt_report['n_significant_after_fdr']}")
    P(f"    verdict                : {mt_report['verdict']}")
    P("")
    P("  Out-of-sample performance (the numbers that matter):")
    P(f"    total PnL              : ${m['total_pnl_$']:,.0f}  ({m['return_pct']:+.1f}%)")
    P(f"    Sharpe                 : {m['sharpe']:.2f}")
    P(f"    Sortino                : {m['sortino']:.2f}")
    P(f"    max drawdown           : ${m['max_drawdown_$']:,.0f}  ({m['max_drawdown_pct']:.1f}%)")
    P(f"    profit factor          : {m['profit_factor']}")
    P(f"    win rate               : {m['win_rate']:.1f}%")
    P(f"    trades                 : {m['n_trades']}")
    P(f"    folds profitable       : {m.get('pct_folds_profitable')}%")
    P("")
    P("  Overfitting check (in-sample vs out-of-sample):")
    d = wf_result.degradation
    P(f"    mean IS Sharpe         : {d.get('is_sharpe_mean')}")
    P(f"    OOS Sharpe             : {d.get('oos_sharpe')}")
    P(f"    Sharpe retention       : {d.get('sharpe_retention_pct')}%   "
      f"(low % = strategy was largely overfit)")
    P("")
    P("  Topstep rule check (on OOS curve):")
    if ts_res is not None:
        P(f"    passed                 : {ts_res.passed}")
        P(f"    max trailing drawdown  : ${ts_res.max_trailing_drawdown:,.0f}")
        P(f"    worst day              : ${ts_res.worst_day:,.0f}")
        if ts_res.failed_rules:
            for fr in ts_res.failed_rules:
                P(f"    !! {fr}")
    P("")
    P("  ACCEPTANCE GATE (your standards, applied to OOS only):")
    for name, (val, thr, ok) in gate["checks"].items():
        mark = "PASS" if ok else "FAIL"
        P(f"    [{mark}] {name:<28} value={val}  threshold={thr}")
    P("")
    verdict = "ACCEPTED" if gate["overall_pass"] else "REJECTED"
    P(f"  >>> OVERALL: {verdict} <<<")
    if not gate["overall_pass"]:
        P("      Do NOT trade this live. Iterate features/labels, get better data")
        P("      (order flow), or accept that no robust edge was found here.")
    P("=" * 70)

    text = "\n".join(lines)
    (out_dir / f"report_{instrument_key}.txt").write_text(text)

    # machine-readable summary
    summary = {
        "instrument": instrument_key,
        "oos_metrics": {k: v for k, v in m.items() if k != "topstep"},
        "gate": {"overall_pass": gate["overall_pass"],
                 "checks": {k: {"value": str(v[0]), "threshold": str(v[1]), "pass": bool(v[2])}
                            for k, v in gate["checks"].items()}},
        "degradation": wf_result.degradation,
    }
    (out_dir / f"summary_{instrument_key}.json").write_text(json.dumps(summary, indent=2, default=str))

    return text, gate["overall_pass"]
