"""
Command-line entry point for every pipeline in this repo. One subcommand per
strategy family / utility:

    python run.py ml                  data -> features -> labels -> discovery
                                       -> walk-forward -> report (ML strategy)
    python run.py trend                trend-following walk-forward pipeline
    python run.py reactive             reactive order-flow momentum (Databento only)
    python run.py structure            market structure strategy (NQ, real futures)
    python run.py structure-variants   structure: base vs RTH-entry variant comparison
    python run.py adaptive             adaptive ML (decay-weighted, regime-aware)
    python run.py sizing               dynamic position-sizing demo (fixed vs dynamic)
    python run.py smoketest <path>     Databento DBN smoke test (prints loaded bars)

Run `python run.py <command> --help` for that command's options.
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config
import research
import strategies as strat
import walkforward as wf
import backtest as bt
import topstep as ts
import report as rep
import data_clients
import synthetic
from adaptive_model import AdaptiveConfig, run_adaptive_backtest


def _bars_per_day() -> int:
    return {"1m": 390, "5m": 78, "1h": 7}.get(config.INTERVAL, 78)


def _get_universe(source: str, dbn_path: str | None = None):
    """Shared loader for the ml / trend / adaptive subcommands."""
    if source == "eodhd":
        return data_clients.load_universe()
    elif source == "databento":
        if not dbn_path:
            sys.exit("ERROR: --dbn-path is required when --source databento")
        print(f"  loading Databento DBN files from: {dbn_path}")
        return data_clients.load_universe_databento(dbn_path, interval=config.INTERVAL)
    elif source == "synthetic":
        print("  generating synthetic universe (offline) ...")
        return synthetic.generate(history_days=400, interval_min=5)
    else:
        sys.exit(f"unknown source: {source}")


# ============================================================================
# ml -- ML strategy pipeline (data -> features -> labels -> discovery -> WF)
# ============================================================================

def _run_instrument_ml(universe, target: str):
    print(f"\n{'#'*70}\n# {target}\n{'#'*70}")

    print("[1/5] building feature matrix ...")
    X, target_df = research.build_feature_matrix(universe, target)

    print("[2/5] generating labels ...")
    y = research.make_labels(target_df)

    # align & drop warmup NaNs
    common     = X.dropna(how="all").index
    X          = X.loc[common]
    y          = y.loc[common]
    target_df  = target_df.loc[common]

    bpd = _bars_per_day()
    total_bars   = len(X)
    print(f"      {total_bars} bars available ({total_bars // bpd} trading days)")

    print("[3/5] discovery / signal ranking ...")
    table = research.rank_features(X, y["fwd_ret"])
    mt    = research.multiple_testing_report(table)
    pool  = research.top_feature_pool(table, k=25)
    print(f"      top signals: {', '.join(pool[:6])} ...")
    print(f"      multiple-testing verdict: {mt['verdict']} "
          f"({mt['n_significant_after_fdr']} survive FDR vs "
          f"{mt['n_expected_by_chance']} expected by chance)")

    print("[4/5] walk-forward optimization (judging OOS only) ...")
    result = wf.run_walk_forward(
        X, y["direction"], target_df,
        instrument_key=target,
        feature_pool=pool,
        n_contracts=1)

    print("[5/5] writing report ...")
    text, accepted = rep.write_report(result, table, mt, target)
    print("\n" + text)
    return accepted


def cmd_ml(args):
    bpd = _bars_per_day()

    # Switch instrument economics to match the data source.
    config.INSTRUMENTS = config.get_instruments(args.source)
    if args.source == "databento":
        print("  [databento] using real futures contract specs "
              "(NQ $20/pt, ES $50/pt, GC $100/pt)")

    if args.quick:
        # ~6 months train, ~1 month test, 24 param samples
        config.WALKFORWARD.train_bars       = bpd * 120   # 120 trading days
        config.WALKFORWARD.test_bars        = bpd * 20    # 20 trading days
        config.WALKFORWARD.step_bars        = bpd * 20
        config.WALKFORWARD.n_param_samples  = 24
        print(f"  [quick mode] train={config.WALKFORWARD.train_bars} bars "
              f"({120}d), test={config.WALKFORWARD.test_bars} bars ({20}d), "
              f"n_samples={config.WALKFORWARD.n_param_samples}")
    else:
        # Full run: ~1yr train, ~3 month test
        config.WALKFORWARD.train_bars       = bpd * 252   # 1 year
        config.WALKFORWARD.test_bars        = bpd * 60    # 3 months
        config.WALKFORWARD.step_bars        = bpd * 60
        config.WALKFORWARD.n_param_samples  = 60
        print(f"  [full mode] train={config.WALKFORWARD.train_bars} bars, "
              f"test={config.WALKFORWARD.test_bars} bars, "
              f"n_samples={config.WALKFORWARD.n_param_samples}")

    universe = _get_universe(args.source, args.dbn_path)
    targets  = args.instruments or config.TRADABLES

    results = {}
    for t in targets:
        if t not in universe:
            print(f"  skipping {t}: not in universe")
            continue
        try:
            results[t] = _run_instrument_ml(universe, t)
        except Exception as e:
            print(f"  !! {t} failed: {e}")
            import traceback; traceback.print_exc()
            results[t] = None

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for t, ok in results.items():
        status = "ACCEPTED" if ok else ("REJECTED" if ok is False else "ERROR")
        print(f"  {t:4s}: {status}")
    print(f"\nReports & plots written to: {config.OUTPUT_DIR}")


# ============================================================================
# trend -- trend-following pipeline
# ============================================================================

def _run_instrument_trend(target_df: pd.DataFrame, target: str):
    print(f"\n{'#'*70}\n# {target} — TREND FOLLOWING\n{'#'*70}")
    bpd = _bars_per_day()
    print(f"  {len(target_df)} bars ({len(target_df)//bpd} trading days)")

    print("[1/3] walk-forward trend optimization ...")
    result = wf.run_trend_walk_forward(
        target_df, instrument_key=target, n_contracts=1)

    print("[2/3] writing report ...")
    m  = result.oos_metrics
    ts_res = m.get("topstep")
    d  = result.degradation

    lines = []
    P = lines.append
    P("=" * 70)
    P(f"  TREND RESULT FOR {target}  (OUT-OF-SAMPLE, walk-forward)")
    P("=" * 70)
    P(f"    total PnL       : ${m['total_pnl_$']:,.0f}  ({m['return_pct']:+.1f}%)")
    P(f"    Sharpe          : {m['sharpe']:.2f}")
    P(f"    Sortino         : {m['sortino']:.2f}")
    P(f"    max drawdown    : ${m['max_drawdown_$']:,.0f}  ({m['max_drawdown_pct']:.1f}%)")
    P(f"    profit factor   : {m['profit_factor']}")
    P(f"    win rate        : {m['win_rate']:.1f}%")
    P(f"    trades          : {m['n_trades']}")
    P(f"    folds profitable: {m.get('pct_folds_profitable')}%")
    P(f"    avg trade $     : ${m['avg_trade_$']:.2f}")
    P(f"    avg win $       : ${m['avg_win_$']:.2f}")
    P(f"    avg loss $      : ${m['avg_loss_$']:.2f}")
    P("")
    P(f"    IS Sharpe mean  : {d.get('is_sharpe_mean')}")
    P(f"    OOS Sharpe      : {d.get('oos_sharpe')}")
    P(f"    Sharpe retention: {d.get('sharpe_retention_pct')}%")
    P("")
    if ts_res is not None:
        P(f"    Topstep passed  : {ts_res.passed}")
        P(f"    max trail DD    : ${ts_res.max_trailing_drawdown:,.0f}")
        P(f"    worst day       : ${ts_res.worst_day:,.0f}")
        for fr in ts_res.failed_rules:
            P(f"    !! {fr}")
    P("")

    # acceptance gate
    gate_checks = {
        "sharpe >= 1.0":         m["sharpe"] >= 1.0,
        "drawdown <= 10%":       abs(m["max_drawdown_pct"]) <= 10.0,
        "profit_factor >= 1.2":  (m["profit_factor"] or 0) >= 1.2,
        "trades >= 100":         m["n_trades"] >= 100,
        "folds profitable >= 55%": m.get("pct_folds_profitable", 0) >= 55.0,
        "topstep passed":        ts_res.passed if ts_res else False,
    }
    accepted = all(gate_checks.values())
    for name, ok in gate_checks.items():
        P(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    P("")
    P(f"  >>> {'ACCEPTED' if accepted else 'REJECTED'} <<<")
    P("=" * 70)

    text = "\n".join(lines)
    print("\n" + text)

    out = config.OUTPUT_DIR
    (out / f"report_trend_{target}.txt").write_text(text)
    result.fold_summaries.to_csv(out / f"folds_trend_{target}.csv", index=False)

    # equity plot
    fig, ax = plt.subplots(figsize=(11, 5))
    eq = result.oos_equity
    ax.plot(eq.index, eq.values, lw=1.2)
    peak = eq.cummax()
    ax.fill_between(eq.index, eq.values, peak.values,
                    where=(eq < peak), alpha=0.2, color="red")
    ax.axhline(config.TOPSTEP.starting_balance, ls="--", c="grey", lw=0.8)
    ax.set_title(f"{target} TREND — OOS equity")
    ax.set_ylabel("Account equity ($)")
    fig.tight_layout()
    fig.savefig(out / f"equity_trend_{target}.png", dpi=120)
    plt.close(fig)
    print(f"  Plot: {out / f'equity_trend_{target}.png'}")

    return accepted


def cmd_trend(args):
    bpd = _bars_per_day()
    if args.quick:
        config.WALKFORWARD.train_bars      = bpd * 120
        config.WALKFORWARD.test_bars       = bpd * 20
        config.WALKFORWARD.step_bars       = bpd * 20
        config.WALKFORWARD.n_param_samples = 40
        print(f"  [quick] train={config.WALKFORWARD.train_bars} bars, "
              f"test={config.WALKFORWARD.test_bars}, "
              f"n_samples={config.WALKFORWARD.n_param_samples}")
    else:
        config.WALKFORWARD.train_bars      = bpd * 252
        config.WALKFORWARD.test_bars       = bpd * 60
        config.WALKFORWARD.step_bars       = bpd * 60
        config.WALKFORWARD.n_param_samples = 80
        print(f"  [full] train={config.WALKFORWARD.train_bars} bars, "
              f"test={config.WALKFORWARD.test_bars}, "
              f"n_samples={config.WALKFORWARD.n_param_samples}")

    universe = _get_universe(args.source, args.dbn_path)
    targets  = args.instruments or config.TRADABLES

    results = {}
    for t in targets:
        if t not in universe:
            print(f"  skipping {t}: not in universe")
            continue
        try:
            results[t] = _run_instrument_trend(universe[t], t)
        except Exception as e:
            import traceback
            print(f"  !! {t} failed: {e}")
            traceback.print_exc()
            results[t] = None

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for t, ok in results.items():
        status = "ACCEPTED" if ok else ("REJECTED" if ok is False else "ERROR")
        print(f"  {t}: {status}")


# ============================================================================
# reactive -- reactive order-flow momentum (Databento only)
# ============================================================================

_REACTIVE_NUMERIC_PARAMS = [
    "score_threshold", "imbalance_threshold", "delta_zscore_window",
    "bar_delta_window", "max_hold_bars", "large_print_threshold",
]


def _param_importance_reactive(result: "wf.ReactiveWFResult", target: str) -> None:
    """
    Four-part analysis written to output/ after a walk-forward run:
      1. Per-parameter bin → avg OOS Sharpe  (which values work best)
      2. Parameter stability (std dev across folds — low = signal, high = noise)
      3. Top-10 IS performers vs their actual OOS Sharpe  (overfitting check)
      4. score_threshold × max_hold_bars heatmap of avg OOS Sharpe
    """
    folds = result.fold_summaries
    if folds.empty or "oos_sharpe" not in folds.columns:
        print("  (no fold data for importance analysis)")
        return

    out   = config.OUTPUT_DIR
    lines = []
    P     = lines.append

    P("=" * 70)
    P(f"  PARAMETER IMPORTANCE ANALYSIS — {target}")
    P("=" * 70)

    # ---- 1. Bin → avg OOS Sharpe ----------------------------------------
    P("\n  1. Parameter value bins vs average OOS Sharpe")
    P("  " + "-" * 66)
    for col in _REACTIVE_NUMERIC_PARAMS:
        if col not in folds.columns:
            continue
        vals = folds[col].dropna()
        if vals.nunique() < 2:
            continue
        bins  = pd.qcut(vals, q=5, duplicates="drop")
        means = folds.groupby(bins, observed=True)["oos_sharpe"].mean()
        P(f"\n  {col}:")
        for interval, avg in means.items():
            P(f"    {str(interval):>22}  ->  avg OOS Sharpe {avg:+.3f}")

    if "trend_filter" in folds.columns:
        P(f"\n  trend_filter:")
        for val, grp in folds.groupby("trend_filter"):
            P(f"    {str(val):>22}  ->  avg OOS Sharpe {grp['oos_sharpe'].mean():+.3f}")

    # ---- 2. Parameter stability -----------------------------------------
    P("\n\n  2. Parameter stability across folds (low std = consistent = real)")
    P("  " + "-" * 66)
    P(f"  {'parameter':<28}  {'mean':>8}  {'std':>8}  {'cv':>8}")
    for col in _REACTIVE_NUMERIC_PARAMS:
        if col not in folds.columns:
            continue
        v = folds[col].dropna()
        if len(v) < 2:
            continue
        mu, sd = v.mean(), v.std()
        cv = sd / abs(mu) if mu != 0 else float("nan")
        P(f"  {col:<28}  {mu:>8.3f}  {sd:>8.3f}  {cv:>8.3f}")

    # ---- 3. Top-10 IS performers vs OOS (overfitting check) --------------
    P("\n\n  3. Top-10 IS-Sharpe folds vs their actual OOS Sharpe")
    P("  " + "-" * 66)
    if "is_sharpe" in folds.columns:
        top10 = folds.nlargest(10, "is_sharpe")[
            ["fold", "is_sharpe", "oos_sharpe", "oos_pnl_$",
             "score_threshold", "max_hold_bars", "trend_filter"]
        ]
        P(f"  {'fold':>4}  {'IS Sharpe':>10}  {'OOS Sharpe':>10}  "
          f"{'OOS PnL':>10}  {'thr':>4}  {'hold':>5}  {'trend':>5}")
        for _, row in top10.iterrows():
            P(f"  {int(row['fold']):>4}  {row['is_sharpe']:>10.3f}  "
              f"{row['oos_sharpe']:>10.3f}  ${row['oos_pnl_$']:>9,.0f}  "
              f"{int(row.get('score_threshold',0)):>4}  "
              f"{int(row.get('max_hold_bars',0)):>5}  "
              f"{str(row.get('trend_filter','?')):>5}")
        corr = folds[["is_sharpe", "oos_sharpe"]].dropna().corr().iloc[0, 1]
        P(f"\n  IS vs OOS Sharpe correlation across all folds: {corr:+.3f}")
        P("  (> 0.3 = mild IS→OOS predictability; < 0 = pure curve-fitting)")

    P("\n" + "=" * 70)
    text = "\n".join(lines)
    print("\n" + text)
    (out / f"param_importance_{target}.txt").write_text(text, encoding="utf-8")

    # ---- 4. Heatmap: score_threshold × max_hold_bars --------------------
    if "score_threshold" in folds.columns and "max_hold_bars" in folds.columns:
        pivot = (folds.groupby(["score_threshold", "max_hold_bars"])["oos_sharpe"]
                      .mean()
                      .unstack(fill_value=np.nan))

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                       vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(r) for r in pivot.index])
        ax.set_xlabel("max_hold_bars")
        ax.set_ylabel("score_threshold")
        ax.set_title(f"{target} — Avg OOS Sharpe by score_threshold × max_hold_bars")
        plt.colorbar(im, ax=ax, label="avg OOS Sharpe")
        fig.tight_layout()
        fig.savefig(out / f"heatmap_reactive_{target}.png", dpi=130)
        plt.close(fig)
        print(f"  heatmap saved → {out / f'heatmap_reactive_{target}.png'}")


def _load_universe_reactive(dbn_path: str) -> dict[str, pd.DataFrame]:
    print(f"  loading Databento bars from: {dbn_path}")
    return data_clients.load_universe_databento(dbn_path, interval=config.INTERVAL)


def _run_instrument_reactive(universe: dict, target: str) -> bool | None:
    print(f"\n{'#'*70}\n# {target} — REACTIVE ORDER-FLOW MOMENTUM\n{'#'*70}")

    if target not in universe:
        print(f"  skipping {target}: not in loaded universe")
        return None

    tdf = universe[target]
    print(f"  {len(tdf)} bars  "
          f"{tdf.index.min().date()} -> {tdf.index.max().date()}")

    print("[1/3] walk-forward optimization ...")
    result = wf.run_reactive_walk_forward(tdf, instrument_key=target,
                                          n_contracts=2, verbose=True)

    print("[2/3] composing report ...")
    m      = result.oos_metrics
    ts_res = m.get("topstep")
    gate   = rep.check_gate(m)

    lines = []
    P = lines.append
    P("=" * 70)
    P(f"  REACTIVE ORDER-FLOW RESULT FOR {target}  (OUT-OF-SAMPLE, walk-forward)")
    P("=" * 70)
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
    P("  Overfitting check (IS vs OOS):")
    d = result.degradation
    P(f"    mean IS Sharpe         : {d.get('is_sharpe_mean')}")
    P(f"    OOS Sharpe             : {d.get('oos_sharpe')}")
    P(f"    Sharpe retention       : {d.get('sharpe_retention_pct')}%"
      f"   (low = overfit to train)")
    P("")
    P("  Topstep rule check (on OOS curve):")
    if ts_res is not None:
        P(f"    passed                 : {ts_res.passed}")
        P(f"    max trailing drawdown  : ${ts_res.max_trailing_drawdown:,.0f}")
        P(f"    worst day              : ${ts_res.worst_day:,.0f}")
        for fr in ts_res.failed_rules:
            P(f"    !! {fr}")
    P("")
    P("  Parameter stability across folds:")
    folds = result.fold_summaries
    for col in ("score_threshold", "imbalance_threshold",
                "delta_zscore_window", "max_hold_bars", "large_print_threshold"):
        if col in folds.columns:
            vals = folds[col].dropna()
            P(f"    {col:<26}: {vals.mean():.3f}  "
              f"(min={vals.min():.3f}, max={vals.max():.3f})")
    P("")
    P("  ACCEPTANCE GATE (OOS only):")
    for name, (val, thr, ok) in gate["checks"].items():
        mark = "PASS" if ok else "FAIL"
        P(f"    [{mark}] {name:<28} value={val}  threshold={thr}")
    P("")
    verdict = "ACCEPTED" if gate["overall_pass"] else "REJECTED"
    P(f"  >>> OVERALL: {verdict} <<<")
    if not gate["overall_pass"]:
        P("      Do NOT trade this live. Tune the score threshold / hold bars,")
        P("      or accept that no robust edge was found in this data window.")
    P("=" * 70)

    text = "\n".join(lines)
    print("\n" + text)

    out = config.OUTPUT_DIR
    (out / f"report_reactive_{target}.txt").write_text(text, encoding="utf-8")
    result.fold_summaries.to_csv(out / f"folds_reactive_{target}.csv", index=False)

    # Equity plot
    eq   = result.oos_equity
    peak = eq.cummax()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eq.index, eq.values, lw=1.2, color="steelblue", label="OOS equity")
    ax.fill_between(eq.index, eq.values, peak.values,
                    where=(eq < peak), alpha=0.20, color="red")
    ax.axhline(config.TOPSTEP.starting_balance, ls="--", c="grey", lw=0.8,
               label="Starting balance")
    ax.set_title(f"{target} — Reactive Order-Flow (walk-forward OOS)")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / f"equity_reactive_{target}.png", dpi=120)
    plt.close(fig)

    # Machine-readable summary
    summary = {
        "instrument": target,
        "strategy":   "reactive_orderflow",
        "oos_metrics": {k: v for k, v in m.items() if k != "topstep"},
        "gate": {
            "overall_pass": gate["overall_pass"],
            "checks": {
                k: {"value": str(v[0]), "threshold": str(v[1]), "pass": bool(v[2])}
                for k, v in gate["checks"].items()
            },
        },
        "degradation":   result.degradation,
        "chosen_params": result.chosen_params,
    }
    (out / f"summary_reactive_{target}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\n[3/4] parameter importance analysis ...")
    _param_importance_reactive(result, target)

    print(f"\n[4/4] outputs written to {out}")
    return gate["overall_pass"]


def cmd_reactive(args):
    # Databento path always uses real futures contract economics
    config.INSTRUMENTS = config.DATABENTO_INSTRUMENTS
    print("  [databento] instrument specs: "
          "NQ $20/pt  ES $50/pt  GC $100/pt")

    bpd = _bars_per_day()
    if args.quick:
        config.WALKFORWARD.train_bars      = bpd * 60   # ~3 months
        config.WALKFORWARD.test_bars       = bpd * 15    # ~3 weeks
        config.WALKFORWARD.step_bars       = bpd * 15
        config.WALKFORWARD.n_param_samples = args.n_samples or 30
    else:
        config.WALKFORWARD.train_bars      = bpd * 120  # ~6 months
        config.WALKFORWARD.test_bars       = bpd * 30   # ~6 weeks
        config.WALKFORWARD.step_bars       = bpd * 30
        config.WALKFORWARD.n_param_samples = args.n_samples or 500
    print(f"  [{'quick' if args.quick else 'full'}] "
          f"train={config.WALKFORWARD.train_bars} bars "
          f"test={config.WALKFORWARD.test_bars} bars "
          f"n_samples={config.WALKFORWARD.n_param_samples}")

    universe = _load_universe_reactive(args.dbn_path)
    targets  = args.instruments or config.TRADABLES

    results: dict[str, bool | None] = {}
    for t in targets:
        try:
            results[t] = _run_instrument_reactive(universe, t)
        except Exception as e:
            import traceback
            print(f"  !! {t} failed: {e}")
            traceback.print_exc()
            results[t] = None

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for t, ok in results.items():
        if ok is True:
            status = "ACCEPTED"
        elif ok is False:
            status = "REJECTED"
        else:
            status = "ERROR / SKIPPED"
        print(f"  {t:4s}: {status}")
    print(f"\nReports & plots written to: {config.OUTPUT_DIR}")


# ============================================================================
# structure -- market structure strategy (NQ, real futures / QQQ proxy)
# ============================================================================
#
# WHY DATABENTO IS THE DEFAULT:
#     FVGs, liquidity grabs and swing levels are price-structure features that
#     matter because real NQ participants watch real NQ levels.  On the QQQ
#     proxy (~$520) those zones sit at the wrong prices and reflect a different
#     participant base (ETF arb flows), so the structure is economically
#     wrong, not just the dollar PnL.  Databento gives real NQ prices
#     (~20,000) where an FVG at 20,120-20,080 is the actual zone traders react
#     to, and the ATR-scaled thresholds (1.5x ATR ~= a 150-point gap) become
#     meaningful instead of noise.
#
#     --source databento uses DATABENTO_INSTRUMENTS['NQ'] ($20/point, tick 0.25).
#     --source eodhd      uses INSTRUMENTS['NQ'] (MNQ $2/point) — ILLUSTRATIVE.

_STRUCTURE_NUMERIC_PARAMS = [
    "fvg_min_atr_mult", "grab_wick_ratio", "grab_atr_mult",
    "max_hold_bars", "extension_threshold",
]


def _pick_sample_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one week of 5m bars starting ~25 % into the dataset
    (allows daily/hourly indicators to be warm).
    """
    tz = config.SESSION.timezone
    et = df.index.tz_convert(tz)
    days = np.unique(et.date)
    if len(days) < 10:
        return df
    start_day = days[max(5, len(days) // 4)]
    end_day   = days[min(len(days) - 1, int(np.searchsorted(days, start_day)) + 5)]
    mask = (et.date >= start_day) & (et.date <= end_day)
    return df[mask]


def _run_diagnostics_structure(df: pd.DataFrame, target: str, source: str) -> None:
    """
    Print FVG zone count, grab count, signal distribution.
    Plot one day of 5m bars with FVG zones shaded and grabs marked.
    """
    out = config.OUTPUT_DIR
    print(f"\n[DIAGNOSTICS] Sampling from {df.index.min().date()} -> {df.index.max().date()}")
    week_df = _pick_sample_week(df)
    print(f"  sample window: {week_df.index.min().date()} to {week_df.index.max().date()}  "
          f"({len(week_df)} bars)")

    diag = strat.diagnose_structure(week_df)

    # ---- print summary stats ------------------------------------------------
    n_bull_fvg   = int(diag["in_bull_fvg"].sum())
    n_bear_fvg   = int(diag["in_bear_fvg"].sum())
    n_bull_grab  = int(diag["bull_grab"].sum())
    n_bear_grab  = int(diag["bear_grab"].sum())
    n_zones      = len(diag["fvg_zones"])

    sig = diag["signal"]
    n_long  = int((sig == 1).sum())
    n_short = int((sig == -1).sum())
    n_flat  = int((sig == 0).sum())
    total   = max(len(sig), 1)

    print(f"\n  FVG zones detected (3-candle, 5m) : {n_zones}")
    print(f"  5m bars inside a bull FVG zone   : {n_bull_fvg}  "
          f"({n_bull_fvg / total * 100:.1f}% of bars)")
    print(f"  5m bars inside a bear FVG zone   : {n_bear_fvg}  "
          f"({n_bear_fvg / total * 100:.1f}% of bars)")
    print(f"  Bullish grab bars                : {n_bull_grab}")
    print(f"  Bearish grab bars                : {n_bear_grab}")
    print(f"\n  Signal distribution (default params):")
    print(f"    long  : {n_long:5d}  ({n_long  / total * 100:.1f}%)")
    print(f"    short : {n_short:5d}  ({n_short / total * 100:.1f}%)")
    print(f"    flat  : {n_flat:5d}  ({n_flat  / total * 100:.1f}%)")

    # ---- pick one day to plot -----------------------------------------------
    tz  = config.SESSION.timezone
    et  = week_df.index.tz_convert(tz)
    days = np.unique(et.date)
    # pick the middle day of the week that has at least 40 bars
    plot_day = None
    for d in days[len(days) // 2:]:
        day_mask = et.date == d
        if day_mask.sum() >= 40:
            plot_day = d
            break
    if plot_day is None and len(days) > 0:
        plot_day = days[0]

    if plot_day is None:
        print("  (not enough data to plot)")
        return

    day_mask = et.date == plot_day
    day_df   = week_df[day_mask].copy()

    print(f"\n  Plotting {plot_day}  ({len(day_df)} bars) ...")
    _plot_one_day_structure(day_df, diag, plot_day, target, out, source)


def _plot_one_day_structure(
    day_df: pd.DataFrame,
    diag: dict,
    plot_day,
    target: str,
    out,
    source: str,
) -> None:
    """
    Candlestick chart of one trading day with:
      - Bullish FVG zones shaded green (alpha 0.15)
      - Bearish FVG zones shaded red   (alpha 0.15)
      - Bullish grab bars marked with '^' at the low
      - Bearish grab bars marked with 'v' at the high
    """
    tz    = config.SESSION.timezone
    n_day = len(day_df)
    if n_day == 0:
        return

    o = day_df["open"].values
    h = day_df["high"].values
    l = day_df["low"].values
    c = day_df["close"].values

    fig, ax = plt.subplots(figsize=(14, 6))

    # --- candlesticks ---
    for i in range(n_day):
        color = "mediumseagreen" if c[i] >= o[i] else "tomato"
        ax.plot([i, i], [l[i], h[i]], color=color, linewidth=0.8, zorder=2)
        body_lo = min(o[i], c[i])
        body_hi = max(o[i], c[i])
        ax.add_patch(mpatches.FancyBboxPatch(
            (i - 0.35, body_lo), 0.70, max(body_hi - body_lo, 0.01),
            boxstyle="square,pad=0", linewidth=0,
            facecolor=color, alpha=0.85, zorder=3))

    price_lo = l.min()
    price_hi = h.max()
    margin   = (price_hi - price_lo) * 0.05

    # --- FVG zones: draw each as a bounded rectangle over its active span ---
    idx_day = day_df.index
    day_lo  = idx_day[0]
    day_hi  = idx_day[-1]

    for zone in diag["fvg_zones"]:
        zl = zone["zone_low"]
        zh = zone["zone_high"]
        d  = zone["direction"]
        a_start = zone.get("active_start_time", zone["formed_time"])
        a_end   = zone.get("active_end_time",   zone["formed_time"])

        # skip zones whose active life does not overlap this day, or whose
        # price band sits outside the plotted range
        if a_end < day_lo or a_start > day_hi:
            continue
        if zh < price_lo - margin or zl > price_hi + margin:
            continue

        x0 = int(np.searchsorted(idx_day, a_start, side="left"))
        x1 = int(np.searchsorted(idx_day, a_end,   side="right")) - 1
        x0 = max(0, min(x0, n_day - 1))
        x1 = max(0, min(x1, n_day - 1))
        if x1 < x0:
            x1 = x0

        color = "green" if d == 1 else "red"
        ax.add_patch(mpatches.Rectangle(
            (x0 - 0.4, zl), (x1 - x0) + 0.8, max(zh - zl, 1e-6),
            linewidth=0.8, edgecolor=color, facecolor=color, alpha=0.14,
            zorder=1, label=("Bull FVG" if d == 1 else "Bear FVG")))

    # --- liquidity grab markers ---
    bull_g = diag["bull_grab"].reindex(idx_day).fillna(False).values.astype(bool)
    bear_g = diag["bear_grab"].reindex(idx_day).fillna(False).values.astype(bool)

    bull_x = np.where(bull_g)[0]
    bear_x = np.where(bear_g)[0]

    if len(bull_x):
        ax.scatter(bull_x, l[bull_x] - margin * 0.6,
                   marker="^", s=80, color="lime", zorder=5,
                   label="Bull grab")
    if len(bear_x):
        ax.scatter(bear_x, h[bear_x] + margin * 0.6,
                   marker="v", s=80, color="red", zorder=5,
                   label="Bear grab")

    # --- signal overlay (background shade) ---
    sig_day = diag["signal"].reindex(idx_day).fillna(0).values
    for i in range(n_day - 1):
        if sig_day[i] == 1:
            ax.axvspan(i, i + 1, alpha=0.06, color="steelblue", zorder=0)
        elif sig_day[i] == -1:
            ax.axvspan(i, i + 1, alpha=0.06, color="salmon", zorder=0)

    # deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for hdl, lbl in zip(handles, labels):
        if lbl not in seen:
            seen[lbl] = hdl
    ax.legend(list(seen.values()), list(seen.keys()), loc="upper left", fontsize=8)

    # x-axis: show ET time labels every ~13 bars (≈ 1 hour)
    tick_step = max(1, n_day // 10)
    ticks     = list(range(0, n_day, tick_step))
    et_idx    = day_df.index.tz_convert(tz)
    labels    = [et_idx[t].strftime("%H:%M") for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7)

    ax.set_xlim(-0.5, n_day - 0.5)
    ax.set_ylim(price_lo - margin, price_hi + margin)
    ax.set_title(
        f"{target} — Market Structure Diagnostics  {plot_day}\n"
        f"Green/red bands = active FVG zones   |   triangles = liquidity grabs   |   "
        f"blue/red shading = strategy long/short signal",
        fontsize=9,
    )
    ax.set_ylabel("Price (NQ futures)" if source == "databento"
                  else "Price (QQQ proxy $)")
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    path = out / f"diagnostics_structure_{target}_{plot_day}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  diagnostic chart saved -> {path}")


def _param_importance_structure(result: "wf.StructureWFResult", target: str) -> None:
    folds = result.fold_summaries
    if folds.empty or "oos_sharpe" not in folds.columns:
        print("  (no fold data for importance analysis)")
        return

    out   = config.OUTPUT_DIR
    lines = []
    P     = lines.append

    P("=" * 70)
    P(f"  PARAMETER IMPORTANCE ANALYSIS — {target} (structure strategy)")
    P("=" * 70)

    # ---- 1. Bin -> avg OOS Sharpe -------------------------------------------
    P("\n  1. Parameter value bins vs average OOS Sharpe")
    P("  " + "-" * 66)
    for col in _STRUCTURE_NUMERIC_PARAMS:
        if col not in folds.columns:
            continue
        vals = folds[col].dropna()
        if vals.nunique() < 2:
            continue
        bins  = pd.qcut(vals, q=5, duplicates="drop")
        means = folds.groupby(bins, observed=True)["oos_sharpe"].mean()
        P(f"\n  {col}:")
        for interval, avg in means.items():
            P(f"    {str(interval):>22}  ->  avg OOS Sharpe {avg:+.3f}")

    # ---- 2. Parameter stability across folds --------------------------------
    P("\n\n  2. Parameter stability across folds (low CV = consistent)")
    P("  " + "-" * 66)
    P(f"  {'parameter':<28}  {'mean':>8}  {'std':>8}  {'cv':>8}")
    for col in _STRUCTURE_NUMERIC_PARAMS:
        if col not in folds.columns:
            continue
        v = folds[col].dropna()
        if len(v) < 2:
            continue
        mu, sd = v.mean(), v.std()
        cv = sd / abs(mu) if mu != 0 else float("nan")
        P(f"  {col:<28}  {mu:>8.3f}  {sd:>8.3f}  {cv:>8.3f}")

    # ---- 3. Top-10 IS performers vs OOS (overfitting check) -----------------
    P("\n\n  3. Top-10 IS-Sharpe folds vs their actual OOS Sharpe")
    P("  " + "-" * 66)
    if "is_sharpe" in folds.columns:
        top10 = folds.nlargest(10, "is_sharpe")[
            ["fold", "is_sharpe", "oos_sharpe", "oos_pnl_$",
             "fvg_min_atr_mult", "max_hold_bars", "extension_threshold"]
        ]
        P(f"  {'fold':>4}  {'IS Sharpe':>10}  {'OOS Sharpe':>10}  "
          f"{'OOS PnL':>10}  {'fvg_mult':>8}  {'hold':>5}  {'ext_thr':>7}")
        for _, row in top10.iterrows():
            P(f"  {int(row['fold']):>4}  {row['is_sharpe']:>10.3f}  "
              f"{row['oos_sharpe']:>10.3f}  ${row['oos_pnl_$']:>9,.0f}  "
              f"{row.get('fvg_min_atr_mult', 0):>8.2f}  "
              f"{int(row.get('max_hold_bars', 0)):>5}  "
              f"{row.get('extension_threshold', 0):>7.2f}")
        corr = folds[["is_sharpe", "oos_sharpe"]].dropna().corr().iloc[0, 1]
        P(f"\n  IS vs OOS Sharpe correlation across all folds: {corr:+.3f}")
        P("  (> 0.3 = mild IS->OOS predictability; < 0 = pure curve-fitting)")

    # ---- 4. fvg_min_atr_mult x max_hold_bars heatmap -----------------------
    if "fvg_min_atr_mult" in folds.columns and "max_hold_bars" in folds.columns:
        # Bin fvg_min_atr_mult into 4 buckets for the pivot
        folds_h = folds.copy()
        try:
            # Auto-binned quantiles (range is data-driven, so don't hardcode labels).
            folds_h["fvg_bin"] = pd.qcut(
                folds_h["fvg_min_atr_mult"], q=4, duplicates="drop")
            pivot = (folds_h.groupby(["fvg_bin", "max_hold_bars"], observed=True)
                            ["oos_sharpe"].mean().unstack(fill_value=np.nan))
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                           vmin=-1.0, vmax=1.0)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=7)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(r) for r in pivot.index])
            ax.set_xlabel("max_hold_bars")
            ax.set_ylabel("fvg_min_atr_mult bin")
            ax.set_title(
                f"{target} — Avg OOS Sharpe by fvg_min_atr_mult x max_hold_bars")
            plt.colorbar(im, ax=ax, label="avg OOS Sharpe")
            fig.tight_layout()
            hm_path = out / f"heatmap_structure_{target}.png"
            fig.savefig(hm_path, dpi=130)
            plt.close(fig)
            print(f"  heatmap saved -> {hm_path}")
        except Exception as e:
            P(f"\n  (heatmap skipped: {e})")

    P("\n" + "=" * 70)
    text = "\n".join(lines)
    print("\n" + text)
    (out / f"param_importance_structure_{target}.txt").write_text(
        text, encoding="utf-8")


def _load_universe_structure(source: str, dbn_path: str | None) -> dict[str, pd.DataFrame]:
    if source == "databento":
        if not dbn_path:
            print("  ERROR: --source databento requires --dbn-path "
                  "(file/dir/zip of trades DBNs).", file=sys.stderr)
            sys.exit(1)
        print(f"  loading Databento bars from: {dbn_path}")
        return data_clients.load_universe_databento(
            dbn_path, interval=config.INTERVAL)

    if source == "eodhd":
        print("  loading EODHD bars ...")
        return data_clients.load_universe(interval=config.INTERVAL)

    print(f"  ERROR: --source must be 'databento' or 'eodhd' (got '{source}').",
          file=sys.stderr)
    sys.exit(1)


def _run_instrument_structure(universe: dict, target: str, diagnose_only: bool,
                              source: str) -> bool | None:
    pv = config.INSTRUMENTS[target].point_value
    if source == "databento":
        data_label = "DATABENTO REAL PRICES / MICRO ECON"
        econ_note  = (f"  Structure detected on REAL {target} prices (~20,000); "
                      f"PnL sized in MICRO (MNQ ${pv:g}/point) to stay within the "
                      f"Topstep daily loss limit.")
        head_econ  = (f"Real {target} prices (Databento) - MICRO economics "
                      f"(MNQ ${pv:g}/point)")
        data_src   = "databento_real_prices_micro_econ"
        econ_meta  = f"MNQ ${pv:g}/point on real {target} prices"
        plot_econ  = f"Real {target} prices (Databento) / MNQ ${pv:g}/point"
    else:
        data_label = "EODHD QQQ PROXY"
        econ_note  = (f"  NOTE: PnL uses MNQ economics (${pv:g}/point) on a QQQ "
                      f"PROXY price.  ILLUSTRATIVE ONLY — FVG levels are QQQ "
                      f"levels, not real {target}.")
        head_econ  = f"QQQ proxy via EODHD — MNQ economics (${pv:g}/point) — ILLUSTRATIVE PnL"
        data_src   = "eodhd_qqq_proxy"
        econ_meta  = f"MNQ ${pv:g}/point — illustrative"
        plot_econ  = f"QQQ proxy / MNQ ${pv:g}/point — illustrative only"

    print(f"\n{'#'*70}\n# {target} — MARKET STRUCTURE STRATEGY ({data_label})\n{'#'*70}")
    print(econ_note)

    if target not in universe:
        print(f"  skipping {target}: not in loaded universe")
        return None

    tdf = universe[target]
    print(f"  {len(tdf)} bars  "
          f"{tdf.index.min().date()} -> {tdf.index.max().date()}")

    # ---- diagnostic pass ----------------------------------------------------
    print("\n[DIAG] running pre-backtest diagnostics ...")
    try:
        _run_diagnostics_structure(tdf, target, source)
    except Exception as e:
        print(f"  (diagnostics failed: {e})")

    if diagnose_only:
        print("\n  --diagnose-only set; stopping before walk-forward.")
        return None

    # ---- walk-forward -------------------------------------------------------
    print("\n[1/3] walk-forward optimization ...")
    sizing = config.DYNAMIC_SIZING if config.DYNAMIC_SIZING.enabled else None
    if sizing is not None:
        print(f"  dynamic sizing ON: micro->mini ladder, anchor={sizing.anchor}, "
              f"safety={sizing.safety_fraction}")
    result = wf.run_structure_walk_forward(
        tdf, instrument_key=target, n_contracts=1, verbose=True, sizing=sizing)

    # ---- report -------------------------------------------------------------
    print("[2/3] composing report ...")
    m      = result.oos_metrics
    ts_res = m.get("topstep")
    gate   = rep.check_gate(m)

    lines = []
    P = lines.append
    P("=" * 70)
    P(f"  MARKET STRUCTURE RESULT FOR {target}  (OOS walk-forward)")
    P(f"  {head_econ}")
    P("=" * 70)
    P("")
    P("  Out-of-sample performance (the only numbers that matter):")
    P(f"    total PnL              : ${m['total_pnl_$']:,.0f}  ({m['return_pct']:+.1f}%)")
    P(f"    Sharpe                 : {m['sharpe']:.2f}")
    P(f"    Sortino                : {m['sortino']:.2f}")
    P(f"    max drawdown           : ${m['max_drawdown_$']:,.0f}  ({m['max_drawdown_pct']:.1f}%)")
    P(f"    profit factor          : {m['profit_factor']}")
    P(f"    win rate               : {m['win_rate']:.1f}%")
    P(f"    trades                 : {m['n_trades']}")
    P(f"    folds profitable       : {m.get('pct_folds_profitable')}%")
    P("")
    P("  Overfitting check (IS vs OOS):")
    d = result.degradation
    P(f"    mean IS Sharpe         : {d.get('is_sharpe_mean')}")
    P(f"    OOS Sharpe             : {d.get('oos_sharpe')}")
    P(f"    Sharpe retention       : {d.get('sharpe_retention_pct')}%")
    P("")
    P("  Topstep rule check (on OOS curve):")
    if ts_res is not None:
        P(f"    passed                 : {ts_res.passed}")
        P(f"    max trailing drawdown  : ${ts_res.max_trailing_drawdown:,.0f}")
        P(f"    worst day              : ${ts_res.worst_day:,.0f}")
        for fr in ts_res.failed_rules:
            P(f"    !! {fr}")
    P("")
    P("  Parameter stability across folds:")
    folds = result.fold_summaries
    for col in _STRUCTURE_NUMERIC_PARAMS:
        if col in folds.columns:
            vals = folds[col].dropna()
            P(f"    {col:<26}: {vals.mean():.3f}  "
              f"(min={vals.min():.3f}, max={vals.max():.3f})")
    P("")
    P("  ACCEPTANCE GATE (OOS only):")
    for name, (val, thr, ok) in gate["checks"].items():
        mark = "PASS" if ok else "FAIL"
        P(f"    [{mark}] {name:<28} value={val}  threshold={thr}")
    P("")
    verdict = "ACCEPTED" if gate["overall_pass"] else "REJECTED"
    P(f"  >>> OVERALL: {verdict} <<<")
    if not gate["overall_pass"]:
        P("      Do NOT trade this live. Tune FVG threshold / hold bars / extension.")
    P("=" * 70)

    text = "\n".join(lines)
    print("\n" + text)

    out = config.OUTPUT_DIR
    (out / f"report_structure_{target}.txt").write_text(text, encoding="utf-8")
    result.fold_summaries.to_csv(
        out / f"folds_structure_{target}.csv", index=False)

    # Equity plot
    eq   = result.oos_equity
    peak = eq.cummax()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eq.index, eq.values, lw=1.2, color="steelblue", label="OOS equity")
    ax.fill_between(eq.index, eq.values, peak.values,
                    where=(eq < peak), alpha=0.20, color="red")
    ax.axhline(config.TOPSTEP.starting_balance, ls="--", c="grey", lw=0.8,
               label="Starting balance")
    ax.set_title(
        f"{target} — Market Structure Strategy (walk-forward OOS)\n"
        f"{plot_econ}")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / f"equity_structure_{target}.png", dpi=120)
    plt.close(fig)

    # Machine-readable summary
    summary = {
        "instrument": target,
        "strategy":   "market_structure",
        "data_source": data_src,
        "economics":   econ_meta,
        "oos_metrics": {k: v for k, v in m.items() if k != "topstep"},
        "gate": {
            "overall_pass": gate["overall_pass"],
            "checks": {
                k: {"value": str(v[0]), "threshold": str(v[1]), "pass": bool(v[2])}
                for k, v in gate["checks"].items()
            },
        },
        "degradation":   result.degradation,
        "chosen_params": result.chosen_params,
    }
    (out / f"summary_structure_{target}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\n[3/3] parameter importance analysis ...")
    _param_importance_structure(result, target)

    print(f"\n  outputs written to {out}")
    return gate["overall_pass"]


def cmd_structure(args):
    # Economics: this strategy uses MICRO (MNQ $2/point) economics in BOTH
    # paths. On the databento path the *prices* are real NQ (~20,000) so the
    # structure detection is correct, but PnL is sized in MNQ so a single bad
    # trade (~100 pts) loses ~$200, well inside the Topstep daily loss limit —
    # at full NQ ($20/pt) that same trade would lose ~$2,000 and blow the day.
    # config.INSTRUMENTS is the micro dict by default; do NOT switch to
    # DATABENTO_INSTRUMENTS ($20/pt) here (that is for the full-size reactive path).
    if args.source == "databento":
        print(f"  [databento] real NQ prices, MNQ economics: "
              f"NQ ${config.INSTRUMENTS['NQ'].point_value:g}/pt (micro) — "
              f"structure on real prices, PnL micro-sized")
    else:
        print(f"  [eodhd] instrument specs: NQ "
              f"${config.INSTRUMENTS['NQ'].point_value:g}/pt  "
              f"(MNQ micro — QQQ proxy, illustrative only)")

    bpd = _bars_per_day()
    if args.quick:
        config.WALKFORWARD.train_bars      = bpd * 60
        config.WALKFORWARD.test_bars       = bpd * 15
        config.WALKFORWARD.step_bars       = bpd * 15
        config.WALKFORWARD.n_param_samples = args.n_samples or 30
    else:
        config.WALKFORWARD.train_bars      = bpd * 120
        config.WALKFORWARD.test_bars       = bpd * 30
        config.WALKFORWARD.step_bars       = bpd * 30
        config.WALKFORWARD.n_param_samples = args.n_samples or 300
    print(f"  [{'quick' if args.quick else 'full'}] "
          f"train={config.WALKFORWARD.train_bars} bars  "
          f"test={config.WALKFORWARD.test_bars} bars  "
          f"n_samples={config.WALKFORWARD.n_param_samples}")

    universe = _load_universe_structure(args.source, args.dbn_path)
    targets  = args.instruments or ["NQ"]

    results: dict[str, bool | None] = {}
    for t in targets:
        try:
            results[t] = _run_instrument_structure(universe, t, args.diagnose_only,
                                                    args.source)
        except Exception as e:
            import traceback
            print(f"  !! {t} failed: {e}")
            traceback.print_exc()
            results[t] = None

    if not args.diagnose_only:
        print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
        for t, ok in results.items():
            if ok is True:
                status = "ACCEPTED"
            elif ok is False:
                status = "REJECTED"
            else:
                status = "ERROR / SKIPPED"
            print(f"  {t:4s}: {status}")
        print(f"\nReports & plots written to: {config.OUTPUT_DIR}")


# ============================================================================
# structure-variants -- base vs RTH-entry variant comparison (+ sizing)
# ============================================================================
#
# The ablation showed RTH-only entries are the winning fix (best Sharpe,
# highest avg_win, ~half the drawdown, passes Topstep) while the target/R:R/
# FVG-lock fixes hurt. So this keeps ONLY RTH-only and tests it on any
# instrument, with an optional dynamic micro->mini sizing overlay.
#
# Variants:
#   base       original strategy (all-session entries, time-stop exit)
#   rth        base + RTH-only entries 09:35-15:30 ET   (the winner)
#   rth_wide   base + wider RTH entries 09:30-15:55 ET  (more trades?)
#   +  with --dynamic-sizing: rth+dyn / rth_wide+dyn (micro->mini ladder)
#
# Economics: micro contract ($/pt from config.INSTRUMENTS) on REAL futures prices.

_SV_ES = "ES"   # set from --instrument at runtime

# EDGE SET — test the four new edge features one at a time vs the base, plus a
# combo. Each is 1 contract so the comparison is about the edge, not sizing.
#   base   reference
#   of     + order-flow confirmation (delta must agree at the entry bar)   [#1]
#   flat   + trade the neutral (bias==0) regime, with 1 (less ambitious) TP [#2]
#   mtp    + multi-TP ratchet exit (n_targets searched 1..3)                [#4]
#   combo  of + flat + mtp together
# (#3, per-trigger win-rate, is REPORTED for every variant, not a variant.)
# of_apply defaults to "momentum" in the strategy, so order-flow now gates ONLY
# momentum entries (it confirms continuation but contradicts FVG reversion).
_SV_OF      = {"orderflow_confirm": True, "of_z_min": 0.3}    # momentum-only by default
_SV_FLAT    = {"trade_flat": True}
_SV_MTP     = {"use_liquidity_targets": True, "min_rr": None}  # n_targets searched
_SV_FLAT_MTP = {**_SV_FLAT, **_SV_MTP}                         # the two winners

_SV_VARIANTS: dict[str, dict] = {
    "base": {
        "label": "base", "n_contracts": 1, "variant": None, "sizing": None,
        "desc": "original strategy, all-session entries, time stop",
    },
    "of": {
        "label": "of", "n_contracts": 1,
        "variant": {"fixed": dict(_SV_OF)}, "sizing": None,
        "desc": "base + order-flow confirm on MOMENTUM only (delta z>=0.3) [#1]",
    },
    "flat": {
        "label": "flat", "n_contracts": 1,
        "variant": {"fixed": dict(_SV_FLAT)}, "sizing": None,
        "desc": "base + trade neutral regime, 1 TP in flat [#2]",
    },
    "mtp": {
        "label": "mtp", "n_contracts": 1,
        "variant": {"fixed": dict(_SV_MTP)}, "sizing": None,
        "desc": "base + multi-TP ratchet exit (n_targets 1..3) [#4]",
    },
    "flat_mtp": {
        "label": "flat_mtp", "n_contracts": 1,
        "variant": {"fixed": dict(_SV_FLAT_MTP)}, "sizing": None,
        "desc": "base + flat regime + multi-TP (the two winners, no order-flow)",
    },
    "mixed": {
        "label": "mixed", "n_contracts": 1,
        "variant": {"fixed": {**_SV_FLAT_MTP, "flat_use_targets": False}},
        "sizing": None,
        "desc": "trend trades = multi-TP ratchet, flat trades = time stop",
    },
    "combo": {
        "label": "combo", "n_contracts": 1,
        "variant": {"fixed": {**_SV_OF, **_SV_FLAT, **_SV_MTP}}, "sizing": None,
        "desc": "flat + multi-TP + momentum-only order-flow",
    },
}


def _load_es_variants(dbn_path: str | None) -> pd.DataFrame:
    if dbn_path:
        print(f"  loading Databento bars from: {dbn_path}")
        uni = data_clients.load_universe_databento(dbn_path, interval=config.INTERVAL)
    else:
        cands = sorted(glob.glob(str(config.CACHE_DIR / "databento_v2_*5m*.pkl")),
                       key=lambda p: Path(p).stat().st_mtime, reverse=True)
        if not cands:
            print("  ERROR: no cached Databento universe in cache/. Pass --dbn-path.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"  loading cached Databento universe: {Path(cands[0]).name}")
        uni = pd.read_pickle(cands[0])
    if _SV_ES not in uni:
        print(f"  ERROR: {_SV_ES} not in universe (keys={list(uni)})", file=sys.stderr)
        sys.exit(1)
    df = uni[_SV_ES]
    print(f"  {_SV_ES}: {len(df)} bars  {df.index.min().date()} -> {df.index.max().date()}")
    return df


def _weeks_in(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 1.0
    return max((idx.max() - idx.min()).total_seconds() / 86400.0 / 7.0, 1.0)


def _rr_join(result) -> pd.DataFrame:
    """Join taken entries (pre-entry R:R) to realised trades by entry_time."""
    tk = getattr(result, "oos_taken", None)
    tr = result.oos_trades
    if tk is None or not len(tk) or tr is None or not len(tr) or "entry_time" not in tr:
        return pd.DataFrame()
    j = tr.merge(tk[["entry_time", "rr", "target", "stop", "is_target_trade"]],
                 on="entry_time", how="left")
    return j


def _summarize_variant(result) -> dict:
    m   = result.oos_metrics
    ts_res = m.get("topstep")
    wks = _weeks_in(result.oos_equity.index)
    j   = _rr_join(result)

    rr_all = rr_win = rr_loss = None
    if len(j):
        jf = j[np.isfinite(j["rr"])]            # drop NaN and inf (degenerate stops)
        rr = jf["rr"]
        rr_all = round(float(rr.mean()), 2) if len(rr) else None
        win = jf[(jf["pnl_$"] > 0)]["rr"]
        los = jf[(jf["pnl_$"] < 0)]["rr"]
        rr_win  = round(float(win.mean()), 2) if len(win) else None
        rr_loss = round(float(los.mean()), 2) if len(los) else None

    sk = getattr(result, "oos_skipped", None)
    n_skip = int(len(sk)) if sk is not None else 0
    skip_wp = (round(100 * float(sk["would_profit"].mean()), 1)
               if (sk is not None and len(sk)) else None)

    return {
        "total_pnl_$":     round(m["total_pnl_$"], 0),
        "weekly_pnl_$":    round(m["total_pnl_$"] / wks, 0),
        "sharpe":          m["sharpe"],
        "max_dd_$":        round(m["max_drawdown_$"], 0),
        "max_tdd_$":       round(ts_res.max_trailing_drawdown, 0) if ts_res else None,
        "worst_day_$":     round(ts_res.worst_day, 0) if ts_res else None,
        "avg_win_$":       round(m["avg_win_$"], 0),
        "avg_loss_$":      round(m["avg_loss_$"], 0),
        "avg_rr":          rr_all,
        "rr_win":          rr_win,
        "rr_loss":         rr_loss,
        "trades":          int(m["n_trades"]),
        "trades_per_week": round(m["n_trades"] / wks, 1),
        "win_rate":        m["win_rate"],
        "skipped_rr":      n_skip,
        "skip_would_profit_pct": skip_wp,
        "topstep_pass":    bool(ts_res.passed) if ts_res else None,
        "topstep_fails":   list(ts_res.failed_rules) if ts_res else [],
        "weeks":           round(wks, 1),
    }


def _contract_advice(name: str, spec: dict, s: dict) -> str | None:
    if s["topstep_pass"] or s["max_tdd_$"] is None:
        return None
    if not any("trailing_drawdown" in f for f in s["topstep_fails"]):
        return None
    limit, tdd, n = config.TOPSTEP.trailing_drawdown, s["max_tdd_$"], spec["n_contracts"]
    if tdd <= 0:
        return None
    safe = int(np.floor(n * limit / tdd))
    if 1 <= safe < n:
        return (f"reduce {name} from {n} to {safe} contract(s) "
                f"(max trailing DD ${tdd:,.0f} > ${limit:,.0f}; ~${tdd/n:,.0f}/contract)")
    if safe < 1:
        return (f"{name} breaches even at 1 contract (max trailing DD ${tdd:,.0f}); "
                f"tighten stops rather than resize")
    return None


def _draw_candles(ax, o, h, l, c):
    for i in range(len(c)):
        col = "mediumseagreen" if c[i] >= o[i] else "tomato"
        ax.plot([i, i], [l[i], h[i]], color=col, lw=0.7, zorder=2)
        lo, hi = min(o[i], c[i]), max(o[i], c[i])
        ax.add_patch(mpatches.Rectangle((i - 0.34, lo), 0.68, max(hi - lo, 1e-6),
                                        linewidth=0, facecolor=col, alpha=0.85, zorder=3))


def _draw_zones(ax, zones, w0, w1, plo, phi, margin):
    seen = set()
    for z in zones:
        a0 = z.get("active_start_idx", z["formed_idx"] + 1)
        a1 = z.get("active_end_idx",   z["formed_idx"] + 1)
        if a1 < w0 or a0 > w1:
            continue
        zl, zh = z["zone_low"], z["zone_high"]
        if zh < plo - margin or zl > phi + margin:
            continue
        x0, x1 = max(a0, w0) - w0, min(a1, w1) - w0
        color = "green" if z["direction"] == 1 else "red"
        lbl = "Bull FVG" if z["direction"] == 1 else "Bear FVG"
        ax.add_patch(mpatches.Rectangle(
            (x0 - 0.4, zl), (x1 - x0) + 0.8, max(zh - zl, 1e-6),
            linewidth=0.6, edgecolor=color, facecolor=color, alpha=0.12,
            zorder=1, label=(lbl if lbl not in seen else None)))
        seen.add(lbl)


def _time_axis(ax, idx_window, tz):
    n = len(idx_window)
    step = max(1, n // 9)
    ticks = list(range(0, n, step))
    et = idx_window.tz_convert(tz)
    ax.set_xticks(ticks)
    ax.set_xticklabels([et[t].strftime("%m-%d %H:%M") for t in ticks],
                       fontsize=6, rotation=30, ha="right")


def _representative_params(spec: dict, chosen: list[dict]) -> dict:
    p = dict(strat.STRUCTURE_DEFAULT_PARAMS)
    numeric = ["fvg_min_atr_mult", "grab_wick_ratio", "grab_atr_mult",
               "max_hold_bars", "extension_threshold", "min_rr"]
    if chosen:
        for k in numeric:
            vals = [c[k] for c in chosen if k in c and c[k] is not None]
            if vals:
                p[k] = float(np.median(vals))
        p["max_hold_bars"] = int(round(p["max_hold_bars"]))
    p.update((spec.get("variant") or {}).get("fixed", {}))
    return p


def _entry_chart(df, zones, trade, rr_est, stop, target, name, n, out_dir, tz, pv):
    idx = df.index
    epos = idx.get_loc(trade["entry_time"]) if trade["entry_time"] in idx \
        else int(idx.searchsorted(trade["entry_time"]))
    xpos = idx.get_loc(trade["exit_time"]) if trade["exit_time"] in idx \
        else int(idx.searchsorted(trade["exit_time"]))
    w0, w1 = max(0, epos - 60), min(len(idx) - 1, epos + 30)
    win = df.iloc[w0:w1 + 1]
    o, h, l, c = (win["open"].values, win["high"].values,
                  win["low"].values, win["close"].values)
    plo, phi = l.min(), h.max()
    margin = (phi - plo) * 0.05 + 1e-6

    sgn = 1.0 if trade["side"] == "long" else -1.0
    risk = abs(trade["entry"] - stop) if (stop is not None and np.isfinite(stop)) else np.nan
    rr_act = (trade["pnl_points"] / risk) if (risk and np.isfinite(risk) and risk > 0) else np.nan
    wl = "WIN" if trade["pnl_$"] > 0 else "LOSS"

    fig, ax = plt.subplots(figsize=(13, 6))
    _draw_candles(ax, o, h, l, c)
    _draw_zones(ax, zones, w0, w1, plo, phi, margin)
    ax.axvline(epos - w0, color="blue", lw=1.4, label="entry")
    if 0 <= xpos - w0 <= (w1 - w0):
        ax.axvline(xpos - w0, color="black", lw=1.2, ls="--", label="exit")
    if stop is not None and np.isfinite(stop):
        ax.axhline(stop, color="red", lw=1.1, ls="--", label="stop")
    if target is not None and np.isfinite(target):
        ax.axhline(target, color="green", lw=1.1, ls="--", label="target")

    et = idx[epos].tz_convert(tz).strftime("%Y-%m-%d %H:%M")
    rr_est_s = f"{rr_est:.2f}" if (rr_est is not None and np.isfinite(rr_est)) else "n/a"
    rr_act_s = f"{rr_act:+.2f}" if np.isfinite(rr_act) else "n/a"
    ax.set_title(
        f"{_SV_ES} {name} trade {n}  [{wl} ${trade['pnl_$']:+,.0f}]  {trade['side']}  {et}\n"
        f"pre-entry R:R {rr_est_s}  ->  achieved R:R {rr_act_s}  "
        f"({trade['pnl_points']:+.2f} pts)",
        fontsize=10)
    ax.set_ylabel(f"Price ({_SV_ES} futures)")
    ax.set_xlim(-0.5, (w1 - w0) + 0.5)
    ax.set_ylim(plo - margin, phi + margin)
    _time_axis(ax, win.index, tz)
    hh, ll = ax.get_legend_handles_labels()
    if hh:
        seen = {}
        for a, b in zip(hh, ll):
            seen.setdefault(b, a)
        ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=8)
    fig.tight_layout()
    tag = idx[epos].tz_convert(tz).strftime("%Y%m%d_%H%M")
    fig.savefig(out_dir / f"{_SV_ES}_{name}_trade_{n}_{tag}.png", dpi=110)
    plt.close(fig)


def _gen_entry_charts(df, result, spec, name, out_dir, tz, n_charts=10):
    tr = result.oos_trades
    if tr is None or not len(tr):
        print(f"    {name}: no OOS trades to chart")
        return
    rep_params = _representative_params(spec, result.chosen_params)
    dbg = strat.generate_structure_signal_debug(df, rep_params, _SV_ES)
    zones = dbg["fvg_zones"]
    j = _rr_join(result)                       # entry_time -> rr/target/stop
    info = {row["entry_time"]: row for _, row in j.iterrows()} if len(j) else {}
    pv = config.INSTRUMENTS[_SV_ES].point_value

    tr = tr.sort_values("entry_time").reset_index(drop=True)
    total = len(tr)
    picks = list(range(total)) if total <= n_charts else \
        list(np.linspace(0, total - 1, n_charts).astype(int))
    for k, ti in enumerate(picks):
        t = tr.iloc[ti]
        row = info.get(t["entry_time"], {})
        _entry_chart(df, zones, t, row.get("rr"), row.get("stop"), row.get("target"),
                     name, k + 1, out_dir, tz, pv)
    print(f"    {name}: {len(picks)} entry charts -> {out_dir}")


def _fmt(val, fmt):
    if val is None:
        return "n/a"
    try:
        return fmt.format(val)
    except (ValueError, TypeError):
        return str(val)


def _render_variant_table(summaries: dict[str, dict]) -> str:
    rows = [
        ("Total OOS PnL",      "total_pnl_$",  "${:,.0f}"),
        ("Weekly avg PnL",     "weekly_pnl_$", "${:,.0f}"),
        ("Sharpe",             "sharpe",       "{:.2f}"),
        ("Max drawdown",       "max_dd_$",     "${:,.0f}"),
        ("Worst day",          "worst_day_$",  "${:,.0f}"),
        ("Avg win $",          "avg_win_$",    "${:,.0f}"),
        ("Avg loss $",         "avg_loss_$",   "${:,.0f}"),
        ("Avg R:R",            "avg_rr",       "{:.2f}"),
        ("Trades skipped (RR)","skipped_rr",   "{:d}"),
        ("Topstep pass",       "topstep_pass", "{}"),
        ("Trades (total)",     "trades",       "{:d}"),
        ("Trades/week",        "trades_per_week", "{:.1f}"),
        ("Win rate %",         "win_rate",     "{:.1f}"),
    ]
    names = list(summaries.keys())
    labels = [_SV_VARIANTS[n]["label"] for n in names]
    cw = 12
    out = [f"{'Metric':<20}" + "".join(f"{lb:>{cw}}" for lb in labels)]
    out.append("-" * len(out[0]))
    for label, key, fmt in rows:
        line = f"{label:<20}"
        for nm in names:
            line += f"{_fmt(summaries[nm].get(key), fmt):>{cw}}"
        out.append(line)
    return "\n".join(out)


def _skip_analysis(result, name: str) -> list[str]:
    sk = getattr(result, "oos_skipped", None)
    out = [f"  Skipped-trade analysis — {name} (R:R filter):"]
    if sk is None or not len(sk):
        out.append("    (no trades skipped by the R:R filter)")
        return out
    per = sk.groupby("fold").agg(
        skipped=("would_profit", "size"),
        would_profit=("would_profit", "sum")).reset_index()
    out.append(f"    {'fold':>5}  {'skipped':>8}  {'would-have-won':>15}  {'win%':>6}")
    for _, r in per.iterrows():
        wp = 100 * r["would_profit"] / r["skipped"] if r["skipped"] else 0
        out.append(f"    {int(r['fold']):>5}  {int(r['skipped']):>8}  "
                   f"{int(r['would_profit']):>15}  {wp:>5.0f}%")
    tot = len(sk); wptot = int(sk["would_profit"].sum())
    pct = 100 * wptot / tot
    out.append(f"    {'ALL':>5}  {tot:>8}  {wptot:>15}  {pct:>5.0f}%")
    verdict = ("filter looks CORRECT (most skipped setups would have lost)"
               if pct < 45 else
               "filter may be TOO AGGRESSIVE (many skipped setups would have won)"
               if pct > 55 else "filter is borderline (skipped setups ~ coin-flip)")
    out.append(f"    -> {pct:.0f}% of skipped setups would have hit target first: {verdict}")
    return out


def _trigger_analysis(result, name: str) -> list[str]:
    """#3: standalone win rate / avg PnL / expectancy per entry trigger."""
    tk = getattr(result, "oos_taken", None)
    tr = result.oos_trades
    out = [f"  Per-trigger breakdown — {name}:"]
    if (tk is None or not len(tk) or "trigger" not in tk
            or tr is None or not len(tr) or "entry_time" not in tr):
        out.append("    (no per-trigger data)")
        return out
    j = tr.merge(tk[["entry_time", "trigger", "regime"]], on="entry_time", how="left")
    j = j[j["trigger"].notna()]
    if not len(j):
        out.append("    (no matched trades)")
        return out

    def _block(col, keys_title):
        lines = [f"    {keys_title:<10} {'trades':>7} {'win%':>6} {'avg$':>8} "
                 f"{'tot$':>9} {'exp$/trade':>11}"]
        for key, g in j.groupby(col):
            n_ = len(g); wr = 100 * (g["pnl_$"] > 0).mean()
            avg = g["pnl_$"].mean(); tot = g["pnl_$"].sum()
            lines.append(f"    {str(key):<10} {n_:>7d} {wr:>5.0f}% {avg:>8.0f} "
                         f"{tot:>9.0f} {avg:>11.1f}")
        return lines

    out += _block("trigger", "trigger")
    if j["regime"].nunique() > 1:
        out.append("")
        out += _block("regime", "regime")
    return out


def cmd_structure_variants(args):
    global _SV_ES
    _SV_ES = args.instrument
    micro = config.INSTRUMENTS[_SV_ES]
    print(f"  instrument: {_SV_ES}  economics: {micro.name} ${micro.point_value:g}/point "
          f"(micro) on real {_SV_ES} futures prices")

    # Restrict to the requested base variants (default: all).
    if args.variants:
        bad = [k for k in args.variants if k not in _SV_VARIANTS]
        if bad:
            print(f"  WARN: ignoring unknown variants {bad} "
                  f"(valid: {list(_SV_VARIANTS)})", file=sys.stderr)
        keep = [k for k in args.variants if k in _SV_VARIANTS]
        if not keep:
            sys.exit("  ERROR: no valid --variants selected")
        sel = {k: _SV_VARIANTS[k] for k in keep}
        _SV_VARIANTS.clear()
        _SV_VARIANTS.update(sel)

    # Add a dynamic-sized (+dyn) copy of EACH selected variant when requested.
    if args.dynamic_sizing:
        sz = config.DynamicSizing(enabled=True, safety_fraction=args.safety,
                                  anchor=args.anchor,
                                  max_trade_risk_frac=args.max_trade_risk_frac,
                                  daily_risk_frac=args.daily_risk_frac)
        for src in list(_SV_VARIANTS):
            base_spec = _SV_VARIANTS[src]
            _SV_VARIANTS[f"{src}_dyn"] = {
                "label": f"{src}+dyn", "n_contracts": 1,
                "variant": base_spec["variant"], "sizing": sz,
                "desc": f"{base_spec['desc']} + dynamic micro->mini sizing "
                        f"(anchor={args.anchor}, safety={args.safety})",
            }
        print(f"  dynamic sizing ON: micro->mini ladder, anchor={args.anchor}, "
              f"safety={args.safety}")

    # Constant-size multiplier for the FIXED (non-dynamic) variants. This is the
    # consistency-preserving way to scale: every bet equal-weight, just larger.
    if args.contracts and args.contracts != 1:
        nc = min(args.contracts, config.TOPSTEP.max_contracts)
        for name, spec in _SV_VARIANTS.items():
            if spec.get("sizing") is None:
                spec["n_contracts"] = nc
                spec["label"] = f"{spec['label']}x{nc}"
        print(f"  fixed variants sized at a CONSTANT {nc} micro(s) "
              f"(capped by TOPSTEP.max_contracts={config.TOPSTEP.max_contracts})")

    bpd = 78
    if args.quick:
        config.WALKFORWARD.train_bars      = bpd * 40
        config.WALKFORWARD.test_bars       = bpd * 15
        config.WALKFORWARD.step_bars       = bpd * 15
        config.WALKFORWARD.n_param_samples = 8
    else:
        config.WALKFORWARD.train_bars      = bpd * 120
        config.WALKFORWARD.test_bars       = bpd * 30
        config.WALKFORWARD.step_bars       = bpd * 30
        config.WALKFORWARD.n_param_samples = args.n_samples
    if args.min_trades is not None:
        config.WALKFORWARD.min_trades_in_sample = args.min_trades
    print(f"  walk-forward: train={config.WALKFORWARD.train_bars} "
          f"test={config.WALKFORWARD.test_bars} step={config.WALKFORWARD.step_bars} "
          f"n_samples={config.WALKFORWARD.n_param_samples} "
          f"min_trades={config.WALKFORWARD.min_trades_in_sample}")

    df = _load_es_variants(args.dbn_path)
    tz, out = config.SESSION.timezone, config.OUTPUT_DIR

    results, summaries = {}, {}
    for name, spec in _SV_VARIANTS.items():
        print(f"\n{'#'*70}\n# {spec['label']}: {spec['desc']}\n{'#'*70}")
        res = wf.run_structure_walk_forward(
            df, instrument_key=_SV_ES, n_contracts=spec["n_contracts"],
            verbose=True, variant=spec["variant"], sizing=spec.get("sizing"))
        results[name], summaries[name] = res, _summarize_variant(res)

    # ---- table + reports ---------------------------------------------------
    lines = ["=" * 92,
             f"  {_SV_ES} STRUCTURE STRATEGY — base vs RTH variants (walk-forward OOS)",
             f"  {micro.name} ${micro.point_value:g}/point micro on real {_SV_ES} prices."
             "   Goal: lift avg_win_$ and weekly PnL.",
             "=" * 92, "", _render_variant_table(summaries), ""]

    # R:R winners vs losers
    lines.append("  Reward:Risk (pre-entry estimate, target trades only):")
    for name in _SV_VARIANTS:
        s = summaries[name]
        if s["avg_rr"] is None:
            lines.append(f"    {_SV_VARIANTS[name]['label']:<20}: (no target trades / no R:R)")
        else:
            lines.append(f"    {_SV_VARIANTS[name]['label']:<20}: "
                         f"avg {s['avg_rr']:.2f}  |  winners {s['rr_win']}  "
                         f"losers {s['rr_loss']}")
    lines.append("")

    # Topstep
    lines.append("  Topstep compliance:")
    for name, spec in _SV_VARIANTS.items():
        s = summaries[name]
        if s["topstep_pass"]:
            lines.append(f"    {_SV_VARIANTS[name]['label']:<20}: PASS")
            continue
        lines.append(f"    {_SV_VARIANTS[name]['label']:<20}: FAIL -> {', '.join(s['topstep_fails'])}")
        adv = _contract_advice(name, spec, s)
        if adv:
            lines.append(f"        suggestion: {adv}")
        if any("consistency" in f for f in s["topstep_fails"]):
            lines.append("        note: consistency rule is scale-invariant; "
                         "reducing contracts will not fix it.")
    lines.append("")

    # Per-trigger breakdown (#3) for every variant
    for name in _SV_VARIANTS:
        lines += _trigger_analysis(results[name], _SV_VARIANTS[name]["label"])
        lines.append("")

    # Skip analysis for any variant whose R:R gate actually skipped trades
    for name in _SV_VARIANTS:
        sk = getattr(results[name], "oos_skipped", None)
        if sk is not None and len(sk):
            lines += _skip_analysis(results[name], _SV_VARIANTS[name]["label"])
            lines.append("")

    text = "\n".join(lines)
    print("\n" + text)
    # ---- dynamic-sizing breakdown (per-trade size distribution) ------------
    dyn_names = [n for n in _SV_VARIANTS if _SV_VARIANTS[n].get("sizing")]
    if dyn_names:
        szlines = ["  Dynamic-sizing breakdown (per-trade entry size):"]
        for name in dyn_names:
            td = results[name].oos_trades
            if td is not None and len(td) and "size" in td:
                dist = ", ".join(f"{k}x{v}" for k, v in td["size"].value_counts().items())
                szlines.append(f"    {_SV_VARIANTS[name]['label']:<12}: {dist}")
            else:
                szlines.append(f"    {_SV_VARIANTS[name]['label']:<12}: (no trades)")
        block = "\n".join(szlines)
        text = text + "\n\n" + block
        print("\n" + block)

    (out / f"variant_comparison_{_SV_ES}.txt").write_text(text, encoding="utf-8")
    pd.DataFrame(summaries).to_csv(out / f"variant_comparison_{_SV_ES}.csv")
    print(f"\n  table -> {out / f'variant_comparison_{_SV_ES}.txt'}")

    # ---- equity overlay ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in _SV_VARIANTS:
        eq = results[name].oos_equity
        ax.plot(eq.index, eq.values, lw=1.1,
                label=f"{_SV_VARIANTS[name]['label']} (${summaries[name]['weekly_pnl_$']:,.0f}/wk)")
    ax.axhline(config.TOPSTEP.starting_balance, ls="--", c="grey", lw=0.8)
    ax.set_title(f"{_SV_ES} variants — walk-forward OOS equity "
                 f"({micro.name} ${micro.point_value:g}/pt on real {_SV_ES} prices)")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"equity_variants_{_SV_ES}.png", dpi=120)
    plt.close(fig)

    # ---- entry charts: 10 sample trades from the defined-exit variant ------
    chart_variant = "rth_te" if "rth_te" in _SV_VARIANTS else next(iter(_SV_VARIANTS))
    if not args.no_charts:
        edir = out / "entry_charts"
        edir.mkdir(exist_ok=True)
        print(f"\n  generating {chart_variant} entry charts ...")
        _gen_entry_charts(df, results[chart_variant], _SV_VARIANTS[chart_variant],
                          chart_variant, edir, tz, n_charts=10)

    print(f"\n  done. outputs in {out}")


# ============================================================================
# adaptive -- adaptive ML (decay-weighted, regime-aware, self-refitting)
# ============================================================================

def _run_instrument_adaptive(universe: dict, target: str,
                             cfg: AdaptiveConfig, warmup: int,
                             discovery_only: bool = False):
    print(f"\n{'#'*70}\n# {target} — ADAPTIVE ML\n{'#'*70}")

    print("[1/3] building features + labels ...")
    X, target_df = research.build_feature_matrix(universe, target)
    y            = research.make_labels(target_df)
    common       = X.dropna(how="all").index
    X, y, target_df = X.loc[common], y.loc[common], target_df.loc[common]

    bpd = _bars_per_day()
    print(f"  {len(X)} bars ({len(X)//bpd} trading days)")

    print("[2/3] discovery (feature pool for adaptive model) ...")
    table = research.rank_features(X, y["fwd_ret"])
    mt    = research.multiple_testing_report(table)
    pool  = research.top_feature_pool(table, k=20)

    # ---- full ranking table (always shown, required for --discovery-only) ----
    print(f"\n  Feature ranking table ({len(table)} features):")
    cols = ["feature", "spearman", "p_value"]
    if "mutual_info" in table.columns:
        cols.append("mutual_info")
    if "rf_importance" in table.columns:
        cols.append("rf_importance")
    with pd.option_context("display.max_rows", None, "display.float_format", "{:.4f}".format):
        print(table[cols].to_string(index=False))

    print(f"\n  Multiple-testing report:")
    print(f"    features tested        : {mt['n_features_tested']}")
    print(f"    significant (raw)      : {mt['n_significant_raw']}")
    print(f"    expected by chance     : {mt['n_expected_by_chance']}")
    print(f"    survive FDR (BH)       : {mt['n_significant_after_fdr']}")
    print(f"    verdict                : {mt['verdict']}")

    if discovery_only:
        print(f"\n  [--discovery-only] exiting after discovery step.")
        return None

    print(f"\n  top signals: {', '.join(pool[:6])} ...")
    print("[3/4] adaptive simulation ...")
    result = run_adaptive_backtest(
        X, y["direction"], target_df,
        instrument_key=target,
        feature_pool=pool,
        cfg=cfg,
        n_contracts=1,
        warmup_bars=warmup,
        verbose=True,
    )

    print("[4/4] reporting ...")
    eq     = result["equity"]
    trades = result["trades"]
    regime = result["regime"]

    rets    = eq.diff().fillna(0.0)
    metrics = bt._compute_metrics(eq, rets, trades, config.INTERVAL,
                                  float(eq.iloc[0]))
    ts_res  = ts.evaluate(eq, config.TOPSTEP)

    lines = []
    P = lines.append
    P("=" * 70)
    P(f"  ADAPTIVE ML RESULT FOR {target}")
    P("=" * 70)
    P(f"    total PnL       : ${metrics['total_pnl_$']:,.0f}  "
      f"({metrics['return_pct']:+.1f}%)")
    P(f"    Sharpe          : {metrics['sharpe']:.2f}")
    P(f"    Sortino         : {metrics['sortino']:.2f}")
    P(f"    max drawdown    : ${metrics['max_drawdown_$']:,.0f}  "
      f"({metrics['max_drawdown_pct']:.1f}%)")
    P(f"    profit factor   : {metrics['profit_factor']}")
    P(f"    win rate        : {metrics['win_rate']:.1f}%")
    P(f"    trades          : {metrics['n_trades']}")
    P(f"    avg trade $     : ${metrics['avg_trade_$']:.2f}")
    P(f"    avg win $       : ${metrics['avg_win_$']:.2f}")
    P(f"    avg loss $      : ${metrics['avg_loss_$']:.2f}")
    P("")
    P(f"    model refits    : {result['n_refits']}")
    P(f"    refit schedule  : every {cfg.refit_every_bars} bars "
      f"(~{cfg.refit_every_bars//bpd:.0f} trading days)")
    P(f"    decay halflife  : {cfg.decay_halflife} bars "
      f"(~{cfg.decay_halflife//bpd:.0f} trading days)")
    P(f"    regime avg scalar: {regime.mean():.2f}  "
      f"(1.0=fully familiar, 0=unfamiliar, goes flat)")
    P(f"    account killed  : {result['killed']}")
    P("")
    P(f"    Topstep passed  : {ts_res.passed}")
    P(f"    max trail DD    : ${ts_res.max_trailing_drawdown:,.0f}")
    P(f"    worst day       : ${ts_res.worst_day:,.0f}")
    for fr in ts_res.failed_rules:
        P(f"    !! {fr}")
    P("")

    gate = {
        "sharpe >= 1.0":          metrics["sharpe"] >= 1.0,
        "drawdown <= 10%":        abs(metrics["max_drawdown_pct"]) <= 10.0,
        "profit_factor >= 1.2":   (metrics["profit_factor"] or 0) >= 1.2,
        "trades >= 50":           metrics["n_trades"] >= 50,
        "topstep passed":         ts_res.passed,
    }
    accepted = all(gate.values())
    for name, ok in gate.items():
        P(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    P("")
    P(f"  >>> {'ACCEPTED' if accepted else 'REJECTED'} <<<")
    P("=" * 70)

    text = "\n".join(lines)
    print("\n" + text)

    out = config.OUTPUT_DIR
    (out / f"report_adaptive_{target}.txt").write_text(text)

    # --- plot: equity + regime scalar + refit markers ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(eq.index, eq.values, lw=1.0, color="steelblue", label="Equity")
    peak = eq.cummax()
    ax1.fill_between(eq.index, eq.values, peak.values,
                     where=(eq < peak), alpha=0.2, color="red")
    ax1.axhline(config.TOPSTEP.starting_balance, ls="--", c="grey", lw=0.8)
    for rb in result["refit_bars"]:
        if rb < len(eq):
            ax1.axvline(eq.index[rb], color="orange",
                        alpha=0.4, lw=0.8, ls=":")
    ax1.set_title(f"{target} Adaptive ML — OOS equity "
                  f"(orange lines = model refits)")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")

    # Regime scalar
    ax2.fill_between(regime.index, 0, regime.values,
                     alpha=0.5, color="green", label="Regime familiarity")
    ax2.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Regime scalar")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out / f"equity_adaptive_{target}.png", dpi=120)
    plt.close(fig)

    # Save trade log
    if len(trades):
        trades.to_csv(out / f"trades_adaptive_{target}.csv", index=False)

    return accepted


def cmd_adaptive(args):
    # Switch instrument economics to match the data source.
    config.INSTRUMENTS = config.get_instruments(args.source)
    if args.source == "databento":
        print("  [databento] using real futures contract specs "
              "(NQ $20/pt, ES $50/pt, GC $100/pt)")

    cfg = AdaptiveConfig(
        decay_halflife     = args.halflife,
        refit_every_bars   = args.refit,
        zscore_threshold   = args.threshold,
        min_train_samples  = args.warmup // 2,
    )

    print(f"  Adaptive config:")
    print(f"    decay halflife : {cfg.decay_halflife} bars")
    print(f"    refit every    : {cfg.refit_every_bars} bars")
    print(f"    zscore threshold: {cfg.zscore_threshold}")
    print(f"    warmup         : {args.warmup} bars")

    if args.source == "synthetic":
        print("  generating synthetic universe ...")
        universe = synthetic.generate(history_days=600, interval_min=5)
    else:
        universe = _get_universe(args.source, args.dbn_path)
    targets  = args.instruments or config.TRADABLES

    results = {}
    for t in targets:
        if t not in universe:
            print(f"  skipping {t}: not in universe")
            continue
        try:
            results[t] = _run_instrument_adaptive(universe, t, cfg, args.warmup,
                                                   discovery_only=args.discovery_only)
        except Exception as e:
            import traceback
            print(f"  !! {t} failed: {e}")
            traceback.print_exc()
            results[t] = None

    if not args.discovery_only:
        print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
        for t, ok in results.items():
            status = "ACCEPTED" if ok else ("REJECTED" if ok is False else "ERROR")
            print(f"  {t}: {status}")


# ============================================================================
# sizing -- dynamic position-sizing demonstration
# ============================================================================
#
# Runs ONE continuous backtest of the structure strategy over the full cached
# series for a chosen instrument and compares:
#     fixed 1 micro      vs      dynamic sizing (1..9 micro -> 1..2 mini)
#
# Dynamic sizing scales with the cushion above the trailing-DD floor (anchor
# "profit" = banked profit, so size starts at 1 micro and grows only once real
# profit is made), capping per-trade worst-case loss at `safety_fraction` of
# the cushion to keep the $2k trailing drawdown intact.
#
# This is a fast, deterministic check of the SIZING engine (no per-fold
# retuning). For a full walk-forward with sizing, pass
# sizing=config.DynamicSizing(...) to wf.run_structure_walk_forward.

# Representative current strategy config (the "fixes" set + sensible R:R).
_SIZING_PARAMS = {
    "rth_only_entries": True, "use_liquidity_targets": True,
    "require_fvg_agreement": True, "min_rr": 2.0, "hold_through_target": False,
}


def _load_sizing_universe(instrument: str) -> pd.DataFrame:
    cands = sorted(glob.glob(str(config.CACHE_DIR / "databento_v2_*5m*.pkl")),
                   key=lambda p: Path(p).stat().st_mtime, reverse=True)
    if not cands:
        print("  ERROR: no cached Databento universe in cache/.", file=sys.stderr)
        sys.exit(1)
    uni = pd.read_pickle(cands[0])
    if instrument not in uni:
        print(f"  ERROR: {instrument} not in {list(uni)}", file=sys.stderr)
        sys.exit(1)
    return uni[instrument]


def _sizing_row(tag, r):
    ts_res = r.topstep
    return (f"  {tag:<16} final ${r.equity.iloc[-1]:>9,.0f}  "
            f"PnL ${r.equity.iloc[-1]-config.TOPSTEP.starting_balance:>+8,.0f}  "
            f"maxDD ${r.metrics['max_drawdown_$']:>+8,.0f}  "
            f"maxTDD ${ts_res.max_trailing_drawdown:>7,.0f}  "
            f"worstDay ${ts_res.worst_day:>+7,.0f}  "
            f"pass={ts_res.passed}  trades={len(r.trades)}")


def cmd_sizing(args):
    inst = args.instrument
    micro = config.INSTRUMENTS[inst]
    mini  = config.DATABENTO_INSTRUMENTS[inst]
    print(f"  {inst}: micro {micro.name} ${micro.point_value:g}/pt  ->  "
          f"mini {mini.name} ${mini.point_value:g}/pt  "
          f"(ladder: 1..9 micro, 1 mini, 2 mini)")
    print(f"  sizing: anchor={args.anchor}  safety_fraction={args.safety}\n")

    df = _load_sizing_universe(inst)
    dbg = strat.generate_structure_signal_debug(df, _SIZING_PARAMS, inst)
    sig, stop = dbg["signal"], dbg["stop"]

    r_fix = bt.run_backtest(df, sig, inst, n_contracts=1, rules=config.TopstepRules())
    SZ = config.DynamicSizing(enabled=True, safety_fraction=args.safety, anchor=args.anchor)
    r_dyn = bt.run_backtest(df, sig, inst, rules=config.TopstepRules(),
                            stops=stop, sizing=SZ)

    print(_sizing_row("fixed 1 micro", r_fix))
    print(_sizing_row(f"dynamic({args.anchor})", r_dyn))

    td = r_dyn.trades
    if len(td):
        print("\n  dynamic size distribution (per-trade entry size):")
        for sz, cnt in td["size"].value_counts().items():
            print(f"    {sz:<10} {cnt}")
        # show size climbing with banked profit
        td = td.sort_values("entry_time").reset_index(drop=True)
        td["cum_pnl"] = td["pnl_$"].cumsum()
        big = td[td["contracts"] >= 10]
        if len(big):
            print(f"\n  first up-size to >=1 mini at cum P&L "
                  f"${big['cum_pnl'].iloc[0]:,.0f} "
                  f"(entry {big['entry_time'].iloc[0]})")
        else:
            print("\n  size never exceeded micros (strategy did not bank enough "
                  "profit to unlock minis — correct safety behaviour).")

    out = config.OUTPUT_DIR
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(r_fix.equity.index, r_fix.equity.values, lw=1.1, color="grey",
            label=f"fixed 1 micro (${r_fix.equity.iloc[-1]-50000:+,.0f})")
    ax.plot(r_dyn.equity.index, r_dyn.equity.values, lw=1.3, color="steelblue",
            label=f"dynamic {args.anchor} (${r_dyn.equity.iloc[-1]-50000:+,.0f})")
    ax.axhline(config.TOPSTEP.starting_balance, ls="--", c="black", lw=0.7)
    ax.axhline(config.TOPSTEP.starting_balance - config.TOPSTEP.trailing_drawdown,
               ls=":", c="red", lw=0.8, label="-$2k floor (start)")
    ax.set_title(f"{inst} — dynamic sizing vs fixed 1 micro (micro->mini, anchor={args.anchor})")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    p = out / f"sizing_{inst}_{args.anchor}.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"\n  equity plot -> {p}")


# ============================================================================
# smoketest -- Databento DBN smoke test
# ============================================================================

def cmd_live(args):
    from live_trader import LiveTrader
    if args.live:
        config.TOPSTEP_API.dry_run = False
    elif args.dry_run:
        config.TOPSTEP_API.dry_run = True
    if not config.TOPSTEP_API.dry_run:
        print("  *** LIVE MODE: orders will be sent to your real Topstep account ***")
        confirm = input(f"  type the instrument key ({args.instrument}) to confirm: ")
        if confirm.strip() != args.instrument:
            sys.exit("  confirmation did not match -- aborting.")
    trader = LiveTrader(args.instrument)
    trader.run_forever()


def cmd_smoketest(args):
    bars = data_clients.load_universe_databento(args.path, interval=args.interval)
    if not bars:
        print("No tradable roots found after filtering.")
        return

    for alias, b in bars.items():
        print(f"\n=== {alias} ===")
        print(f"rows: {len(b):,} | range: {b.index.min()} -> {b.index.max()}")
        cols = ["open", "high", "low", "close", "volume", "delta", "cum_delta", "trade_count"]
        cols = [c for c in cols if c in b.columns]
        print(b[cols].head(8).to_string())
        if "delta" in b.columns:
            print(f"  total delta: {b['delta'].sum():,.0f}")
            print(f"  delta range: {b['delta'].min():,.0f} to {b['delta'].max():,.0f}")


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Quant research system -- one subcommand per pipeline.")
    sub = ap.add_subparsers(dest="command", required=True)

    # ml
    p = sub.add_parser("ml", help="ML strategy: data -> features -> labels -> "
                                  "discovery -> walk-forward -> report")
    p.add_argument("--source", choices=["eodhd", "databento", "synthetic"], default="synthetic")
    p.add_argument("--dbn-path", default=None,
                   help="path to Databento .dbn/.dbn.zst file, zip, or folder (required for --source databento)")
    p.add_argument("--instruments", nargs="*", default=None,
                   help="subset of NQ ES GC (default: all in config.TRADABLES)")
    p.add_argument("--quick", action="store_true",
                   help="fewer param samples — faster but less thorough")
    p.set_defaults(func=cmd_ml)

    # trend
    p = sub.add_parser("trend", help="Trend-following walk-forward pipeline")
    p.add_argument("--source", choices=["eodhd", "databento", "synthetic"], default="eodhd")
    p.add_argument("--dbn-path", default=None,
                   help="path to Databento .dbn/.dbn.zst file, zip, or folder (required for --source databento)")
    p.add_argument("--instruments", nargs="*", default=None)
    p.add_argument("--quick", action="store_true")
    p.set_defaults(func=cmd_trend)

    # reactive
    p = sub.add_parser("reactive", help="Reactive order-flow momentum — Databento only")
    p.add_argument("--dbn-path", required=True,
                   help="Databento data: .dbn file, .dbn.zst, zip, or directory")
    p.add_argument("--instruments", nargs="*", default=None,
                   help="subset of NQ ES GC (default: all in config.TRADABLES)")
    p.add_argument("--quick", action="store_true",
                   help="smaller windows + fewer samples for fast iteration")
    p.add_argument("--n-samples", type=int, default=None,
                   help="override n_param_samples (default: 500 full / 30 quick)")
    p.set_defaults(func=cmd_reactive)

    # structure
    p = sub.add_parser("structure", help="Market structure strategy — real NQ "
                                         "futures (Databento) by default")
    p.add_argument("--source", default="databento", choices=["databento", "eodhd"],
                   help="data source (default: databento — real NQ prices)")
    p.add_argument("--dbn-path", default=None,
                   help="Databento data (.dbn/.zst/zip/dir); required when "
                        "--source databento")
    p.add_argument("--instruments", nargs="*", default=None,
                   help="subset of instruments (default: NQ)")
    p.add_argument("--quick", action="store_true",
                   help="smaller windows + fewer samples for fast iteration")
    p.add_argument("--n-samples", type=int, default=None,
                   help="override n_param_samples")
    p.add_argument("--diagnose-only", action="store_true",
                   help="print diagnostics and plot; skip the walk-forward")
    p.set_defaults(func=cmd_structure)

    # structure-variants
    p = sub.add_parser("structure-variants",
                       help="Structure: base vs RTH-entry variant comparison")
    p.add_argument("--dbn-path", default=None)
    p.add_argument("--instrument", default="ES", help="NQ / ES / GC (default ES)")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-charts", action="store_true")
    p.add_argument("--dynamic-sizing", action="store_true",
                   help="also run dynamic-sized (micro->mini ladder) copies of the RTH variants")
    p.add_argument("--anchor", default="profit", choices=["profit", "floor"],
                   help="dynamic-sizing cushion anchor (default profit)")
    p.add_argument("--safety", type=float, default=0.5,
                   help="dynamic-sizing risk fraction of the cushion (default 0.5)")
    p.add_argument("--max-trade-risk-frac", type=float, default=0.2,
                   help="cap a single trade's worst-case at this fraction of the "
                        "$2k trailing DD (0.2=$400/trade; lower=safer, less upside)")
    p.add_argument("--daily-risk-frac", type=float, default=0.8,
                   help="cap a single trade's worst-case at this fraction of the "
                        "remaining daily-loss budget (default 0.8)")
    p.add_argument("--min-trades", type=int, default=None,
                   help="override WALKFORWARD.min_trades_in_sample (lower = fewer "
                        "skipped folds on sparse RTH variants; default keeps config)")
    p.add_argument("--variants", nargs="*", default=None,
                   help="subset of base variants to run (base rth te rth_te). "
                        "With --dynamic-sizing each selected variant also gets a "
                        "+dyn copy. Default: all.")
    p.add_argument("--contracts", type=int, default=1,
                   help="CONSTANT micro-contract multiplier for the fixed (non-dynamic) "
                        "variants. The right way to scale a consistent strategy: "
                        "linear PnL + same Sharpe (e.g. base --contracts 3). Capped "
                        "by TOPSTEP.max_contracts.")
    p.set_defaults(func=cmd_structure_variants)

    # adaptive
    p = sub.add_parser("adaptive", help="Adaptive ML — decay-weighted, "
                                        "regime-aware, self-refitting")
    p.add_argument("--source", choices=["eodhd", "databento", "synthetic"], default="eodhd")
    p.add_argument("--dbn-path", default=None,
                   help="path to Databento .dbn/.dbn.zst file, zip, or folder (required for --source databento)")
    p.add_argument("--instruments", nargs="*", default=None)
    p.add_argument("--halflife", type=int, default=1560,
                   help="decay half-life in bars (~20 trading days at 5m)")
    p.add_argument("--refit", type=int, default=390,
                   help="refit every N bars (~1 trading day at 5m)")
    p.add_argument("--threshold", type=float, default=0.08,
                   help="signal confidence threshold")
    p.add_argument("--warmup", type=int, default=3000,
                   help="bars before first fit (~38 trading days at 5m)")
    p.add_argument("--discovery-only", action="store_true",
                   help="run features -> labels -> discovery only, print rankings, then exit")
    p.set_defaults(func=cmd_adaptive)

    # sizing
    p = sub.add_parser("sizing", help="Dynamic position-sizing demonstration "
                                      "(generic micro -> mini, risk-based)")
    p.add_argument("--instrument", default="NQ")
    p.add_argument("--anchor", default="profit", choices=["profit", "floor"])
    p.add_argument("--safety", type=float, default=0.5)
    p.set_defaults(func=cmd_sizing)

    # live
    p = sub.add_parser("live", help="Live trading via TopstepX / ProjectX Gateway "
                                    "(mtp+dyn structure strategy)")
    p.add_argument("--instrument", default="NQ", choices=["NQ", "ES", "GC"])
    p.add_argument("--live", action="store_true",
                   help="disable dry-run and send real orders (overrides "
                        "config.TOPSTEP_API.dry_run / TOPSTEPX_DRY_RUN)")
    p.add_argument("--dry-run", action="store_true",
                   help="force dry-run even if TOPSTEPX_DRY_RUN=false in the environment")
    p.set_defaults(func=cmd_live)

    # smoketest
    p = sub.add_parser("smoketest", help="Databento DBN smoke test")
    p.add_argument("path", help="DBN file, zip, or folder (for example GLBX-...)")
    p.add_argument("--interval", default="5m", help="bar interval: 1m, 5m, 1h")
    p.set_defaults(func=cmd_smoketest)

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
