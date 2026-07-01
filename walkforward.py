"""
Walk-forward optimization -- the honest core of the whole system.

For each fold:
    [-------- TRAIN (in-sample) --------][--- TEST (out-of-sample) ---]

  1. Randomly sample N parameter sets, fit on TRAIN, score on embedded
     VALIDATION slice (last 30% of TRAIN). Keep the best.
  2. Refit best params on full TRAIN, evaluate ONCE on TEST.
  3. Stitch TEST equity curves. That concatenated OOS curve is the only
     number we trust.

NaN / blown-account guard: if the account equity goes to zero or negative
during a fold, we floor it at $1 and flag the fold rather than propagating
NaN through all subsequent folds.

One optimizer per strategy family (all sharing this same honest methodology --
fit on train, validate on embedded holdout, evaluate ONCE on test; equity
carried forward across folds; only OOS numbers matter):

  * run_walk_forward           -- ML strategy (searches model hyperparameters)
  * run_trend_walk_forward     -- trend-following (searches rule parameters,
                                   no classifier / features needed)
  * run_reactive_walk_forward  -- reactive order-flow (Databento only)
  * run_structure_walk_forward -- market structure (Databento or EODHD proxy)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

import config
import strategies as strat
import backtest as bt
import topstep as ts


# ============================================================================
# ML strategy walk-forward
# ============================================================================

@dataclass
class WalkForwardResult:
    oos_equity: pd.Series
    fold_summaries: pd.DataFrame
    oos_metrics: dict
    is_metrics_mean: dict
    chosen_params: list[dict]
    degradation: dict


def _sample_params(rng, feature_pool: list[str]) -> dict:
    k = rng.integers(3, min(8, len(feature_pool)) + 1)
    feats = list(rng.choice(feature_pool, size=k, replace=False))
    model = "logit"
    return {
        "model": str(model),
        "features": feats,
        "max_features": len(feats),
        "threshold": float(rng.choice([0.05, 0.08, 0.10, 0.12, 0.15])),
        "rf_depth": int(rng.choice([3, 4, 5])),
        "C": float(rng.choice([0.25, 0.5, 1.0, 2.0])),
    }


def _objective(metrics: dict) -> float:
    """
    In-sample selection objective. Rewards risk-adjusted return, penalises
    tiny-sample overfits and large drawdowns.
    Returns -inf if Sharpe is negative — we refuse to forward a losing config.
    """
    if metrics["n_trades"] < config.WALKFORWARD.min_trades_in_sample:
        return -np.inf
    sharpe = metrics["sharpe"]
    pf = metrics["profit_factor"] or 0.0
    dd_pen = abs(metrics["max_drawdown_pct"]) / 100.0
    return sharpe + 0.25 * min(pf, 3.0) - 2.0 * dd_pen


def run_walk_forward(features: pd.DataFrame, labels: pd.Series,
                     target_df: pd.DataFrame, instrument_key: str,
                     feature_pool: list[str], n_contracts: int = 1,
                     wf: config.WalkForward | None = None,
                     verbose: bool = True) -> WalkForwardResult:
    wf = wf or config.WALKFORWARD
    rng = np.random.default_rng(wf.random_seed)

    common = (features.index
                      .intersection(labels.index)
                      .intersection(target_df.index))
    X   = features.loc[common]
    y   = labels.loc[common]
    tdf = target_df.loc[common]
    n   = len(common)

    starts = range(0, n - wf.train_bars - wf.test_bars + 1, wf.step_bars)

    fold_rows     = []
    oos_curves    = []
    chosen        = []
    is_metric_list = []
    last_equity   = config.TOPSTEP.starting_balance
    oos_trades_all = []
    skipped_folds  = 0

    for fold_id, s in enumerate(starts):
        tr0, tr1 = s, s + wf.train_bars
        te0, te1 = tr1, min(tr1 + wf.test_bars, n)
        if te1 - te0 < 50:
            continue

        X_tr,  y_tr  = X.iloc[tr0:tr1],  y.iloc[tr0:tr1]
        X_te         = X.iloc[te0:te1]
        tdf_tr       = tdf.iloc[tr0:tr1]
        tdf_te       = tdf.iloc[te0:te1]

        # embedded validation: fit on first 70% of train, select on last 30%
        split   = int(len(X_tr) * 0.70)
        X_fit,  y_fit  = X_tr.iloc[:split], y_tr.iloc[:split]
        X_val          = X_tr.iloc[split:]
        tdf_val        = tdf_tr.iloc[split:]

        best_obj, best_params, best_val = -np.inf, None, None
        for _ in range(wf.n_param_samples):
            params = _sample_params(rng, feature_pool)
            fitted = strat.fit_strategy(X_fit, y_fit, params)
            if fitted is None:
                continue
            sig_val = fitted.predict_signal(X_val)
            vm  = bt.quick_metrics(tdf_val, sig_val, instrument_key, n_contracts)
            obj = _objective(vm)
            if obj > best_obj:
                best_obj, best_params, best_val = obj, params, vm

        # Skip this fold entirely if no config beat the Sharpe > 0 gate
        if best_params is None or best_obj == -np.inf:
            skipped_folds += 1
            if verbose:
                print(f"  fold {fold_id:2d} | skipped — no param set found positive "
                      f"validation Sharpe (n_samples={wf.n_param_samples})")
            continue

        # refit on full train, evaluate once on test
        fitted = strat.fit_strategy(X_tr, y_tr, best_params)
        if fitted is None:
            skipped_folds += 1
            continue

        sig_te = fitted.predict_signal(X_te)

        # carry equity forward; guard against blown account producing NaN
        safe_equity = max(last_equity, 1.0)
        rules_fold  = config.TopstepRules(**{**config.TOPSTEP.__dict__})
        rules_fold.starting_balance = safe_equity

        res_te      = bt.run_backtest(tdf_te, sig_te, instrument_key,
                                      n_contracts, rules=rules_fold)

        end_eq = float(res_te.equity.iloc[-1])
        # NaN / negative guard
        if not np.isfinite(end_eq) or end_eq <= 0:
            end_eq = safe_equity   # treat this fold as flat, don't propagate NaN
        last_equity = end_eq

        oos_curves.append(res_te.equity.clip(lower=1.0).ffill())
        if len(res_te.trades):
            oos_trades_all.append(res_te.trades)
        chosen.append(best_params)
        is_metric_list.append(best_val)

        m = res_te.metrics
        fold_rows.append({
            "fold":         fold_id,
            "train_start":  X.index[tr0],
            "test_start":   X.index[te0],
            "test_end":     X.index[te1 - 1],
            "is_sharpe":    best_val["sharpe"],
            "oos_sharpe":   m["sharpe"],
            "oos_pnl_$":    m["total_pnl_$"],
            "oos_trades":   m["n_trades"],
            "oos_win_rate": m["win_rate"],
            "oos_max_dd_pct": m["max_drawdown_pct"],
            "model":        best_params["model"],
        })
        if verbose:
            pnl = m['total_pnl_$']
            pnl_str = f"${pnl:+,.0f}" if np.isfinite(pnl) else "$nan"
            print(f"  fold {fold_id:2d} | IS Sharpe {best_val['sharpe']:+.2f} "
                  f"-> OOS Sharpe {m['sharpe']:+.2f} | OOS PnL {pnl_str} "
                  f"| {m['n_trades']} trades")

    if not oos_curves:
        raise RuntimeError(
            f"No valid folds produced — all {skipped_folds} folds skipped.\n"
            "This usually means the model finds no edge in this data with these "
            "features. Try: more n_param_samples, different features, or check "
            "that the data loaded correctly."
        )

    if skipped_folds:
        print(f"  ({skipped_folds} folds skipped — no positive-Sharpe config found)")

    # stitch OOS equity
    oos_equity = pd.concat(oos_curves)
    oos_equity = (oos_equity[~oos_equity.index.duplicated(keep="last")]
                             .sort_index()
                             .ffill())

    oos_rets   = oos_equity.diff().fillna(0.0)
    folds      = pd.DataFrame(fold_rows)
    oos_trades = (pd.concat(oos_trades_all, ignore_index=True)
                  if oos_trades_all else pd.DataFrame(columns=["pnl_$"]))

    oos_metrics = bt._compute_metrics(
        oos_equity, oos_rets, oos_trades,
        config.INTERVAL, oos_equity.iloc[0])
    oos_metrics["pct_folds_profitable"] = round(
        float((folds["oos_pnl_$"] > 0).mean()) * 100, 1) if len(folds) else 0.0

    oos_metrics["topstep"] = ts.evaluate(oos_equity, config.TOPSTEP)

    is_mean = {
        k: round(float(np.mean([m[k] for m in is_metric_list
                                if m.get(k) is not None])), 3)
        for k in ("sharpe", "profit_factor")
        if any(m.get(k) is not None for m in is_metric_list)
    }
    oos_s = oos_metrics["sharpe"]
    is_s  = is_mean.get("sharpe")
    degradation = {
        "is_sharpe_mean":       is_s,
        "oos_sharpe":           oos_s,
        "sharpe_retention_pct": (round(oos_s / is_s * 100, 1)
                                 if is_s and is_s != 0 else None),
    }

    return WalkForwardResult(
        oos_equity=oos_equity,
        fold_summaries=folds,
        oos_metrics=oos_metrics,
        is_metrics_mean=is_mean,
        chosen_params=chosen,
        degradation=degradation,
    )


# ============================================================================
# Trend-following walk-forward
# ============================================================================
#
# Same honest methodology, but searching over trend rule parameters instead of
# ML model hyperparameters. No classifier, no features DataFrame needed —
# just raw OHLCV.

@dataclass
class TrendWalkForwardResult:
    oos_equity: pd.Series
    fold_summaries: pd.DataFrame
    oos_metrics: dict
    chosen_params: list[dict]
    degradation: dict


def _trend_objective(metrics: dict, min_trades: int = 8) -> float:
    if metrics["n_trades"] < min_trades:
        return -np.inf
    sharpe = metrics["sharpe"]
    pf     = metrics["profit_factor"] or 0.0
    dd     = abs(metrics["max_drawdown_pct"]) / 100.0
    return sharpe + 0.3 * min(pf, 3.0) - 2.0 * dd


def run_trend_walk_forward(target_df: pd.DataFrame,
                           instrument_key: str,
                           n_contracts: int = 1,
                           wf: config.WalkForward | None = None,
                           verbose: bool = True) -> TrendWalkForwardResult:
    wf  = wf  or config.WALKFORWARD
    rng = np.random.default_rng(wf.random_seed)

    n = len(target_df)
    starts = range(0, n - wf.train_bars - wf.test_bars + 1, wf.step_bars)

    fold_rows      = []
    oos_curves     = []
    chosen         = []
    is_metric_list = []
    oos_trades_all = []
    last_equity    = config.TOPSTEP.starting_balance
    skipped        = 0

    for fold_id, s in enumerate(starts):
        tr0, tr1 = s, s + wf.train_bars
        te0, te1 = tr1, min(tr1 + wf.test_bars, n)
        if te1 - te0 < 50:
            continue

        tdf_tr = target_df.iloc[tr0:tr1]
        tdf_te = target_df.iloc[te0:te1]

        # embedded validation: fit/select on last 30% of train
        split   = int(len(tdf_tr) * 0.70)
        tdf_fit = tdf_tr.iloc[:split]
        tdf_val = tdf_tr.iloc[split:]

        best_obj, best_params, best_val = -np.inf, None, None

        for _ in range(wf.n_param_samples):
            params = strat.sample_trend_params(rng)
            # generate signal on fit slice, score on val slice
            sig_fit = strat.generate_trend_signal(tdf_fit, params)
            sig_val = strat.generate_trend_signal(
                pd.concat([tdf_fit.iloc[-200:], tdf_val]),   # warmup context
                params
            ).iloc[200:]   # trim warmup

            if len(sig_val) == 0:
                continue

            vm  = bt.quick_metrics(tdf_val, sig_val, instrument_key, n_contracts)
            obj = _trend_objective(vm)
            if obj > best_obj:
                best_obj, best_params, best_val = obj, params, vm

        if best_params is None:
            skipped += 1
            if verbose:
                print(f"  fold {fold_id:2d} | skipped — no valid config")
            continue

        # generate signal on full train for warmup, extract test slice
        warmup = tdf_tr.iloc[-200:]
        sig_full = strat.generate_trend_signal(
            pd.concat([warmup, tdf_te]), best_params
        ).iloc[200:]

        safe_eq    = max(last_equity, 1.0)
        rules_fold = config.TopstepRules(**{**config.TOPSTEP.__dict__})
        rules_fold.starting_balance = safe_eq

        res_te  = bt.run_backtest(tdf_te, sig_full, instrument_key,
                                  n_contracts, rules=rules_fold)

        end_eq = float(res_te.equity.iloc[-1])
        if not np.isfinite(end_eq) or end_eq <= 0:
            end_eq = safe_eq
        last_equity = end_eq

        oos_curves.append(res_te.equity.clip(lower=1.0).ffill())
        if len(res_te.trades):
            oos_trades_all.append(res_te.trades)
        chosen.append(best_params)
        is_metric_list.append(best_val)

        m = res_te.metrics
        pnl_str = f"${m['total_pnl_$']:+,.0f}" if np.isfinite(m['total_pnl_$']) else "$nan"
        fold_rows.append({
            "fold":            fold_id,
            "test_start":      target_df.index[te0],
            "test_end":        target_df.index[te1 - 1],
            "is_sharpe":       best_val["sharpe"],
            "oos_sharpe":      m["sharpe"],
            "oos_pnl_$":       m["total_pnl_$"],
            "oos_trades":      m["n_trades"],
            "oos_win_rate":    m["win_rate"],
            "oos_max_dd_pct":  m["max_drawdown_pct"],
            "breakout_bars":   best_params["breakout_bars"],
            "stop_atr_mult":   best_params["stop_atr_mult"],
            "horizon_bars":    best_params["horizon_bars"],
        })
        if verbose:
            print(f"  fold {fold_id:2d} | IS {best_val['sharpe']:+.2f} "
                  f"-> OOS {m['sharpe']:+.2f} | {pnl_str} "
                  f"| {m['n_trades']} trades "
                  f"| bb={best_params['breakout_bars']} "
                  f"stop={best_params['stop_atr_mult']}x "
                  f"h={best_params['horizon_bars']}")

    if not oos_curves:
        raise RuntimeError(
            f"No valid folds — {skipped} skipped.\n"
            "Try increasing n_param_samples or check data length."
        )

    if skipped:
        print(f"  ({skipped} folds skipped)")

    oos_equity = pd.concat(oos_curves)
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")].sort_index().ffill()
    oos_rets   = oos_equity.diff().fillna(0.0)
    folds      = pd.DataFrame(fold_rows)
    oos_trades = (pd.concat(oos_trades_all, ignore_index=True)
                  if oos_trades_all else pd.DataFrame(columns=["pnl_$"]))

    oos_metrics = bt._compute_metrics(
        oos_equity, oos_rets, oos_trades,
        config.INTERVAL, oos_equity.iloc[0])
    oos_metrics["pct_folds_profitable"] = round(
        float((folds["oos_pnl_$"] > 0).mean()) * 100, 1) if len(folds) else 0.0

    oos_metrics["topstep"] = ts.evaluate(oos_equity, config.TOPSTEP)

    is_sharpes = [m["sharpe"] for m in is_metric_list]
    is_mean    = round(float(np.mean(is_sharpes)), 3) if is_sharpes else None
    oos_s      = oos_metrics["sharpe"]
    degradation = {
        "is_sharpe_mean": is_mean,
        "oos_sharpe":     oos_s,
        "sharpe_retention_pct": (round(oos_s / is_mean * 100, 1)
                                 if is_mean and is_mean != 0 else None),
    }

    return TrendWalkForwardResult(
        oos_equity=oos_equity,
        fold_summaries=folds,
        oos_metrics=oos_metrics,
        chosen_params=chosen,
        degradation=degradation,
    )


# ============================================================================
# Reactive order-flow walk-forward
# ============================================================================
#
# No ML model, no feature matrix, no labels. The signal is purely rule-based.

# Bars of train history prepended before each test fold so that rolling
# indicators (ATR 100-bar percentile being the longest) are fully warmed up.
_REACTIVE_LOOKBACK_WARM = 110


@dataclass
class ReactiveWFResult:
    oos_equity:     pd.Series
    fold_summaries: pd.DataFrame
    oos_metrics:    dict
    is_metrics_mean: dict
    chosen_params:  list[dict]
    degradation:    dict


def _sample_reactive_params(rng: np.random.Generator) -> dict:
    return {
        "score_threshold":       int(rng.choice([2, 3, 4])),
        "imbalance_threshold":   float(rng.uniform(0.52, 0.65)),
        "delta_zscore_window":   int(rng.integers(10, 50)),
        "bar_delta_window":      int(rng.integers(8, 25)),
        "max_hold_bars":         int(rng.integers(4, 30)),
        "large_print_threshold": float(rng.uniform(0.03, 0.25)),
        "trend_filter":          bool(rng.choice([True, False])),
    }


def _reactive_objective(metrics: dict) -> float:
    """Reward risk-adjusted return; reject configs that barely trade or lose."""
    if metrics["n_trades"] < config.WALKFORWARD.min_trades_in_sample:
        return -np.inf
    sharpe = metrics["sharpe"]
    if sharpe <= 0:
        return -np.inf
    pf     = metrics["profit_factor"] or 0.0
    dd_pen = abs(metrics["max_drawdown_pct"]) / 100.0
    return sharpe + 0.25 * min(pf, 3.0) - 2.0 * dd_pen


def run_reactive_walk_forward(
    target_df: pd.DataFrame,
    instrument_key: str,
    n_contracts: int = 1,
    wf: config.WalkForward | None = None,
    verbose: bool = True,
) -> ReactiveWFResult:
    """
    target_df must be a Databento bar DataFrame (OHLCV + order-flow columns).
    n_contracts=2 allows size_scalar=0.5 to trade 1 contract (half size).
    """
    strat._check_reactive_columns(target_df)
    wf  = wf or config.WALKFORWARD
    rng = np.random.default_rng(wf.random_seed)

    tdf = target_df.copy()
    n   = len(tdf)

    starts = range(0, n - wf.train_bars - wf.test_bars + 1, wf.step_bars)

    fold_rows       = []
    oos_curves      = []
    chosen          = []
    is_metric_list  = []
    last_equity     = config.TOPSTEP.starting_balance
    oos_trades_all  = []
    skipped_folds   = 0

    for fold_id, s in enumerate(starts):
        tr0, tr1 = s, s + wf.train_bars
        te0, te1 = tr1, min(tr1 + wf.test_bars, n)
        if te1 - te0 < 50:
            continue

        tdf_tr  = tdf.iloc[tr0:tr1]
        tdf_te  = tdf.iloc[te0:te1]

        # Validation: run signal on full train, score on the last 30 %.
        # Running on the full train slice means rolling lookbacks are warm
        # by the time we reach the val portion.
        split   = int(wf.train_bars * 0.70)
        val_idx = tdf_tr.index[split:]
        tdf_val = tdf_tr.iloc[split:]

        best_obj, best_params, best_val = -np.inf, None, None
        for _ in range(wf.n_param_samples):
            params = _sample_reactive_params(rng)
            try:
                sig_tr, scl_tr = strat.generate_reactive_signal(tdf_tr, params, instrument_key)
            except Exception:
                continue
            sig_val = sig_tr.loc[val_idx]
            scl_val = scl_tr.loc[val_idx]
            vm      = bt.quick_metrics(tdf_val, sig_val * scl_val,
                                       instrument_key, n_contracts)
            obj     = _reactive_objective(vm)
            if obj > best_obj:
                best_obj, best_params, best_val = obj, params, vm

        if best_params is None or best_obj == -np.inf:
            skipped_folds += 1
            if verbose:
                print(f"  fold {fold_id:2d} | skipped — no param set found positive "
                      f"validation Sharpe (n_samples={wf.n_param_samples})")
            continue

        # Test: prepend enough train history so rolling indicators are warm.
        warmup_start  = max(0, te0 - _REACTIVE_LOOKBACK_WARM)
        tdf_warmed    = tdf.iloc[warmup_start:te1]
        sig_w, scl_w  = strat.generate_reactive_signal(tdf_warmed, best_params, instrument_key)
        sig_te  = sig_w.loc[tdf_te.index]
        scl_te  = scl_w.loc[tdf_te.index]

        safe_equity = max(last_equity, 1.0)
        rules_fold  = config.TopstepRules(**{**config.TOPSTEP.__dict__})
        rules_fold.starting_balance = safe_equity

        res_te = bt.run_backtest(tdf_te, sig_te * scl_te,
                                 instrument_key, n_contracts,
                                 rules=rules_fold)

        end_eq = float(res_te.equity.iloc[-1])
        if not np.isfinite(end_eq) or end_eq <= 0:
            end_eq = safe_equity
        last_equity = end_eq

        oos_curves.append(res_te.equity.clip(lower=1.0).ffill())
        if len(res_te.trades):
            oos_trades_all.append(res_te.trades)
        chosen.append(best_params)
        is_metric_list.append(best_val)

        m = res_te.metrics
        fold_rows.append({
            "fold":                  fold_id,
            "train_start":           tdf.index[tr0],
            "test_start":            tdf.index[te0],
            "test_end":              tdf.index[te1 - 1],
            "is_sharpe":             best_val["sharpe"],
            "oos_sharpe":            m["sharpe"],
            "oos_pnl_$":             m["total_pnl_$"],
            "oos_trades":            m["n_trades"],
            "oos_win_rate":          m["win_rate"],
            "oos_max_dd_pct":        m["max_drawdown_pct"],
            "score_threshold":       best_params["score_threshold"],
            "imbalance_threshold":   round(best_params["imbalance_threshold"],   3),
            "delta_zscore_window":   best_params["delta_zscore_window"],
            "bar_delta_window":      best_params.get("bar_delta_window", 12),
            "max_hold_bars":         best_params["max_hold_bars"],
            "large_print_threshold": round(best_params["large_print_threshold"], 3),
            "trend_filter":          best_params.get("trend_filter", True),
        })
        if verbose:
            pnl_str = f"${m['total_pnl_$']:+,.0f}"
            print(
                f"  fold {fold_id:2d} | "
                f"IS {best_val['sharpe']:+.2f} -> OOS {m['sharpe']:+.2f} | "
                f"PnL {pnl_str} | {m['n_trades']} trades | "
                f"thr={best_params['score_threshold']} "
                f"hold={best_params['max_hold_bars']} "
                f"dz_win={best_params['delta_zscore_window']}"
            )

    if not oos_curves:
        raise RuntimeError(
            f"No valid folds produced — all {skipped_folds} skipped.\n"
            "Check that the Databento data contains sufficient order-flow history "
            "and that the bar count exceeds train_bars + test_bars."
        )

    if skipped_folds:
        print(f"  ({skipped_folds} fold(s) skipped — no positive-Sharpe config found)")

    # Stitch OOS equity
    oos_equity = pd.concat(oos_curves)
    oos_equity = (oos_equity[~oos_equity.index.duplicated(keep="last")]
                             .sort_index().ffill())

    oos_rets   = oos_equity.diff().fillna(0.0)
    folds      = pd.DataFrame(fold_rows)
    oos_trades = (pd.concat(oos_trades_all, ignore_index=True)
                  if oos_trades_all else pd.DataFrame(columns=["pnl_$"]))

    oos_metrics = bt._compute_metrics(
        oos_equity, oos_rets, oos_trades,
        config.INTERVAL, oos_equity.iloc[0])
    oos_metrics["pct_folds_profitable"] = (
        round(float((folds["oos_pnl_$"] > 0).mean()) * 100, 1)
        if len(folds) else 0.0
    )

    oos_metrics["topstep"] = ts.evaluate(oos_equity, config.TOPSTEP)

    is_mean = {
        k: round(float(np.mean([m[k] for m in is_metric_list
                                if m.get(k) is not None])), 3)
        for k in ("sharpe", "profit_factor")
        if any(m.get(k) is not None for m in is_metric_list)
    }
    oos_s = oos_metrics["sharpe"]
    is_s  = is_mean.get("sharpe")
    degradation = {
        "is_sharpe_mean":       is_s,
        "oos_sharpe":           oos_s,
        "sharpe_retention_pct": (round(oos_s / is_s * 100, 1)
                                 if is_s and is_s != 0 else None),
    }

    return ReactiveWFResult(
        oos_equity      = oos_equity,
        fold_summaries  = folds,
        oos_metrics     = oos_metrics,
        is_metrics_mean = is_mean,
        chosen_params   = chosen,
        degradation     = degradation,
    )


# ============================================================================
# Market structure walk-forward
# ============================================================================
#
# Data source: EODHD 5-minute QQQ proxy bars, or Databento real futures bars.

# Need enough history to warm the daily EMA20 (20 trading days = ~1560 bars)
# plus the hourly ATR14 (14 hourly bars).  We prepend 1600 bars as a safe margin.
_STRUCTURE_LOOKBACK_WARM = 1600


@dataclass
class StructureWFResult:
    oos_equity:      pd.Series
    fold_summaries:  pd.DataFrame
    oos_metrics:     dict
    is_metrics_mean: dict
    chosen_params:   list[dict]
    degradation:     dict
    oos_trades:      pd.DataFrame        # concatenated OOS trades (entry/exit/pnl)
    oos_stops:       pd.Series           # per-bar stop level over the OOS period
    oos_taken:       pd.DataFrame        # taken entries w/ pre-entry R:R + target
    oos_skipped:     pd.DataFrame        # R:R-skipped entries + retrospective + fold


def _sample_structure_params(rng: np.random.Generator, variant: dict | None = None) -> dict:
    # Base ranges (per the agreed spec):
    #   grab_wick_ratio  -> 0.50-0.58   (best OOS sits at the low end)
    #   fvg_min_atr_mult -> 1.0-1.2     (smaller gaps work better)
    #   grab_atr_mult    -> 0.20-0.60   (Variant A lowers the floor to 0.15)
    # A variant may override the grab floor and inject fixed params that are
    # applied to EVERY sampled set (so the variant's behaviour is held constant
    # while the tunable params are searched identically to the base).
    variant = variant or {}
    grab_lo = float(variant.get("grab_atr_min", 0.20))
    p = {
        "fvg_min_atr_mult":    float(rng.uniform(1.1, 1.15)),
        "grab_wick_ratio":     float(rng.uniform(0.50, 0.54)),
        "grab_atr_mult":       float(rng.uniform(grab_lo, 0.45)),
        "max_hold_bars":       int(rng.integers(18, 24)),
        "extension_threshold": float(rng.uniform(1.4, 1.8)),
        # Fix 2 — new tunables (inert unless the variant enables liquidity targets)
        "n_targets":           int(rng.integers(1, 4)),     # 1..3 TP levels
        "min_rr":              float(rng.uniform(1.5, 3.0)),
    }
    p.update(variant.get("fixed", {}))
    return p


def _structure_selection_objective(
    tdf_val: pd.DataFrame,
    eff_signal: pd.Series,
    instrument_key: str,
    n_contracts: int,
    full_metrics: dict,
) -> float:
    """
    Overfit-aware selection score (replaces the old peak-Sharpe objective).

    Picking the single highest-Sharpe config out of hundreds of random samples
    reliably curve-fits validation noise — the observed IS/OOS Sharpe correlation
    was about -0.85 (high IS => bad OOS). So rather than rewarding peak Sharpe we
    reward configs that are *robust*:

      * profitable in BOTH halves of the validation window (regime robustness)
      * backed by enough trades (statistical reliability, not a few lucky ones)
      * low drawdown

    and we CAP the Sharpe and profit-factor contributions so a single lucky
    spike cannot dominate the ranking.
    """
    m = full_metrics
    floor = config.WALKFORWARD.min_trades_in_sample
    if m["n_trades"] < floor:
        return -np.inf
    if m["sharpe"] <= 0:
        return -np.inf

    pf = m["profit_factor"] or 0.0
    dd = abs(m["max_drawdown_pct"]) / 100.0

    # Capped, drawdown-heavy base — peak Sharpe alone is not trusted.
    base = min(m["sharpe"], 2.5) + 0.20 * min(pf, 3.0) - 2.5 * dd

    # Split-half consistency: a config that only works in one half is fragile.
    half = len(eff_signal) // 2
    if half >= 10:
        m1 = bt.quick_metrics(tdf_val.iloc[:half], eff_signal.iloc[:half],
                              instrument_key, n_contracts)
        m2 = bt.quick_metrics(tdf_val.iloc[half:], eff_signal.iloc[half:],
                              instrument_key, n_contracts)
        if not (m1["total_pnl_$"] > 0 and m2["total_pnl_$"] > 0):
            base -= 1.0

    # Thin-trade penalty beyond the hard minimum (more trades = more reliable).
    if m["n_trades"] < 2 * floor:
        base -= 0.5 * (2 * floor - m["n_trades"]) / floor

    return base


def run_structure_walk_forward(
    target_df: pd.DataFrame,
    instrument_key: str,
    n_contracts: int = 1,
    wf: config.WalkForward | None = None,
    verbose: bool = True,
    variant: dict | None = None,
    sizing: "config.DynamicSizing | None" = None,
) -> StructureWFResult:
    """
    target_df must be a 5-minute OHLCV DataFrame (real NQ/ES/GC bars).

    `variant` (optional) holds the variant spec: {"grab_atr_min": float,
    "fixed": {param overrides applied to every sampled set}}. With variant=None
    the search and behaviour are exactly the base strategy.
    """
    wf  = wf or config.WALKFORWARD
    rng = np.random.default_rng(wf.random_seed)

    tdf = target_df.copy()
    n   = len(tdf)

    starts = range(0, n - wf.train_bars - wf.test_bars + 1, wf.step_bars)

    fold_rows      = []
    oos_curves     = []
    chosen         = []
    is_metric_list = []
    last_equity    = config.TOPSTEP.starting_balance
    oos_trades_all = []
    oos_stop_parts = []
    oos_taken_all  = []
    oos_skipped_all = []
    skipped_folds  = 0

    for fold_id, s in enumerate(starts):
        tr0, tr1 = s, s + wf.train_bars
        te0, te1 = tr1, min(tr1 + wf.test_bars, n)
        if te1 - te0 < 50:
            continue

        tdf_tr = tdf.iloc[tr0:tr1]
        tdf_te = tdf.iloc[te0:te1]

        # Validation: signal on full train slice, score only the last 30 %
        split   = int(wf.train_bars * 0.70)
        val_idx = tdf_tr.index[split:]
        tdf_val = tdf_tr.iloc[split:]

        best_obj, best_params, best_val = -np.inf, None, None
        for _ in range(wf.n_param_samples):
            params = _sample_structure_params(rng, variant)
            try:
                sig_tr, scl_tr = strat.generate_structure_signal(
                    tdf_tr, params, instrument_key)
            except Exception:
                continue
            sig_val = sig_tr.loc[val_idx]
            scl_val = scl_tr.loc[val_idx]
            eff_val = sig_val * scl_val
            vm      = bt.quick_metrics(
                tdf_val, eff_val, instrument_key, n_contracts)
            obj = _structure_selection_objective(
                tdf_val, eff_val, instrument_key, n_contracts, vm)
            if obj > best_obj:
                best_obj, best_params, best_val = obj, params, vm

        if best_params is None or best_obj == -np.inf:
            skipped_folds += 1
            if verbose:
                print(f"  fold {fold_id:2d} | skipped — no positive-Sharpe config "
                      f"(n_samples={wf.n_param_samples})")
            continue

        # Test: prepend history so daily EMAs are warm before the test window.
        # Use the debug API so we also capture the per-bar stop level for charts.
        warmup_start = max(0, te0 - _STRUCTURE_LOOKBACK_WARM)
        tdf_warmed   = tdf.iloc[warmup_start:te1]
        dbg_w  = strat.generate_structure_signal_debug(
            tdf_warmed, best_params, instrument_key)
        sig_te  = dbg_w["signal"].loc[tdf_te.index]
        scl_te  = dbg_w["size"].loc[tdf_te.index]
        stop_te = dbg_w["stop"].loc[tdf_te.index]
        oos_stop_parts.append(stop_te)

        # Taken/skipped entries that fall inside this fold's TEST window only
        # (the warmup prefix is excluded so each entry is counted once).
        te_start = tdf.index[te0]
        tk = dbg_w.get("taken")
        if tk is not None and len(tk):
            tk = tk[tk["entry_time"] >= te_start].copy()
            tk["fold"] = fold_id
            oos_taken_all.append(tk)
        sk = dbg_w.get("skipped")
        if sk is not None and len(sk):
            sk = sk[sk["skip_time"] >= te_start].copy()
            sk["fold"] = fold_id
            oos_skipped_all.append(sk)

        safe_equity = max(last_equity, 1.0)
        rules_fold  = config.TopstepRules(**{**config.TOPSTEP.__dict__})
        rules_fold.starting_balance = safe_equity

        res_te = bt.run_backtest(
            tdf_te, sig_te * scl_te, instrument_key, n_contracts,
            rules=rules_fold, stops=stop_te, sizing=sizing)

        end_eq = float(res_te.equity.iloc[-1])
        if not np.isfinite(end_eq) or end_eq <= 0:
            end_eq = safe_equity
        last_equity = end_eq

        oos_curves.append(res_te.equity.clip(lower=1.0).ffill())
        if len(res_te.trades):
            oos_trades_all.append(res_te.trades)
        chosen.append(best_params)
        is_metric_list.append(best_val)

        m = res_te.metrics
        fold_rows.append({
            "fold":                fold_id,
            "train_start":         tdf.index[tr0],
            "test_start":          tdf.index[te0],
            "test_end":            tdf.index[te1 - 1],
            "is_sharpe":           best_val["sharpe"],
            "oos_sharpe":          m["sharpe"],
            "oos_pnl_$":           m["total_pnl_$"],
            "oos_trades":          m["n_trades"],
            "oos_win_rate":        m["win_rate"],
            "oos_max_dd_pct":      m["max_drawdown_pct"],
            "fvg_min_atr_mult":    round(best_params["fvg_min_atr_mult"],    3),
            "grab_wick_ratio":     round(best_params["grab_wick_ratio"],     3),
            "grab_atr_mult":       round(best_params["grab_atr_mult"],       3),
            "max_hold_bars":       best_params["max_hold_bars"],
            "extension_threshold": round(best_params["extension_threshold"], 3),
            "min_rr":              (round(float(best_params["min_rr"]), 3)
                                    if best_params.get("min_rr") is not None else None),
            "n_targets":           int(best_params.get("n_targets", 1)),
        })

        if verbose:
            pnl_str = f"${m['total_pnl_$']:+,.0f}"
            print(
                f"  fold {fold_id:2d} | "
                f"IS {best_val['sharpe']:+.2f} -> OOS {m['sharpe']:+.2f} | "
                f"PnL {pnl_str} | {m['n_trades']} trades | "
                f"fvg_mult={best_params['fvg_min_atr_mult']:.2f}  "
                f"hold={best_params['max_hold_bars']}  "
                f"ext_thr={best_params['extension_threshold']:.2f}"
            )

    if not oos_curves:
        raise RuntimeError(
            f"No valid folds — all {skipped_folds} skipped.\n"
            "Check that EODHD data spans at least train_bars + test_bars "
            "and that min_trades_in_sample is not too tight."
        )

    if skipped_folds:
        print(f"  ({skipped_folds} fold(s) skipped — no positive-Sharpe config found)")

    oos_equity = pd.concat(oos_curves)
    oos_equity = (oos_equity[~oos_equity.index.duplicated(keep="last")]
                             .sort_index().ffill())

    oos_rets   = oos_equity.diff().fillna(0.0)
    folds      = pd.DataFrame(fold_rows)
    oos_trades = (pd.concat(oos_trades_all, ignore_index=True)
                  if oos_trades_all else pd.DataFrame(columns=["pnl_$"]))
    oos_stops  = (pd.concat(oos_stop_parts) if oos_stop_parts
                  else pd.Series(dtype=float))
    if len(oos_stops):
        oos_stops = oos_stops[~oos_stops.index.duplicated(keep="last")].sort_index()
    oos_taken   = (pd.concat(oos_taken_all, ignore_index=True)
                   if oos_taken_all else pd.DataFrame())
    oos_skipped = (pd.concat(oos_skipped_all, ignore_index=True)
                   if oos_skipped_all else pd.DataFrame())

    oos_metrics = bt._compute_metrics(
        oos_equity, oos_rets, oos_trades,
        config.INTERVAL, oos_equity.iloc[0])
    oos_metrics["pct_folds_profitable"] = (
        round(float((folds["oos_pnl_$"] > 0).mean()) * 100, 1)
        if len(folds) else 0.0
    )

    oos_metrics["topstep"] = ts.evaluate(oos_equity, config.TOPSTEP)

    is_mean = {
        k: round(float(np.mean([m[k] for m in is_metric_list
                                if m.get(k) is not None])), 3)
        for k in ("sharpe", "profit_factor")
        if any(m.get(k) is not None for m in is_metric_list)
    }
    oos_s = oos_metrics["sharpe"]
    is_s  = is_mean.get("sharpe")
    degradation = {
        "is_sharpe_mean":       is_s,
        "oos_sharpe":           oos_s,
        "sharpe_retention_pct": (
            round(oos_s / is_s * 100, 1) if is_s and is_s != 0 else None
        ),
    }

    return StructureWFResult(
        oos_equity      = oos_equity,
        fold_summaries  = folds,
        oos_metrics     = oos_metrics,
        is_metrics_mean = is_mean,
        chosen_params   = chosen,
        degradation     = degradation,
        oos_trades      = oos_trades,
        oos_stops       = oos_stops,
        oos_taken       = oos_taken,
        oos_skipped     = oos_skipped,
    )


# ============================================================================
# EXPERIMENTAL: session-gap reversal walk-forward
# ============================================================================
#
# Same honest fit-train / select-embedded-validation / evaluate-once-OOS
# methodology as the other optimizers above, EXCEPT the trade-count floor used
# to reject a sampled config is the `min_trades` ARGUMENT (default 3), not
# config.WALKFORWARD.min_trades_in_sample (default 10). This is a deliberate
# BYPASS of the usual gate: strategies.generate_session_gap_signal fires at
# most once per calendar day, so the normal floor would reject nearly every
# fold before we ever saw whether the idea is worth pursuing. Read the
# resulting trade counts accordingly -- a handful of trades per fold is a
# first look, not a statistically reliable sample.

@dataclass
class SessionGapWFResult:
    oos_equity: pd.Series
    fold_summaries: pd.DataFrame
    oos_metrics: dict
    chosen_params: list[dict]
    degradation: dict
    min_trades_used: int


def _session_gap_objective(metrics: dict, min_trades: int) -> float:
    if metrics["n_trades"] < min_trades:
        return -np.inf
    sharpe = metrics["sharpe"]
    pf     = metrics["profit_factor"] or 0.0
    dd     = abs(metrics["max_drawdown_pct"]) / 100.0
    return sharpe + 0.3 * min(pf, 3.0) - 2.0 * dd


def run_session_gap_walk_forward(target_df: pd.DataFrame,
                                 instrument_key: str,
                                 n_contracts: int = 1,
                                 wf: config.WalkForward | None = None,
                                 verbose: bool = True,
                                 min_trades: int = 3) -> SessionGapWFResult:
    wf  = wf or config.WALKFORWARD
    rng = np.random.default_rng(wf.random_seed)

    n = len(target_df)
    starts = range(0, n - wf.train_bars - wf.test_bars + 1, wf.step_bars)

    fold_rows, oos_curves, chosen, is_metric_list, oos_trades_all = [], [], [], [], []
    last_equity = config.TOPSTEP.starting_balance
    skipped = 0

    for fold_id, s in enumerate(starts):
        tr0, tr1 = s, s + wf.train_bars
        te0, te1 = tr1, min(tr1 + wf.test_bars, n)
        if te1 - te0 < 50:
            continue

        tdf_tr = target_df.iloc[tr0:tr1]
        tdf_te = target_df.iloc[te0:te1]

        split   = int(len(tdf_tr) * 0.70)
        tdf_fit = tdf_tr.iloc[:split]
        tdf_val = tdf_tr.iloc[split:]

        best_obj, best_params, best_val = -np.inf, None, None
        for _ in range(wf.n_param_samples):
            params = strat.sample_session_gap_params(rng)
            # 200-bar warmup context so the London/day-boundary state is sane
            # at the start of the validation slice.
            sig_val = strat.generate_session_gap_signal(
                pd.concat([tdf_fit.iloc[-200:], tdf_val]), params, instrument_key
            )[0].iloc[200:]
            if len(sig_val) == 0:
                continue
            vm  = bt.quick_metrics(tdf_val, sig_val, instrument_key, n_contracts)
            obj = _session_gap_objective(vm, min_trades)
            if obj > best_obj:
                best_obj, best_params, best_val = obj, params, vm

        if best_params is None:
            skipped += 1
            if verbose:
                print(f"  fold {fold_id:2d} | skipped -- no config cleared "
                      f"min_trades={min_trades} on validation")
            continue

        warmup = tdf_tr.iloc[-200:]
        sig_full = strat.generate_session_gap_signal(
            pd.concat([warmup, tdf_te]), best_params, instrument_key
        )[0].iloc[200:]

        safe_eq    = max(last_equity, 1.0)
        rules_fold = config.TopstepRules(**{**config.TOPSTEP.__dict__})
        rules_fold.starting_balance = safe_eq

        res_te = bt.run_backtest(tdf_te, sig_full, instrument_key,
                                 n_contracts, rules=rules_fold)

        end_eq = float(res_te.equity.iloc[-1])
        if not np.isfinite(end_eq) or end_eq <= 0:
            end_eq = safe_eq
        last_equity = end_eq

        oos_curves.append(res_te.equity.clip(lower=1.0).ffill())
        if len(res_te.trades):
            oos_trades_all.append(res_te.trades)
        chosen.append(best_params)
        is_metric_list.append(best_val)

        m = res_te.metrics
        pnl_str = f"${m['total_pnl_$']:+,.0f}" if np.isfinite(m['total_pnl_$']) else "$nan"
        fold_rows.append({
            "fold":            fold_id,
            "test_start":      target_df.index[te0],
            "test_end":        target_df.index[te1 - 1],
            "is_sharpe":       best_val["sharpe"],
            "oos_sharpe":      m["sharpe"],
            "oos_pnl_$":       m["total_pnl_$"],
            "oos_trades":      m["n_trades"],
            "oos_win_rate":    m["win_rate"],
            "oos_max_dd_pct":  m["max_drawdown_pct"],
            "sweep_atr_mult":  best_params["sweep_atr_mult"],
            "stop_atr_mult":   best_params["stop_atr_mult"],
        })
        if verbose:
            print(f"  fold {fold_id:2d} | IS {best_val['sharpe']:+.2f} "
                  f"-> OOS {m['sharpe']:+.2f} | {pnl_str} | {m['n_trades']} trades "
                  f"(min_trades={min_trades}) | sweep={best_params['sweep_atr_mult']}x "
                  f"stop={best_params['stop_atr_mult']}x")

    if not oos_curves:
        raise RuntimeError(
            f"No valid folds -- {skipped} skipped even with min_trades={min_trades}.\n"
            "Try a lower --min-trades, more --n-samples, or a longer data window."
        )
    if skipped:
        print(f"  ({skipped} fold(s) skipped even with the relaxed min_trades={min_trades})")

    oos_equity = pd.concat(oos_curves)
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")].sort_index().ffill()
    oos_rets   = oos_equity.diff().fillna(0.0)
    folds      = pd.DataFrame(fold_rows)
    oos_trades = (pd.concat(oos_trades_all, ignore_index=True)
                  if oos_trades_all else pd.DataFrame(columns=["pnl_$"]))

    oos_metrics = bt._compute_metrics(
        oos_equity, oos_rets, oos_trades, config.INTERVAL, oos_equity.iloc[0])
    oos_metrics["pct_folds_profitable"] = round(
        float((folds["oos_pnl_$"] > 0).mean()) * 100, 1) if len(folds) else 0.0
    oos_metrics["topstep"] = ts.evaluate(oos_equity, config.TOPSTEP)

    is_sharpes = [m["sharpe"] for m in is_metric_list]
    is_mean    = round(float(np.mean(is_sharpes)), 3) if is_sharpes else None
    oos_s      = oos_metrics["sharpe"]
    degradation = {
        "is_sharpe_mean": is_mean,
        "oos_sharpe":     oos_s,
        "sharpe_retention_pct": (round(oos_s / is_mean * 100, 1)
                                 if is_mean and is_mean != 0 else None),
    }

    return SessionGapWFResult(
        oos_equity=oos_equity,
        fold_summaries=folds,
        oos_metrics=oos_metrics,
        chosen_params=chosen,
        degradation=degradation,
        min_trades_used=min_trades,
    )


# ============================================================================
# EXPERIMENTAL: ORB pullback / VWAP bands / news fade walk-forward
# ============================================================================
#
# Same shape as run_session_gap_walk_forward just above (raw OHLCV, no
# features, a `min_trades` bypass of the usual config.WALKFORWARD floor) --
# kept as separate functions rather than one generic helper to match this
# file's existing per-strategy-family convention (trend/reactive/structure
# above are likewise separate despite sharing structure).

@dataclass
class OrbPullbackWFResult:
    oos_equity: pd.Series
    fold_summaries: pd.DataFrame
    oos_metrics: dict
    chosen_params: list[dict]
    degradation: dict
    min_trades_used: int


def _orb_pullback_objective(metrics: dict, min_trades: int) -> float:
    if metrics["n_trades"] < min_trades:
        return -np.inf
    sharpe = metrics["sharpe"]
    pf     = metrics["profit_factor"] or 0.0
    dd     = abs(metrics["max_drawdown_pct"]) / 100.0
    return sharpe + 0.3 * min(pf, 3.0) - 2.0 * dd


def run_orb_pullback_walk_forward(target_df: pd.DataFrame,
                                  instrument_key: str,
                                  n_contracts: int = 1,
                                  wf: config.WalkForward | None = None,
                                  verbose: bool = True,
                                  min_trades: int = 5) -> OrbPullbackWFResult:
    wf  = wf or config.WALKFORWARD
    rng = np.random.default_rng(wf.random_seed)

    n = len(target_df)
    starts = range(0, n - wf.train_bars - wf.test_bars + 1, wf.step_bars)

    fold_rows, oos_curves, chosen, is_metric_list, oos_trades_all = [], [], [], [], []
    last_equity = config.TOPSTEP.starting_balance
    skipped = 0

    for fold_id, s in enumerate(starts):
        tr0, tr1 = s, s + wf.train_bars
        te0, te1 = tr1, min(tr1 + wf.test_bars, n)
        if te1 - te0 < 50:
            continue

        tdf_tr = target_df.iloc[tr0:tr1]
        tdf_te = target_df.iloc[te0:te1]
        split   = int(len(tdf_tr) * 0.70)
        tdf_fit = tdf_tr.iloc[:split]
        tdf_val = tdf_tr.iloc[split:]

        best_obj, best_params, best_val = -np.inf, None, None
        for _ in range(wf.n_param_samples):
            params = strat.sample_orb_pullback_params(rng)
            sig_val = strat.generate_orb_pullback_signal(
                pd.concat([tdf_fit.iloc[-200:], tdf_val]), params, instrument_key
            )[0].iloc[200:]
            if len(sig_val) == 0:
                continue
            vm  = bt.quick_metrics(tdf_val, sig_val, instrument_key, n_contracts)
            obj = _orb_pullback_objective(vm, min_trades)
            if obj > best_obj:
                best_obj, best_params, best_val = obj, params, vm

        if best_params is None:
            skipped += 1
            if verbose:
                print(f"  fold {fold_id:2d} | skipped -- no config cleared "
                      f"min_trades={min_trades} on validation")
            continue

        warmup = tdf_tr.iloc[-200:]
        sig_full = strat.generate_orb_pullback_signal(
            pd.concat([warmup, tdf_te]), best_params, instrument_key
        )[0].iloc[200:]

        safe_eq    = max(last_equity, 1.0)
        rules_fold = config.TopstepRules(**{**config.TOPSTEP.__dict__})
        rules_fold.starting_balance = safe_eq

        res_te = bt.run_backtest(tdf_te, sig_full, instrument_key,
                                 n_contracts, rules=rules_fold)

        end_eq = float(res_te.equity.iloc[-1])
        if not np.isfinite(end_eq) or end_eq <= 0:
            end_eq = safe_eq
        last_equity = end_eq

        oos_curves.append(res_te.equity.clip(lower=1.0).ffill())
        if len(res_te.trades):
            oos_trades_all.append(res_te.trades)
        chosen.append(best_params)
        is_metric_list.append(best_val)

        m = res_te.metrics
        pnl_str = f"${m['total_pnl_$']:+,.0f}" if np.isfinite(m['total_pnl_$']) else "$nan"
        fold_rows.append({
            "fold":            fold_id,
            "test_start":      target_df.index[te0],
            "test_end":        target_df.index[te1 - 1],
            "is_sharpe":       best_val["sharpe"],
            "oos_sharpe":      m["sharpe"],
            "oos_pnl_$":       m["total_pnl_$"],
            "oos_trades":      m["n_trades"],
            "oos_win_rate":    m["win_rate"],
            "sma_period":      best_params["sma_period"],
            "stop_atr_mult":   best_params["stop_atr_mult"],
        })
        if verbose:
            print(f"  fold {fold_id:2d} | IS {best_val['sharpe']:+.2f} "
                  f"-> OOS {m['sharpe']:+.2f} | {pnl_str} | {m['n_trades']} trades "
                  f"(min_trades={min_trades})")

    if not oos_curves:
        raise RuntimeError(
            f"No valid folds -- {skipped} skipped even with min_trades={min_trades}.\n"
            "Try a lower --min-trades, more --n-samples, or a longer data window."
        )
    if skipped:
        print(f"  ({skipped} fold(s) skipped even with the relaxed min_trades={min_trades})")

    oos_equity = pd.concat(oos_curves)
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")].sort_index().ffill()
    oos_rets   = oos_equity.diff().fillna(0.0)
    folds      = pd.DataFrame(fold_rows)
    oos_trades = (pd.concat(oos_trades_all, ignore_index=True)
                  if oos_trades_all else pd.DataFrame(columns=["pnl_$"]))

    oos_metrics = bt._compute_metrics(
        oos_equity, oos_rets, oos_trades, config.INTERVAL, oos_equity.iloc[0])
    oos_metrics["pct_folds_profitable"] = round(
        float((folds["oos_pnl_$"] > 0).mean()) * 100, 1) if len(folds) else 0.0
    oos_metrics["topstep"] = ts.evaluate(oos_equity, config.TOPSTEP)

    is_sharpes = [m["sharpe"] for m in is_metric_list]
    is_mean    = round(float(np.mean(is_sharpes)), 3) if is_sharpes else None
    oos_s      = oos_metrics["sharpe"]
    degradation = {
        "is_sharpe_mean": is_mean,
        "oos_sharpe":     oos_s,
        "sharpe_retention_pct": (round(oos_s / is_mean * 100, 1)
                                 if is_mean and is_mean != 0 else None),
    }

    return OrbPullbackWFResult(
        oos_equity=oos_equity,
        fold_summaries=folds,
        oos_metrics=oos_metrics,
        chosen_params=chosen,
        degradation=degradation,
        min_trades_used=min_trades,
    )


@dataclass
class VwapBandsWFResult:
    oos_equity: pd.Series
    fold_summaries: pd.DataFrame
    oos_metrics: dict
    chosen_params: list[dict]
    degradation: dict
    min_trades_used: int


def _vwap_bands_objective(metrics: dict, min_trades: int) -> float:
    if metrics["n_trades"] < min_trades:
        return -np.inf
    sharpe = metrics["sharpe"]
    pf     = metrics["profit_factor"] or 0.0
    dd     = abs(metrics["max_drawdown_pct"]) / 100.0
    return sharpe + 0.25 * min(pf, 3.0) - 2.0 * dd


def run_vwap_bands_walk_forward(target_df: pd.DataFrame,
                                instrument_key: str,
                                n_contracts: int = 1,
                                wf: config.WalkForward | None = None,
                                verbose: bool = True,
                                min_trades: int = 10) -> VwapBandsWFResult:
    wf  = wf or config.WALKFORWARD
    rng = np.random.default_rng(wf.random_seed)

    n = len(target_df)
    starts = range(0, n - wf.train_bars - wf.test_bars + 1, wf.step_bars)

    fold_rows, oos_curves, chosen, is_metric_list, oos_trades_all = [], [], [], [], []
    last_equity = config.TOPSTEP.starting_balance
    skipped = 0

    for fold_id, s in enumerate(starts):
        tr0, tr1 = s, s + wf.train_bars
        te0, te1 = tr1, min(tr1 + wf.test_bars, n)
        if te1 - te0 < 50:
            continue

        tdf_tr = target_df.iloc[tr0:tr1]
        tdf_te = target_df.iloc[te0:te1]
        split   = int(len(tdf_tr) * 0.70)
        tdf_fit = tdf_tr.iloc[:split]
        tdf_val = tdf_tr.iloc[split:]

        best_obj, best_params, best_val = -np.inf, None, None
        for _ in range(wf.n_param_samples):
            params = strat.sample_vwap_bands_params(rng)
            # VWAP is session-anchored (18:00 ET), so prepend full sessions of
            # warmup rather than a fixed bar count for a sane running VWAP/SD.
            sig_val = strat.generate_vwap_bands_signal(
                pd.concat([tdf_fit.iloc[-300:], tdf_val]), params, instrument_key
            )[0].iloc[300:]
            if len(sig_val) == 0:
                continue
            vm  = bt.quick_metrics(tdf_val, sig_val, instrument_key, n_contracts)
            obj = _vwap_bands_objective(vm, min_trades)
            if obj > best_obj:
                best_obj, best_params, best_val = obj, params, vm

        if best_params is None:
            skipped += 1
            if verbose:
                print(f"  fold {fold_id:2d} | skipped -- no config cleared "
                      f"min_trades={min_trades} on validation")
            continue

        warmup = tdf_tr.iloc[-300:]
        sig_full = strat.generate_vwap_bands_signal(
            pd.concat([warmup, tdf_te]), best_params, instrument_key
        )[0].iloc[300:]

        safe_eq    = max(last_equity, 1.0)
        rules_fold = config.TopstepRules(**{**config.TOPSTEP.__dict__})
        rules_fold.starting_balance = safe_eq

        res_te = bt.run_backtest(tdf_te, sig_full, instrument_key,
                                 n_contracts, rules=rules_fold)

        end_eq = float(res_te.equity.iloc[-1])
        if not np.isfinite(end_eq) or end_eq <= 0:
            end_eq = safe_eq
        last_equity = end_eq

        oos_curves.append(res_te.equity.clip(lower=1.0).ffill())
        if len(res_te.trades):
            oos_trades_all.append(res_te.trades)
        chosen.append(best_params)
        is_metric_list.append(best_val)

        m = res_te.metrics
        pnl_str = f"${m['total_pnl_$']:+,.0f}" if np.isfinite(m['total_pnl_$']) else "$nan"
        fold_rows.append({
            "fold":            fold_id,
            "test_start":      target_df.index[te0],
            "test_end":        target_df.index[te1 - 1],
            "is_sharpe":       best_val["sharpe"],
            "oos_sharpe":      m["sharpe"],
            "oos_pnl_$":       m["total_pnl_$"],
            "oos_trades":      m["n_trades"],
            "oos_win_rate":    m["win_rate"],
            "entry_sd":        best_params["entry_sd"],
            "rsi_window":      best_params["rsi_window"],
        })
        if verbose:
            print(f"  fold {fold_id:2d} | IS {best_val['sharpe']:+.2f} "
                  f"-> OOS {m['sharpe']:+.2f} | {pnl_str} | {m['n_trades']} trades "
                  f"(min_trades={min_trades})")

    if not oos_curves:
        raise RuntimeError(
            f"No valid folds -- {skipped} skipped even with min_trades={min_trades}.\n"
            "Try a lower --min-trades, more --n-samples, or a longer data window."
        )
    if skipped:
        print(f"  ({skipped} fold(s) skipped even with the relaxed min_trades={min_trades})")

    oos_equity = pd.concat(oos_curves)
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")].sort_index().ffill()
    oos_rets   = oos_equity.diff().fillna(0.0)
    folds      = pd.DataFrame(fold_rows)
    oos_trades = (pd.concat(oos_trades_all, ignore_index=True)
                  if oos_trades_all else pd.DataFrame(columns=["pnl_$"]))

    oos_metrics = bt._compute_metrics(
        oos_equity, oos_rets, oos_trades, config.INTERVAL, oos_equity.iloc[0])
    oos_metrics["pct_folds_profitable"] = round(
        float((folds["oos_pnl_$"] > 0).mean()) * 100, 1) if len(folds) else 0.0
    oos_metrics["topstep"] = ts.evaluate(oos_equity, config.TOPSTEP)

    is_sharpes = [m["sharpe"] for m in is_metric_list]
    is_mean    = round(float(np.mean(is_sharpes)), 3) if is_sharpes else None
    oos_s      = oos_metrics["sharpe"]
    degradation = {
        "is_sharpe_mean": is_mean,
        "oos_sharpe":     oos_s,
        "sharpe_retention_pct": (round(oos_s / is_mean * 100, 1)
                                 if is_mean and is_mean != 0 else None),
    }

    return VwapBandsWFResult(
        oos_equity=oos_equity,
        fold_summaries=folds,
        oos_metrics=oos_metrics,
        chosen_params=chosen,
        degradation=degradation,
        min_trades_used=min_trades,
    )


@dataclass
class NewsFadeWFResult:
    oos_equity: pd.Series
    fold_summaries: pd.DataFrame
    oos_metrics: dict
    chosen_params: list[dict]
    degradation: dict
    min_trades_used: int


def _news_fade_objective(metrics: dict, min_trades: int) -> float:
    if metrics["n_trades"] < min_trades:
        return -np.inf
    sharpe = metrics["sharpe"]
    pf     = metrics["profit_factor"] or 0.0
    dd     = abs(metrics["max_drawdown_pct"]) / 100.0
    return sharpe + 0.3 * min(pf, 3.0) - 2.0 * dd


def run_news_fade_walk_forward(target_df: pd.DataFrame,
                               instrument_key: str,
                               n_contracts: int = 1,
                               wf: config.WalkForward | None = None,
                               verbose: bool = True,
                               min_trades: int = 2) -> NewsFadeWFResult:
    wf  = wf or config.WALKFORWARD
    rng = np.random.default_rng(wf.random_seed)

    n = len(target_df)
    starts = range(0, n - wf.train_bars - wf.test_bars + 1, wf.step_bars)

    fold_rows, oos_curves, chosen, is_metric_list, oos_trades_all = [], [], [], [], []
    last_equity = config.TOPSTEP.starting_balance
    skipped = 0

    # Needs enough trailing daily bars for daily_atr_window (default 14 days)
    # to be warm before we trust the spike-detection threshold.
    warm = 20 * 78

    for fold_id, s in enumerate(starts):
        tr0, tr1 = s, s + wf.train_bars
        te0, te1 = tr1, min(tr1 + wf.test_bars, n)
        if te1 - te0 < 50:
            continue

        tdf_tr = target_df.iloc[tr0:tr1]
        tdf_te = target_df.iloc[te0:te1]
        split   = int(len(tdf_tr) * 0.70)
        tdf_fit = tdf_tr.iloc[:split]
        tdf_val = tdf_tr.iloc[split:]

        best_obj, best_params, best_val = -np.inf, None, None
        for _ in range(wf.n_param_samples):
            params = strat.sample_news_fade_params(rng)
            warmup_ctx = tdf_fit.iloc[-warm:] if len(tdf_fit) > warm else tdf_fit
            sig_val = strat.generate_news_fade_signal(
                pd.concat([warmup_ctx, tdf_val]), params, instrument_key
            )[0].iloc[len(warmup_ctx):]
            if len(sig_val) == 0:
                continue
            vm  = bt.quick_metrics(tdf_val, sig_val, instrument_key, n_contracts)
            obj = _news_fade_objective(vm, min_trades)
            if obj > best_obj:
                best_obj, best_params, best_val = obj, params, vm

        if best_params is None:
            skipped += 1
            if verbose:
                print(f"  fold {fold_id:2d} | skipped -- no config cleared "
                      f"min_trades={min_trades} on validation")
            continue

        warmup_ctx = tdf_tr.iloc[-warm:] if len(tdf_tr) > warm else tdf_tr
        sig_full = strat.generate_news_fade_signal(
            pd.concat([warmup_ctx, tdf_te]), best_params, instrument_key
        )[0].iloc[len(warmup_ctx):]

        safe_eq    = max(last_equity, 1.0)
        rules_fold = config.TopstepRules(**{**config.TOPSTEP.__dict__})
        rules_fold.starting_balance = safe_eq

        res_te = bt.run_backtest(tdf_te, sig_full, instrument_key,
                                 n_contracts, rules=rules_fold)

        end_eq = float(res_te.equity.iloc[-1])
        if not np.isfinite(end_eq) or end_eq <= 0:
            end_eq = safe_eq
        last_equity = end_eq

        oos_curves.append(res_te.equity.clip(lower=1.0).ffill())
        if len(res_te.trades):
            oos_trades_all.append(res_te.trades)
        chosen.append(best_params)
        is_metric_list.append(best_val)

        m = res_te.metrics
        pnl_str = f"${m['total_pnl_$']:+,.0f}" if np.isfinite(m['total_pnl_$']) else "$nan"
        fold_rows.append({
            "fold":            fold_id,
            "test_start":      target_df.index[te0],
            "test_end":        target_df.index[te1 - 1],
            "is_sharpe":       best_val["sharpe"],
            "oos_sharpe":      m["sharpe"],
            "oos_pnl_$":       m["total_pnl_$"],
            "oos_trades":      m["n_trades"],
            "oos_win_rate":    m["win_rate"],
            "spike_atr_mult":  best_params["spike_atr_mult"],
            "fib_level":       best_params["fib_level"],
        })
        if verbose:
            print(f"  fold {fold_id:2d} | IS {best_val['sharpe']:+.2f} "
                  f"-> OOS {m['sharpe']:+.2f} | {pnl_str} | {m['n_trades']} trades "
                  f"(min_trades={min_trades})")

    if not oos_curves:
        raise RuntimeError(
            f"No valid folds -- {skipped} skipped even with min_trades={min_trades}.\n"
            "Try a lower --min-trades, more --n-samples, or a longer data window."
        )
    if skipped:
        print(f"  ({skipped} fold(s) skipped even with the relaxed min_trades={min_trades})")

    oos_equity = pd.concat(oos_curves)
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")].sort_index().ffill()
    oos_rets   = oos_equity.diff().fillna(0.0)
    folds      = pd.DataFrame(fold_rows)
    oos_trades = (pd.concat(oos_trades_all, ignore_index=True)
                  if oos_trades_all else pd.DataFrame(columns=["pnl_$"]))

    oos_metrics = bt._compute_metrics(
        oos_equity, oos_rets, oos_trades, config.INTERVAL, oos_equity.iloc[0])
    oos_metrics["pct_folds_profitable"] = round(
        float((folds["oos_pnl_$"] > 0).mean()) * 100, 1) if len(folds) else 0.0
    oos_metrics["topstep"] = ts.evaluate(oos_equity, config.TOPSTEP)

    is_sharpes = [m["sharpe"] for m in is_metric_list]
    is_mean    = round(float(np.mean(is_sharpes)), 3) if is_sharpes else None
    oos_s      = oos_metrics["sharpe"]
    degradation = {
        "is_sharpe_mean": is_mean,
        "oos_sharpe":     oos_s,
        "sharpe_retention_pct": (round(oos_s / is_mean * 100, 1)
                                 if is_mean and is_mean != 0 else None),
    }

    return NewsFadeWFResult(
        oos_equity=oos_equity,
        fold_summaries=folds,
        oos_metrics=oos_metrics,
        chosen_params=chosen,
        degradation=degradation,
        min_trades_used=min_trades,
    )
