"""
Backtesting engine.

Bar-by-bar, long/short/flat. Key correctness properties:

  * NO LOOK-AHEAD: the signal for bar t is acted on at bar t's close -> the
    position is held INTO bar t+1, and PnL is realised on the t->t+1 move.
    (Signals themselves are produced from features known at/of bar t.)
  * Realistic costs: round-trip commission + slippage in ticks on every change
    of position size.
  * Topstep-aware: stops trading the instant the account would be liquidated.
  * Session flattening: closes positions before the bell (no overnight).

Returns a BacktestResult with the equity curve, trade list, and a full metrics
dict. This same engine is used both for single runs and inside walk-forward.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

import config
import topstep as ts


@dataclass
class BacktestResult:
    equity: pd.Series                 # account $ over time
    returns: pd.Series                # per-bar account returns
    trades: pd.DataFrame
    metrics: dict
    topstep: ts.TopstepResult
    n_contracts: int


ANNUALISATION = {  # bars per year for Sharpe scaling
    "1m": 252 * 390,
    "5m": 252 * 78,
    "1h": 252 * 7,
}


def _flatten_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True where we must be flat (after the flatten-before-close cutoff)."""
    et = index.tz_convert(config.SESSION.timezone)
    minutes = np.asarray(et.hour) * 60 + np.asarray(et.minute)
    close_m = 16 * 60
    cutoff = close_m - config.SESSION.flatten_before_close_min
    return minutes >= cutoff


def choose_dynamic_units(sizing: "config.DynamicSizing", rules: config.TopstepRules,
                         micro_point_value: float, cushion: float,
                         entry_px: float, stop_px: float,
                         remaining_daily: float | None = None) -> int:
    """
    Largest ladder rung (see config.DynamicSizing.ladder_units) whose worst-case
    stop-out fits the TIGHTEST of:
      - safety_fraction * cushion        (anchor: profit or room-to-floor)
      - max_trade_risk_frac * trailing_drawdown  (absolute per-trade cap)
      - daily_risk_frac * remaining_daily         (fraction of the day's loss budget)

    Shared by the backtest engine (run_backtest) and live_trader.py so paper
    and live sizing decisions are identical given the same inputs.
    """
    if not np.isfinite(stop_px):
        return sizing.min_units
    d_pts = abs(entry_px - stop_px)
    per_unit_risk = micro_point_value * d_pts
    if per_unit_risk <= 0:
        return sizing.min_units
    budget = sizing.safety_fraction * max(cushion, 0.0)
    if sizing.max_trade_risk_frac is not None:
        budget = min(budget, sizing.max_trade_risk_frac * rules.trailing_drawdown)
    if remaining_daily is not None and sizing.daily_risk_frac is not None:
        budget = min(budget, sizing.daily_risk_frac * max(remaining_daily, 0.0))
    max_u  = budget / per_unit_risk
    chosen = sizing.min_units
    for rung in sizing.ladder_units:
        if rung <= max_u:
            chosen = rung
    return chosen


def run_backtest(target_df: pd.DataFrame, signal: pd.Series, instrument_key: str,
                 n_contracts: int = 1, rules: config.TopstepRules | None = None,
                 interval: str | None = None,
                 stops: pd.Series | None = None,
                 sizing: "config.DynamicSizing | None" = None) -> BacktestResult:
    """
    target_df : OHLCV of the traded instrument (aligned to signal.index)
    signal    : desired position in {-1, 0, +1} per bar (the model's call at bar t)
    stops     : per-bar stop price (used only for risk-based dynamic sizing)
    sizing    : DynamicSizing policy. When enabled (and stops given), position
                size at each entry scales micro->mini with the cushion above the
                trailing-DD floor; otherwise the fixed n_contracts is used.

    Internally the position is tracked in MICRO-EQUIVALENT UNITS (1 mini = 10
    micros). For the fixed-size path a "unit" is just a contract of the base
    instrument, so behaviour is unchanged.
    """
    rules = rules or config.TOPSTEP
    interval = interval or config.INTERVAL
    inst = config.INSTRUMENTS[instrument_key]

    df = target_df.loc[signal.index]
    close = df["close"].values
    n = len(close)

    sig = signal.fillna(0).clip(-1, 1).values.astype(float)
    flat = _flatten_mask(df.index)
    sig[flat] = 0.0
    sig_dir = np.sign(sig)   # desired DIRECTION; size decided below

    dynamic = bool(sizing and sizing.enabled and stops is not None)

    if dynamic:
        micro     = inst                                  # config.INSTRUMENTS = micro specs
        mini      = config.DATABENTO_INSTRUMENTS.get(instrument_key, micro)
        micro_pv  = micro.point_value
        pv_unit   = micro_pv                              # $ per point per micro-unit
        stop_arr  = (stops.reindex(df.index).values
                     if stops is not None else np.full(n, np.nan))
        micro_leg = micro.commission_rt + micro.tick_size * micro.slippage_ticks * micro_pv
        mini_leg  = mini.commission_rt + mini.tick_size * mini.slippage_ticks * mini.point_value
        daily_lim  = rules.daily_loss_limit

        def _leg_cost(units: float) -> float:
            u = int(abs(round(units)))
            if u == 0:
                return 0.0
            if u <= 9:                                    # micros
                return u * micro_leg
            return (u // 10) * mini_leg                   # minis (10 or 20 units)
    else:
        pv_unit   = inst.point_value
        fixed_leg = inst.commission_rt + inst.tick_size * inst.slippage_ticks * pv_unit
        fixed_u   = float(min(n_contracts, rules.max_contracts))

        def _leg_cost(units: float) -> float:
            return abs(units) * fixed_leg

    def _size_label(units: float) -> str:
        u = int(abs(round(units)))
        if not dynamic:
            return f"{u}x"
        return f"{u} micro" if u <= 9 else f"{u // 10} mini"

    equity = np.empty(n)
    equity[0] = rules.starting_balance
    position = 0.0          # signed micro-equivalent units
    trade_log = []
    entry_price = None
    entry_time = None
    entry_units = 0.0
    killed = False

    running_peak   = rules.starting_balance
    trail_stops_at = rules.starting_balance + rules.trailing_drawdown
    liq_floor      = rules.starting_balance - rules.trailing_drawdown
    dd             = rules.trailing_drawdown

    soft_stop    = (0.8 * rules.daily_loss_limit
                    if rules.daily_loss_limit is not None else None)
    et_dates     = np.asarray(df.index.tz_convert(config.SESSION.timezone).date)
    day_start_eq = equity[0]
    current_day  = et_dates[0]
    day_halted   = False

    def _choose_units(cushion: float, entry_px: float, stop_px: float,
                      remaining_daily: float | None) -> int:
        return choose_dynamic_units(sizing, rules, micro_pv, cushion,
                                    entry_px, stop_px, remaining_daily)

    for i in range(1, n):
        prev_pos = position
        d_dir = sig_dir[i - 1]   # decided at bar i-1, held into bar i

        if et_dates[i] != current_day:
            current_day  = et_dates[i]
            day_start_eq = equity[i - 1]
            day_halted   = False

        if killed or day_halted:
            d_dir = 0.0

        # ---- resolve desired position size (units) ----
        if d_dir == 0.0:
            desired = 0.0
        elif np.sign(prev_pos) == d_dir and prev_pos != 0:
            desired = prev_pos                       # hold existing size
        elif dynamic:
            if sizing.anchor == "profit":
                cushion = max(equity[i - 1] - rules.starting_balance, 0.0)
            else:                                   # "floor": room before liquidation
                liq_now = max(running_peak - dd, liq_floor)
                cushion = equity[i - 1] - liq_now
            # remaining daily-loss budget at this entry (None if no daily limit)
            if daily_lim is not None:
                day_loss = max(0.0, day_start_eq - equity[i - 1])
                remaining_daily = daily_lim - day_loss
            else:
                remaining_daily = None
            desired = d_dir * _choose_units(cushion, close[i - 1],
                                            stop_arr[i - 1], remaining_daily)
        else:
            desired = d_dir * fixed_u

        # ---- cost + trade logging on any change ----
        change_cost = 0.0
        if desired != prev_pos:
            # close the old leg (exit / flip), open the new leg (entry / flip)
            if prev_pos != 0 and (np.sign(desired) != np.sign(prev_pos) or desired == 0):
                change_cost += _leg_cost(prev_pos)
                pnl_pts = (close[i - 1] - entry_price) * np.sign(prev_pos)
                trade_log.append({
                    "entry_time": entry_time, "exit_time": df.index[i - 1],
                    "side": "long" if prev_pos > 0 else "short",
                    "contracts": abs(prev_pos),
                    "size": _size_label(prev_pos),
                    "entry": entry_price, "exit": close[i - 1],
                    "pnl_points": pnl_pts,
                    "pnl_$": pnl_pts * pv_unit * abs(prev_pos),
                })
            if desired != 0 and (prev_pos == 0 or np.sign(desired) != np.sign(prev_pos)):
                change_cost += _leg_cost(desired)
                entry_price = close[i - 1]
                entry_time = df.index[i - 1]
                entry_units = desired

        position = desired

        move = close[i] - close[i - 1]
        gross = position * move * pv_unit
        equity[i] = equity[i - 1] + gross - change_cost

        # Topstep trailing-DD kill check (intraday).
        running_peak = max(running_peak, min(equity[i], trail_stops_at))
        liq = max(running_peak - dd, liq_floor)
        if not killed and equity[i] <= liq:
            killed = True
            position = 0.0

        if soft_stop is not None and not killed and not day_halted:
            if equity[i] - day_start_eq <= -soft_stop:
                day_halted = True

    eq = pd.Series(equity, index=df.index)
    rets = eq.diff().fillna(0.0)
    trades = pd.DataFrame(trade_log)

    metrics = _compute_metrics(eq, rets, trades, interval, rules.starting_balance)
    topstep_res = ts.evaluate(eq, rules)
    metrics["topstep_passed"] = topstep_res.passed
    metrics["topstep_failed_rules"] = topstep_res.failed_rules

    return BacktestResult(eq, rets, trades, metrics, topstep_res, int(n_contracts))


def quick_metrics(target_df: pd.DataFrame, signal: pd.Series, instrument_key: str,
                  n_contracts: int = 1, interval: str | None = None) -> dict:
    """
    Fast, VECTORISED approximation used only for parameter SELECTION inside the
    walk-forward search (no Topstep kill logic, no per-trade log). ~100x faster
    than the event-driven engine. The final out-of-sample evaluation always uses
    the full run_backtest().
    """
    interval = interval or config.INTERVAL
    inst = config.INSTRUMENTS[instrument_key]
    df = target_df.loc[signal.index]
    close = df["close"].values
    sig = signal.fillna(0).clip(-1, 1).values.astype(float)
    flat = _flatten_mask(df.index)
    sig[flat] = 0.0
    pos = sig * min(n_contracts, config.TOPSTEP.max_contracts)

    held = np.empty_like(pos); held[0] = 0.0; held[1:] = pos[:-1]   # acted next bar
    move = np.empty_like(close); move[0] = 0.0; move[1:] = np.diff(close)
    gross = held * move * inst.point_value
    dpos = np.empty_like(pos); dpos[0] = abs(pos[0]); dpos[1:] = np.abs(np.diff(pos))
    cost = dpos * (inst.commission_rt + inst.tick_size * inst.slippage_ticks * inst.point_value)
    net = gross - cost

    start = config.TOPSTEP.starting_balance
    eq = start + np.cumsum(net)
    ann = ANNUALISATION.get(interval, 252 * 78)
    r = net / start
    sd = r.std()
    sharpe = float(r.mean() / sd * np.sqrt(ann)) if sd > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd_pct = float(((eq - peak) / peak).min()) * 100
    pos_pnl = net[net > 0].sum(); neg_pnl = -net[net < 0].sum()
    pf = float(pos_pnl / neg_pnl) if neg_pnl > 0 else np.inf
    n_trades = int((dpos > 0).sum())
    return {"sharpe": round(sharpe, 3), "max_drawdown_pct": round(dd_pct, 2),
            "profit_factor": round(pf, 3) if np.isfinite(pf) else None,
            "n_trades": n_trades, "total_pnl_$": round(float(eq[-1] - start), 2)}


def _compute_metrics(eq: pd.Series, rets: pd.Series, trades: pd.DataFrame,
                     interval: str, start: float) -> dict:
    total_pnl = float(eq.iloc[-1] - start)
    ann = ANNUALISATION.get(interval, 252 * 78)

    # account-return Sharpe/Sortino (per-bar $ returns scaled to account)
    r = rets / start
    mu, sd = r.mean(), r.std()
    sharpe = float(mu / sd * np.sqrt(ann)) if sd > 0 else 0.0
    downside = r[r < 0].std()
    sortino = float(mu / downside * np.sqrt(ann)) if downside and downside > 0 else 0.0

    # drawdown
    peak = eq.cummax()
    dd = eq - peak
    max_dd = float(dd.min())
    max_dd_pct = float((dd / peak).min())

    # trade stats
    if len(trades):
        wins = trades[trades["pnl_$"] > 0]["pnl_$"]
        losses = trades[trades["pnl_$"] < 0]["pnl_$"]
        gross_win = wins.sum()
        gross_loss = -losses.sum()
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else np.inf
        win_rate = float((trades["pnl_$"] > 0).mean())
        avg_trade = float(trades["pnl_$"].mean())
        n_trades = int(len(trades))
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
    else:
        profit_factor = win_rate = avg_trade = avg_win = avg_loss = 0.0
        n_trades = 0

    return {
        "total_pnl_$": round(total_pnl, 2),
        "return_pct": round(total_pnl / start * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_$": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct * 100, 2),
        "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else None,
        "win_rate": round(win_rate * 100, 2),
        "n_trades": n_trades,
        "avg_trade_$": round(avg_trade, 2),
        "avg_win_$": round(avg_win, 2),
        "avg_loss_$": round(avg_loss, 2),
        "final_equity_$": round(float(eq.iloc[-1]), 2),
    }
