"""
Topstep rule simulation.

Given an equity curve (account $ over time) plus the timestamps, this checks
whether the run would have survived a Topstep account:

  * trailing maximum drawdown (the big one -- kills the account if equity falls
    `trailing_drawdown` below its running peak)
  * daily loss limit
  * profit target (for passing a combine)
  * consistency rule (no single day too large a share of total profit)

The backtester calls `evaluate()` after producing an equity curve. It also
exposes `trailing_stop_breached()` so the engine can stop trading the instant
the account would be dead, rather than continuing to trade a blown account.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

import config


@dataclass
class TopstepResult:
    passed: bool
    failed_rules: list[str]
    hit_profit_target: bool
    max_trailing_drawdown: float
    worst_day: float
    best_day: float
    consistency_ratio: float          # best day / total profit
    killed_at: pd.Timestamp | None    # when the account would have died


def evaluate(equity: pd.Series, rules: config.TopstepRules | None = None
             ) -> TopstepResult:
    """`equity` = account balance over time (index = bar timestamps)."""
    rules = rules or config.TOPSTEP
    failed: list[str] = []
    killed_at = None

    start = rules.starting_balance
    eq = equity.copy()

    # --- trailing drawdown (intraday peak) ---
    # Topstep's actual rule: the floor trails the peak only until the account
    # reaches start + trailing_drawdown.  At that point the floor locks at
    # starting_balance permanently and never moves higher.
    # Example (50K account, $2K DD):
    #   equity $49K -> liq $47K (trailing)
    #   equity $51K -> liq $49K (trailing, but capped)
    #   equity $52K -> liq $50K (locked — floor never exceeds starting balance)
    #   equity $55K -> liq $50K (still locked)
    trail_stops_at  = start + rules.trailing_drawdown
    true_peak       = eq.cummax()                            # for max-DD reporting
    effective_peak  = true_peak.clip(upper=trail_stops_at)  # capped for liq calc
    liq_level       = (effective_peak - rules.trailing_drawdown).clip(lower=start - rules.trailing_drawdown)
    breach = eq <= liq_level
    max_tdd = float((true_peak - eq).max())
    if breach.any():
        killed_at = eq.index[breach.argmax()]
        failed.append(f"trailing_drawdown breached at {killed_at} "
                      f"(max trailing DD ${max_tdd:,.0f} >= ${rules.trailing_drawdown:,.0f})")

    # --- per-day stats ---
    et = eq.index.tz_convert(config.SESSION.timezone)
    day = pd.Series(et.date, index=eq.index)
    daily_pnl = eq.groupby(day).last() - eq.groupby(day).first()
    worst_day = float(daily_pnl.min()) if len(daily_pnl) else 0.0
    best_day = float(daily_pnl.max()) if len(daily_pnl) else 0.0

    if rules.daily_loss_limit is not None and worst_day <= -rules.daily_loss_limit:
        failed.append(f"daily_loss_limit breached (worst day ${worst_day:,.0f})")

    total_profit = float(eq.iloc[-1] - start)
    hit_target = (rules.profit_target is not None) and (total_profit >= rules.profit_target)

    consistency_ratio = (best_day / total_profit) if total_profit > 0 else np.inf
    if (rules.consistency_max_day_fraction is not None and total_profit > 0
            and consistency_ratio > rules.consistency_max_day_fraction):
        failed.append(f"consistency rule: best day is {consistency_ratio:.0%} of "
                      f"total profit (> {rules.consistency_max_day_fraction:.0%})")

    passed = (len(failed) == 0)
    return TopstepResult(
        passed=passed,
        failed_rules=failed,
        hit_profit_target=hit_target,
        max_trailing_drawdown=max_tdd,
        worst_day=worst_day,
        best_day=best_day,
        consistency_ratio=float(consistency_ratio) if np.isfinite(consistency_ratio) else float("nan"),
        killed_at=killed_at,
    )
