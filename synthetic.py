"""
Synthetic market data generator.

Used to validate the ENTIRE pipeline offline, without spending EODHD calls and
without network. It produces correlated 5-min bars for NQ/ES/GC/VIX/DXY/BONDS
with realistic-ish properties:

  * intraday volatility seasonality (U-shape)
  * cross-asset correlation (ES & NQ move together; gold ~ -DXY; VIX ~ -equities)
  * a SMALL, GENUINE predictive signal: lagged ES return weakly leads NQ.

That last point matters: it lets you confirm the discovery + walk-forward layers
can actually find a real edge when one exists -- and, just as important, that
they report the edge as *modest*, not fantastical. If the pipeline ever reports
a huge Sharpe on this data, something is leaking.

This is for plumbing validation ONLY. Real conclusions require real data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config


def _intraday_vol_curve(index: pd.DatetimeIndex) -> np.ndarray:
    """U-shaped vol multiplier across the RTH session."""
    et = index.tz_convert("America/New_York")
    minutes = et.hour * 60 + et.minute
    open_m, close_m = 9 * 60 + 30, 16 * 60
    frac = np.clip((minutes - open_m) / (close_m - open_m), 0, 1)
    # high at open & close, low midday
    return 0.6 + 0.9 * (np.cos(frac * np.pi * 2) * 0.5 + 0.5)


def generate(history_days: int = 400, interval_min: int = 5, seed: int = 42
             ) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # Build an RTH-only 5-min index over business days.
    end = pd.Timestamp.now(tz="UTC").normalize()
    days = pd.bdate_range(end - pd.Timedelta(days=history_days), end, tz="UTC")
    stamps = []
    for d in days:
        et_open = pd.Timestamp(f"{d.date()} 09:30", tz="America/New_York")
        et_close = pd.Timestamp(f"{d.date()} 16:00", tz="America/New_York")
        rng_idx = pd.date_range(et_open, et_close, freq=f"{interval_min}min")
        stamps.append(rng_idx.tz_convert("UTC"))
    index = pd.DatetimeIndex(np.concatenate([s.values for s in stamps])).tz_localize("UTC")
    n = len(index)

    volm = _intraday_vol_curve(index)

    # --- common market factor (drives ES & NQ together) ---
    base_vol = 0.0006
    market = rng.standard_normal(n) * base_vol * volm

    # gold has its own factor + small inverse link to a "dollar" factor
    dollar = rng.standard_normal(n) * base_vol * 0.7 * volm
    gold_idio = rng.standard_normal(n) * base_vol * 0.9 * volm

    # --- ES returns ---
    es_ret = market + rng.standard_normal(n) * base_vol * 0.4 * volm

    # --- NQ returns: correlated to ES + a SMALL genuine lead from lagged ES ---
    nq_idio = rng.standard_normal(n) * base_vol * 0.6 * volm
    nq_ret = 1.05 * market + nq_idio
    # the planted edge: yesterday's-bar ES return weakly predicts this bar's NQ
    lead = np.zeros(n)
    lead[1:] = es_ret[:-1]
    nq_ret = nq_ret + 0.18 * lead   # small but real

    # --- Gold ---
    gc_ret = gold_idio - 0.5 * dollar

    # --- VIX proxy: inverse to equities, mean-reverting level ---
    vix_ret = -3.0 * market + rng.standard_normal(n) * base_vol * 2.0
    # --- DXY proxy ---
    dxy_ret = dollar + rng.standard_normal(n) * base_vol * 0.3
    # --- Bonds proxy: mildly inverse to equities ---
    bond_ret = -0.4 * market + rng.standard_normal(n) * base_vol * 0.5

    def to_ohlcv(ret: np.ndarray, p0: float, vol_scale: float) -> pd.DataFrame:
        price = p0 * np.exp(np.cumsum(ret))
        # build OHLC around the close path
        close = price
        open_ = np.empty_like(close); open_[0] = p0; open_[1:] = close[:-1]
        wig = np.abs(rng.standard_normal(n)) * base_vol * volm * close
        high = np.maximum(open_, close) + wig
        low = np.minimum(open_, close) - wig
        vol = (rng.gamma(2.0, 1.0, n) * vol_scale * volm).round()
        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
            index=index,
        )

    return {
        "NQ":    to_ohlcv(nq_ret, 18000.0, 2000),
        "ES":    to_ohlcv(es_ret, 5200.0, 5000),
        "GC":    to_ohlcv(gc_ret, 2350.0, 800),
        "VIX":   to_ohlcv(vix_ret, 15.0, 300),
        "DXY":   to_ohlcv(dxy_ret, 28.0, 400),
        "BONDS": to_ohlcv(bond_ret, 92.0, 600),
    }
