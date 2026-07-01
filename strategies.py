"""
Strategy signal generators -- everything that turns bars/features into a
position signal in {-1, 0, +1}. Four independent families live here, one per
section:

  * ML          (fit_strategy / FittedStrategy) -- standardised logistic
    regression or small random forest, fit ONLY on in-sample data (the
    walk-forward layer guarantees this). Fully described by a PARAMS dict,
    which is what the optimizer searches over.

  * TREND       (generate_trend_signal) -- volatility-normalized breakout +
    volume confirmation + trend filter. Naturally regime-adaptive: rides
    trending markets, gets stopped out quickly in chop. No model to overfit;
    only the rule parameters are optimized.

  * REACTIVE    (generate_reactive_signal) -- order-flow momentum. No ML, no
    forward labels, no prediction. Each bar is scored -5..+5 across five
    independent order-flow signals; a position is taken when enough signals
    agree. Requires real Databento columns (delta, cum_delta, buy_vol,
    sell_vol, large_buy, large_sell, volume) -- fails loudly on EODHD data.

  * STRUCTURE   (generate_structure_signal) -- multi-timeframe market
    structure (daily EMA trend bias, 5-min 3-candle Fair Value Gaps, swing
    level / liquidity-grab confirmation). Price-structure based; intended for
    real NQ futures bars (Databento) where the detected zones sit at the
    prices real participants react to.

  * SESSION GAP (generate_session_gap_signal) -- EXPERIMENTAL. London-range
    sweep-and-fail reversal, entered only in a narrow NY-open window (default
    08:30-09:00 ET). Fires at most once per calendar day, so a normal
    walk-forward fold will rarely clear the usual min_trades_in_sample floor
    -- see walkforward.run_session_gap_walk_forward and `python run.py
    session-gap --min-trades`.

  * ORB PULLBACK (generate_orb_pullback_signal) -- EXPERIMENTAL. 15-minute
    opening-range breakout (08:30-08:45 ET), confirmed by a 5-minute close
    beyond the range with an aligned 20-period SMA slope, entered on a
    pullback BACK to the broken edge (not the initial break). Fires at most
    once per calendar day.

  * VWAP BANDS (generate_vwap_bands_signal) -- EXPERIMENTAL. Mean-reversion
    off an anchored intraday VWAP (session anchor 18:00 ET) and its
    volume-weighted standard-deviation bands: fade a 2.5-sigma touch back
    toward 1-sigma, gated by an RSI extreme, hard-stopped past 3-sigma. Fires
    whenever price reaches the outer band, so trades far more often than the
    other experimental strategies.

  * NEWS FADE (generate_news_fade_signal) -- EXPERIMENTAL. Fades an abnormally
    large 08:30 ET candle (the NFP/CPI release time) back toward its 61.8%
    Fibonacci retracement. Uses candle-size-vs-daily-ATR as a proxy for "a
    market-moving release happened" rather than an economic calendar (this
    repo has no calendar data source) -- see the docstring above the function
    for how that adapts the original 1-minute/3x-ATR spec to 5-minute bars.
    Fires only on days with an outsized 08:30 candle, so it is the sparsest
    of the four.

All are causal (zero look-ahead): every indicator only uses past/current bar
information.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import pandas as pd

import config

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# ============================================================================
# ML strategy
# ============================================================================
#
# Design goals:
#   * fit ONLY on in-sample data (the walk-forward layer guarantees this)
#   * standardisation and feature selection happen inside fit() on train data
#     -> no information from the test set leaks in
#   * a confidence threshold controls how often we trade (high threshold =
#     fewer, higher-conviction trades = lower turnover/costs)
#
# Two model backends:
#   * "logit"  : standardised logistic regression (fast, robust, interpretable)
#   * "forest" : small random forest (captures nonlinearity; needs sklearn)

ML_DEFAULT_PARAMS = {
    "model": "logit",        # "logit" | "forest"
    "features": None,        # list[str]; None -> use all provided
    "threshold": 0.10,       # trade only if |P(up)-0.5| > threshold
    "max_features": 12,      # cap the feature count (regularisation)
    "rf_depth": 4,           # forest depth (if model == forest)
    "C": 1.0,                # logit regularisation strength
}


@dataclass
class FittedStrategy:
    params: dict
    model: object
    scaler: object
    cols: list[str]

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        cols = [c for c in self.cols if c in X.columns]
        Xc = X[cols].replace([np.inf, -np.inf], np.nan)
        idx = Xc.index
        Xf = Xc.fillna(0.0).values
        Xs = self.scaler.transform(Xf)
        proba_up = self.model.predict_proba(Xs)[:, 1]
        thr = self.params["threshold"]
        sig = np.where(proba_up > 0.5 + thr, 1,
                       np.where(proba_up < 0.5 - thr, -1, 0))
        # do not take a position where inputs were missing
        valid = Xc.notna().all(axis=1).values
        sig = np.where(valid, sig, 0)
        return pd.Series(sig, index=idx, dtype=float)


def fit_strategy(X_train: pd.DataFrame, y_train: pd.Series, params: dict | None = None
                 ) -> FittedStrategy | None:
    """
    y_train: binary direction in {-1, +1} (0s should be dropped by the caller, or
    are treated as the negative class). Returns None if training isn't possible.
    """
    if not _HAS_SK:
        raise RuntimeError("scikit-learn is required for the strategy layer.")
    p = {**ML_DEFAULT_PARAMS, **(params or {})}

    cols = p["features"] or list(X_train.columns)
    cols = cols[: p["max_features"]]
    X = X_train[cols].replace([np.inf, -np.inf], np.nan)

    # align & clean
    mask = X.notna().all(axis=1) & y_train.notna() & (y_train != 0)
    Xc = X[mask]
    yc = (y_train[mask] > 0).astype(int)
    if len(Xc) < 100 or yc.nunique() < 2:
        return None

    scaler = StandardScaler().fit(Xc.values)
    Xs = scaler.transform(Xc.values)

    if p["model"] == "forest":
        model = RandomForestClassifier(
            n_estimators=150, max_depth=p["rf_depth"], min_samples_leaf=50,
            n_jobs=-1, random_state=0, class_weight="balanced")
    else:
        model = LogisticRegression(
            C=p["C"], max_iter=500, class_weight="balanced")
    model.fit(Xs, yc.values)
    return FittedStrategy(params=p, model=model, scaler=scaler, cols=cols)


# ============================================================================
# Trend-following strategy
# ============================================================================
#
# Instead of predicting direction with a classifier, this identifies confirmed
# momentum and trades WITH it. Key difference from the ML approach:
#
#   ML approach:   predict where price will go -> trade that direction
#   Trend approach: wait for price to show where it's going -> follow it
#
# Signal logic (per bar):
#   1. Volatility-normalized breakout: price crosses N-bar high/low by > k*ATR
#   2. Volume confirmation: volume above M-bar average (participation check)
#   3. Trend filter: price above/below longer EMA (avoid counter-trend entries)
#   4. Time filter: no entries in last 30min (avoid close chop)
#
# Position management:
#   - Enter on breakout confirmation
#   - Exit on opposite breakout OR ATR trailing stop OR time stop (horizon bars)
#   - Never hold overnight (session flatten enforced by backtest engine)
#
# Parameters searched by walk-forward optimizer:
#   - breakout_bars: lookback for high/low channel (10-50)
#   - atr_mult: how far beyond channel before entry (0.1-0.5)
#   - trend_ema: longer EMA for trend filter (50-200)
#   - vol_filter: require volume > N-bar average (True/False)
#   - stop_atr_mult: trailing stop distance in ATR (1.0-3.0)
#   - horizon_bars: max hold time in bars (6-48)

DEFAULT_TREND_PARAMS = {
    "breakout_bars":  20,     # channel lookback
    "atr_mult":       0.15,   # breakout must exceed channel by this * ATR
    "trend_ema":      100,    # trend filter EMA period
    "vol_filter":     True,   # require above-average volume
    "vol_lookback":   20,     # volume average lookback
    "stop_atr_mult":  2.0,    # trailing stop in ATR units
    "horizon_bars":   24,     # max hold (time stop)
}


def _trend_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              window: int = 14) -> np.ndarray:
    n = len(close)
    pc = np.empty(n); pc[0] = close[0]; pc[1:] = close[:-1]
    tr = np.maximum.reduce([high - low, np.abs(high - pc), np.abs(low - pc)])
    atr = np.full(n, np.nan)
    for i in range(window - 1, n):
        atr[i] = tr[max(0, i - window + 1):i + 1].mean()
    return atr


def generate_trend_signal(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Generate a -1/0/+1 signal series from OHLCV using trend-following rules.
    Fully causal — only uses information available at bar close.
    """
    p = {**DEFAULT_TREND_PARAMS, **params}
    n = len(df)

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    v = df["volume"].values

    # Pre-compute indicators
    atr = _trend_atr(h, l, c, window=14)

    # Rolling channel (shift 1 so we use yesterday's high/low, not today's)
    bb = p["breakout_bars"]
    chan_high = np.full(n, np.nan)
    chan_low  = np.full(n, np.nan)
    for i in range(bb, n):
        chan_high[i] = h[i - bb:i].max()   # highest high of prev bb bars
        chan_low[i]  = l[i - bb:i].min()   # lowest low of prev bb bars

    # Trend filter EMA
    ema_period = p["trend_ema"]
    ema = np.full(n, np.nan)
    ema[ema_period - 1] = c[:ema_period].mean()
    alpha = 2.0 / (ema_period + 1)
    for i in range(ema_period, n):
        ema[i] = c[i] * alpha + ema[i - 1] * (1 - alpha)

    # Volume filter
    vl = p["vol_lookback"]
    vol_avg = np.full(n, np.nan)
    for i in range(vl, n):
        vol_avg[i] = v[i - vl:i].mean()

    # Timezone-aware time filter: no new entries in last 30 min
    et = df.index.tz_convert("America/New_York")
    minutes = np.array(et.hour) * 60 + np.array(et.minute)
    too_late = minutes >= (16 * 60 - 30)   # after 15:30 ET

    # State machine: track position and trailing stop
    signal   = np.zeros(n)
    position = 0       # -1, 0, +1
    stop_px  = np.nan
    bars_held = 0

    for i in range(max(bb, ema_period, vl) + 1, n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            signal[i] = position
            continue

        # --- manage existing position ---
        if position != 0:
            bars_held += 1
            # update trailing stop
            if position == 1:
                new_stop = c[i] - p["stop_atr_mult"] * a
                stop_px = max(stop_px, new_stop) if np.isfinite(stop_px) else new_stop
                if l[i] <= stop_px or bars_held >= p["horizon_bars"]:
                    position = 0; stop_px = np.nan; bars_held = 0
            else:  # short
                new_stop = c[i] + p["stop_atr_mult"] * a
                stop_px = min(stop_px, new_stop) if np.isfinite(stop_px) else new_stop
                if h[i] >= stop_px or bars_held >= p["horizon_bars"]:
                    position = 0; stop_px = np.nan; bars_held = 0

        # --- look for new entry (only when flat) ---
        if position == 0 and not too_late[i]:
            vol_ok = (not p["vol_filter"]) or (v[i] > vol_avg[i])
            trend_up   = np.isfinite(ema[i]) and c[i] > ema[i]
            trend_down = np.isfinite(ema[i]) and c[i] < ema[i]

            long_breakout  = (np.isfinite(chan_high[i]) and
                              c[i] > chan_high[i] + p["atr_mult"] * a)
            short_breakout = (np.isfinite(chan_low[i]) and
                              c[i] < chan_low[i] - p["atr_mult"] * a)

            if long_breakout and trend_up and vol_ok:
                position = 1
                stop_px  = c[i] - p["stop_atr_mult"] * a
                bars_held = 0
            elif short_breakout and trend_down and vol_ok:
                position = -1
                stop_px  = c[i] + p["stop_atr_mult"] * a
                bars_held = 0

        signal[i] = position

    return pd.Series(signal, index=df.index, dtype=float)


def sample_trend_params(rng) -> dict:
    """Random parameter set for walk-forward search."""
    return {
        "breakout_bars": int(rng.choice([10, 15, 20, 30, 40, 50])),
        "atr_mult":      float(rng.choice([0.05, 0.10, 0.15, 0.20, 0.30])),
        "trend_ema":     int(rng.choice([50, 75, 100, 150, 200])),
        "vol_filter":    bool(rng.choice([True, False])),
        "vol_lookback":  int(rng.choice([10, 20, 30])),
        "stop_atr_mult": float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])),
        "horizon_bars":  int(rng.choice([12, 24, 36, 48])),
    }


# ============================================================================
# Reactive order-flow momentum strategy
# ============================================================================
#
# No ML, no forward labels, no prediction. Each bar is scored -5 to +5 across
# five independent order-flow signals. A position is taken when enough signals
# agree and held until the evidence fades or a time stop fires.
#
# Requires real Databento order-flow columns:
#     delta, cum_delta, buy_vol, sell_vol, large_buy, large_sell, volume
#
# Fails loudly if those columns are missing — do not attempt to run on EODHD data.

REACTIVE_REQUIRED_COLS = frozenset({
    "open", "high", "low", "close", "volume",
    "delta", "cum_delta", "buy_vol", "sell_vol", "large_buy", "large_sell",
})

REACTIVE_DEFAULT_PARAMS: dict = {
    "score_threshold":       2,      # signals that must agree before entry
    "imbalance_threshold":   0.55,   # buy fraction above this = bullish pressure
    "delta_zscore_window":   20,     # lookback for cum_delta z-score (Signal 1)
    "bar_delta_window":      12,     # lookback for per-bar delta z-score (Signal 4)
    "max_hold_bars":         12,     # hard time stop (bars)
    "large_print_threshold": 0.10,   # (large_buy-large_sell)/volume threshold
    "trend_filter":          True,   # enable multi-timeframe trend filters
}


def _check_reactive_columns(df: pd.DataFrame) -> None:
    missing = REACTIVE_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Reactive strategy requires real order-flow columns. "
            f"Missing: {sorted(missing)}. "
            f"Use --source databento to load the correct data."
        )


def _reactive_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _reactive_rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, np.nan)
    return (tp * vol).rolling(window).sum() / vol.rolling(window).sum()


def generate_reactive_signal(
    df: pd.DataFrame,
    params: dict | None = None,
    instrument_key: str = "",
) -> tuple[pd.Series, pd.Series]:
    """
    Score-based entry + explicit exit rules with a hard time stop.

    Parameters
    ----------
    df     : bars DataFrame with OHLCV + order-flow columns (Databento path)
    params : override dict; missing keys fall back to REACTIVE_DEFAULT_PARAMS

    Returns
    -------
    signal      : pd.Series  ∈ {-1, 0, +1}, index = df.index
    size_scalar : pd.Series  ∈ {0.5, 1.0},  index = df.index
                  Multiply together → effective position passed to backtest.
                  With n_contracts=2: 1.0→2 contracts, 0.5→1 contract.
    """
    _check_reactive_columns(df)
    p = {**REACTIVE_DEFAULT_PARAMS, **(params or {})}

    score_thr     = int(p["score_threshold"])
    imbal_thr     = float(p["imbalance_threshold"])
    dz_win        = int(p["delta_zscore_window"])
    bdz_win       = int(p.get("bar_delta_window", 12))
    max_hold      = int(p["max_hold_bars"])
    lp_thr        = float(p["large_print_threshold"])
    trend_filter  = bool(p.get("trend_filter", True))

    close     = df["close"].astype(float)
    delta     = df["delta"].astype(float)
    cum_delta = df["cum_delta"].astype(float)
    buy_vol   = df["buy_vol"].astype(float)
    sell_vol  = df["sell_vol"].astype(float)
    vol       = df["volume"].astype(float)
    lbuy      = df["large_buy"].astype(float)
    lsell     = df["large_sell"].astype(float)

    # ---- causal indicator pre-computations --------------------------------

    vwap     = _reactive_rolling_vwap(df, 20)
    cd_ema20 = cum_delta.ewm(span=20, adjust=False).mean()

    # Signal 1: cumulative delta z-score — sustained directional flow
    cdz_mu   = cum_delta.rolling(dz_win).mean()
    cdz_sd   = cum_delta.rolling(dz_win).std()
    cum_delta_z = (cum_delta - cdz_mu) / (cdz_sd + 1e-8)

    # Signal 4: per-bar delta z-score — immediate aggressive flow this bar
    bdz_mu  = delta.rolling(bdz_win).mean()
    bdz_sd  = delta.rolling(bdz_win).std()
    bar_delta_z = (delta - bdz_mu) / (bdz_sd + 1e-8)

    total_vol = buy_vol + sell_vol + 1e-8
    bs_imbal  = buy_vol / total_vol

    lp = (lbuy - lsell) / (vol + 1e-8)   # large-print net pressure, per bar

    atr      = _reactive_atr(df, 20)
    atr_80th = atr.rolling(100).quantile(0.80)
    # GC-specific: tighter filter targeting consistency-rule breaches from
    # concentrated single-day moves during gold vol spikes.
    atr_90th_200 = atr.rolling(200).quantile(0.90) if instrument_key == "GC" else None

    # ---- five score components, each ∈ {-1, 0, +1} -----------------------

    # 1. sustained flow: cum_delta z-score vs rolling window
    s1 = np.where(cum_delta_z > 0.5,  1.0,
         np.where(cum_delta_z < -0.5, -1.0, 0.0))

    # 2. buy/sell volume imbalance
    s2 = np.where(bs_imbal > imbal_thr,            1.0,
         np.where(bs_imbal < (1.0 - imbal_thr),   -1.0, 0.0))

    # 3. price vs rolling VWAP
    s3 = np.sign(close - vwap)

    # 4. immediate aggression: per-bar delta z-score (distinct from Signal 1)
    s4 = np.where(bar_delta_z > 0.5,  1.0,
         np.where(bar_delta_z < -0.5, -1.0, 0.0))

    # 5. large-print pressure (absorption / aggression)
    s5 = np.where(lp > lp_thr,  1.0,
         np.where(lp < -lp_thr, -1.0, 0.0))

    score_s = pd.Series(s1 + s2 + s3 + s4 + s5, index=df.index).fillna(0.0)

    # ---- multi-timeframe trend filters ------------------------------------
    # All three filters are gated by `trend_filter`. When False, arrays default
    # to the most permissive values so existing entry logic is unchanged.

    n = len(df)

    if trend_filter:
        # ------------------------------------------------------------------
        # Filter 1 — medium-term trend (150-bar EMA ≈ 1.5 trading sessions)
        # close > EMA_150              → uptrend  : long only
        # close < EMA_150              → downtrend: short only
        # |close − EMA_150| / ATR < 0.5 → ambiguous: both dirs, thr += 1
        # ------------------------------------------------------------------
        ema_150    = close.ewm(span=150, adjust=False).mean()
        trend_norm = (close - ema_150) / (atr + 1e-8)

        trend_up   = (trend_norm >  0.5).fillna(False).values
        trend_dn   = (trend_norm < -0.5).fillna(False).values
        trend_ambg = (~trend_up & ~trend_dn)              # |norm| <= 0.5

        # direction gates (True = direction is permitted this bar)
        allow_long_v  = (~trend_dn).values.astype(bool)   # uptrend or ambiguous
        allow_short_v = (~trend_up).values.astype(bool)   # downtrend or ambiguous
        # ambiguous zone raises the bar for both directions
        ambig_bonus_v = trend_ambg.astype(int)

        # ------------------------------------------------------------------
        # Filter 2 — long-term regime (500-bar cum-delta z-score ≈ 2–3 days)
        # z > +1: bullish → short thr += 1
        # z < -1: bearish → long  thr += 1
        # neutral: no adjustment
        # ------------------------------------------------------------------
        cd_500_mu = cum_delta.rolling(500).mean()
        cd_500_sd = cum_delta.rolling(500).std()
        cd_500_z  = ((cum_delta - cd_500_mu) / (cd_500_sd + 1e-8)).fillna(0.0)

        regime_bull = (cd_500_z >  1.0).values   # buyers in control → penalise shorts
        regime_bear = (cd_500_z < -1.0).values   # sellers in control → penalise longs

        long_regime_adj_v  = regime_bear.astype(int)   # +1 to long thr when bearish
        short_regime_adj_v = regime_bull.astype(int)   # +1 to short thr when bullish

        # ------------------------------------------------------------------
        # Filter 3 — flow alignment (6-bar vs 50-bar delta direction)
        # Disagreement between short-term and medium-term flow → half size
        # ------------------------------------------------------------------
        delta_6  = delta.rolling(6).sum().fillna(0.0)
        delta_50 = delta.rolling(50).sum().fillna(0.0)
        # misaligned when the two sums have opposite signs (excluding zero)
        flow_misaligned_v = (
            (np.sign(delta_6) != np.sign(delta_50)) &
            (delta_6 != 0) & (delta_50 != 0)
        ).values.astype(bool)

    else:
        # trend filter OFF — preserve original entry behaviour exactly
        allow_long_v       = np.ones(n, dtype=bool)
        allow_short_v      = np.ones(n, dtype=bool)
        ambig_bonus_v      = np.zeros(n, dtype=int)
        long_regime_adj_v  = np.zeros(n, dtype=int)
        short_regime_adj_v = np.zeros(n, dtype=int)
        flow_misaligned_v  = np.zeros(n, dtype=bool)

    # ---- pre-compute exit-condition arrays (avoids per-bar pandas overhead) --

    score_v          = score_s.values.astype(float)
    cum_delta_v      = cum_delta.fillna(0.0).values
    price_below_vwap = (close < vwap).fillna(False).values
    price_above_vwap = (close > vwap).fillna(False).values
    delta_z_v        = bar_delta_z.fillna(0.0).values
    atr_high_v       = (atr > atr_80th).fillna(False).values
    gc_vol_spike_v   = ((atr > atr_90th_200).fillna(False).values
                        if atr_90th_200 is not None
                        else np.zeros(n, dtype=bool))

    # ---- stateful signal loop ---------------------------------------------
    # position is decided at bar i based on data through bar i (causal),
    # then acted at bar i's close and held into bar i+1 by the backtest engine.

    signal_v      = np.zeros(n, dtype=float)
    size_scalar_v = np.ones(n, dtype=float)

    position  = 0   # current desired position: -1, 0, or +1
    bars_held = 0

    for i in range(n):
        sc = score_v[i]

        # --- exit check (evaluated before any new entry) ---
        if position != 0:
            do_exit = bars_held >= max_hold   # hard time stop

            if not do_exit:
                if position == 1:   # long exits
                    do_exit = (sc < 0
                               or cum_delta_v[i] < 0
                               or price_below_vwap[i])
                else:               # short exits
                    do_exit = (sc > 0
                               or cum_delta_v[i] > 0
                               or price_above_vwap[i])

            if do_exit:
                position  = 0
                bars_held = 0

        # --- entry check (only when flat after potential exit) ---
        # Compose filter adjustments: ambiguous zone and regime counter-trend
        # each add +1 to the relevant side's effective threshold.
        if position == 0:
            eff_long_thr  = score_thr + ambig_bonus_v[i] + long_regime_adj_v[i]
            eff_short_thr = score_thr + ambig_bonus_v[i] + short_regime_adj_v[i]

            if allow_long_v[i] and sc >= eff_long_thr:
                position = 1
            elif allow_short_v[i] and sc <= -eff_short_thr:
                position = -1

        if position != 0:
            bars_held += 1

        signal_v[i] = position

        # --- size scalar: shrink when evidence starts to disagree ---
        ss = 1.0
        # selling pressure building while long
        if position == 1 and delta_z_v[i] < 0:
            ss *= 0.5
        # general high-vol regime (ATR > 80th pct / 100 bars)
        if atr_high_v[i]:
            ss *= 0.5
        # GC-specific: extreme vol spike (ATR > 90th pct / 200 bars)
        if gc_vol_spike_v[i]:
            ss *= 0.5
        # flow alignment (Filter 3): short and medium flow disagree → half size
        if flow_misaligned_v[i]:
            ss *= 0.5
        size_scalar_v[i] = ss

    return (
        pd.Series(signal_v,      index=df.index, name="signal"),
        pd.Series(size_scalar_v, index=df.index, name="size_scalar"),
    )


# ============================================================================
# Multi-timeframe market structure strategy (NQ / real futures bars)
# ============================================================================
#
# Price-structure based — works on any 5-minute OHLCV.  Intended for REAL NQ
# futures bars (Databento, ~20,000) where the detected zones sit at the prices
# real participants react to.  It also runs on the EODHD QQQ proxy, but there the
# FVG/swing levels are QQQ levels (~$520) and are economically meaningless for NQ
# — illustrative only.
#
# All timeframes are causal (zero look-ahead):
#   Daily : EMA5/EMA20 trend bias + extension score (uses prev completed day)
#   5-min : 3-candle Fair Value Gap zones (detected directly on 5m bars)
#   5-min : swing level confirmation + liquidity grab detection
#
# A bullish FVG is the classic 3-candle gap: candle 2 is a strong bullish impulse
# and candle 3's low prints above candle 1's high, leaving an unfilled gap
# [C1.high, C3.low].  Mirror for bearish.  Zones expire after 50 bars, are
# invalidated on a close beyond the far boundary, and are capped at 5 per side.
#
# Entry: trend_bias==+1, extension_score<threshold, inside active bull FVG OR
#        recent bullish grab, 5m bar closes green.  Mirror for shorts.
#
# NOTE: PnL is computed in the economics of config.INSTRUMENTS[key], which the
# runner sets from the data source (Databento NQ = $20/point; EODHD = MNQ $2/pt).

STRUCTURE_DEFAULT_PARAMS: dict = {
    "fvg_min_atr_mult":    1.1,   # FVG gap must span this many ATR_20 to be valid
    "grab_wick_ratio":     0.60,  # lower wick / bar range >= this for a grab
    "grab_atr_mult":       0.30,  # grab must penetrate swing_low by this many ATRs
    "max_hold_bars":       12,    # hard time stop (5m bars), exit_mode == "time"
    "extension_threshold": 1.5,   # |daily extension| above this => no entry (chasing)
    # ---- variant knobs (base values reproduce the original behaviour) ----
    "near_miss_atr":          0.0,    # Variant A: count price within N*ATR of an
                                      #   FVG zone as "inside" (0.0 = strict band)
    "exit_mode":              "time", # Variant B: "time" (max_hold) or "trail"
    "trail_atr_mult":         1.5,    # Variant B: trailing-stop distance in ATR
    "per_trade_stop_dollars": None,   # Variant C: hard $ stop per contract (None=off)
    # ---- "fixes" knobs (base values reproduce the original behaviour) ----
    "rth_only_entries":       False,  # Fix 1: restrict new entries to the RTH window
    "rth_start_min":          575,    #   earliest entry, ET minutes-of-day (09:35)
    "rth_end_min":            930,    #   latest entry, ET minutes-of-day (15:30)
    # Fix 2 / "mtp": multi-TP ratchet exit. LOCKED ON as the production default
    # after the base/of/flat/mtp/flat_mtp/mixed/combo ablation in
    # structure-variants -- mtp (+ dynamic sizing, see config.DYNAMIC_SIZING)
    # is the winning combo. n_targets is still searched 1..3 by the walk-forward
    # optimizer (see walkforward._sample_structure_params).
    "use_liquidity_targets":  True,   # Fix 2: liquidity-target exit + trailing
    "min_rr":                 None,   # Fix 2: skip entries below this reward/risk
    "n_targets":              1,      # Fix 2: number of TP levels (ratcheting trail)
    "require_fvg_agreement":  False,  # Fix 3: active FVG dir must match trade dir
    # ---- new edge knobs (base values reproduce the original behaviour) ----
    "orderflow_confirm":      False,  # #1: require delta agreement at the entry bar
    "of_window":              20,     #     lookback for the per-bar delta z-score
    "of_z_min":               0.0,    #     min delta z-score in the trade direction
    "of_apply":               "momentum",  # which triggers order-flow gates:
                                      #   "momentum" (default; delta confirms
                                      #   continuation but contradicts FVG
                                      #   reversion) or "all"
    "trade_flat":             False,  # #2: also enter when daily bias == 0 (neutral)
    "flat_use_targets":       True,   # flat-regime exit: True=multi-TP ratchet,
                                      #   False=time stop (flat trades do worse on
                                      #   the ratchet, so False keeps them on the
                                      #   time stop while trend trades use targets)
}

_SWING_LOOKBACK  = 3   # bars each side for swing confirmation
_N_SWING_KEEP    = 5   # rolling history of recent swing levels
_GRAB_VALID_BARS = 2   # bars after a grab bar where entry is still valid

# ---- liquidity-target exit (Fix 2) ----------------------------------------
_RTH_START_MIN     = 9 * 60 + 35    # 09:35 ET — earliest new entry
_RTH_END_MIN       = 15 * 60 + 30   # 15:30 ET — latest new entry
_TARGET_LOOKBACK   = 100            # bars to scan for an unswept swing target
_SKIP_RETRO_BARS   = 200            # forward horizon for skipped-trade retrospective
# Multi-take-profit ratchet: the trail starts at trail_atr_mult and TIGHTENS by
# _TRAIL_TIGHTEN each time a TP level is passed, down to a floor of _TRAIL_FLOOR
# ATR. Full exit on the final TP or when the (ratcheting) trail is hit.
_TRAIL_TIGHTEN     = 0.6            # trail multiplier shrink per TP passed
_TRAIL_FLOOR       = 0.25           # tightest the trail can get (ATR)
_FLAT_N_TARGETS    = 1              # neutral-regime trades: 1 (least ambitious) TP

# ---- FVG (3-candle Fair Value Gap) detection constants --------------------
_FVG_ATR_WINDOW       = 20    # ATR period used by FVG rules (ATR_20)
_IMPULSE_BODY_FRAC    = 0.5   # candle-2 body must exceed this fraction of its range
_IMPULSE_BODY_ATR     = 1.0   # candle-2 body must exceed this * ATR_20
_FVG_MAX_AGE_BARS     = 200   # a zone is dropped this many bars after it forms
                              # (~17h: a morning FVG still matters that afternoon)
_FVG_MAX_ACTIVE       = 5     # most this many active zones per side (keep newest)

# ---- volatility regime filter ---------------------------------------------
# When 20-bar ATR is above its 90th percentile over the trailing 500 bars, the
# market is in an extreme-volatility regime (e.g. the April-2026 tariff whipsaws
# with 300+ point swings). Structure breaks down there, so we sit out entirely.
_VOL_REGIME_WINDOW = 500
_VOL_REGIME_PCT    = 0.90

# Bar interval in minutes — used to reject 3-candle patterns that straddle a
# session boundary (overnight / weekend gaps are NOT fair value gaps).
_INTERVAL_MIN = {"1m": 1, "5m": 5, "1h": 60}.get(config.INTERVAL, 5)


def _contiguity_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """
    True at bar i when bars i-2, i-1, i are contiguous in time (no session gap
    between any of them). A gap larger than 1.5x the bar interval marks a session
    boundary, so a 3-candle FVG pattern spanning it is discarded.
    """
    n = len(index)
    contig = np.zeros(n, dtype=bool)
    if n < 3:
        return contig
    dt_min  = index.to_series().diff().dt.total_seconds().to_numpy() / 60.0
    max_gap = _INTERVAL_MIN * 1.5
    ok_prev = dt_min <= max_gap            # gap from bar i-1 to bar i is normal
    contig[2:] = ok_prev[2:] & ok_prev[1:-1]
    return contig


# ---------------------------------------------------------------------------
# Resampling helpers
# ---------------------------------------------------------------------------

def _resample_daily(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5m bars to completed daily bars (ET calendar days)."""
    tz = config.SESSION.timezone
    df_et = df_5m.tz_convert(tz)
    d = df_et.resample("1D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    ).dropna(subset=["close"])
    return d[d["volume"] > 0]


def _structure_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _orderflow_ok(df_5m: pd.DataFrame, window: int, z_min: float):
    """
    Per-bar order-flow confirmation from real Databento delta (#1).

    Returns (ok_long, ok_short) boolean arrays. A long entry is confirmed when
    the bar's signed delta, z-scored over `window` bars, is >= z_min (aggressive
    buyers actually showed up); short requires <= -z_min. Falls back to all-True
    when the bars carry no 'delta' column (e.g. the EODHD proxy path), so the
    confirmation simply has no effect there.
    """
    n = len(df_5m)
    if "delta" not in df_5m.columns:
        ok = np.ones(n, dtype=bool)
        return ok, ok.copy()
    delta = df_5m["delta"].astype(float)
    mu = delta.rolling(window).mean()
    sd = delta.rolling(window).std()
    z  = ((delta - mu) / (sd + 1e-9)).fillna(0.0).values
    ok_long  = z >= z_min
    ok_short = z <= -z_min
    return ok_long, ok_short


# ---------------------------------------------------------------------------
# Daily bias (mapped caually to 5m bars)
# ---------------------------------------------------------------------------

def _daily_bias(
    df_d: pd.DataFrame,
    idx_5m: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    """
    Daily EMA5/EMA20 trend bias and extension score, forward-filled to 5m bars.
    shift(1) ensures only completed daily bars are visible at any 5m bar.
    """
    if len(df_d) < 5:
        zeros = pd.Series(0.0, index=idx_5m)
        return zeros, zeros.copy()

    ema5  = df_d["close"].ewm(span=5,  adjust=False).mean()
    ema20 = df_d["close"].ewm(span=20, adjust=False).mean()
    atr_d = _structure_atr(df_d, 14)

    slope5  = ema5.diff()
    slope20 = ema20.diff()

    bias = pd.Series(
        np.where(
            (ema5 > ema20) & (slope5 > 0) & (slope20 > 0),  1.0,
            np.where(
                (ema5 < ema20) & (slope5 < 0) & (slope20 < 0), -1.0, 0.0
            )
        ),
        index=df_d.index,
    )
    ext = (df_d["close"] - ema20) / (atr_d + 1e-8)

    # shift(1): use previous completed day's values only
    trend_5m = bias.shift(1).reindex(idx_5m, method="ffill").fillna(0.0)
    ext_5m   = ext.shift(1).reindex(idx_5m, method="ffill").fillna(0.0)
    return trend_5m, ext_5m


# ---------------------------------------------------------------------------
# 3-candle Fair Value Gap detection (on 5-minute bars directly)
# ---------------------------------------------------------------------------

def _find_fvg_zones(
    df_5m: pd.DataFrame,
    atr_20: np.ndarray,
    contiguous: np.ndarray,
    fvg_min_atr_mult: float,
) -> list[dict]:
    """
    Classic 3-candle FVG, scanned with a rolling window over 5m bars.

    At bar i the window is (C1=i-2, C2=i-1, C3=i):
      bullish FVG -- C2 is a strong bullish impulse and C3.low > C1.high
                     => unfilled gap = [C1.high, C3.low]
      bearish FVG -- C2 is a strong bearish impulse and C3.high < C1.low
                     => unfilled gap = [C3.high, C1.low]

    "Strong impulse" (C2):
        body > _IMPULSE_BODY_FRAC * range   AND   body > _IMPULSE_BODY_ATR * ATR_20

    Gap must be at least fvg_min_atr_mult * ATR_20 wide.  Patterns that straddle
    a session boundary (contiguous[i] is False) are rejected — overnight/weekend
    gaps are not fair value gaps.

    The pattern completes at C3's close (bar i), so the zone is first eligible
    for a retest at bar i+1 (handled in _fvg_activity).

    Returns a list of dicts: {formed_idx, direction(+1/-1), zone_low, zone_high}.
    """
    n = len(df_5m)
    o = df_5m["open"].values
    h = df_5m["high"].values
    l = df_5m["low"].values
    c = df_5m["close"].values

    zones: list[dict] = []
    for i in range(2, n):
        if not contiguous[i]:
            continue
        a2, a3 = atr_20[i - 1], atr_20[i]
        if not (np.isfinite(a2) and np.isfinite(a3)) or a2 <= 0 or a3 <= 0:
            continue

        # --- candle 2 must be a strong impulse ---
        body2 = abs(c[i - 1] - o[i - 1])
        rng2  = h[i - 1] - l[i - 1]
        if rng2 <= 0:
            continue
        if not (body2 > _IMPULSE_BODY_FRAC * rng2 and body2 > _IMPULSE_BODY_ATR * a2):
            continue

        min_gap = fvg_min_atr_mult * a3

        if c[i - 1] > o[i - 1]:                      # bullish impulse
            gap = l[i] - h[i - 2]                     # C3.low - C1.high
            if gap > 0 and gap >= min_gap:
                zones.append({
                    "formed_idx": i,
                    "direction":  1,
                    "zone_low":   float(h[i - 2]),
                    "zone_high":  float(l[i]),
                })
        elif c[i - 1] < o[i - 1]:                    # bearish impulse
            gap = l[i - 2] - h[i]                      # C1.low - C3.high
            if gap > 0 and gap >= min_gap:
                zones.append({
                    "formed_idx": i,
                    "direction":  -1,
                    "zone_low":   float(h[i]),
                    "zone_high":  float(l[i - 2]),
                })

    return zones


def _fvg_activity(
    df_5m: pd.DataFrame,
    zones: list[dict],
    atr: np.ndarray | None = None,
    entry_tol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward sweep that resolves the lifecycle of every FVG zone:

      * a zone is active from formed_idx + 1 onward
      * it EXPIRES once _FVG_MAX_AGE_BARS bars have elapsed since formation
      * it is INVALIDATED when a bar closes fully beyond its far boundary
            bull: close < zone_low      bear: close > zone_high
      * at most _FVG_MAX_ACTIVE zones per side are kept (newest formed win)

    A 5m bar counts as "in" a zone when its close lies within the zone band,
    optionally widened by entry_tol * ATR on each side (Variant A "near miss"
    capture). With entry_tol == 0 this is exactly the strict band. Invalidation
    always uses the strict boundary, so widening the entry band never keeps a
    dead zone alive.

    Returns
    -------
    in_bull, in_bear : bool[n]
    bull_stop        : float[n]  highest active bull zone_low (nearest support)
    bear_stop        : float[n]  lowest  active bear zone_high (nearest resistance)
    """
    n     = len(df_5m)
    close = df_5m["close"].values
    if atr is None or entry_tol <= 0.0:
        tol = np.zeros(n)
    else:
        tol = entry_tol * np.nan_to_num(np.asarray(atr, dtype=float), nan=0.0)

    by_formed: dict[int, list[dict]] = defaultdict(list)
    for z in zones:
        by_formed[z["formed_idx"]].append(z)

    in_bull   = np.zeros(n, dtype=bool)
    in_bear   = np.zeros(n, dtype=bool)
    bull_stop = np.full(n, np.nan)
    bear_stop = np.full(n, np.nan)

    active_bull: list[dict] = []
    active_bear: list[dict] = []

    def _retire(z, i):
        z.setdefault("active_end_idx", i)

    for i in range(n):
        # 1. activate zones formed on the previous bar
        for z in by_formed.get(i - 1, ()):
            z["active_start_idx"] = i
            (active_bull if z["direction"] == 1 else active_bear).append(z)

        # 2. expire stale zones (age measured from formation)
        kept = []
        for z in active_bull:
            if i - z["formed_idx"] > _FVG_MAX_AGE_BARS:
                _retire(z, i)
            else:
                kept.append(z)
        active_bull = kept
        kept = []
        for z in active_bear:
            if i - z["formed_idx"] > _FVG_MAX_AGE_BARS:
                _retire(z, i)
            else:
                kept.append(z)
        active_bear = kept

        # 3. cap active count (keep the most recently formed)
        if len(active_bull) > _FVG_MAX_ACTIVE:
            active_bull.sort(key=lambda z: z["formed_idx"])
            for z in active_bull[:-_FVG_MAX_ACTIVE]:
                _retire(z, i)
            active_bull = active_bull[-_FVG_MAX_ACTIVE:]
        if len(active_bear) > _FVG_MAX_ACTIVE:
            active_bear.sort(key=lambda z: z["formed_idx"])
            for z in active_bear[:-_FVG_MAX_ACTIVE]:
                _retire(z, i)
            active_bear = active_bear[-_FVG_MAX_ACTIVE:]

        px = close[i]
        t  = tol[i]

        # 4. in-zone detection + nearest stop (band widened by entry_tol*ATR)
        best = np.nan
        for z in active_bull:
            if (z["zone_low"] - t) <= px <= (z["zone_high"] + t):
                in_bull[i] = True
                if np.isnan(best) or z["zone_low"] > best:
                    best = z["zone_low"]      # highest support = nearest stop
        bull_stop[i] = best

        best = np.nan
        for z in active_bear:
            if (z["zone_low"] - t) <= px <= (z["zone_high"] + t):
                in_bear[i] = True
                if np.isnan(best) or z["zone_high"] < best:
                    best = z["zone_high"]     # lowest resistance = nearest stop
        bear_stop[i] = best

        # 5. invalidate zones price has closed beyond (drop for future bars)
        kept = []
        for z in active_bull:
            if px < z["zone_low"]:
                _retire(z, i)
            else:
                kept.append(z)
        active_bull = kept
        kept = []
        for z in active_bear:
            if px > z["zone_high"]:
                _retire(z, i)
            else:
                kept.append(z)
        active_bear = kept

    # zones still active at the end run to the last bar
    for z in active_bull + active_bear:
        z.setdefault("active_end_idx", n - 1)

    return in_bull, in_bear, bull_stop, bear_stop


# ---------------------------------------------------------------------------
# Swing levels on 5m bars (causal: swing at bar i confirmed at i+SWING_LOOKBACK)
# ---------------------------------------------------------------------------

def _causal_swing_levels(
    df_5m: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the nearest confirmed swing low and swing high at each 5m bar.

    A swing low at bar i is confirmed once bars i+1 .. i+SWING_LOOKBACK are
    seen, so at bar i+SWING_LOOKBACK. Values are forward-filled.

    nearest_sl[t] = highest recent confirmed swing low  (closest support from below)
    nearest_sh[t] = lowest  recent confirmed swing high (closest resistance from above)
    """
    low  = df_5m["low"].values
    high = df_5m["high"].values
    n    = len(low)
    lb   = _SWING_LOOKBACK

    nearest_sl = np.full(n, np.nan)
    nearest_sh = np.full(n, np.nan)

    recent_lows  : list[float] = []
    recent_highs : list[float] = []

    for confirm_bar in range(lb, n - lb):
        i = confirm_bar - lb   # the candidate swing bar

        left_l  = low[i - lb : i]     if i >= lb else np.array([])
        right_l = low[i + 1 : i + lb + 1]
        left_h  = high[i - lb : i]    if i >= lb else np.array([])
        right_h = high[i + 1 : i + lb + 1]

        if (len(left_l) == lb and len(right_l) == lb and
                low[i] < left_l.min() and low[i] < right_l.min()):
            recent_lows.append(low[i])
            if len(recent_lows) > _N_SWING_KEEP:
                recent_lows.pop(0)

        if (len(left_h) == lb and len(right_h) == lb and
                high[i] > left_h.max() and high[i] > right_h.max()):
            recent_highs.append(high[i])
            if len(recent_highs) > _N_SWING_KEEP:
                recent_highs.pop(0)

        if recent_lows:
            nearest_sl[confirm_bar] = max(recent_lows)   # highest = nearest from below
        if recent_highs:
            nearest_sh[confirm_bar] = min(recent_highs)  # lowest = nearest from above

    # Forward-fill so every bar has the most recent confirmed swing
    for t in range(1, n):
        if np.isnan(nearest_sl[t]) and not np.isnan(nearest_sl[t - 1]):
            nearest_sl[t] = nearest_sl[t - 1]
        if np.isnan(nearest_sh[t]) and not np.isnan(nearest_sh[t - 1]):
            nearest_sh[t] = nearest_sh[t - 1]

    return nearest_sl, nearest_sh


def _swing_points(df_5m: pd.DataFrame) -> tuple:
    """
    Confirmed swing highs and lows for liquidity-target identification (Fix 2).

    Returns (sh_idx, sh_lvl, sh_sweep, sl_idx, sl_lvl, sl_sweep) as numpy arrays
    sorted by confirmation bar. A swing at candidate bar j is CONFIRMED at
    j + _SWING_LOOKBACK (causal — only known then). `*_sweep` is the first bar
    within the next _TARGET_LOOKBACK bars at which price trades back through the
    level (high >= swing-high / low <= swing-low); np.inf if never swept inside
    that horizon. A swing high is "unswept at bar i" iff sh_sweep > i.
    """
    low  = df_5m["low"].values
    high = df_5m["high"].values
    n    = len(low)
    lb   = _SWING_LOOKBACK

    sh_idx, sh_lvl, sl_idx, sl_lvl = [], [], [], []
    for confirm_bar in range(lb, n - lb):
        j = confirm_bar - lb
        if j < lb:
            continue
        ll, rl = low[j - lb:j], low[j + 1:j + lb + 1]
        lh, rh = high[j - lb:j], high[j + 1:j + lb + 1]
        if (len(rl) == lb and low[j] < ll.min() and low[j] < rl.min()):
            sl_idx.append(confirm_bar); sl_lvl.append(float(low[j]))
        if (len(rh) == lb and high[j] > lh.max() and high[j] > rh.max()):
            sh_idx.append(confirm_bar); sh_lvl.append(float(high[j]))

    def _sweep(idxs, lvls, is_high):
        out = []
        for c, lv in zip(idxs, lvls):
            hit = np.inf
            end = min(c + _TARGET_LOOKBACK, n - 1)
            for k in range(c + 1, end + 1):
                if (is_high and high[k] >= lv) or ((not is_high) and low[k] <= lv):
                    hit = k
                    break
            out.append(hit)
        return np.asarray(out, dtype=float)

    return (np.asarray(sh_idx, dtype=int), np.asarray(sh_lvl, dtype=float),
            _sweep(sh_idx, sh_lvl, True),
            np.asarray(sl_idx, dtype=int), np.asarray(sl_lvl, dtype=float),
            _sweep(sl_idx, sl_lvl, False))


def _nearest_unswept_target(side: int, i: int, entry_px: float,
                            idx_arr, lvl_arr, sweep_arr) -> float:
    """
    Nearest unswept swing target for an entry at bar i.
      side=+1 (long)  -> nearest unswept swing HIGH above entry
      side=-1 (short) -> nearest unswept swing LOW  below entry
    Considers only swings confirmed within the last _TARGET_LOOKBACK bars.
    Returns np.nan if none qualifies.
    """
    if idx_arr.size == 0:
        return np.nan
    lo = int(np.searchsorted(idx_arr, i - _TARGET_LOOKBACK, side="left"))
    hi = int(np.searchsorted(idx_arr, i, side="right"))
    if hi <= lo:
        return np.nan
    lv = lvl_arr[lo:hi]
    unswept = sweep_arr[lo:hi] > i
    if side == 1:
        cand = lv[(lv > entry_px) & unswept]
        return float(cand.min()) if cand.size else np.nan
    cand = lv[(lv < entry_px) & unswept]
    return float(cand.max()) if cand.size else np.nan


def _unswept_targets(side: int, i: int, entry_px: float,
                     idx_arr, lvl_arr, sweep_arr, n_targets: int) -> list[float]:
    """
    The nearest `n_targets` unswept swing levels in the trade direction, ordered
    nearest-first (TP1, TP2, ...). For longs: swing HIGHs above entry ascending;
    for shorts: swing LOWs below entry descending. Empty list if none qualify.
    """
    if idx_arr.size == 0 or n_targets < 1:
        return []
    lo = int(np.searchsorted(idx_arr, i - _TARGET_LOOKBACK, side="left"))
    hi = int(np.searchsorted(idx_arr, i, side="right"))
    if hi <= lo:
        return []
    lv = lvl_arr[lo:hi]
    unswept = sweep_arr[lo:hi] > i
    if side == 1:
        cand = np.unique(lv[(lv > entry_px) & unswept])      # ascending
        return [float(x) for x in cand[:n_targets]]
    cand = np.unique(lv[(lv < entry_px) & unswept])[::-1]    # descending (nearest first)
    return [float(x) for x in cand[:n_targets]]


def _retro_profit(i: int, side: int, stop: float, target: float,
                  h: np.ndarray, l: np.ndarray, n: int) -> bool:
    """
    Retrospective check for an R:R-skipped trade: would it have hit the target
    before the stop within the forward horizon? Conservative — if a bar touches
    both stop and target, the stop is assumed hit first (counts as a loss).
    """
    end = min(i + _SKIP_RETRO_BARS, n - 1)
    for k in range(i + 1, end + 1):
        if side == 1:
            if l[k] <= stop:
                return False
            if h[k] >= target:
                return True
        else:
            if h[k] >= stop:
                return False
            if l[k] <= target:
                return True
    return False


# ---------------------------------------------------------------------------
# Liquidity grab detection (causal)
# ---------------------------------------------------------------------------

def _liquidity_grabs(
    df_5m: pd.DataFrame,
    atr_5m: np.ndarray,
    grab_wick_ratio: float,
    grab_atr_mult: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect bullish and bearish liquidity grabs on 5m bars.

    Bullish grab at bar t:
      - Low penetrates nearest swing low by >= grab_atr_mult * ATR
      - Close recovers above the swing low
      - Lower wick (min(open,close) - low) >= grab_wick_ratio * bar_range

    Grab remains valid for GRAB_VALID_BARS bars after the grab bar.

    Returns
    -------
    bull_grab_active : bool[n]
    bear_grab_active : bool[n]
    bull_grab_stop   : float[n]  stop = low of the grab bar
    bear_grab_stop   : float[n]  stop = high of the grab bar
    """
    o = df_5m["open"].values
    h = df_5m["high"].values
    l = df_5m["low"].values
    c = df_5m["close"].values
    n = len(c)

    nearest_sl, nearest_sh = _causal_swing_levels(df_5m)

    bull_grab_active = np.zeros(n, dtype=bool)
    bear_grab_active = np.zeros(n, dtype=bool)
    bull_grab_stop_v = np.full(n, np.nan)
    bear_grab_stop_v = np.full(n, np.nan)

    for t in range(1, n):
        bar_range = h[t] - l[t]
        if bar_range < 1e-8:
            continue
        atr_t = atr_5m[t]
        if atr_t <= 0 or np.isnan(atr_t):
            continue

        # Bullish grab
        sl = nearest_sl[t]
        if not np.isnan(sl) and sl > 0:
            penetration  = sl - l[t]
            wick_below   = min(o[t], c[t]) - l[t]
            if (penetration >= grab_atr_mult * atr_t and
                    c[t] > sl and
                    wick_below / bar_range >= grab_wick_ratio):
                stop = l[t]
                for k in range(t, min(t + _GRAB_VALID_BARS + 1, n)):
                    bull_grab_active[k] = True
                    if np.isnan(bull_grab_stop_v[k]):
                        bull_grab_stop_v[k] = stop

        # Bearish grab
        sh = nearest_sh[t]
        if not np.isnan(sh) and sh > 0:
            penetration  = h[t] - sh
            wick_above   = h[t] - max(o[t], c[t])
            if (penetration >= grab_atr_mult * atr_t and
                    c[t] < sh and
                    wick_above / bar_range >= grab_wick_ratio):
                stop = h[t]
                for k in range(t, min(t + _GRAB_VALID_BARS + 1, n)):
                    bear_grab_active[k] = True
                    if np.isnan(bear_grab_stop_v[k]):
                        bear_grab_stop_v[k] = stop

    return bull_grab_active, bear_grab_active, bull_grab_stop_v, bear_grab_stop_v


# ---------------------------------------------------------------------------
# Main signal generation
# ---------------------------------------------------------------------------

def _compute_structure_arrays(
    df_5m: pd.DataFrame,
    params: dict | None,
    instrument_key: str,
    record: bool = False,
) -> dict:
    """
    Core market-structure engine. Returns raw numpy arrays plus the per-bar
    context used by both the public signal wrappers and the chart/near-miss
    diagnostics. All logic is causal.

    Variant knobs (see STRUCTURE_DEFAULT_PARAMS):
      near_miss_atr          -- Variant A: widen the FVG entry band by N*ATR_20
      exit_mode == "trail"   -- Variant B: ATR trailing stop + bias-flip exit
      per_trade_stop_dollars -- Variant C: per-contract $ stop cap
      rth_only_entries       -- Fix 1: only enter 09:35-15:30 ET
      use_liquidity_targets  -- Fix 2: unswept-swing target + adaptive trailing
                                stop + R:R filter (replaces the time stop; falls
                                back to the time stop when no target exists)
      require_fvg_agreement  -- Fix 3: active FVG direction must match trade dir

    `record=True` (set by the debug wrapper) collects taken/skipped entry lists
    incl. the R:R-skip retrospective; validation runs with record=False for speed
    but still applies the R:R filter so behaviour is identical.
    """
    p        = {**STRUCTURE_DEFAULT_PARAMS, **(params or {})}
    ext_thr  = float(p["extension_threshold"])
    max_hold = int(p["max_hold_bars"])
    fvg_mult = float(p["fvg_min_atr_mult"])
    wr       = float(p["grab_wick_ratio"])
    gm       = float(p["grab_atr_mult"])
    near_tol = float(p.get("near_miss_atr", 0.0))
    exit_mode = str(p.get("exit_mode", "time"))
    trail_mult = float(p.get("trail_atr_mult", 1.5))
    per_trade_stop = p.get("per_trade_stop_dollars", None)
    rth_only      = bool(p.get("rth_only_entries", False))
    use_targets   = bool(p.get("use_liquidity_targets", False))
    require_agree = bool(p.get("require_fvg_agreement", False))
    n_targets     = max(1, int(p.get("n_targets", 1)))      # #4: TP ladder depth
    of_confirm    = bool(p.get("orderflow_confirm", False)) # #1
    of_window     = int(p.get("of_window", 20))
    of_z_min      = float(p.get("of_z_min", 0.0))
    of_apply      = str(p.get("of_apply", "momentum"))      # gate which triggers
    trade_flat    = bool(p.get("trade_flat", False))        # #2
    flat_use_tgts = bool(p.get("flat_use_targets", True))   # flat exit style
    _mrr          = p.get("min_rr", None)
    min_rr        = float(_mrr) if _mrr is not None else None

    # Per-contract $ stop -> price distance (Variant C).
    stop_pts = None
    if per_trade_stop is not None:
        inst = config.INSTRUMENTS.get(instrument_key)
        pv   = inst.point_value if inst is not None else 1.0
        stop_pts = float(per_trade_stop) / pv

    # --- build multi-timeframe context (all causal) ---
    df_d = _resample_daily(df_5m)
    trend_5m, ext_5m = _daily_bias(df_d, df_5m.index)

    atr_20_s   = _structure_atr(df_5m, _FVG_ATR_WINDOW)
    atr_20     = atr_20_s.values
    contiguous = _contiguity_mask(df_5m.index)
    fvg_zones  = _find_fvg_zones(df_5m, atr_20, contiguous, fvg_mult)
    in_bull_fvg, in_bear_fvg, bull_fvg_stop, bear_fvg_stop = \
        _fvg_activity(df_5m, fvg_zones, atr=atr_20, entry_tol=near_tol)

    vol_p90     = atr_20_s.rolling(_VOL_REGIME_WINDOW).quantile(_VOL_REGIME_PCT)
    vol_extreme = (atr_20_s > vol_p90).fillna(False).values

    atr_5m = _structure_atr(df_5m, 14).fillna(0.0).values
    bull_grab, bear_grab, bull_grab_stop, bear_grab_stop = \
        _liquidity_grabs(df_5m, atr_5m, wr, gm)

    ema5_5m    = df_5m["close"].ewm(span=5, adjust=False).mean().values
    ema5_slope = np.diff(ema5_5m, prepend=ema5_5m[0])
    ema20_5m    = df_5m["close"].ewm(span=20, adjust=False).mean().values
    ema20_slope = np.diff(ema20_5m, prepend=ema20_5m[0])

    o = df_5m["open"].values
    h = df_5m["high"].values
    l = df_5m["low"].values
    c = df_5m["close"].values
    n = len(c)

    # RTH entry window (Fix 1): minutes-of-day in ET.
    if rth_only:
        et   = df_5m.index.tz_convert(config.SESSION.timezone)
        mins = np.asarray(et.hour) * 60 + np.asarray(et.minute)
        rs   = int(p.get("rth_start_min", _RTH_START_MIN))
        re_  = int(p.get("rth_end_min",   _RTH_END_MIN))
        rth_ok = (mins >= rs) & (mins <= re_)
    else:
        rth_ok = np.ones(n, dtype=bool)

    # Swing points for liquidity targets (Fix 2) — only computed when used.
    if use_targets:
        sh_idx, sh_lvl, sh_sweep, sl_idx, sl_lvl, sl_sweep = _swing_points(df_5m)
    else:
        _empty_i, _empty_f = np.asarray([], dtype=int), np.asarray([], dtype=float)
        sh_idx = sl_idx = _empty_i
        sh_lvl = sl_lvl = sh_sweep = sl_sweep = _empty_f

    # Order-flow confirmation arrays (#1). All-True when off or no delta column.
    if of_confirm:
        of_ok_long, of_ok_short = _orderflow_ok(df_5m, of_window, of_z_min)
    else:
        of_ok_long = of_ok_short = np.ones(n, dtype=bool)

    trend_v = trend_5m.values
    ext_v   = ext_5m.values

    signal_v      = np.zeros(n, dtype=float)
    size_scalar_v = np.ones(n,  dtype=float)
    stop_v        = np.full(n, np.nan)
    target_v      = np.full(n, np.nan)

    taken: list[dict] = []
    skipped: list[dict] = []

    # --- state ---
    position          = 0
    bars_held         = 0
    stop_level        = np.nan
    targets: list[float] = []     # #4: TP ladder (nearest-first); [] = no targets
    tp_stage          = 0         # how many TP levels have been passed
    is_target_trade   = False
    entry_price       = np.nan
    extreme_close     = np.nan
    entry_bar_range   = np.nan
    momentum_exited   = False
    momentum_exit_dir = 0

    def _flatten():
        nonlocal position, bars_held, stop_level, targets, tp_stage
        nonlocal is_target_trade, entry_price, extreme_close, momentum_exited
        position = 0; bars_held = 0
        stop_level = np.nan; targets = []; tp_stage = 0
        is_target_trade = False
        entry_price = np.nan; extreme_close = np.nan
        momentum_exited = False

    for i in range(n):

        # ---- volatility regime gate: extreme vol => sit out (force flat) ----
        if vol_extreme[i]:
            _flatten()
            signal_v[i] = 0.0
            size_scalar_v[i] = 0.0
            continue

        # ---- momentum-waning exit (secondary, all variants) ----
        if position != 0 and bars_held >= 3:
            last3 = [h[max(0, i - k)] - l[max(0, i - k)] for k in range(3)]
            if (all(r < entry_bar_range * 0.85 for r in last3)
                    and abs(ema5_slope[i]) < 0.10 * (atr_5m[i] + 1e-8)):
                momentum_exit_dir = position
                _flatten()
                momentum_exited = True

        # ---- hard structural stop (also carries the Variant C $ cap) ----
        if position != 0 and not np.isnan(stop_level):
            if (position == 1 and c[i] < stop_level) or \
               (position == -1 and c[i] > stop_level):
                _flatten()

        # ---- exit: multi-TP ratchet (Fix 2/#4) > Variant B trail > time stop ----
        # The trail starts at trail_mult ATR and TIGHTENS by _TRAIL_TIGHTEN for
        # each TP level already passed (floor _TRAIL_FLOOR). Full exit when the
        # final TP is reached or the ratcheting trail is hit.
        if position != 0 and use_targets and is_target_trade:
            a    = atr_5m[i] + 1e-8
            mult = max(trail_mult * (_TRAIL_TIGHTEN ** tp_stage), _TRAIL_FLOOR)
            if position == 1:
                extreme_close = max(extreme_close, c[i])
                if c[i] < extreme_close - mult * a:
                    _flatten()
                else:
                    while tp_stage < len(targets) and c[i] >= targets[tp_stage]:
                        tp_stage += 1
                    if tp_stage >= len(targets):     # passed the final TP
                        _flatten()
            else:
                extreme_close = min(extreme_close, c[i])
                if c[i] > extreme_close + mult * a:
                    _flatten()
                else:
                    while tp_stage < len(targets) and c[i] <= targets[tp_stage]:
                        tp_stage += 1
                    if tp_stage >= len(targets):
                        _flatten()

        elif position != 0 and (not use_targets) and exit_mode == "trail":
            a = atr_5m[i] + 1e-8
            if position == 1:
                extreme_close = max(extreme_close, c[i])
                if c[i] < extreme_close - trail_mult * a or trend_v[i] == -1.0:
                    _flatten()
            else:
                extreme_close = min(extreme_close, c[i])
                if c[i] > extreme_close + trail_mult * a or trend_v[i] == 1.0:
                    _flatten()

        elif position != 0 and bars_held >= max_hold:
            # time stop: base behaviour, and the Fix-2 fallback when no target
            _flatten()

        # ---- entry / re-entry ----
        if position == 0 and (not rth_only or rth_ok[i]):
            tb  = trend_v[i]
            ext = ext_v[i]

            in_fvg_l = bool(in_bull_fvg[i]); in_fvg_s = bool(in_bear_fvg[i])
            grab_l   = bool(bull_grab[i]);   grab_s   = bool(bear_grab[i])
            has_bull_struct = in_fvg_l or grab_l
            has_bear_struct = in_fvg_s or grab_s

            # Fix 3: active FVG direction must agree with the trade direction
            long_block  = require_agree and in_fvg_s and not in_fvg_l
            short_block = require_agree and in_fvg_l and not in_fvg_s

            bull_momentum = (
                i >= 1 and not has_bull_struct
                and ema5_slope[i] > 0 and ema20_slope[i] > 0
                and c[i] > ema20_5m[i] and c[i - 1] <= ema20_5m[i - 1]
            )
            bear_momentum = (
                i >= 1 and not has_bear_struct
                and ema5_slope[i] < 0 and ema20_slope[i] < 0
                and c[i] < ema20_5m[i] and c[i - 1] >= ema20_5m[i - 1]
            )

            # Direction by regime:
            #   trend (bias=+/-1): structure OR momentum, extension-gated
            #   flat  (bias= 0)  : structure ONLY (no momentum, no ext gate); take
            #                      only an unambiguous side (#2). Targets are fewer
            #                      (less ambitious) in flat — see nt below.
            trend_long  = (tb == 1.0 and ext < ext_thr and (has_bull_struct or bull_momentum))
            trend_short = (tb == -1.0 and ext > -ext_thr and (has_bear_struct or bear_momentum))
            flat_long   = (trade_flat and tb == 0.0 and has_bull_struct and not has_bear_struct)
            flat_short  = (trade_flat and tb == 0.0 and has_bear_struct and not has_bull_struct)

            want_long  = (trend_long or flat_long) and not long_block
            want_short = (trend_short or flat_short) and not short_block
            side = 1 if want_long else (-1 if want_short else 0)

            # #3: trigger tag (priority fvg > grab > momentum) — needed before
            # the order-flow gate so we can apply it per-trigger.
            trig = None
            if side == 1:
                trig = "fvg" if in_fvg_l else ("grab" if grab_l else "momentum")
            elif side == -1:
                trig = "fvg" if in_fvg_s else ("grab" if grab_s else "momentum")

            # #1: order-flow confirmation. By default gates the MOMENTUM trigger
            # only (delta confirms continuation, but it contradicts FVG/grab
            # reversion entries — confirming those filtered out the winners).
            if side != 0 and of_confirm and (of_apply == "all" or trig == "momentum"):
                if (side == 1 and not of_ok_long[i]) or \
                   (side == -1 and not of_ok_short[i]):
                    side = 0

            if side != 0:
                is_flat = (tb == 0.0)
                if side == 1:
                    allow = not (momentum_exited and momentum_exit_dir == 1) or in_fvg_l
                else:
                    allow = not (momentum_exited and momentum_exit_dir == -1) or in_fvg_s

                if allow:
                    # structural stop (+ Variant C cap)
                    if side == 1:
                        if grab_l and not np.isnan(bull_grab_stop[i]):
                            s_lvl = bull_grab_stop[i]
                        elif in_fvg_l and not np.isnan(bull_fvg_stop[i]):
                            s_lvl = bull_fvg_stop[i]
                        else:
                            s_lvl = c[i] - 2.0 * (atr_5m[i] + 1e-8)
                        if stop_pts is not None:
                            s_lvl = max(s_lvl, c[i] - stop_pts)
                    else:
                        if grab_s and not np.isnan(bear_grab_stop[i]):
                            s_lvl = bear_grab_stop[i]
                        elif in_fvg_s and not np.isnan(bear_fvg_stop[i]):
                            s_lvl = bear_fvg_stop[i]
                        else:
                            s_lvl = c[i] + 2.0 * (atr_5m[i] + 1e-8)
                        if stop_pts is not None:
                            s_lvl = min(s_lvl, c[i] + stop_pts)

                    # #4: collect the TP ladder (fewer targets in flat) + R:R gate.
                    # Flat-regime trades skip the ratchet (and fall back to the
                    # time stop) when flat_use_targets is False — they perform
                    # worse on the multi-TP exit than trend trades do.
                    nt = _FLAT_N_TARGETS if is_flat else n_targets
                    tps, rr, is_tt, do_enter = [], np.nan, False, True
                    if use_targets and (not is_flat or flat_use_tgts):
                        if side == 1:
                            tps = _unswept_targets(1, i, c[i], sh_idx, sh_lvl, sh_sweep, nt)
                        else:
                            tps = _unswept_targets(-1, i, c[i], sl_idx, sl_lvl, sl_sweep, nt)
                        if tps:
                            risk   = abs(c[i] - s_lvl)
                            reward = abs(tps[0] - c[i])          # R:R measured to TP1
                            rr     = reward / risk if risk > 1e-9 else np.inf
                            is_tt  = True
                            if (min_rr is not None) and rr < min_rr:
                                do_enter = False
                                if record:
                                    skipped.append({
                                        "skip_idx":  i, "skip_time": df_5m.index[i],
                                        "direction": "long" if side == 1 else "short",
                                        "entry": float(c[i]), "stop": float(s_lvl),
                                        "target": float(tps[0]), "rr": float(rr),
                                        "trigger": trig,
                                        "would_profit": _retro_profit(
                                            i, side, s_lvl, tps[0], h, l, n),
                                    })
                        # no target -> fall back to time stop (enter, is_tt=False)

                    if do_enter:
                        position         = side
                        bars_held        = 0
                        momentum_exited  = False
                        entry_price      = c[i]
                        extreme_close    = c[i]
                        entry_bar_range  = (h[i] - l[i]) + 1e-8
                        stop_level       = s_lvl
                        targets          = tps if is_tt else []
                        tp_stage         = 0
                        is_target_trade  = is_tt
                        if record:
                            taken.append({
                                "entry_idx":  i, "entry_time": df_5m.index[i],
                                "direction":  "long" if side == 1 else "short",
                                "entry":   float(c[i]), "stop": float(s_lvl),
                                "target":  float(tps[0]) if is_tt else np.nan,
                                "rr":      float(rr) if is_tt else np.nan,
                                "is_target_trade": is_tt,
                                "trigger": trig,
                                "regime":  "flat" if is_flat else "trend",
                                "n_targets": len(tps) if is_tt else 0,
                            })

        if position != 0:
            bars_held += 1

        signal_v[i] = position
        stop_v[i]   = stop_level if position != 0 else np.nan
        if position != 0 and targets:
            target_v[i] = targets[min(tp_stage, len(targets) - 1)]
        else:
            target_v[i] = np.nan

    return {
        "signal":      signal_v,
        "size":        size_scalar_v,
        "stop":        stop_v,
        "target":      target_v,
        "trend":       trend_v,
        "ext":         ext_v,
        "in_bull_fvg": in_bull_fvg,
        "in_bear_fvg": in_bear_fvg,
        "bull_grab":   bull_grab.astype(bool),
        "bear_grab":   bear_grab.astype(bool),
        "atr_20":      atr_20,
        "vol_extreme": vol_extreme,
        "fvg_zones":   fvg_zones,
        "taken":       taken,
        "skipped":     skipped,
    }


def generate_structure_signal(
    df_5m: pd.DataFrame,
    params: dict | None = None,
    instrument_key: str = "NQ",
) -> tuple[pd.Series, pd.Series]:
    """
    Multi-timeframe market structure entry/exit signal.

    Returns
    -------
    signal      : pd.Series  in {-1, 0, +1}
    size_scalar : pd.Series  (1.0; reserved for risk scaling)
    """
    a = _compute_structure_arrays(df_5m, params, instrument_key)
    return (
        pd.Series(a["signal"], index=df_5m.index, name="signal"),
        pd.Series(a["size"],   index=df_5m.index, name="size_scalar"),
    )


def generate_structure_signal_debug(
    df_5m: pd.DataFrame,
    params: dict | None = None,
    instrument_key: str = "NQ",
) -> dict:
    """
    Same engine as generate_structure_signal but also returns the per-bar stop
    level, target level, decision context, and the taken/skipped entry records
    (used for entry/near-miss charts and the R:R report). Series keys: signal,
    size, stop, target, trend, ext, in_bull_fvg, in_bear_fvg, bull_grab,
    bear_grab, vol_extreme; plus 'fvg_zones' (list), 'atr_20' (Series),
    'taken' (DataFrame) and 'skipped' (DataFrame).
    """
    a   = _compute_structure_arrays(df_5m, params, instrument_key, record=True)
    idx = df_5m.index
    out = {
        "fvg_zones": a["fvg_zones"],
        "atr_20":    pd.Series(a["atr_20"], index=idx),
        "taken":     pd.DataFrame(a["taken"]),
        "skipped":   pd.DataFrame(a["skipped"]),
    }
    for k in ("signal", "size", "stop", "target", "trend", "ext",
              "in_bull_fvg", "in_bear_fvg", "bull_grab", "bear_grab",
              "vol_extreme"):
        out[k] = pd.Series(a[k], index=idx)
    return out


def diagnose_structure(
    df_5m: pd.DataFrame,
    params: dict | None = None,
) -> dict:
    """
    Return intermediate detection results for a slice of 5m bars.

    Used by the `structure` run subcommand to:
      - print FVG count, grab count, signal distribution
      - plot one day with zones shaded and grabs marked
    """
    p        = {**STRUCTURE_DEFAULT_PARAMS, **(params or {})}
    fvg_mult = float(p["fvg_min_atr_mult"])
    wr       = float(p["grab_wick_ratio"])
    gm       = float(p["grab_atr_mult"])

    df_d = _resample_daily(df_5m)
    trend_5m, ext_5m = _daily_bias(df_d, df_5m.index)

    atr_20     = _structure_atr(df_5m, _FVG_ATR_WINDOW).values
    contiguous = _contiguity_mask(df_5m.index)
    fvg_zones  = _find_fvg_zones(df_5m, atr_20, contiguous, fvg_mult)
    in_bull_fvg, in_bear_fvg, bull_fs, bear_fs = _fvg_activity(df_5m, fvg_zones)

    atr_5m = _structure_atr(df_5m, 14).fillna(0.0).values
    bull_grab, bear_grab, bull_gs, bear_gs = _liquidity_grabs(df_5m, atr_5m, wr, gm)
    signal, _ = generate_structure_signal(df_5m, params)

    # Annotate zones with timestamps so the plotter can draw bounded rectangles.
    idx = df_5m.index
    n   = len(idx)
    for z in fvg_zones:
        fi = z["formed_idx"]
        z["formed_time"] = idx[fi]
        a0 = z.get("active_start_idx", min(fi + 1, n - 1))
        a1 = z.get("active_end_idx",   n - 1)
        z["active_start_time"] = idx[min(a0, n - 1)]
        z["active_end_time"]   = idx[min(a1, n - 1)]

    return {
        "fvg_zones":   fvg_zones,                                   # list of dicts
        "trend":       trend_5m,                                     # pd.Series
        "extension":   ext_5m,                                       # pd.Series
        "in_bull_fvg": pd.Series(in_bull_fvg, index=df_5m.index),
        "in_bear_fvg": pd.Series(in_bear_fvg, index=df_5m.index),
        "bull_grab":   pd.Series(bull_grab,   index=df_5m.index),
        "bear_grab":   pd.Series(bear_grab,   index=df_5m.index),
        "signal":      signal,
    }


# ============================================================================
# EXPERIMENTAL: London/NY session-gap reversal
# ============================================================================
#
# Gold-motivated idea: London often drives a directional move between 03:00
# and 08:00 ET that gets faded once NY liquidity arrives. Rules:
#
#   1. Track the London session's high/low/VWAP as it forms (03:00-08:00 ET).
#   2. In the NY-open window (08:30-09:00 ET), watch each completed 15-minute
#      candle. If price sweeps outside the London range by more than
#      sweep_atr_mult * ATR intrabar, but that 15m candle's own CLOSE ends up
#      back inside the range (wick swept out, close failed to hold outside),
#      fade the sweep back toward the London session's VWAP.
#   3. Stop = the sweep extreme +/- stop_atr_mult * ATR. Target = London VWAP.
#   4. At most ONE trade per calendar day (ET) -- this fires roughly once a
#      day, not once a bar, so expect thin per-fold trade counts.
#
# Causality note: "the 15-minute candle's close" is read directly off the
# LAST 5-minute bar of that 15-minute block (its close IS the 15m candle's
# close), so no resampling/look-ahead is needed -- the check only fires on
# bars where (minute + 5) % 15 == 0.

SESSION_GAP_DEFAULT_PARAMS: dict = {
    "london_start_min":  3 * 60,        # 03:00 ET
    "london_end_min":    8 * 60,        # 08:00 ET
    "entry_start_min":   8 * 60 + 30,   # 08:30 ET
    "entry_end_min":     9 * 60,        # 09:00 ET
    "sweep_atr_mult":    1.0,   # sweep must clear the London range by this many ATR
    "stop_atr_mult":     0.5,   # stop = sweep extreme +/- this many ATR
    "atr_window":        14,
    "max_hold_bars":     48,    # safety time stop (~4h of 5m bars) if neither hit
}


def _minutes_et(index: pd.DatetimeIndex) -> np.ndarray:
    et = index.tz_convert(config.SESSION.timezone)
    return np.asarray(et.hour) * 60 + np.asarray(et.minute)


def generate_session_gap_signal(
    df_5m: pd.DataFrame,
    params: dict | None = None,
    instrument_key: str = "",
) -> tuple[pd.Series, pd.Series]:
    """
    London-range sweep-and-fail reversal, entered only in a narrow NY-open
    window. At most one trade per calendar day (ET).

    Returns (signal, size_scalar) like the other strategies -- size_scalar is
    always 1.0 (no risk scaling implemented for this experimental strategy).
    """
    p = {**SESSION_GAP_DEFAULT_PARAMS, **(params or {})}
    n = len(df_5m)
    idx = df_5m.index
    minutes = _minutes_et(idx)
    day = np.asarray(idx.tz_convert(config.SESSION.timezone).date)

    h = df_5m["high"].values
    l = df_5m["low"].values
    c = df_5m["close"].values
    v = df_5m["volume"].values

    atr = _structure_atr(df_5m, int(p["atr_window"])).values

    lon_start, lon_end = int(p["london_start_min"]), int(p["london_end_min"])
    ent_start, ent_end = int(p["entry_start_min"]), int(p["entry_end_min"])
    sweep_mult, stop_mult = float(p["sweep_atr_mult"]), float(p["stop_atr_mult"])
    max_hold = int(p["max_hold_bars"])

    signal = np.zeros(n, dtype=float)
    position, bars_held = 0, 0
    stop_level = target_level = np.nan
    traded_day = None

    london_hi = london_lo = np.nan
    lon_tp_vol_sum = lon_vol_sum = 0.0
    block_hi = block_lo = np.nan
    cur_day = day[0] if n else None

    for i in range(n):
        if day[i] != cur_day:
            cur_day = day[i]
            london_hi = london_lo = np.nan
            lon_tp_vol_sum = lon_vol_sum = 0.0

        m = minutes[i]

        # accumulate the London session range + VWAP inputs, causally
        if lon_start <= m < lon_end:
            london_hi = h[i] if np.isnan(london_hi) else max(london_hi, h[i])
            london_lo = l[i] if np.isnan(london_lo) else min(london_lo, l[i])
            tp = (h[i] + l[i] + c[i]) / 3.0
            lon_tp_vol_sum += tp * v[i]
            lon_vol_sum    += v[i]

        # track the CURRENT (still-forming) 15-minute block's range
        if m % 15 == 0:
            block_hi, block_lo = h[i], l[i]
        else:
            block_hi = h[i] if np.isnan(block_hi) else max(block_hi, h[i])
            block_lo = l[i] if np.isnan(block_lo) else min(block_lo, l[i])

        # ---- manage an existing position ----
        if position != 0:
            bars_held += 1
            hit_stop = (position == 1 and l[i] <= stop_level) or \
                       (position == -1 and h[i] >= stop_level)
            hit_target = (position == 1 and h[i] >= target_level) or \
                         (position == -1 and l[i] <= target_level)
            if hit_stop or hit_target or bars_held >= max_hold:
                position, bars_held = 0, 0
                stop_level = target_level = np.nan

        # ---- entry: only at the close of a completed 15m candle in-window ----
        is_block_close = ((m + 5) % 15 == 0)
        a = atr[i]
        if (position == 0 and traded_day != cur_day and is_block_close
                and ent_start <= m < ent_end and np.isfinite(a) and a > 0
                and np.isfinite(london_hi) and np.isfinite(london_lo)):
            vwap = (lon_tp_vol_sum / lon_vol_sum) if lon_vol_sum > 0 \
                else (london_hi + london_lo) / 2.0

            swept_up  = block_hi > london_hi + sweep_mult * a
            swept_dn  = block_lo < london_lo - sweep_mult * a
            failed_up = swept_up and c[i] <= london_hi   # wick out, 15m closed back in
            failed_dn = swept_dn and c[i] >= london_lo

            if failed_up and not failed_dn:
                position = -1
                stop_level, target_level = block_hi + stop_mult * a, vwap
                bars_held, traded_day = 0, cur_day
            elif failed_dn and not failed_up:
                position = 1
                stop_level, target_level = block_lo - stop_mult * a, vwap
                bars_held, traded_day = 0, cur_day

        signal[i] = position

    return (pd.Series(signal, index=idx, name="signal"),
            pd.Series(np.ones(n), index=idx, name="size_scalar"))


def sample_session_gap_params(rng) -> dict:
    """Random parameter set for walk-forward search."""
    return {
        "sweep_atr_mult": float(rng.choice([0.5, 0.75, 1.0, 1.25, 1.5])),
        "stop_atr_mult":  float(rng.choice([0.25, 0.5, 0.75, 1.0])),
        "atr_window":     int(rng.choice([10, 14, 20])),
        "max_hold_bars":  int(rng.choice([24, 36, 48, 60])),
    }


# ============================================================================
# EXPERIMENTAL: multi-timeframe ORB pullback
# ============================================================================
#
#   1. Opening range = the 08:30-08:45 ET 15-minute block's high/low.
#   2. Once formed, wait for a 5-minute bar to CLOSE completely beyond it --
#      that's the breakout confirmation, not the entry.
#   3. At confirmation, require the 20-period SMA's slope to agree with the
#      breakout direction (a flat SMA kills the setup for the day).
#   4. Arm a pending "limit order" back at the broken edge and wait for price
#      to pull back to it (fill = the entry, not the breakout bar itself).
#   5. Manage the fill with an ATR trailing stop + time stop (this is a trend
#      -following exit, matching the strategy's own "Trend Following" label).
#
# At most one FILL per day; a breakout in the opposite direction before fill
# re-arms the pending order to the new direction (the original call is
# treated as a fakeout).

ORB_PULLBACK_DEFAULT_PARAMS: dict = {
    "or_start_min":       8 * 60 + 30,  # 08:30 ET
    "or_end_min":         8 * 60 + 45,  # 08:45 ET
    "sma_period":         20,
    "slope_lookback":     3,      # bars back to measure the SMA slope over
    "slope_min_atr_frac": 0.05,   # SMA must move >= this * ATR over slope_lookback bars
    "max_wait_bars":      24,     # cancel the pending pullback order after this many bars
    "stop_atr_mult":      1.5,    # trailing stop distance once filled, in ATR
    "max_hold_bars":      36,
    "atr_window":         14,
}


def generate_orb_pullback_signal(
    df_5m: pd.DataFrame,
    params: dict | None = None,
    instrument_key: str = "",
) -> tuple[pd.Series, pd.Series]:
    p = {**ORB_PULLBACK_DEFAULT_PARAMS, **(params or {})}
    n = len(df_5m)
    idx = df_5m.index
    minutes = _minutes_et(idx)
    day = np.asarray(idx.tz_convert(config.SESSION.timezone).date)

    h = df_5m["high"].values
    l = df_5m["low"].values
    c = df_5m["close"].values

    atr = _structure_atr(df_5m, int(p["atr_window"])).values
    sma = df_5m["close"].rolling(int(p["sma_period"])).mean().values

    or_start, or_end = int(p["or_start_min"]), int(p["or_end_min"])
    slope_lb   = int(p["slope_lookback"])
    slope_frac = float(p["slope_min_atr_frac"])
    max_wait   = int(p["max_wait_bars"])
    stop_mult  = float(p["stop_atr_mult"])
    max_hold   = int(p["max_hold_bars"])

    signal = np.zeros(n, dtype=float)
    position, bars_held = 0, 0
    stop_level = np.nan
    extreme = np.nan

    or_hi = or_lo = np.nan
    pending_dir = 0          # 0 = none armed, +1/-1 = waiting for a pullback fill
    pending_level = np.nan
    bars_since_confirm = 0
    filled_today = False
    cur_day = day[0] if n else None

    for i in range(n):
        if day[i] != cur_day:
            cur_day = day[i]
            or_hi = or_lo = np.nan
            pending_dir = 0
            filled_today = False

        m = minutes[i]

        if or_start <= m < or_end:
            or_hi = h[i] if np.isnan(or_hi) else max(or_hi, h[i])
            or_lo = l[i] if np.isnan(or_lo) else min(or_lo, l[i])

        # ---- manage an existing position (ATR trailing stop + time stop) ----
        if position != 0:
            bars_held += 1
            a = atr[i] + 1e-8
            if position == 1:
                extreme = max(extreme, c[i])
                if c[i] < extreme - stop_mult * a or bars_held >= max_hold:
                    position, bars_held = 0, 0
                    stop_level = extreme = np.nan
            else:
                extreme = min(extreme, c[i])
                if c[i] > extreme + stop_mult * a or bars_held >= max_hold:
                    position, bars_held = 0, 0
                    stop_level = extreme = np.nan

        # ---- breakout confirmation (re)arms the pending pullback order ----
        if (position == 0 and not filled_today and m >= or_end
                and np.isfinite(or_hi) and np.isfinite(or_lo)
                and i >= slope_lb and np.isfinite(sma[i]) and np.isfinite(sma[i - slope_lb])):
            slope = sma[i] - sma[i - slope_lb]
            a = atr[i]
            slope_ok_long  = np.isfinite(a) and a > 0 and slope >  slope_frac * a
            slope_ok_short = np.isfinite(a) and a > 0 and slope < -slope_frac * a

            armed_this_bar = False
            if c[i] > or_hi and slope_ok_long and pending_dir != 1:
                pending_dir, pending_level, bars_since_confirm = 1, or_hi, 0
                armed_this_bar = True
            elif c[i] < or_lo and slope_ok_short and pending_dir != -1:
                pending_dir, pending_level, bars_since_confirm = -1, or_lo, 0
                armed_this_bar = True
            elif pending_dir != 0:
                bars_since_confirm += 1
                if bars_since_confirm > max_wait:
                    pending_dir = 0

            # ---- pullback fill (never on the bar the order was just armed --
            # that bar's own wick reaching back to the level isn't a genuine
            # subsequent pullback, just noise inside the confirming candle) ----
            if not armed_this_bar:
                if pending_dir == 1 and l[i] <= pending_level:
                    position, stop_level, extreme = 1, or_lo, c[i]
                    bars_held, pending_dir, filled_today = 0, 0, True
                elif pending_dir == -1 and h[i] >= pending_level:
                    position, stop_level, extreme = -1, or_hi, c[i]
                    bars_held, pending_dir, filled_today = 0, 0, True

        signal[i] = position

    return (pd.Series(signal, index=idx, name="signal"),
            pd.Series(np.ones(n), index=idx, name="size_scalar"))


def sample_orb_pullback_params(rng) -> dict:
    return {
        "sma_period":         int(rng.choice([10, 15, 20, 30])),
        "slope_lookback":     int(rng.choice([2, 3, 5])),
        "slope_min_atr_frac": float(rng.choice([0.0, 0.02, 0.05, 0.10])),
        "max_wait_bars":      int(rng.choice([12, 18, 24, 36])),
        "stop_atr_mult":      float(rng.choice([1.0, 1.5, 2.0, 2.5])),
        "max_hold_bars":      int(rng.choice([24, 36, 48, 60])),
    }


# ============================================================================
# EXPERIMENTAL: VWAP standard-deviation band mean-reversion
# ============================================================================
#
# Anchored intraday VWAP (session anchor 18:00 ET -- the electronic futures
# session open) plus volume-weighted standard-deviation bands, computed with
# the standard cumulative-moment formula (mean = cumsum(vol*tp)/cumsum(vol),
# var = cumsum(vol*tp^2)/cumsum(vol) - mean^2) so the bands only ever use
# same-session history up to and including the current bar -- no look-ahead.
#
# Fade a 2.5-sigma touch back toward 1-sigma, gated by RSI (>75 / <25 on
# whatever bar interval this is run on -- the original idea specified a
# 1-minute RSI; this repo's bars are 5-minute, so the RSI window is in 5m
# bars here, not literally 1-minute ticks). Hard stop just past 3-sigma.

VWAP_BANDS_DEFAULT_PARAMS: dict = {
    "session_anchor_hour_et": 18,    # 18:00 ET session anchor
    "entry_sd":     2.5,
    "target_sd":    1.0,
    "stop_sd":      3.0,
    "rsi_window":   14,
    "rsi_overbought": 75.0,
    "rsi_oversold":   25.0,
    "max_hold_bars":  48,
}


def _vwap_bands_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up   = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / (down + 1e-12)
    return 100 - 100 / (1 + rs)


def generate_vwap_bands_signal(
    df_5m: pd.DataFrame,
    params: dict | None = None,
    instrument_key: str = "",
) -> tuple[pd.Series, pd.Series]:
    p = {**VWAP_BANDS_DEFAULT_PARAMS, **(params or {})}
    idx = df_5m.index
    n = len(df_5m)

    h = df_5m["high"].values
    l = df_5m["low"].values
    c = df_5m["close"].values
    v = df_5m["volume"].astype(float)
    tp = (df_5m["high"] + df_5m["low"] + df_5m["close"]) / 3.0

    anchor_h = int(p["session_anchor_hour_et"])
    et = idx.tz_convert(config.SESSION.timezone)
    session_date = (et - pd.Timedelta(hours=anchor_h)).date

    vol_tp  = v * tp
    vol_tp2 = v * tp * tp
    cum_vol   = v.groupby(session_date).cumsum()
    cum_vtp   = vol_tp.groupby(session_date).cumsum()
    cum_vtp2  = vol_tp2.groupby(session_date).cumsum()

    vwap = (cum_vtp / cum_vol).values
    var  = (cum_vtp2 / cum_vol).values - vwap ** 2
    sd   = np.sqrt(np.clip(var, 0.0, None))

    rsi = _vwap_bands_rsi(df_5m["close"], int(p["rsi_window"])).values

    entry_sd, target_sd, stop_sd = float(p["entry_sd"]), float(p["target_sd"]), float(p["stop_sd"])
    rsi_ob, rsi_os = float(p["rsi_overbought"]), float(p["rsi_oversold"])
    max_hold = int(p["max_hold_bars"])

    signal = np.zeros(n, dtype=float)
    position, bars_held = 0, 0
    stop_level = target_level = np.nan

    for i in range(n):
        if not np.isfinite(vwap[i]) or not np.isfinite(sd[i]) or sd[i] <= 0:
            signal[i] = position
            continue

        if position != 0:
            bars_held += 1
            hit_stop = (position == 1 and l[i] <= stop_level) or \
                       (position == -1 and h[i] >= stop_level)
            hit_target = (position == 1 and h[i] >= target_level) or \
                         (position == -1 and l[i] <= target_level)
            if hit_stop or hit_target or bars_held >= max_hold:
                position, bars_held = 0, 0
                stop_level = target_level = np.nan

        if position == 0 and np.isfinite(rsi[i]):
            upper_entry, upper_target, upper_stop = (vwap[i] + entry_sd * sd[i],
                                                      vwap[i] + target_sd * sd[i],
                                                      vwap[i] + stop_sd * sd[i])
            lower_entry, lower_target, lower_stop = (vwap[i] - entry_sd * sd[i],
                                                      vwap[i] - target_sd * sd[i],
                                                      vwap[i] - stop_sd * sd[i])
            if c[i] >= upper_entry and rsi[i] > rsi_ob:
                position = -1
                stop_level, target_level = upper_stop, upper_target
                bars_held = 0
            elif c[i] <= lower_entry and rsi[i] < rsi_os:
                position = 1
                stop_level, target_level = lower_stop, lower_target
                bars_held = 0

        signal[i] = position

    return (pd.Series(signal, index=idx, name="signal"),
            pd.Series(np.ones(n), index=idx, name="size_scalar"))


def sample_vwap_bands_params(rng) -> dict:
    return {
        "entry_sd":       float(rng.choice([2.0, 2.25, 2.5, 2.75, 3.0])),
        "target_sd":      float(rng.choice([0.5, 1.0, 1.5])),
        "stop_sd":        float(rng.choice([3.0, 3.5, 4.0])),
        "rsi_window":     int(rng.choice([10, 14, 20])),
        "rsi_overbought": float(rng.choice([70.0, 75.0, 80.0])),
        "rsi_oversold":   float(rng.choice([20.0, 25.0, 30.0])),
        "max_hold_bars":  int(rng.choice([24, 36, 48, 60])),
    }


# ============================================================================
# EXPERIMENTAL: macro-catalyst news fade
# ============================================================================
#
# The original idea keys off NFP/CPI at 08:30 ET using a real economic
# calendar ("if the first 1-minute candle moves more than 3x the standard
# daily ATR"). This repo has no calendar data source, so instead of trying to
# know WHICH day is a release day, this uses the release's own signature as
# the trigger: an abnormally large 08:30 ET candle relative to the recent
# daily ATR. That catches real NFP/CPI days without a calendar, at the cost
# of also catching any other 08:30 candle that happens to be unusually large
# for an unrelated reason. Also adapted: the original spec measures a
# 1-minute candle against 3x daily ATR; this runs on 5-minute bars, so
# `spike_atr_mult` defaults much lower (a 5m candle covers 5x the time, so
# reaching the same multiple of daily ATR is a much higher bar) -- tune it
# for your instrument rather than trusting the default.

NEWS_FADE_DEFAULT_PARAMS: dict = {
    "release_min_et":   8 * 60 + 30,  # 08:30 ET -- NFP/CPI release time
    "spike_atr_mult":   1.5,          # 5m spike range >= this * daily ATR
    "daily_atr_window": 14,
    "fib_level":        0.618,
    "max_wait_bars":    3,            # bars after the spike to wait for the retrace fill
    "stop_atr_mult":    0.25,         # extra buffer beyond the spike extreme, in 5m ATR
    "max_hold_bars":    24,
}


def generate_news_fade_signal(
    df_5m: pd.DataFrame,
    params: dict | None = None,
    instrument_key: str = "",
) -> tuple[pd.Series, pd.Series]:
    p = {**NEWS_FADE_DEFAULT_PARAMS, **(params or {})}
    idx = df_5m.index
    n = len(df_5m)
    minutes = _minutes_et(idx)

    o = df_5m["open"].values
    h = df_5m["high"].values
    l = df_5m["low"].values
    c = df_5m["close"].values

    atr5 = _structure_atr(df_5m, 14).fillna(0.0).values

    df_d = _resample_daily(df_5m)
    daily_atr = _structure_atr(df_d, int(p["daily_atr_window"]))
    # shift(1): only the prior COMPLETED day's ATR is known intraday today
    daily_atr_5m = daily_atr.shift(1).reindex(idx, method="ffill").values

    release_min = int(p["release_min_et"])
    spike_mult  = float(p["spike_atr_mult"])
    fib         = float(p["fib_level"])
    max_wait    = int(p["max_wait_bars"])
    stop_mult   = float(p["stop_atr_mult"])
    max_hold    = int(p["max_hold_bars"])

    signal = np.zeros(n, dtype=float)
    position, bars_held = 0, 0
    stop_level = target_level = np.nan

    pending_dir = 0
    pending_level = np.nan
    pending_deadline = -1

    for i in range(n):
        if position != 0:
            bars_held += 1
            hit_stop = (position == 1 and l[i] <= stop_level) or \
                       (position == -1 and h[i] >= stop_level)
            hit_target = (position == 1 and h[i] >= target_level) or \
                         (position == -1 and l[i] <= target_level)
            if hit_stop or hit_target or bars_held >= max_hold:
                position, bars_held = 0, 0
                stop_level = target_level = np.nan

        # ---- detect the news candle exactly at the release minute ----
        armed_this_bar = False
        if minutes[i] == release_min and np.isfinite(daily_atr_5m[i]) and daily_atr_5m[i] > 0:
            bar_range = h[i] - l[i]
            if bar_range >= spike_mult * daily_atr_5m[i] and c[i] != o[i]:
                rng = h[i] - l[i]
                if c[i] > o[i]:      # bullish spike -> fade short toward the fib retrace
                    pending_dir   = -1
                    pending_level = h[i] - fib * rng
                    stop_level_arm = h[i] + stop_mult * atr5[i]
                else:                # bearish spike -> fade long toward the fib retrace
                    pending_dir   = 1
                    pending_level = l[i] + fib * rng
                    stop_level_arm = l[i] - stop_mult * atr5[i]
                pending_deadline = i + max_wait
                pending_target   = o[i]     # target: full reversion to the pre-spike level
                pending_stop     = stop_level_arm
                armed_this_bar   = True

        # ---- wait for the retrace fill (never on the spike bar itself -- the
        # fib level sits INSIDE that candle's own high/low by construction, so
        # its own wick would always trivially satisfy the fill check) ----
        if not armed_this_bar and position == 0 and pending_dir != 0:
            if i > pending_deadline:
                pending_dir = 0
            elif pending_dir == -1 and l[i] <= pending_level:
                position, stop_level, target_level = -1, pending_stop, pending_target
                bars_held, pending_dir = 0, 0
            elif pending_dir == 1 and h[i] >= pending_level:
                position, stop_level, target_level = 1, pending_stop, pending_target
                bars_held, pending_dir = 0, 0

        signal[i] = position

    return (pd.Series(signal, index=idx, name="signal"),
            pd.Series(np.ones(n), index=idx, name="size_scalar"))


def sample_news_fade_params(rng) -> dict:
    return {
        "spike_atr_mult":   float(rng.choice([0.75, 1.0, 1.25, 1.5, 2.0])),
        "daily_atr_window": int(rng.choice([10, 14, 20])),
        "fib_level":        float(rng.choice([0.5, 0.618, 0.786])),
        "max_wait_bars":    int(rng.choice([2, 3, 5, 8])),
        "stop_atr_mult":    float(rng.choice([0.0, 0.25, 0.5])),
        "max_hold_bars":    int(rng.choice([12, 24, 36])),
    }
