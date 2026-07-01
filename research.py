"""
Research pipeline -- feature engineering, label generation, and signal
discovery. These three steps always run back-to-back (build features, build
the prediction target, then rank candidate features against it) so they live
together.

  * FEATURES  -- the "atomic units" of the market. EODHD gives OHLCV only (no
    true order flow), so we derive the richest possible microstructure
    PROXIES from OHLCV, plus intermarket relationships and time-of-day
    structure. When real Databento order-flow columns are present (delta,
    cum_delta, buy_vol, sell_vol, trade_count, large_buy, large_sell),
    orderflow_features() adds genuine microstructure features on top. Every
    feature is causal (uses only past/current bar info, no .shift(-n)) so
    there is no look-ahead leakage into the predictors.

  * LABELS -- the prediction TARGET. Two label types: forward_return (simple
    signed return over the next N bars) and triple_barrier (profit-take / stop
    -loss at +/- k*ATR plus a time limit, mirroring how a real bracket order
    resolves). Labels are necessarily forward-looking -- but only ever used as
    the y in training, never as a feature.

  * DISCOVERY -- ranks features by their (in-sample) relationship to the
    target and contextualises that with a multiple-testing reality check: if
    you test 80 features, several will look predictive by pure chance.
    Descriptive only -- it does NOT decide the live strategy, the walk-forward
    optimizer does that on out-of-sample data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

import config

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# ============================================================================
# Features
# ============================================================================

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / (down + 1e-12)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def instrument_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Features computed from a single OHLCV frame."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    ret = np.log(c / c.shift(1))
    f = pd.DataFrame(index=df.index)

    # --- momentum / returns over several horizons ---
    for k in (1, 3, 6, 12, 24):
        f[f"{prefix}_ret_{k}"] = np.log(c / c.shift(k))
    f[f"{prefix}_rsi_14"] = _rsi(c, 14)
    ema_f, ema_s = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    f[f"{prefix}_macd"] = (ema_f - ema_s) / c

    # --- volatility & regime ---
    rv = ret.rolling(20).std()
    f[f"{prefix}_rvol_20"] = rv
    atr = _atr(df, 14)
    f[f"{prefix}_atr_norm"] = atr / c
    # Garman-Klass intrabar volatility (uses OHLC -> richer than close-only)
    gk = 0.5 * (np.log(h / l)) ** 2 - (2 * np.log(2) - 1) * (np.log(c / o)) ** 2
    f[f"{prefix}_gk_vol"] = np.sqrt(gk.clip(lower=0))
    f[f"{prefix}_vol_regime"] = (rv - rv.rolling(200).mean()) / (rv.rolling(200).std() + 1e-12)

    # --- volume features ---
    vol_z = (v - v.rolling(50).mean()) / (v.rolling(50).std() + 1e-12)
    f[f"{prefix}_vol_z"] = vol_z
    f[f"{prefix}_vol_roc"] = v.pct_change(6)

    # --- ORDER-FLOW PROXIES from OHLCV (the best you can do without tape) ---
    # Close Location Value: where in the bar's range did we close? +1 = top
    clv = ((c - l) - (h - c)) / ((h - l) + 1e-12)
    f[f"{prefix}_clv"] = clv
    # signed-volume pressure: CLV * volume, cumulated -> a delta-like proxy
    signed_vol = clv * v
    f[f"{prefix}_buy_pressure"] = signed_vol.rolling(12).sum() / (v.rolling(12).sum() + 1e-12)
    # body vs range: trend vs indecision
    f[f"{prefix}_body_frac"] = (c - o).abs() / ((h - l) + 1e-12)

    # --- mean-reversion / location ---
    ma = c.rolling(50).mean()
    f[f"{prefix}_dist_ma50"] = (c - ma) / (rv * c + 1e-12)
    # rolling VWAP distance
    tp = (h + l + c) / 3
    vwap = (tp * v).rolling(50).sum() / (v.rolling(50).sum() + 1e-12)
    f[f"{prefix}_vwap_dist"] = (c - vwap) / (atr + 1e-12)

    return f


# Columns that signal real order-flow data is available (Databento path).
_OF_COLS = frozenset({"delta", "cum_delta", "buy_vol", "sell_vol",
                      "trade_count", "large_buy", "large_sell"})


def orderflow_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Genuine microstructure features from bar-aggregated order flow.
    Gracefully no-ops if any required column is missing (EODHD path falls
    through to instrument_features only).

    All computations are strictly causal: only .rolling() / .diff() on past
    bars, never .shift(-n).
    """
    if not _OF_COLS.issubset(df.columns):
        return pd.DataFrame(index=df.index)

    f = pd.DataFrame(index=df.index)
    c           = df["close"]
    delta       = df["delta"].astype(float)
    cum_delta   = df["cum_delta"].astype(float)
    buy_vol     = df["buy_vol"].astype(float)
    sell_vol    = df["sell_vol"].astype(float)
    volume      = df["volume"].astype(float)
    trade_count = df["trade_count"].astype(float)
    large_buy   = df["large_buy"].astype(float)
    large_sell  = df["large_sell"].astype(float)

    # --- delta z-score (20 and 50 bar windows) ---
    for w in (20, 50):
        mu = delta.rolling(w).mean()
        sd = delta.rolling(w).std()
        f[f"{prefix}_delta_z{w}"] = (delta - mu) / (sd + 1e-8)

    # --- cumulative delta momentum: rate of change over 3/6/12 bars ---
    for k in (3, 6, 12):
        f[f"{prefix}_cdelta_roc{k}"] = cum_delta.diff(k)

    # --- delta divergence: price up but delta down = bearish (-1), vice-versa = bullish (+1) ---
    price_dir = np.sign(c.diff())
    delta_dir = np.sign(delta)
    # When they disagree the divergence has the sign of delta (counter-trend signal).
    # 0 when they agree (no divergence).
    f[f"{prefix}_delta_div"] = np.where(price_dir != delta_dir, delta_dir, 0.0)

    # --- buy/sell imbalance ratio, rolling 6 and 12 bars ---
    for w in (6, 12):
        bv_r = buy_vol.rolling(w).sum()
        sv_r = sell_vol.rolling(w).sum()
        f[f"{prefix}_bs_imbal_{w}"] = bv_r / (bv_r + sv_r + 1e-8)

    # --- large-print pressure, rolling 6 bars ---
    lb_r = large_buy.rolling(6).sum()
    ls_r = large_sell.rolling(6).sum()
    v_r  = volume.rolling(6).sum()
    f[f"{prefix}_large_pressure_6"] = (lb_r - ls_r) / (v_r + 1e-8)

    # --- trade intensity z-score (20 bars) ---
    mu_tc = trade_count.rolling(20).mean()
    sd_tc = trade_count.rolling(20).std()
    f[f"{prefix}_tcount_z20"] = (trade_count - mu_tc) / (sd_tc + 1e-8)

    # --- delta exhaustion: |delta| relative to its 20-bar absolute range ---
    delta_abs = delta.abs()
    delta_max = delta_abs.rolling(20).max()
    f[f"{prefix}_delta_exhaust_20"] = delta_abs / (delta_max + 1e-8)

    return f


def intermarket_features(closes: pd.DataFrame) -> pd.DataFrame:
    """
    `closes` is a DataFrame with one column per alias (NQ, ES, GC, VIX, ...).
    Builds spreads, ratios, lead-lag and correlations. These are where a lot of
    short-term edge actually lives.
    """
    f = pd.DataFrame(index=closes.index)
    rets = np.log(closes / closes.shift(1))

    if {"ES", "NQ"}.issubset(closes.columns):
        ratio = closes["NQ"] / closes["ES"]
        f["es_nq_ratio_z"] = (ratio - ratio.rolling(100).mean()) / (ratio.rolling(100).std() + 1e-12)
        # LEAD-LAG: lagged ES return as a predictor for NQ (and vice-versa)
        for lag in (1, 2, 3):
            f[f"es_ret_lag{lag}"] = rets["ES"].shift(lag)
            f[f"nq_ret_lag{lag}"] = rets["NQ"].shift(lag)
        # short-window correlation between ES & NQ returns (regime signal)
        f["es_nq_corr_50"] = rets["ES"].rolling(50).corr(rets["NQ"])

    if "VIX" in closes.columns:
        f["vix_chg_6"] = np.log(closes["VIX"] / closes["VIX"].shift(6))
        f["vix_level_z"] = ((closes["VIX"] - closes["VIX"].rolling(200).mean())
                            / (closes["VIX"].rolling(200).std() + 1e-12))
    if "DXY" in closes.columns:
        f["dxy_chg_6"] = np.log(closes["DXY"] / closes["DXY"].shift(6))
    if "BONDS" in closes.columns:
        f["bonds_chg_6"] = np.log(closes["BONDS"] / closes["BONDS"].shift(6))
    if {"GC", "DXY"}.issubset(closes.columns):
        f["gc_dxy_corr_50"] = rets["GC"].rolling(50).corr(rets["DXY"])

    return f


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    et = index.tz_convert(config.SESSION.timezone)
    minutes = np.asarray(et.hour) * 60 + np.asarray(et.minute)
    open_m = 9 * 60 + 30
    f = pd.DataFrame(index=index)
    # cyclical encoding of time-of-day
    frac = (minutes - open_m) / (16 * 60 - open_m)
    f["tod_sin"] = np.sin(2 * np.pi * frac)
    f["tod_cos"] = np.cos(2 * np.pi * frac)
    f["minutes_since_open"] = np.clip(minutes - open_m, 0, None)
    f["is_first_30m"] = (f["minutes_since_open"] <= 30).astype(float)
    f["is_last_30m"] = (minutes >= 16 * 60 - 30).astype(float)
    f["dow"] = np.asarray(et.dayofweek, dtype=float)
    return f


def build_feature_matrix(universe: dict[str, pd.DataFrame], target: str
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the full feature matrix for predicting `target` (e.g. "NQ").

    Returns (features, target_ohlcv) aligned on the target's bar index.
    All non-target series are reindexed onto the target grid (forward-fill within
    session) so intermarket features line up causally.
    """
    target_df = universe[target].copy()
    idx = target_df.index

    # align all closes onto the target grid
    closes = {}
    for alias, df in universe.items():
        closes[alias] = df["close"].reindex(idx).ffill(limit=3)
    closes = pd.DataFrame(closes)

    blocks = [time_features(idx), intermarket_features(closes)]

    # per-instrument features for the target + the other tradables
    feat_sources = set([target] + config.TRADABLES)
    for alias in feat_sources:
        if alias in universe:
            src = universe[alias].reindex(idx).ffill(limit=3)
            blocks.append(instrument_features(src, prefix=alias.lower()))

    # real order-flow features (Databento path only; no-ops gracefully on EODHD)
    of_block = orderflow_features(target_df, prefix=target.lower())
    if not of_block.empty:
        blocks.append(of_block)

    feats = pd.concat(blocks, axis=1)
    # replace infinities, drop the warmup region with NaNs
    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats, target_df


# ============================================================================
# Labels
# ============================================================================

def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return np.log(close.shift(-horizon) / close)


def triple_barrier(df: pd.DataFrame, horizon: int, tp_mult: float, sl_mult: float,
                   atr_window: int) -> pd.Series:
    """
    Returns a Series in {-1, 0, +1} aligned to df.index.

    For each bar i: look forward up to `horizon` bars. Using the bar's ATR to set
    barriers, +1 if the high crosses +tp first, -1 if the low crosses -sl first,
    else 0.
    """
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n = len(c)

    # ATR for barrier widths
    pc = np.empty(n); pc[0] = c[0]; pc[1:] = c[:-1]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    atr = pd.Series(tr).rolling(atr_window).mean().values

    out = np.zeros(n)
    for i in range(n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            out[i] = np.nan
            continue
        entry = c[i]
        up = entry + tp_mult * a
        dn = entry - sl_mult * a
        end = min(i + horizon, n - 1)
        label = 0
        for j in range(i + 1, end + 1):
            hit_up = h[j] >= up
            hit_dn = l[j] <= dn
            if hit_up and hit_dn:
                # ambiguous bar: assume the stop is touched first (conservative)
                label = -1
                break
            if hit_up:
                label = 1
                break
            if hit_dn:
                label = -1
                break
        out[i] = label
    return pd.Series(out, index=df.index)


def make_labels(target_df: pd.DataFrame, cfg: config.LabelConfig | None = None
                ) -> pd.DataFrame:
    cfg = cfg or config.LABELS
    c = target_df["close"]
    y = pd.DataFrame(index=target_df.index)
    y["fwd_ret"] = forward_return(c, cfg.horizon_bars)
    y["tb"] = triple_barrier(target_df, cfg.horizon_bars,
                             cfg.tp_atr_mult, cfg.sl_atr_mult, cfg.atr_window)
    # binary direction for classifiers (drop the 0s at train time)
    y["direction"] = np.sign(y["fwd_ret"])
    return y


# ============================================================================
# Discovery
# ============================================================================

def _clean(X: pd.DataFrame, y: pd.Series):
    data = X.copy()
    data["__y__"] = y
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data.drop(columns="__y__"), data["__y__"]


def rank_features(features: pd.DataFrame, target: pd.Series,
                  rf_sample: int = 20000) -> pd.DataFrame:
    """
    Returns a table ranking each feature by:
      * Spearman rho with the target (rank correlation, robust to outliers)
      * p-value of that correlation
      * mutual information (nonlinear dependence)
      * random-forest importance (multivariate, nonlinear)
    """
    X, y = _clean(features, target)
    rows = []
    for col in X.columns:
        rho, p = stats.spearmanr(X[col], y)
        rows.append({"feature": col, "spearman": rho, "p_value": p})
    table = pd.DataFrame(rows)

    if _HAS_SK and len(X) > 200:
        # subsample for speed
        if len(X) > rf_sample:
            samp = X.sample(rf_sample, random_state=0)
            ys = y.loc[samp.index]
        else:
            samp, ys = X, y
        mi = mutual_info_regression(samp.values, ys.values, random_state=0)
        table = table.merge(
            pd.DataFrame({"feature": X.columns, "mutual_info": mi}), on="feature")
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            n_jobs=-1, random_state=0)
        rf.fit(samp.values, ys.values)
        table = table.merge(
            pd.DataFrame({"feature": X.columns, "rf_importance": rf.feature_importances_}),
            on="feature")

    table["abs_spearman"] = table["spearman"].abs()
    table = table.sort_values("abs_spearman", ascending=False).reset_index(drop=True)
    return table


def multiple_testing_report(table: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    How many features look 'significant', vs how many you'd expect by chance.
    A big gap between observed and expected is mild evidence of real signal;
    a small gap means you're probably looking at noise.
    """
    n = len(table)
    observed = int((table["p_value"] < alpha).sum())
    expected_random = alpha * n
    # Benjamini-Hochberg FDR control
    pvals = table["p_value"].values
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    n_bh = int(np.max(np.where(passed)[0]) + 1) if passed.any() else 0
    return {
        "n_features_tested": n,
        "n_significant_raw": observed,
        "n_expected_by_chance": round(expected_random, 1),
        "n_significant_after_fdr": n_bh,
        "verdict": ("likely real signal" if n_bh > expected_random * 1.5
                    else "weak / possibly noise"),
    }


def top_feature_pool(table: pd.DataFrame, k: int = 25) -> list[str]:
    """The feature names the optimizer will be allowed to choose from."""
    return table["feature"].head(k).tolist()
