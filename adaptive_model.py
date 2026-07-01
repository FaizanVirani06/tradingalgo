"""
Adaptive ML strategy — decay-weighted, regime-aware, self-refitting.

WHAT CHANGED FROM V1:
  - Replaced probability-threshold signal with LINEAR SCORE signal.
    Logistic regression with weak features collapses all probabilities to ~0.5,
    making the threshold gate useless. Instead we use the raw decision function
    (the linear combination of features before the sigmoid), z-score it over a
    rolling window, and trade when |z| exceeds a threshold in standard deviations.
    This is equivalent to asking "is the model's current reading unusually bullish
    or bearish relative to its recent history?" — which works even when absolute
    probabilities are uninformative.

  - Fixed NaN PnL bug: metrics now computed against first equity value, not
    starting_balance constant, so zero-trade runs don't divide by zero.

  - Regime scalar no longer collapses signal via round(). Instead it scales
    continuously: a 0.7 scalar means trade 70% size (0.7 contracts rounds to 1
    when n_contracts=1, but the logic is correct for multi-contract sizing).

Core ideas (unchanged):
  1. DECAY-WEIGHTED TRAINING: recent bars get exponentially more weight.
  2. EDGE DETECTOR: rolling Sharpe of bar PnL — flags degradation.
  3. REGIME FINGERPRINT: compares current market state to training state.
     Reduces size when regime is unfamiliar.
  4. ROLLING REFIT: refits on a schedule, always using decay-weighted history.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class AdaptiveConfig:
    # Decay half-life in bars (weight halves every N bars going back)
    # 5m bars: 1560 ≈ 20 trading days, 780 ≈ 10 days
    decay_halflife: int = 1560
    min_weight: float  = 0.05

    # Edge detector
    edge_window_bars: int         = 500    # ~6 trading days at 5m
    edge_degraded_threshold: float = -0.5  # rolling Sharpe below this → degraded

    # Regime fingerprint
    regime_window: int       = 200
    regime_max_distance: float = 0.6   # beyond this → scale to 0

    # Refit schedule
    refit_every_bars: int = 390   # ~1 trading day at 5m

    # Signal: z-score threshold in standard deviations
    # Trade when |score_z| > this. Lower = more trades, higher = fewer/stronger.
    zscore_threshold: float = 1.5
    min_hold_bars: int = 6

    # Features always included (anchors)
    anchor_features: list = field(default_factory=lambda: [
        "es_ret_1", "es_clv", "nq_ret_1", "gc_atr_norm", "dxy_chg_6"
    ])
    max_features: int  = 10
    C: float           = 1.0    # less regularisation so scores spread more
    min_train_samples: int = 500

    # Rolling window for z-score normalisation of scores
    score_zscore_window: int = 500


ADAPTIVE_CFG = AdaptiveConfig()


# --------------------------------------------------------------------------
# Decay weights
# --------------------------------------------------------------------------
def compute_decay_weights(n: int, halflife: int,
                          min_weight: float = 0.05) -> np.ndarray:
    bars_ago = np.arange(n - 1, -1, -1, dtype=float)
    w = np.exp(-np.log(2) * bars_ago / halflife)
    w = np.clip(w, min_weight, 1.0)
    return w / w.sum()


# --------------------------------------------------------------------------
# Regime fingerprint
# --------------------------------------------------------------------------
def compute_regime_fingerprint(X: np.ndarray, window: int) -> np.ndarray:
    recent = X[-window:]
    means  = np.nanmean(recent, axis=0)
    stds   = np.nanstd(recent, axis=0)
    stds   = np.where(stds == 0, 1.0, stds)
    return np.concatenate([means, stds])


def regime_distance(cur: np.ndarray, train: np.ndarray) -> float:
    norm = np.linalg.norm(train) + 1e-8
    dist = np.linalg.norm(cur - train) / norm
    return float(1 / (1 + np.exp(-5 * (dist - 0.5))))


def regime_size_scalar(distance: float, max_dist: float = 0.6) -> float:
    if distance >= max_dist:
        return 0.0
    return 1.0 - (distance / max_dist)


# --------------------------------------------------------------------------
# Edge detector
# --------------------------------------------------------------------------
class EdgeDetector:
    def __init__(self, window: int = 500, threshold: float = -0.5,
                 ann: int = 252 * 78):
        self.window    = window
        self.threshold = threshold
        self.ann       = ann
        self._buf      = []
        self.rolling_sharpe = 0.0
        self.degraded  = False

    def update(self, bar_pnl: float):
        self._buf.append(bar_pnl)
        if len(self._buf) > self.window:
            self._buf.pop(0)
        if len(self._buf) >= 50:
            r  = np.array(self._buf)
            sd = r.std()
            self.rolling_sharpe = float(r.mean() / sd * np.sqrt(self.ann)) if sd > 0 else 0.0
            self.degraded = self.rolling_sharpe < self.threshold

    def status(self) -> str:
        s = "DEGRADED" if self.degraded else "OK"
        return f"EdgeDetector [{s}] sharpe={self.rolling_sharpe:.2f}"


# --------------------------------------------------------------------------
# Adaptive model
# --------------------------------------------------------------------------
class AdaptiveModel:
    def __init__(self, feature_cols: list[str], cfg: AdaptiveConfig = ADAPTIVE_CFG):
        self.cfg               = cfg
        self.feature_cols      = feature_cols
        self.pipeline          = None
        self.feature_cols_fitted: list[str] = []
        self.train_fp          = None
        self.fitted            = False
        self.bars_since_refit  = 0
        self.n_fits            = 0
        # Rolling buffer of raw scores for z-score normalisation
        self._score_buf: list[float] = []

    def _select_features(self, cols_available: list[str]) -> list[str]:
        anchors = [f for f in self.cfg.anchor_features if f in cols_available]
        others  = [f for f in self.feature_cols
                   if f not in anchors and f in cols_available]
        return (anchors + others)[:self.cfg.max_features]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> bool:
        cols = self._select_features(list(X.columns))
        Xc   = X[cols].replace([np.inf, -np.inf], np.nan)
        mask = Xc.notna().all(axis=1) & y.notna() & (y != 0)
        Xc, yc = Xc[mask], y[mask]

        if len(Xc) < self.cfg.min_train_samples:
            return False
        yb = (yc > 0).astype(int)
        if yb.nunique() < 2:
            return False

        weights = compute_decay_weights(len(Xc), self.cfg.decay_halflife,
                                        self.cfg.min_weight)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(C=self.cfg.C, max_iter=1000,
                                          class_weight="balanced", random_state=0))
        ])
        pipe.fit(Xc.values, yb.values, clf__sample_weight=weights)

        self.pipeline          = pipe
        self.feature_cols_fitted = cols
        self.train_fp          = compute_regime_fingerprint(
            Xc.values, min(self.cfg.regime_window, len(Xc)))
        self.fitted            = True
        self.bars_since_refit  = 0
        self.n_fits           += 1
        # Reset score buffer on refit so z-score adapts to new model
        self._score_buf        = []
        return True

    def predict(self, X_row: pd.DataFrame) -> tuple[float, float]:
        """
        Returns (signal, regime_scalar) for a single bar (or small window).
        Signal is in {-1, 0, +1}, regime_scalar in [0, 1].

        Uses the LINEAR SCORE (decision function) z-scored over recent history,
        not the probability. This avoids the probability-collapse problem.
        """
        if not self.fitted:
            return 0.0, 0.0

        cols = [c for c in self.feature_cols_fitted if c in X_row.columns]
        Xc   = X_row[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Raw linear score: positive = model leans bullish, negative = bearish
        scaler = self.pipeline.named_steps["scaler"]
        clf    = self.pipeline.named_steps["clf"]
        Xs     = scaler.transform(Xc.values)
        score  = float((Xs @ clf.coef_[0])[0])   # single bar

        # Update rolling score buffer for z-score
        self._score_buf.append(score)
        if len(self._score_buf) > self.cfg.score_zscore_window:
            self._score_buf.pop(0)

        # Z-score the score relative to recent history
        if len(self._score_buf) >= 30:
            buf = np.array(self._score_buf)
            mu, sd = buf.mean(), buf.std()
            z = (score - mu) / (sd + 1e-8)
        else:
            z = 0.0   # not enough history yet

        # Signal based on z-score threshold
        thr = self.cfg.zscore_threshold
        if z > thr:
            sig = 1.0
        elif z < -thr:
            sig = -1.0
        else:
            sig = 0.0

        # Regime scalar
        regime_sc = 1.0
        if self.train_fp is not None and len(Xc) >= 5:
            cur_fp    = compute_regime_fingerprint(Xc.values,
                                                    min(self.cfg.regime_window, len(Xc)))
            dist      = regime_distance(cur_fp, self.train_fp)
            regime_sc = regime_size_scalar(dist, self.cfg.regime_max_distance)

        if regime_sc < 0.1:
            sig = 0.0

        return sig, regime_sc

    def should_refit(self) -> bool:
        return self.bars_since_refit >= self.cfg.refit_every_bars

    def tick(self):
        self.bars_since_refit += 1


# --------------------------------------------------------------------------
# Adaptive simulation
# --------------------------------------------------------------------------
#
# IMPORTANT ACCOUNTING NOTES (read before trusting any dollar figure):
#
#  1. SINGLE SOURCE OF TRUTH = the equity curve. Per-bar PnL is marked from the
#     position actually HELD across each bar. Trade statistics (win rate, profit
#     factor, avg win/loss) are RECONSTRUCTED from that same held-position path
#     afterwards, so they always reconcile with the equity. (The previous version
#     logged trades in an ad-hoc way that did NOT match the equity — its win-rate
#     and profit-factor numbers were unreliable.)
#
#  2. ECONOMIC-SCALE CAVEAT (still unfixed, by design): features and prices come
#     from an ETF proxy (e.g. QQQ ~ $555) but PnL is multiplied by the FUTURES
#     point value (NQ = $20/pt). A 1-point QQQ move is NOT a 1-point NQ move, so
#     the absolute dollar figures here are on the wrong scale. They are internally
#     consistent (good enough to compare configs against each other) but are NOT
#     a faithful dollar P&L for trading NQ. The correct fix is to compute PnL from
#     PROXY RETURNS x real-contract notional; ask for the returns-based rework when
#     you want true dollars. Until then, read Sharpe / win-rate / shape, not $.
#
#  3. NaN-robust: a non-finite bar move contributes zero PnL rather than poisoning
#     the whole equity curve (which is what produced the earlier $nan reports —
#     a single gap in the aligned proxy series turned every downstream bar to NaN).
#
def _reconstruct_trades(held_pos: np.ndarray, close: np.ndarray,
                        index, pv: float, regime: np.ndarray) -> "pd.DataFrame":
    """
    Rebuild the trade list from the realised held-position path so that the sum
    of trade PnL reconciles with the equity's gross PnL by construction.

    held_pos[i] = number of contracts held DURING bar i (signed).
    A trade is a maximal run of constant non-zero held_pos. PnL over the run
    [a..b] is (close[b] - close[a-1]) * sign * size, matching the per-bar marks.
    """
    rows = []
    n = len(held_pos)
    i = 0
    while i < n:
        v = held_pos[i]
        if v == 0:
            i += 1
            continue
        a = i
        while i + 1 < n and held_pos[i + 1] == v:
            i += 1
        b = i
        entry_ref = close[a - 1] if a > 0 else close[a]
        exit_ref  = close[b]
        if np.isfinite(entry_ref) and np.isfinite(exit_ref):
            pnl_pts = (exit_ref - entry_ref) * np.sign(v)
            rows.append({
                "entry_time": index[a], "exit_time": index[b],
                "side": "long" if v > 0 else "short",
                "contracts": abs(v),
                "entry": entry_ref, "exit": exit_ref,
                "pnl_pts": pnl_pts,
                "pnl_$": pnl_pts * pv * abs(v),
                "bars_held": b - a + 1,
                "regime_sc": regime[a],
            })
        i = b + 1
    return pd.DataFrame(rows)


def run_adaptive_backtest(features: pd.DataFrame,
                          labels: pd.Series,
                          target_df: pd.DataFrame,
                          instrument_key: str,
                          feature_pool: list[str],
                          cfg: AdaptiveConfig = ADAPTIVE_CFG,
                          n_contracts: int = 1,
                          warmup_bars: int = 3000,
                          verbose: bool = True) -> dict:
    """
    Continuous live-style simulation. The model only ever sees data up to the
    current bar, AND we embargo the most recent `horizon_bars` labels at each
    fit (those labels look into the future, so training on them leaks).
    """
    import config
    from backtest import ANNUALISATION, _flatten_mask

    inst      = config.INSTRUMENTS[instrument_key]
    pv        = inst.point_value
    tick_cost = inst.tick_size * inst.slippage_ticks * pv
    comm      = inst.commission_rt
    ann       = ANNUALISATION.get(config.INTERVAL, 252 * 78)
    embargo   = int(getattr(config.LABELS, "horizon_bars", 1))  # label look-ahead

    common = (features.index
                      .intersection(labels.index)
                      .intersection(target_df.index))
    X   = features.loc[common]
    y   = labels.loc[common]
    tdf = target_df.loc[common]
    n   = len(common)
    close = tdf["close"].values.astype(float)

    flat_mask = _flatten_mask(tdf.index)
    model     = AdaptiveModel(feature_pool, cfg)
    edge_det  = EdgeDetector(cfg.edge_window_bars, cfg.edge_degraded_threshold, ann)

    start_bal  = config.TOPSTEP.starting_balance
    equity     = np.empty(n); equity[0] = start_bal
    position   = 0.0          # contracts we will hold INTO the next bar
    bars_held  = 0
    sig_log    = np.zeros(n)
    regime_log = np.ones(n)
    held_log   = np.zeros(n)  # contracts held DURING each bar (for reconstruction)
    refit_log  = []
    running_pk = start_bal
    killed     = False
    max_pos    = min(n_contracts, config.TOPSTEP.max_contracts)

    def _fit_to(i: int) -> bool:
        # embargo the last `embargo` rows: their labels peek past bar i
        hi = max(0, i - embargo)
        if hi < cfg.min_train_samples:
            return False
        return model.fit(X.iloc[:hi], y.iloc[:hi])

    if verbose:
        print(f"  Adaptive sim: {n} bars | warmup={warmup_bars} "
              f"| refit_every={cfg.refit_every_bars} | zscore_thr={cfg.zscore_threshold} "
              f"| label_embargo={embargo}")

    for i in range(1, n):
        bar_time = tdf.index[i]

        # --- refit logic ---
        if i == warmup_bars:
            ok = _fit_to(i)
            if verbose:
                print(f"  bar {i:6d} | initial fit {'OK' if ok else 'FAILED'}")
            refit_log.append(i)
        elif i > warmup_bars and model.should_refit():
            was_deg = edge_det.degraded
            ok = _fit_to(i)
            if verbose:
                reason = "DEGRADED->refit" if was_deg else "scheduled"
                print(f"  bar {i:6d} ({bar_time.date()}) | {reason} "
                      f"| fit #{model.n_fits} | {edge_det.status()}")
            refit_log.append(i)

        model.tick()

        # --- signal ---
        if not model.fitted or i < warmup_bars:
            sig_val, regime_sc = 0.0, 1.0
        else:
            sig_val, regime_sc = model.predict(X.iloc[i:i + 1])

        if flat_mask[i] or killed:
            sig_val = 0.0
        sig_log[i]    = sig_val
        regime_log[i] = regime_sc

        # --- decide desired position (respect min hold) ---
        prev_pos = position
        held_log[i] = prev_pos          # position HELD across bar i
        if bars_held < cfg.min_hold_bars and position != 0:
            desired = position
        else:
            desired = sig_val * max_pos

        delta = abs(desired - prev_pos)
        change_cost = delta * (comm + tick_cost) if delta > 0 else 0.0
        if delta > 0:
            bars_held = 0
        else:
            bars_held += 1
        position = desired

        # --- mark PnL from the position held across i-1 -> i (NaN-safe) ---
        move = close[i] - close[i - 1]
        if not np.isfinite(move):
            move = 0.0
        bar_pnl   = prev_pos * move * pv - change_cost
        equity[i] = equity[i - 1] + bar_pnl
        edge_det.update(bar_pnl)

        # --- Topstep trailing-DD kill ---
        running_pk = max(running_pk, equity[i])
        liq = max(running_pk - config.TOPSTEP.trailing_drawdown,
                  start_bal  - config.TOPSTEP.trailing_drawdown)
        if not killed and equity[i] <= liq:
            killed = True
            position = 0.0
            if verbose:
                print(f"  bar {i:6d} | !! TOPSTEP DD BREACHED at {bar_time}")

    eq     = pd.Series(equity, index=tdf.index)
    trades = _reconstruct_trades(held_log, close, tdf.index, pv, regime_log)

    return {
        "equity":        eq,
        "trades":        trades,
        "signal":        pd.Series(sig_log,    index=tdf.index),
        "regime":        pd.Series(regime_log, index=tdf.index),
        "refit_bars":    refit_log,
        "n_refits":      model.n_fits,
        "killed":        killed,
        "edge_detector": edge_det,
    }
