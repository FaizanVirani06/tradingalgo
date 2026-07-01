"""
Central configuration for the quant research system.

Everything that you might want to tune lives here so the rest of the code
stays clean. READ THE TOPSTEP SECTION CAREFULLY and set it to YOUR actual
plan -- the defaults are illustrative, not gospel.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()   # loads .env next to this file, if present -- see .env.example
except ImportError:
    pass

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"          # raw API responses cached here
OUTPUT_DIR = ROOT / "output"        # reports, equity curves, plots
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# EODHD API
# --------------------------------------------------------------------------
# The key is read from the environment. NEVER hardcode it, never commit it.
#   export EODHD_API_KEY=your_key          (mac/linux)
#   setx   EODHD_API_KEY your_key          (windows)
# or put it in a .env file next to this project (copy .env.example -> .env;
# .env is gitignored).
EODHD_API_KEY = os.environ.get("EODHD_API_KEY")
EODHD_BASE_URL = "https://eodhd.com/api"

# EODHD symbols. Futures symbology on EODHD varies; these are sensible
# defaults using liquid proxies you DEFINITELY have access to on the $100
# plan. Swap the continuous-future tickers in once you confirm them in your
# account (the client will tell you if a symbol 404s).
#
# Primary tradables:
#   NQ  -> Nasdaq-100   (proxy: QQQ.US intraday, or NDX index)
#   ES  -> S&P 500      (proxy: SPY.US intraday, or GSPC.INDX)
#   GC  -> Gold         (XAUUSD.FOREX spot)
# Intermarket context (predictors only, never traded):
#   VIX -> VIX.INDX or VIXY.US
#   DXY -> dollar index (proxy: UUP.US or EUR via EURUSD.FOREX inverted)
#   BONDS -> TLT.US (20y treasuries) as a rates proxy
SYMBOLS = {
    "NQ":    "QQQ.US",
    "ES":    "SPY.US",
    "GC":    "GLD.US",
    "VIX":   "VIXY.US",
    "DXY":   "UUP.US",
    "BONDS": "TLT.US",
}

# Which of the above you actually want to *trade* (generate signals for).
# The others are context features only.
TRADABLES = ["NQ", "ES", "GC"]

# Intraday bar interval EODHD supports: "1m" or "5m".
INTERVAL = "5m"

# How much history to pull (EODHD 1m starts ~Oct 2020, varies by ticker).
HISTORY_DAYS = 365 * 2


# --------------------------------------------------------------------------
# TopstepX / ProjectX Gateway API  (LIVE EXECUTION -- real money, real orders)
# --------------------------------------------------------------------------
# Credentials are read from the environment. NEVER hardcode them, never commit
# them, never put them in a place that gets logged.
#   export TOPSTEPX_API_KEY=your_api_key         (mac/linux)
#   setx   TOPSTEPX_API_KEY your_api_key         (windows)
#   export TOPSTEPX_USERNAME=your_login
#   export TOPSTEPX_ACCOUNT_ID=12345             (optional -- auto-resolved if unset)
# or put them in a .env file next to this project (copy .env.example -> .env;
# .env is gitignored -- never commit it). The API key is generated from the
# TopstepX platform (Settings -> API Keys) -- it is NOT your account password.
#
# `dry_run` is the master arm/disarm switch for live_trader.py. It defaults to
# True: the live loop will authenticate, resolve the account/contract, poll
# bars, compute the mtp+dyn signal, and print every order it WOULD place --
# but nothing is sent to the account until you deliberately set it False (env
# var TOPSTEPX_DRY_RUN=false, or pass --live on the `run.py live` CLI). Flip it
# only after you've watched a few dry-run cycles and ideally validated against
# a Topstep evaluation/practice account first.
@dataclass
class TopstepAPI:
    api_key: str | None = os.environ.get("TOPSTEPX_API_KEY")
    username: str | None = os.environ.get("TOPSTEPX_USERNAME")
    account_id: int | None = (
        int(os.environ["TOPSTEPX_ACCOUNT_ID"])
        if os.environ.get("TOPSTEPX_ACCOUNT_ID") else None
    )
    # Gateway host: ProjectX Gateway is multi-tenant (each prop firm gets its
    # own subdomain). Override via TOPSTEPX_GATEWAY_URL if yours differs.
    base_url: str = os.environ.get("TOPSTEPX_GATEWAY_URL", "https://api.topstepx.com")
    # Master safety switch -- see docstring above. TOPSTEPX_DRY_RUN=false to arm.
    dry_run: bool = os.environ.get("TOPSTEPX_DRY_RUN", "true").strip().lower() not in (
        "false", "0", "no",
    )
    # How often the live loop checks for a new closed bar (seconds). Should be
    # small relative to the bar interval (5m bars -> poll every 15-30s) so a
    # new bar is acted on promptly without hammering the REST API.
    poll_interval_sec: int = 20
    # Bars of history to request on each refresh so rolling indicators (daily
    # EMA20, 500-bar ATR percentile, etc.) are warm -- mirrors
    # walkforward._STRUCTURE_LOOKBACK_WARM.
    warmup_bars: int = 1600

TOPSTEP_API = TopstepAPI()


# --------------------------------------------------------------------------
# Instrument economics  (point value = $ per 1.0 move in the FUTURE you trade)
# --------------------------------------------------------------------------
# IMPORTANT: features/signals are derived from the EODHD proxy series, but
# PnL is computed in the economics of the actual futures contract you trade
# on Topstep. Set `point_value`, `tick_size`, and `commission_rt` to the
# contract you really trade. Below are MICRO contracts (sane for a Topstep
# 50K account). For full-size, multiply point values by 10.
@dataclass
class Instrument:
    name: str
    point_value: float      # $ per 1.0 index point
    tick_size: float        # minimum price increment (index points)
    commission_rt: float    # round-trip commission per contract ($)
    slippage_ticks: float   # assumed slippage per side, in ticks

INSTRUMENTS = {
    # Micro E-mini Nasdaq (MNQ): $2 / point, tick 0.25  -- DO NOT change to 20.0 (that is full NQ)
    "NQ": Instrument("MNQ", point_value=2.0, tick_size=0.25, commission_rt=1.40, slippage_ticks=1.0),
    # Micro E-mini S&P (MES): $5 / point, tick 0.25  -- DO NOT change to 50.0 (that is full ES)
    "ES": Instrument("MES", point_value=5.0, tick_size=0.25, commission_rt=1.40, slippage_ticks=1.0),
    # Micro Gold (MGC): $10 / point, tick 0.10  -- DO NOT change to 100.0 (that is full GC)
    "GC": Instrument("MGC", point_value=10.0, tick_size=0.10, commission_rt=1.60, slippage_ticks=1.0),
}

# Instrument economics for the Databento path (real CME futures prices).
# Prices are ~20,000 NQ / ~6,000 ES / ~4,500 GC so point values are larger.
DATABENTO_INSTRUMENTS = {
    # E-mini Nasdaq (NQ): $20 / point, tick 0.25  -- DO NOT change to 2.0 (that is MNQ)
    "NQ": Instrument("NQ", point_value=20.0, tick_size=0.25, commission_rt=2.80, slippage_ticks=1.0),
    # E-mini S&P 500 (ES): $50 / point, tick 0.25  -- DO NOT change to 5.0 (that is MES)
    "ES": Instrument("ES", point_value=50.0, tick_size=0.25, commission_rt=2.80, slippage_ticks=1.0),
    # Gold (GC): $100 / point, tick 0.10  -- DO NOT change to 10.0 (that is MGC)
    "GC": Instrument("GC", point_value=100.0, tick_size=0.10, commission_rt=3.20, slippage_ticks=1.0),
}


def get_instruments(source: str) -> dict:
    """Return the correct instrument economics dict for the given data source."""
    return DATABENTO_INSTRUMENTS if source == "databento" else INSTRUMENTS


# --------------------------------------------------------------------------
# Topstep account rules  -- SET THESE TO YOUR REAL PLAN
# --------------------------------------------------------------------------
# These defaults approximate a Topstep 50K account. Topstep changes terms and
# has multiple account sizes; CONFIRM yours. The backtest enforces these and
# will mark a run FAILED if it would have breached.
@dataclass
class TopstepRules:
    starting_balance: float = 50_000.0
    # Trailing max drawdown: the account is killed if equity falls this far
    # below its running PEAK (intraday peak by default).
    trailing_drawdown: float = 2_000.0
    trailing_intraday: bool = True       # True = peak tracked intraday; False = EOD only
    # Daily loss limit (combine-style). Set None to disable.
    daily_loss_limit: float | None = 1_000.0
    # Profit target to "pass" (combine). Set None if you're already funded.
    profit_target: float | None = 3_000.0
    # Max contracts held at once (per instrument).
    max_contracts: int = 5
    # Consistency rule: no single day may exceed this fraction of total profit.
    # Topstep payout consistency is often ~50% (0.5). Set None to disable.
    consistency_max_day_fraction: float | None = 0.5

TOPSTEP = TopstepRules()


# --------------------------------------------------------------------------
# Topstep account-size plan table  (for live_trader.py auto-detection)
# --------------------------------------------------------------------------
# Topstep's rule NUMBERS (trailing DD / daily loss / profit target) are not
# exposed anywhere in the ProjectX Gateway API -- they're Topstep's own layer
# on top of the generic gateway, and the account-search response only gives
# you `balance`. So live_trader.py detects account size by matching the
# CURRENT balance to the nearest known plan tier below and loads that tier's
# rules into config.TOPSTEP at startup (see live_trader.LiveTrader.start()).
#
# Figures below are Topstep's published Trading Combine / Express Funded
# numbers as of mid-2026 ("Daily Loss Limit is exactly half the Max Loss
# Limit on every account" per Topstep's help center). The 25K tier's profit
# target and max_contracts could NOT be confirmed from public docs (Topstep's
# official parameter table currently only lists 50K/100K/150K) -- profit
# target is left None (it only affects combine pass/fail bookkeeping, not the
# live kill-switch) and max_contracts is set conservatively low. VERIFY BOTH
# against your own Topstep dashboard (Account -> Rules) before relying on them.
TOPSTEP_PLANS: dict[float, TopstepRules] = {
    25_000.0: TopstepRules(
        starting_balance=25_000.0, trailing_drawdown=1_000.0, trailing_intraday=True,
        daily_loss_limit=500.0, profit_target=2_000.0, max_contracts=2,
        consistency_max_day_fraction=0.5,
    ),
    50_000.0: TopstepRules(
        starting_balance=50_000.0, trailing_drawdown=2_000.0, trailing_intraday=True,
        daily_loss_limit=1_000.0, profit_target=3_000.0, max_contracts=5,
        consistency_max_day_fraction=0.5,
    ),
    100_000.0: TopstepRules(
        starting_balance=100_000.0, trailing_drawdown=3_000.0, trailing_intraday=True,
        daily_loss_limit=2_000.0, profit_target=6_000.0, max_contracts=10,
        consistency_max_day_fraction=0.5,
    ),
    150_000.0: TopstepRules(
        starting_balance=150_000.0, trailing_drawdown=4_500.0, trailing_intraday=True,
        daily_loss_limit=3_000.0, profit_target=9_000.0, max_contracts=15,
        consistency_max_day_fraction=0.5,
    ),
}

# Optional explicit override -- set TOPSTEPX_ACCOUNT_SIZE=25000 to skip
# balance-based nearest-tier detection entirely (recommended once you know
# your real tier, since balance drifts with P&L and could land ambiguously
# between two tiers e.g. right after a big trade).
TOPSTEPX_ACCOUNT_SIZE: float | None = (
    float(os.environ["TOPSTEPX_ACCOUNT_SIZE"])
    if os.environ.get("TOPSTEPX_ACCOUNT_SIZE") else None
)


def resolve_topstep_plan(current_balance: float) -> TopstepRules:
    """Nearest known Topstep plan tier to `current_balance` (or the explicit
    TOPSTEPX_ACCOUNT_SIZE override, if set)."""
    target = TOPSTEPX_ACCOUNT_SIZE if TOPSTEPX_ACCOUNT_SIZE is not None else current_balance
    nearest = min(TOPSTEP_PLANS, key=lambda size: abs(size - target))
    return TOPSTEP_PLANS[nearest]


# --------------------------------------------------------------------------
# Dynamic position sizing  (risk-based, generic micro -> mini ladder)
# --------------------------------------------------------------------------
# Size scales with the *cushion* above the live trailing-drawdown floor
# (cushion = equity - liquidation_level). Near the floor we trade the minimum
# (1 micro); as realised profit lifts the cushion we scale up the ladder.
#
# The ladder is expressed in MICRO-EQUIVALENT UNITS, where 1 mini = 10 micros
# (the mini's point value is 10x the micro's: MNQ $2 -> NQ $20, MES $5 -> ES
# $50, MGC $10 -> GC $100). Units 1..9 are micros; 10 = 1 mini; 20 = 2 minis.
# So the ladder reproduces "1..9 MNQ, then 1 NQ, then 2 NQ" for any instrument.
#
# At each entry we pick the LARGEST rung whose worst-case stop-out
# (stop_distance_points * micro_point_value * units) stays within
# `safety_fraction` of the current cushion. This keeps a single full stop-out
# from breaching the $2k trailing drawdown while letting size grow with profit.
@dataclass
class DynamicSizing:
    # LOCKED ON as the production default: mtp (multi-TP ratchet exit, see
    # strategies.STRUCTURE_DEFAULT_PARAMS) + dynamic sizing was the winning
    # combo out of the base/of/flat/mtp/flat_mtp/mixed/combo ablation run via
    # `python run.py structure-variants`.
    enabled: bool = True
    safety_fraction: float = 0.5          # risk <= this fraction of the cushion
    ladder_units: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20)  # micro-equiv units
    min_units: int = 1                    # never size below this (1 micro)
    # How the "cushion" (deployable room) is measured:
    #   "floor"  -> equity - liquidation_floor  (room before you are killed;
    #               ~$2k from day one, so size can be > 1 micro immediately when
    #               stops are tight). This is the literal "distance from max DD".
    #   "profit" -> max(equity - starting_balance, 0)  (banked profit only; size
    #               starts at 1 micro and grows ONLY once real profit is made).
    # Default "profit": matches "start at 1 MNQ, scale once profits are made,
    # keep DD <= $2k". ("floor" deploys the full $2k starting room from day one
    # and can blow the account on early losses — use only deliberately.)
    anchor: str = "profit"
    # ---- hard caps (the anchor alone does NOT bound drawdown / daily loss) ----
    # The profit anchor lets per-trade risk grow without bound as profit banks,
    # so a single sized loss can give back a big chunk (observed -$3,599 on base)
    # and blow a daily limit (-$1,436). These two caps make ANY config safe to
    # size. Set either to None to disable.
    #   max_trade_risk_frac: a single trade's worst-case stop-out may not exceed
    #       this fraction of the trailing drawdown (0.2 * $2k = $400/trade),
    #       REGARDLESS of how much profit has banked.
    #   daily_risk_frac: a single trade's worst-case may not exceed this fraction
    #       of the REMAINING daily-loss budget (so a day can't be blown in one or
    #       a few sized trades; pairs with the engine's 80% daily soft-stop).
    max_trade_risk_frac: float | None = 0.2
    daily_risk_frac: float | None = 0.8

DYNAMIC_SIZING = DynamicSizing()


# --------------------------------------------------------------------------
# Trading session (Eastern). Restrict to regular hours to avoid thin-liquidity
# noise; set rth_only=False to trade the full session.
# --------------------------------------------------------------------------
@dataclass
class Session:
    rth_only: bool = True
    rth_start: str = "09:30"
    rth_end: str = "16:00"
    timezone: str = "America/New_York"
    # Flatten everything this many minutes before the close (no overnight risk).
    flatten_before_close_min: int = 5

SESSION = Session()


# --------------------------------------------------------------------------
# Walk-forward optimization settings
# --------------------------------------------------------------------------
@dataclass
class WalkForward:
    train_bars: int = 5000          # in-sample window length (bars)
    test_bars: int = 1000           # out-of-sample window length (bars)
    step_bars: int = 1000           # how far to slide each fold (== test_bars -> non-overlapping OOS)
    n_param_samples: int = 80       # random parameter sets tried per fold
    min_trades_in_sample: int = 10  # reject params that barely trade
    random_seed: int = 7

WALKFORWARD = WalkForward()


# --------------------------------------------------------------------------
# Labels / prediction target
# --------------------------------------------------------------------------
@dataclass
class LabelConfig:
    horizon_bars: int = 24           # predict return over the next N bars
    # Triple-barrier: profit-take / stop in units of recent volatility (ATR-like)
    tp_atr_mult: float = 1.5
    sl_atr_mult: float = 1.0
    atr_window: int = 20

LABELS = LabelConfig()


# --------------------------------------------------------------------------
# "Meets your standards" acceptance gate -- applied to OUT-OF-SAMPLE results ONLY
# --------------------------------------------------------------------------
# A strategy is only declared acceptable if its aggregated walk-forward
# out-of-sample performance clears ALL of these. Tighten/loosen to taste, but
# do NOT apply these to in-sample numbers -- that defeats the purpose.
@dataclass
class AcceptanceGate:
    min_sharpe: float = 1.0
    max_drawdown_pct: float = 0.30          # of starting balance
    min_profit_factor: float = 1.1
    min_trades: int = 100
    min_pct_oos_folds_profitable: float = 0.52
    must_pass_topstep: bool = True

GATE = AcceptanceGate()
