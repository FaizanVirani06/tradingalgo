"""
Data clients -- everything that touches the network or raw vendor files.

Two independent sources, both producing the same shape of output
({alias: OHLCV DataFrame}, UTC-indexed) so downstream code never cares which
one it's looking at:

  * EODHD     -- intraday OHLCV via REST API (proxy tickers, e.g. QQQ for NQ).
                 Network is only touched here; every response is cached to
                 disk so you never burn the same API call twice.
  * Databento -- real CME futures order flow from `trades` schema DBN files.
                 Aggregates raw trades into bars carrying OHLCV plus genuine
                 microstructure columns (delta, cum_delta, buy_vol/sell_vol,
                 trade_count, large_buy/large_sell).
"""
from __future__ import annotations
import time
import hashlib
import re
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List

import requests
import numpy as np
import pandas as pd

import config

try:
    import databento as db
    _HAS_DB = True
except Exception:
    _HAS_DB = False


# ============================================================================
# EODHD (proxy intraday OHLCV via REST)
# ============================================================================

class EODHDError(RuntimeError):
    pass


def _cache_path(symbol: str, interval: str, start: str, end: str) -> Path:
    key = f"{symbol}_{interval}_{start}_{end}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    safe = symbol.replace(".", "_").replace("/", "_")
    return config.CACHE_DIR / f"{safe}_{interval}_{h}.pkl"


def _interval_to_eodhd(interval: str) -> str:
    return {"1m": "1m", "5m": "5m", "1h": "1h"}.get(interval, interval)


def fetch_intraday(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch intraday bars from EODHD, chunking into <=500-day windows to stay
    under the 600-day per-request limit. Results cached to disk.
    """
    cache = _cache_path(symbol, interval, start, end)
    if cache.exists():
        print(f"    (cached)")
        return pd.read_pickle(cache)

    api_key = config.EODHD_API_KEY
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt   = pd.Timestamp(end,   tz="UTC")

    chunk_days = 500      # safely under the 600-day API cap
    chunks = []
    cursor = start_dt

    while cursor < end_dt:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end_dt)
        from_ts = int(cursor.timestamp())
        to_ts   = int(chunk_end.timestamp())

        url = (
            f"https://eodhd.com/api/intraday/{symbol}"
            f"?interval={interval}&from={from_ts}&to={to_ts}"
            f"&api_token={api_key}&fmt=json"
        )
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            raise EODHDError(
                f"[{symbol}] HTTP {resp.status_code}: {resp.text[:200]}"
                "  (is this symbol on your plan? try a proxy from config.SYMBOLS)"
            )

        data = resp.json()
        if isinstance(data, list) and data:
            chunks.append(pd.DataFrame(data))
        elif isinstance(data, dict) and "error" in data:
            raise EODHDError(f"[{symbol}] API error: {data}")

        cursor = chunk_end
        time.sleep(0.5)   # be polite to the API

    if not chunks:
        raise EODHDError(f"[{symbol}] No data returned for {start} -> {end}")

    df = (pd.concat(chunks)
            .drop_duplicates("timestamp")
            .sort_values("timestamp"))

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = (df.set_index("datetime")
            .rename(columns={
                "open": "open", "high": "high",
                "low": "low", "close": "close", "volume": "volume"
            })[["open", "high", "low", "close", "volume"]])
    df = df[~df.index.duplicated(keep="first")]

    df.to_pickle(cache)
    return df


def load_universe(symbols: Dict[str, str] | None = None,
                  interval: str | None = None,
                  history_days: int | None = None) -> Dict[str, pd.DataFrame]:
    """
    Pull every configured symbol and return {alias: ohlcv_df}.
    Aliases are the keys of config.SYMBOLS (NQ, ES, GC, VIX, ...).
    """
    symbols      = symbols      or config.SYMBOLS
    interval     = interval     or config.INTERVAL
    history_days = history_days or config.HISTORY_DAYS

    end   = pd.Timestamp.now(tz="UTC").normalize()
    start = end - pd.Timedelta(days=history_days)
    start_s = start.strftime("%Y-%m-%d")
    end_s   = end.strftime("%Y-%m-%d")

    out: Dict[str, pd.DataFrame] = {}
    for alias, ticker in symbols.items():
        print(f"  fetching {alias:6s} ({ticker}) ...")
        out[alias] = fetch_intraday(ticker, interval, start_s, end_s)
    return out


# ============================================================================
# Databento (real order-flow ingestion from `trades` schema DBN files)
# ============================================================================
#
# INPUT can be any of:
#   * a single .dbn / .dbn.zst file
#   * a directory containing many daily .dbn(.zst) files
#   * a directory containing per-day .zip files (each holding a DBN)  <-- batch layout
#   * the top-level batch .zip itself
#
# CRITICAL FILTERING:
#   * action == "T" (trades only)
#   * OUTRIGHT contracts only (drop calendar spreads like "GCG6-GCJ6")
#   * front month chosen per root as the highest-volume outright contract,
#     re-evaluated per day then stitched into a continuous series
#
# Output matches the EODHD path: {alias: dataframe} with lowercase OHLCV
# columns plus order-flow columns, indexed by UTC bar timestamp. Cached to
# pickle.

ROOTS = {"NQ": "NQ", "ES": "ES", "GC": "GC"}
_OUTRIGHT_RE = re.compile(r"^[A-Z]{2}[FGHJKMNQUVXZ]\d$")
CACHE_VERSION = 2


def _is_outright(sym: str) -> bool:
    return bool(_OUTRIGHT_RE.match(str(sym)))


def _root_of(sym: str) -> str | None:
    m = re.match(r"^([A-Z]{2})[FGHJKMNQUVXZ]\d$", str(sym))
    return m.group(1) if m else None


def _gather_dbn_files(root: str | Path, workdir: Path) -> List[Path]:
    """
    Return a list of readable .dbn/.dbn.zst paths, extracting zips as needed
    into `workdir`. Handles: single file, dir of dbn, dir of per-day zips,
    or a single top-level batch zip.
    """
    root = Path(root)
    dbns: List[Path] = []

    def _extract_zip(zp: Path):
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if name.endswith((".dbn", ".dbn.zst")):
                    target = workdir / Path(name).name
                    with zf.open(name) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    dbns.append(target)
                elif name.endswith(".zip"):
                    inner = workdir / Path(name).name
                    with zf.open(name) as src, open(inner, "wb") as dst:
                        dst.write(src.read())
                    _extract_zip(inner)

    if root.is_file():
        if root.suffix == ".zip":
            _extract_zip(root)
        else:
            dbns.append(root)
    else:  # directory
        for p in sorted(root.rglob("*")):
            if p.suffix == ".zip":
                _extract_zip(p)
            elif p.name.endswith((".dbn", ".dbn.zst")):
                dbns.append(p)

    # de-dup, keep stable order by filename (which sorts by date for Databento)
    seen, uniq = set(), []
    for p in sorted(dbns, key=lambda x: x.name):
        if p.name not in seen:
            seen.add(p.name); uniq.append(p)
    return uniq


def load_trades_dbn(path: str | Path) -> pd.DataFrame:
    """Read ONE trades DBN file into a clean DataFrame of OUTRIGHT trades."""
    if not _HAS_DB:
        raise RuntimeError("pip install databento")
    store = db.DBNStore.from_file(str(path))
    df = store.to_df()

    ts = pd.to_datetime(df["ts_event"], utc=True) if "ts_event" in df.columns \
        else pd.to_datetime(df.index, utc=True)
    # Keep tz-aware timestamps so downstream session conversion works.
    df = df.assign(ts=ts)

    if "action" in df.columns:
        df = df[df["action"] == "T"]
    df = df[df["symbol"].map(_is_outright)]
    df["root"] = df["symbol"].map(_root_of)
    df = df[df["root"].isin(ROOTS.values())]
    size = pd.to_numeric(df["size"], errors="coerce").astype("float64")
    df["size"] = size
    df["signed_size"] = np.where(df["side"] == "B", size, -size)
    return df[["ts", "root", "symbol", "price", "size", "side", "signed_size"]]


def _front_month_per_day(d: pd.DataFrame) -> pd.DataFrame:
    """
    For one root: choose, for each calendar day, the highest-volume outright
    symbol (the active front month), and keep only that symbol's trades.
    Stitches across the roll automatically.
    """
    d = d.copy()
    d["day"] = d["ts"].dt.tz_convert(config.SESSION.timezone).dt.date
    # most-traded symbol per day
    vol = d.groupby(["day", "symbol"])["size"].sum().reset_index()
    front = vol.sort_values("size").groupby("day").tail(1)[["day", "symbol"]]
    front = front.rename(columns={"symbol": "front"})
    d = d.merge(front, on="day", how="left")
    return d[d["symbol"] == d["front"]].drop(columns=["front"])


def aggregate_orderflow(df: pd.DataFrame, interval: str = "5min",
                        large_pct: float = 0.90) -> Dict[str, pd.DataFrame]:
    """Resample clean trades into bars with OHLCV + order-flow columns, per root."""
    out: Dict[str, pd.DataFrame] = {}
    rule = {"1m": "1min", "5m": "5min", "1h": "1h"}.get(interval, interval)

    for alias, root in ROOTS.items():
        d = df[df["root"] == root]
        if d.empty:
            continue
        d = _front_month_per_day(d).set_index("ts").sort_index()

        large_thr = d["size"].quantile(large_pct)
        d = d.assign(
            buy_vol=np.where(d["side"] == "B", d["size"], 0),
            sell_vol=np.where(d["side"] == "A", d["size"], 0),
            large_buy=np.where((d["side"] == "B") & (d["size"] >= large_thr), d["size"], 0),
            large_sell=np.where((d["side"] == "A") & (d["size"] >= large_thr), d["size"], 0),
        )
        g = d.resample(rule)
        bars = pd.DataFrame({
            "open":  g["price"].first(), "high": g["price"].max(),
            "low":   g["price"].min(),   "close": g["price"].last(),
            "volume": g["size"].sum(),
            "buy_vol": g["buy_vol"].sum(), "sell_vol": g["sell_vol"].sum(),
            "delta": g["signed_size"].sum(), "trade_count": g["price"].count(),
            "large_buy": g["large_buy"].sum(), "large_sell": g["large_sell"].sum(),
        }).dropna(subset=["close"])

        day = bars.index.tz_convert(config.SESSION.timezone).date
        bars["cum_delta"] = bars.groupby(day)["delta"].cumsum()
        out[alias] = bars

    return out


def load_universe_databento(path: str | Path,
                            interval: str | None = None) -> Dict[str, pd.DataFrame]:
    """
    Entry point: a file/dir/zip of daily trades DBNs -> {alias: bar df}, cached.
    """
    interval = interval or config.INTERVAL
    key = hashlib.md5(str(Path(path).resolve()).encode()).hexdigest()[:8]
    cache = config.CACHE_DIR / (
        f"databento_v{CACHE_VERSION}_{Path(path).stem}_{interval}_{key}.pkl"
    )
    if cache.exists():
        print("    (cached databento bars)")
        return pd.read_pickle(cache)

    with tempfile.TemporaryDirectory() as tmp:
        files = _gather_dbn_files(path, Path(tmp))
        if not files:
            raise RuntimeError(f"No .dbn files found under {path}")
        print(f"  found {len(files)} daily DBN file(s)")

        frames = []
        for i, f in enumerate(files, 1):
            try:
                frames.append(load_trades_dbn(f))
            except Exception as e:
                print(f"    !! skipped {f.name}: {e}")
            if i % 10 == 0 or i == len(files):
                print(f"    read {i}/{len(files)} files ...")

    if not frames:
        raise RuntimeError(
            "No DBN files were successfully parsed. "
            "Install the databento package and verify your files are trades-schema DBN(.zst)."
        )

    alltrades = pd.concat(frames, ignore_index=True).sort_values("ts")
    print(f"  {len(alltrades):,} clean outright trades total")
    bars = aggregate_orderflow(alltrades, interval)
    for alias, b in bars.items():
        print(f"    {alias}: {len(b)} bars  "
              f"{b.index.min().date()} -> {b.index.max().date()}")
    pd.to_pickle(bars, cache)
    return bars
