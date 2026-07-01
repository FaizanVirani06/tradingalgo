# Quant Research System — NQ / ES / GC

A self-contained framework that ingests market data, discovers short-term leading
indicators, builds trading strategies, and **judges them honestly on out-of-sample
data** with Topstep account rules enforced. It also drives a live TopstepX /
ProjectX Gateway account with the strategy that survived that process.

This is a **research and validation tool, not a money printer.** Its single most
important job is to stop you from fooling yourself. A backtest that looks great
in-sample means almost nothing; this system reports out-of-sample performance and
how much of the in-sample edge *survived* (the overfitting tax). Many honest runs
on real data will come back **REJECTED** — that is the tool working, not failing.

---

## Layout

| File | Role |
|---|---|
| `config.py` | **Edit this.** Symbols, Topstep rules/plan table, contract economics, dynamic sizing, acceptance gate, TopstepX API config. |
| `data_clients.py` | EODHD (proxy OHLCV) + Databento (real order flow) historical data loaders. |
| `synthetic.py` | Offline data with a *small planted edge* — validates the engine with no API calls. |
| `research.py` | Feature engineering, label generation (forward return / triple barrier), and signal discovery (FDR multiple-testing check). |
| `strategies.py` | All four signal generators: ML (logit/forest), trend-following, reactive order-flow, and market structure (the production strategy). |
| `backtest.py` | Event-driven, no look-ahead, realistic costs, Topstep-aware engine + metrics + dynamic sizing math. |
| `topstep.py` | Trailing drawdown / daily loss / consistency rule simulation (used to grade backtests). |
| `walkforward.py` | The honest core: in-sample fit + selection, single out-of-sample evaluation, per strategy family. |
| `report.py` | Verdict, equity-curve plot, CSV/JSON outputs, acceptance gate. |
| `run.py` | CLI entry point — one subcommand per pipeline (see below). |
| `topstep_client.py` | REST client for the TopstepX / ProjectX Gateway API (auth, accounts, contracts, orders, positions, bars). |
| `live_trader.py` | The live trading loop: production strategy signal + dynamic sizing + Topstep rule kill-switch wired to `topstep_client`. |
| `adaptive_model.py` | Decay-weighted, regime-aware, self-refitting ML variant (used by `run.py adaptive`). |

**Production strategy:** market structure with the mtp exit fix (`strategies.STRUCTURE_DEFAULT_PARAMS["use_liquidity_targets"] = True`) plus dynamic micro→mini sizing (`config.DYNAMIC_SIZING.enabled = True`). This was chosen from a base/of/flat/mtp/flat_mtp/mixed/combo ablation run via `run.py structure-variants`.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in your real keys -- .env is gitignored
```

Required for research/backtesting: `EODHD_API_KEY`.
Required for live trading: `TOPSTEPX_API_KEY`, `TOPSTEPX_USERNAME` (see [config.py](config.py) for the full list and what each does).

> **Security:** `.env` is gitignored and must stay that way. If a real key or API
> secret is ever committed, pasted into a chat, or emailed, regenerate it — treat
> it like a password. Never put real credentials in `config.py` itself.

---

## Running it — research / backtesting

```bash
python run.py ml --source synthetic --quick --instruments NQ   # validate the engine offline, no API calls
python run.py ml --source eodhd --instruments NQ                # ML strategy on real EODHD data
python run.py trend --source eodhd --instruments NQ             # trend-following
python run.py structure --dbn-path <path>                       # market structure (Databento, real NQ prices)
python run.py structure-variants --instrument ES                # base vs mtp/flat/dyn ablation
python run.py adaptive --source eodhd --instruments NQ          # decay-weighted adaptive ML
python run.py reactive --dbn-path <path>                        # order-flow momentum (Databento only)
python run.py sizing --instrument NQ                            # dynamic sizing demo
python run.py smoketest <dbn_path>                               # Databento DBN loader smoke test
```

Add `--quick` to any pipeline for a faster, reduced search while iterating.
Outputs land in `output/`: `report_*.txt`, `equity_*.png`, `folds_*.csv`, `summary_*.json`.

### How to read the result

- **Out-of-sample Sharpe / PnL / drawdown** — the only numbers that matter. In-sample
  is shown only for the overfitting comparison.
- **Sharpe retention %** — out-of-sample Sharpe ÷ in-sample Sharpe. Low (or negative)
  means the strategy was mostly curve-fit.
- **Topstep check** — did the out-of-sample equity curve ever breach trailing drawdown,
  the daily loss limit, or the consistency rule.
- **Acceptance gate** — your standards (`config.AcceptanceGate`), applied to
  out-of-sample only. `ACCEPTED` means *worth paper-trading next*, not *deploy live*.

---

## Running it — live trading

```bash
python run.py live --instrument ES              # dry-run by default (prints, never orders)
python run.py live --instrument ES --live        # arms real order submission (asks for confirmation)
```

What it does each bar close: pulls live bars from TopstepX, runs the same
production structure-strategy signal used in backtests, sizes with the same
dynamic-sizing function the backtest was tuned with, and rebalances your
position via the broker API. Every cycle re-checks your **real account
balance** against Topstep's rules for your auto-detected plan tier
(`config.TOPSTEP_PLANS`) and force-flattens on a breach.

**Known gaps, read before leaving this unattended:**
- Stops are strategy-level (re-evaluated every bar close), not resting orders
  at the exchange. If the process isn't running, an open position has no
  protection.
- It only runs while the process is running: sleep, reboot, lost network, or
  a crash all stop it cold. Run it somewhere that stays up (see Deployment
  below), and don't let your workstation sleep while it's live.
- `config.TOPSTEP_PLANS`' 25K tier numbers (profit target, max contracts)
  could not be confirmed from Topstep's public docs — verify against your own
  dashboard before trusting them for anything beyond the (confirmed) trailing
  DD / daily loss kill-switch.

---

## Deployment (e.g. a DigitalOcean droplet)

```bash
git clone <your-repo-url> quant_system_pkg
cd quant_system_pkg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env      # fill in real keys, TOPSTEPX_DRY_RUN=true first
```

Run it under something that survives SSH disconnects and restarts on crash/reboot
(a bare `python run.py live` in your SSH session dies the moment you disconnect).
A `systemd` unit is the simplest reliable option on a droplet:

```ini
# /etc/systemd/system/quant-live.service
[Unit]
Description=Quant live trading (structure mtp+dyn)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/root/quant_system_pkg
EnvironmentFile=/root/quant_system_pkg/.env
ExecStart=/root/quant_system_pkg/.venv/bin/python run.py live --instrument ES
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now quant-live
journalctl -u quant-live -f      # tail the live output
```

Watch it in dry-run for a while over SSH (`journalctl -u quant-live -f`) before
flipping `TOPSTEPX_DRY_RUN=false` in `.env` and `systemctl restart quant-live`.

---

## Before you trust a single number — known limitations

1. **EODHD has no real order flow.** The `*_clv`, `*_buy_pressure`, `*_body_frac`
   features in `research.py` are **proxies** derived from OHLCV, not the tape.
   Real order-flow features come from the Databento path (`data_clients.py`).
2. **Proxy series vs futures economics.** EODHD features come from liquid ETF/spot
   proxies (QQQ, SPY, GLD) while PnL is computed in micro futures economics
   (MNQ/MES/MGC). Confirm `config.INSTRUMENTS` matches what you actually trade.
3. **Overfitting is the default outcome.** Walk-forward out-of-sample testing is a
   guardrail, not a guarantee. Resist the urge to keep tuning until something passes.
4. **Regime change.** A model trained on one market regime can fail in another.
5. **Backtested edge ≠ live edge.** Slippage, fills, latency, and execution gaps
   (see "Live trading" above) differ live. Paper trade any accepted strategy first.

This software is for research and education. It is not financial advice. Trading
futures involves substantial risk of loss.

---

## Extending it

- **Add real order flow:** new functions in `research.py::orderflow_features`.
- **Smarter strategy search:** replace the random parameter samplers in
  `walkforward.py` with a genetic algorithm or `optuna`.
- **More instruments:** add to `config.SYMBOLS`, `config.TRADABLES`, `config.INSTRUMENTS`.
- **Tighter standards:** edit `config.AcceptanceGate`.
- **Broker-side protective stops:** `topstep_client.place_order` already supports
  `order_type="stop"` — `live_trader.py` doesn't use it yet (see the "known gaps"
  note above).
