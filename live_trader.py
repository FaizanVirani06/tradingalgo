"""
Live trading loop -- wires the locked-in production strategy (market
structure, mtp exit fix + dynamic micro->mini sizing; see
strategies.STRUCTURE_DEFAULT_PARAMS and config.DYNAMIC_SIZING) to a real
TopstepX / ProjectX Gateway account via topstep_client.py.

SAFETY MODEL
------------
  * config.TOPSTEP_API.dry_run (default True) is the master arm/disarm switch.
    In dry-run mode every decision is computed and logged exactly as it would
    be live, but place_order / close_position / partial_close_position are
    never called. Flip TOPSTEPX_DRY_RUN=false (or `run.py live --live`) only
    after you've reviewed dry-run output and ideally validated on an
    evaluation/practice account.
  * Every loop iteration re-checks the Topstep trailing-drawdown and daily-loss
    rules against the ACTUAL broker-reported account balance (config.TOPSTEP
    -- set this to your real plan, see config.py). A breach flattens every
    open position and halts new entries for the rest of the process's life;
    it does not auto-resume. A day-loss breach halts new entries for the
    remainder of the trading day but does not force-flatten (mirrors the
    backtest engine's 80% soft-stop behaviour).
  * Position size is decided with the SAME function the backtest uses
    (backtest.choose_dynamic_units), so live sizing matches what was tuned.
  * Stops here are STRATEGY-LEVEL (checked every bar close), not resting
    stop orders at the exchange -- identical semantics to the backtest
    engine. If the process dies, any open position stays open at the broker
    with no protective order behind it; the loop re-flattens or re-manages it
    on restart based on strategy state, but a crash between bars is a real gap
    in protection. Consider this before leaving it unattended for long stretches.

STATE PERSISTENCE
------------------
A small JSON file (config.OUTPUT_DIR / "live_state.json") tracks the running
equity peak and day-start balance across restarts, since ProjectX Gateway
doesn't expose "peak since account open" directly.
"""
from __future__ import annotations
import json
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import config
import backtest as bt
import strategies as strat
from topstep_client import TopstepXClient, ContractRef, TopstepXError


_STATE_PATH = config.OUTPUT_DIR / "live_state.json"


@dataclass
class LiveState:
    running_peak: float
    day_start_balance: float
    current_day: str          # ISO date string, ET calendar day
    plan_starting_balance: float   # which plan tier this state was computed under
    killed: bool = False


def _fresh_state(starting_balance: float) -> LiveState:
    today = pd.Timestamp.now(tz=config.SESSION.timezone).date().isoformat()
    return LiveState(running_peak=starting_balance,
                     day_start_balance=starting_balance,
                     current_day=today, plan_starting_balance=starting_balance,
                     killed=False)


def _load_state(starting_balance: float) -> LiveState:
    if _STATE_PATH.exists():
        try:
            state = LiveState(**json.loads(_STATE_PATH.read_text()))
            if state.plan_starting_balance == starting_balance:
                return state
            print(f"  [live_trader] saved state was for a ${state.plan_starting_balance:,.0f} "
                  f"plan, detected ${starting_balance:,.0f} now -- resetting live_state.json")
        except Exception:
            pass
    return _fresh_state(starting_balance)


def _save_state(state: LiveState) -> None:
    _STATE_PATH.write_text(json.dumps(asdict(state), indent=2))


class LiveTrader:
    """One instance per (instrument, account). Call run_forever() to start."""

    def __init__(self, instrument_key: str, api_cfg: "config.TopstepAPI | None" = None):
        if instrument_key not in config.INSTRUMENTS:
            raise ValueError(f"unknown instrument_key '{instrument_key}' "
                            f"(expected one of {list(config.INSTRUMENTS)})")
        self.instrument_key = instrument_key
        self.micro = config.INSTRUMENTS[instrument_key]          # e.g. MNQ $2/pt
        self.mini  = config.DATABENTO_INSTRUMENTS[instrument_key]  # e.g. NQ $20/pt
        self.api_cfg = api_cfg or config.TOPSTEP_API
        self.client = TopstepXClient(self.api_cfg)

        self.account_id: int | None = None
        self.micro_contract: ContractRef | None = None
        self.mini_contract: ContractRef | None = None
        self.data_live: bool = True

        self.position_units = 0.0     # signed micro-equivalent units (our belief)
        self._last_bar_time: pd.Timestamp | None = None
        self.state: LiveState | None = None   # set in start(), once account size is known

    # -- setup ------------------------------------------------------------

    def start(self) -> None:
        mode = "DRY RUN (no orders will be sent)" if self.api_cfg.dry_run else "*** LIVE -- REAL ORDERS ***"
        print(f"  [live_trader] mode: {mode}")
        self.client.authenticate()
        self.account_id = self.client.resolve_account_id()
        bal = self.client.account_balance(self.account_id)
        if bal is None:
            raise TopstepXError("could not read account balance -- cannot detect plan size")
        print(f"  [live_trader] account {self.account_id}  balance=${bal:,.2f}")

        # Auto-detect the Topstep plan tier (trailing DD / daily loss / profit
        # target / max contracts) from the account balance -- see
        # config.TOPSTEP_PLANS / config.resolve_topstep_plan. This REPLACES
        # config.TOPSTEP for the lifetime of this process.
        plan = config.resolve_topstep_plan(bal)
        config.TOPSTEP = plan
        override = " (TOPSTEPX_ACCOUNT_SIZE override)" if config.TOPSTEPX_ACCOUNT_SIZE else ""
        print(f"  [live_trader] detected plan tier: ${plan.starting_balance:,.0f}{override}  "
              f"trailing_dd=${plan.trailing_drawdown:,.0f}  "
              f"daily_loss=${plan.daily_loss_limit:,.0f}  "
              f"max_contracts={plan.max_contracts}"
              + (f"  profit_target=${plan.profit_target:,.0f}" if plan.profit_target else
                 "  profit_target=UNCONFIRMED for this tier -- check your Topstep dashboard"))
        if abs(bal - plan.starting_balance) > 0.25 * plan.trailing_drawdown:
            print(f"  [live_trader] WARNING: balance ${bal:,.2f} is more than 25% of the "
                  f"tier's trailing DD away from ${plan.starting_balance:,.0f} -- double-check "
                  f"the detected tier is right (set TOPSTEPX_ACCOUNT_SIZE to override).")
        self.state = _load_state(plan.starting_balance)

        self.micro_contract, micro_live = self.client.resolve_contract(self.micro.name)
        self.mini_contract,  mini_live  = self.client.resolve_contract(self.mini.name)
        # Bars must come from whichever data tier actually has contracts
        # registered for this account (sim/eval vs live/funded).
        self.data_live = micro_live and mini_live
        print(f"  [live_trader] {self.instrument_key}: micro={self.micro_contract.name}"
              f"({self.micro_contract.contract_id}, live={micro_live})  "
              f"mini={self.mini_contract.name}({self.mini_contract.contract_id}, live={mini_live})")

        self.position_units = self._broker_position_units()
        print(f"  [live_trader] starting position: {self.position_units:+.0f} micro-equiv units")

    # -- broker state reconciliation --------------------------------------

    def _broker_position_units(self) -> float:
        """Signed micro-equivalent units currently held, per the broker."""
        positions = self.client.search_open_positions(self.account_id)
        units = 0.0
        for p in positions:
            size = float(p.get("size", 0))
            sign = 1.0 if int(p.get("type", 1)) == 1 else -1.0  # 1=Long, 2=Short
            if p.get("contractId") == self.micro_contract.contract_id:
                units += sign * size
            elif p.get("contractId") == self.mini_contract.contract_id:
                units += sign * size * 10
        return units

    def _account_balance(self) -> float:
        bal = self.client.account_balance(self.account_id)
        if bal is None:
            raise TopstepXError("could not read account balance from /api/Account/search")
        return float(bal)

    # -- kill-switch: Topstep rules checked against the REAL account -------

    def _check_rules(self, balance: float) -> None:
        rules = config.TOPSTEP
        today = pd.Timestamp.now(tz=config.SESSION.timezone).date().isoformat()
        if today != self.state.current_day:
            self.state.current_day = today
            self.state.day_start_balance = balance

        self.state.running_peak = max(self.state.running_peak,
                                      min(balance, rules.starting_balance + rules.trailing_drawdown))
        liq = max(self.state.running_peak - rules.trailing_drawdown,
                 rules.starting_balance - rules.trailing_drawdown)

        if not self.state.killed and balance <= liq:
            print(f"  [live_trader] !!! TRAILING DRAWDOWN BREACHED: "
                  f"balance ${balance:,.2f} <= liq ${liq:,.2f} -- FLATTENING AND HALTING !!!")
            self.state.killed = True
            self._flatten_all()

        day_loss = max(0.0, self.state.day_start_balance - balance)
        soft_stop = (0.8 * rules.daily_loss_limit
                    if rules.daily_loss_limit is not None else None)
        day_halted = soft_stop is not None and day_loss >= soft_stop
        if day_halted:
            print(f"  [live_trader] daily soft-stop hit (day loss ${day_loss:,.0f} >= "
                  f"${soft_stop:,.0f}) -- no new entries until the next trading day")
        self._day_halted = day_halted
        _save_state(self.state)

    # -- order execution ----------------------------------------------------

    def _leg(self, units: float) -> tuple[ContractRef, int]:
        u = int(abs(round(units)))
        if u <= 9:
            return self.micro_contract, u
        return self.mini_contract, u // 10

    def _flatten_all(self) -> None:
        for c in (self.micro_contract, self.mini_contract):
            if c is None:
                continue
            if self.api_cfg.dry_run:
                print(f"  [dry-run] would close_position({c.name})")
            else:
                try:
                    self.client.close_position(self.account_id, c.contract_id)
                except TopstepXError as e:
                    print(f"  [live_trader] !! close_position({c.name}) failed: {e}")
        self.position_units = 0.0

    def _rebalance(self, desired_units: float, ref_px: float, tag: str) -> None:
        current = self.position_units
        if desired_units == current:
            return

        # Close the existing leg first if we're flattening or flipping direction.
        if current != 0 and (desired_units == 0 or np.sign(desired_units) != np.sign(current)):
            c, _ = self._leg(current)
            if self.api_cfg.dry_run:
                print(f"  [dry-run] would close_position({c.name})  "
                      f"(was {current:+.0f} units @ ~{ref_px:.2f})")
            else:
                self.client.close_position(self.account_id, c.contract_id)

        # Open the new leg.
        if desired_units != 0 and (current == 0 or np.sign(desired_units) != np.sign(current)):
            c, size = self._leg(desired_units)
            side = "buy" if desired_units > 0 else "sell"
            if self.api_cfg.dry_run:
                print(f"  [dry-run] would place_order(side={side}, size={size}, "
                      f"contract={c.name}, tag={tag})  ~{ref_px:.2f}")
            else:
                self.client.place_order(self.account_id, c.contract_id, side=side,
                                        size=size, order_type="market", custom_tag=tag)

        self.position_units = desired_units

    # -- signal -> desired position ------------------------------------------

    def _desired_units(self, dbg: dict, balance: float) -> float:
        sig  = float(dbg["signal"].iloc[-1])
        stop = float(dbg["stop"].iloc[-1])
        if sig == 0:
            return 0.0
        if np.sign(self.position_units) == sig and self.position_units != 0:
            return self.position_units  # hold existing size

        if not config.DYNAMIC_SIZING.enabled:
            return sig * float(min(1, config.TOPSTEP.max_contracts))

        rules = config.TOPSTEP
        if config.DYNAMIC_SIZING.anchor == "profit":
            cushion = max(balance - rules.starting_balance, 0.0)
        else:
            liq_now = max(self.state.running_peak - rules.trailing_drawdown,
                         rules.starting_balance - rules.trailing_drawdown)
            cushion = balance - liq_now

        remaining_daily = None
        if rules.daily_loss_limit is not None:
            day_loss = max(0.0, self.state.day_start_balance - balance)
            remaining_daily = rules.daily_loss_limit - day_loss

        entry_px = float(dbg["_close_ref"])
        units = bt.choose_dynamic_units(config.DYNAMIC_SIZING, rules, self.micro.point_value,
                                        cushion, entry_px, stop, remaining_daily)
        return sig * units

    # -- one bar-close cycle ------------------------------------------------

    def process_bar(self, bars_5m: pd.DataFrame) -> None:
        dbg = strat.generate_structure_signal_debug(bars_5m, None, self.instrument_key)
        dbg["_close_ref"] = float(bars_5m["close"].iloc[-1])

        self.position_units = self._broker_position_units()
        balance = self._account_balance()
        self._check_rules(balance)

        if self.state.killed:
            return  # already flattened; no new entries until manually restarted

        desired = self._desired_units(dbg, balance)
        if getattr(self, "_day_halted", False) and desired != 0 and self.position_units == 0:
            print("  [live_trader] day-halted: skipping new entry")
            return

        if desired != self.position_units:
            tag = f"structure-mtp-dyn-{self.instrument_key}"
            print(f"  [live_trader] {bars_5m.index[-1]}  signal={dbg['signal'].iloc[-1]:+.0f}  "
                  f"stop={dbg['stop'].iloc[-1]:.2f}  {self.position_units:+.0f} -> {desired:+.0f} units")
            self._rebalance(desired, dbg["_close_ref"], tag)

    # -- main loop ----------------------------------------------------------

    def run_forever(self) -> None:
        self.start()
        print(f"  [live_trader] polling every {self.api_cfg.poll_interval_sec}s "
              f"for a new closed {config.INTERVAL} bar ...")
        while True:
            try:
                bars = self.client.retrieve_bars(
                    self.mini_contract.contract_id,
                    unit_number=int(config.INTERVAL.rstrip("mh")),
                    unit="minute" if config.INTERVAL.endswith("m") else "hour",
                    limit=self.api_cfg.warmup_bars, live=self.data_live,
                )
                if len(bars) and bars.index[-1] != self._last_bar_time:
                    self._last_bar_time = bars.index[-1]
                    self.process_bar(bars)
            except TopstepXError as e:
                print(f"  [live_trader] !! API error: {e}")
            except Exception:
                print("  [live_trader] !! unexpected error in loop:")
                traceback.print_exc()
            time.sleep(self.api_cfg.poll_interval_sec)
