"""
TopstepX / ProjectX Gateway REST client -- the only module that talks to your
live broker account. Everything else in this repo (backtest, walk-forward,
strategies) never touches this file.

IMPORTANT -- VERIFY BEFORE GOING LIVE: this client is written against the
public ProjectX Gateway API shape (route names, auth flow, and request/response
fields below) as documented for TopstepX and other ProjectX-partner firms.
ProjectX Gateway is multi-tenant and per-tenant details can differ. Before
flipping config.TOPSTEP_API.dry_run to False:
  1. Confirm your gateway base URL (config.TOPSTEP_API.base_url /
     TOPSTEPX_GATEWAY_URL) is correct for your firm.
  2. Run a few `python run.py live --dry-run` cycles and read the printed
     "would place" lines against what you'd expect.
  3. If your firm gives you an evaluation/practice account, point
     TOPSTEPX_ACCOUNT_ID at it first.

Auth flow: POST /api/Auth/loginKey with {userName, apiKey} -> bearer token.
The token is short-lived; this client re-authenticates automatically.
"""
from __future__ import annotations
import time
from dataclasses import dataclass

import requests
import pandas as pd

import config


class TopstepXError(RuntimeError):
    pass


# Order type / side codes per the ProjectX Gateway API.
_ORDER_TYPE = {"limit": 1, "market": 2, "stop": 4}
_ORDER_SIDE = {"buy": 0, "sell": 1}

# Bar unit codes for /api/History/retrieveBars.
_BAR_UNIT = {"second": 1, "minute": 2, "hour": 3, "day": 4}


@dataclass
class ContractRef:
    contract_id: str
    name: str
    tick_size: float | None = None
    tick_value: float | None = None


class TopstepXClient:
    """Thin, synchronous REST wrapper. One instance per live-trading session."""

    def __init__(self, cfg: "config.TopstepAPI | None" = None):
        self.cfg = cfg or config.TOPSTEP_API
        if not self.cfg.api_key or not self.cfg.username:
            raise TopstepXError(
                "TOPSTEPX_API_KEY / TOPSTEPX_USERNAME are not set. Generate an "
                "API key from the TopstepX platform (Settings -> API Keys) and "
                "export TOPSTEPX_API_KEY / TOPSTEPX_USERNAME (see config.py)."
            )
        self._session = requests.Session()
        self._token: str | None = None
        self._token_at: float = 0.0

    # -- auth -----------------------------------------------------------

    def _url(self, path: str) -> str:
        return self.cfg.base_url.rstrip("/") + path

    def authenticate(self) -> None:
        resp = self._session.post(
            self._url("/api/Auth/loginKey"),
            json={"userName": self.cfg.username, "apiKey": self.cfg.api_key},
            timeout=15,
        )
        data = self._safe_json("/api/Auth/loginKey", resp)
        token = data.get("token")
        if not data.get("success", True) or not token:
            raise TopstepXError(f"authentication failed: {data.get('errorMessage') or data}")
        self._token = token
        self._token_at = time.time()

    def _headers(self) -> dict:
        # Re-authenticate every 20 minutes; ProjectX Gateway session tokens
        # are short-lived.
        if self._token is None or (time.time() - self._token_at) > 20 * 60:
            self.authenticate()
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    def _safe_json(self, path: str, resp: requests.Response) -> dict:
        try:
            data = resp.json()
        except ValueError:
            raise TopstepXError(
                f"{path}: non-JSON response (HTTP {resp.status_code}): {resp.text[:300]}"
            )
        if resp.status_code >= 400:
            raise TopstepXError(
                f"{path}: HTTP {resp.status_code}: {data.get('errorMessage') if isinstance(data, dict) else data}"
            )
        return data

    def _post(self, path: str, body: dict) -> dict:
        resp = self._session.post(self._url(path), json=body, headers=self._headers(), timeout=20)
        data = self._safe_json(path, resp)
        if isinstance(data, dict) and data.get("success") is False:
            raise TopstepXError(f"{path} failed: {data.get('errorMessage') or data}")
        return data

    # -- accounts ---------------------------------------------------------

    def search_accounts(self, only_active: bool = True) -> list[dict]:
        return self._post("/api/Account/search",
                          {"onlyActiveAccounts": only_active}).get("accounts", [])

    def resolve_account_id(self) -> int:
        """config.TOPSTEP_API.account_id if set, else the sole active account."""
        if self.cfg.account_id:
            return self.cfg.account_id
        accounts = self.search_accounts()
        if not accounts:
            raise TopstepXError("no active accounts returned by /api/Account/search")
        if len(accounts) > 1:
            print(f"  WARNING: {len(accounts)} active accounts found; using the first "
                  f"({accounts[0].get('id')}). Set TOPSTEPX_ACCOUNT_ID to pin one.")
        return accounts[0]["id"]

    def account_balance(self, account_id: int) -> float | None:
        for a in self.search_accounts():
            if a.get("id") == account_id:
                return a.get("balance")
        return None

    # -- contracts ----------------------------------------------------------

    def search_contracts(self, search_text: str, live: bool = True) -> list[dict]:
        return self._post("/api/Contract/search",
                          {"searchText": search_text, "live": live}).get("contracts", [])

    def resolve_contract(self, search_text: str, live: bool | None = None) -> tuple[ContractRef, bool]:
        """
        Find a contract by root symbol / search text.

        `live` selects the sim vs live data/contract subscription tier. If
        left as None, tries live=True first and falls back to live=False
        (eval / practice accounts are typically registered on the sim tier,
        so contracts only show up there). Returns (contract, live_used) so
        the caller can reuse the same tier for historical bars.
        """
        tiers = [live] if live is not None else [True, False]
        last_tier = tiers[-1]
        for tier in tiers:
            contracts = self.search_contracts(search_text, live=tier)
            active = [c for c in contracts if c.get("activeContract")]
            pool = active or contracts
            if pool:
                c = pool[0]
                return ContractRef(contract_id=c["id"], name=c.get("name", search_text),
                                   tick_size=c.get("tickSize"), tick_value=c.get("tickValue")), tier
            last_tier = tier
        raise TopstepXError(f"no contracts found for '{search_text}' "
                           f"(tried live={tiers}) -- check the symbol/root name "
                           f"and that your account has an active data subscription")

    # -- historical bars ------------------------------------------------

    # Trading hours per session day, used only to size the startTime lookback
    # window generously enough to cover `limit` bars (RTH ~6.5h/day).
    _RTH_HOURS_PER_DAY = 6.5
    _UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}

    def retrieve_bars(self, contract_id: str, unit_number: int = 5,
                      unit: str = "minute", limit: int = 1600,
                      live: bool = True) -> pd.DataFrame:
        """Returns an OHLCV DataFrame matching the shape used everywhere else
        in this repo (lowercase columns, UTC tz-aware index).

        retrieveBars requires an explicit [startTime, endTime] window (not
        just `limit`), so we size the window to comfortably cover `limit`
        bars of trading history, then keep only the most recent `limit` rows
        of whatever comes back.
        """
        end = pd.Timestamp.now('UTC')
        if unit == "day":
            calendar_days = limit
        else:
            bar_seconds = unit_number * self._UNIT_SECONDS[unit]
            bars_per_trading_day = max((self._RTH_HOURS_PER_DAY * 3600) / bar_seconds, 1.0)
            calendar_days = (limit / bars_per_trading_day) * 1.6  # weekend/holiday buffer
        start = end - pd.Timedelta(days=max(calendar_days, 1.0) + 3)

        body = {
            "contractId": contract_id, "live": live,
            "startTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endTime":   end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "unit": _BAR_UNIT[unit], "unitNumber": unit_number,
            "limit": limit, "includePartialBar": False,
        }
        data = self._post("/api/History/retrieveBars", body)
        bars = data.get("bars", [])
        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(bars).rename(columns={
            "t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
        })
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()[["open", "high", "low", "close", "volume"]]
        return df.astype(float).tail(limit)

    # -- orders -----------------------------------------------------------

    def place_order(self, account_id: int, contract_id: str, side: str, size: int,
                    order_type: str = "market", limit_price: float | None = None,
                    stop_price: float | None = None, custom_tag: str | None = None) -> dict:
        body = {
            "accountId": account_id, "contractId": contract_id,
            "type": _ORDER_TYPE[order_type], "side": _ORDER_SIDE[side], "size": int(size),
        }
        if limit_price is not None:
            body["limitPrice"] = float(limit_price)
        if stop_price is not None:
            body["stopPrice"] = float(stop_price)
        if custom_tag:
            body["customTag"] = custom_tag
        return self._post("/api/Order/place", body)

    def cancel_order(self, account_id: int, order_id: int) -> dict:
        return self._post("/api/Order/cancel", {"accountId": account_id, "orderId": order_id})

    def search_open_orders(self, account_id: int) -> list[dict]:
        return self._post("/api/Order/searchOpen", {"accountId": account_id}).get("orders", [])

    # -- positions --------------------------------------------------------

    def search_open_positions(self, account_id: int) -> list[dict]:
        return self._post("/api/Position/searchOpen",
                          {"accountId": account_id}).get("positions", [])

    def close_position(self, account_id: int, contract_id: str) -> dict:
        return self._post("/api/Position/closeContract",
                          {"accountId": account_id, "contractId": contract_id})

    def partial_close_position(self, account_id: int, contract_id: str, size: int) -> dict:
        return self._post("/api/Position/partialCloseContract",
                          {"accountId": account_id, "contractId": contract_id, "size": int(size)})
