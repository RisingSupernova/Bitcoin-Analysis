#!/usr/bin/env python3
"""
Bitcoin Analyst Daily Report - GitHub Actions Version
Fetches market data from 12 API sources, runs multi-agent analysis via Claude,
and emails an HTML report with BTC allocation recommendation.

Data sources:
  1. FRED DFII10  - 10Y real yields
  2. FRED DTWEXBGS - Trade-weighted dollar index
  3. FRED VIXCLS  - VIX volatility index
  4. CoinGecko    - BTC price, market cap, ATH, supply
  5. Fear & Greed - Crypto sentiment index (30-day)
  6. Mempool      - Hash rate (1 month)
  7. Mempool      - Difficulty adjustment
  8. CFTC COT     - Bitcoin futures positioning (Financial Futures)
  9. Blockchain   - Hash rate backup (30 days)
 10. SoSoValue   - Spot BTC ETF flows (daily + historical)
 11. CryptoCompare - BTC OHLCV daily candles (365 bars, for technical indicators)
 12. CryptoCompare - BTC OHLCV 4-hour candles (200 bars, for technical indicators)

Secrets required (set in GitHub repo settings):
  ANTHROPIC_API_KEY  - Claude API key
  FRED_API_KEY       - FRED (St. Louis Fed) API key
  GMAIL_ADDRESS      - Gmail address to send from/to
  GMAIL_APP_PASSWORD - Gmail App Password (not your normal password)
"""

import json
import math
import os
import re
import smtplib
import sys
import time
import traceback
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import quote

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
CLAUDE_MODEL = "claude-sonnet-4-6"
PROMPT_TEMPLATE = Path(__file__).parent / "prompt_template.txt"
DATA_DIR = Path(__file__).parent / "data"
PREV_REC_FILE = DATA_DIR / "previous_recommendation.json"
HISTORY_FILE = DATA_DIR / "history.json"
MAX_HISTORY_DAYS = 45  # ~6 weeks. BTC Fear&Greed already gives 30d lookback;
                       # 45d of price/macro/mining trends is optimal for
                       # trend analysis without burning >1600 tokens.


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def http_get(url: str, headers: dict | None = None, timeout: int = 30) -> dict | str:
    """Simple HTTP GET. Returns parsed JSON (dict/list) or raw text (str)."""
    req = Request(url, headers=headers or {})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
    except (URLError, HTTPError) as e:
        print(f"  ⚠ HTTP error for {url[:80]}: {e}")
        return {}


def http_post(url: str, body: dict, headers: dict | None = None, timeout: int = 30) -> dict | str:
    """Simple HTTP POST with JSON body. Returns parsed JSON or raw text."""
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "User-Agent": "BTC-Analyst/1.0"}
    if headers:
        hdrs.update(headers)
    req = Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
    except (URLError, HTTPError) as e:
        print(f"  ⚠ HTTP error for {url[:80]}: {e}")
        return {}


def fetch_fred(series_id: str) -> dict:
    """Fetch latest FRED observation, skipping missing '.' values."""
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={FRED_API_KEY}"
        f"&file_type=json&sort_order=desc&limit=5"
    )
    data = http_get(url)
    if isinstance(data, dict) and "observations" in data:
        for obs in data["observations"]:
            if obs.get("value") not in (None, ".", ""):
                return obs
    return {}


# ---------------------------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------------------------
def fetch_all_data() -> dict:
    """Fetch all 9 data sources, returning structured dict."""
    print("📡 Fetching data from 12 sources...")

    results = {}

    # 1-3: FRED macro data
    for name, series in [("dfii10", "DFII10"), ("dollar", "DTWEXBGS"), ("vix", "VIXCLS")]:
        print(f"  Fetching FRED {series}...")
        results[name] = fetch_fred(series)
        status = "OK" if results[name].get("value") else "MISSING"
        val = results[name].get("value", "N/A")
        print(f"    → {status}: {val}")

    # 4: CoinGecko BTC
    print("  Fetching CoinGecko BTC...")
    cg_url = (
        "https://api.coingecko.com/api/v3/coins/bitcoin"
        "?localization=false&tickers=false&community_data=false&developer_data=false"
    )
    cg = http_get(cg_url)
    results["coingecko"] = cg if isinstance(cg, dict) else {}
    md = results["coingecko"].get("market_data", {})
    price = md.get("current_price", {}).get("usd", "N/A")
    print(f"    → {'OK' if price != 'N/A' else 'MISSING'}: ${price}")

    # 5: Fear & Greed Index (30 days)
    print("  Fetching Fear & Greed Index...")
    fng = http_get("https://api.alternative.me/fng/?limit=30")
    results["fng"] = fng if isinstance(fng, dict) else {}
    fng_data = results["fng"].get("data", [])
    fng_val = fng_data[0].get("value", "N/A") if fng_data else "N/A"
    fng_class = fng_data[0].get("value_classification", "N/A") if fng_data else "N/A"
    print(f"    → {'OK' if fng_val != 'N/A' else 'MISSING'}: {fng_val} ({fng_class})")

    # 6: Mempool hash rate (1 month)
    print("  Fetching Mempool hash rate...")
    results["mempool_hash"] = http_get("https://mempool.space/api/v1/mining/hashrate/1m")
    status = "OK" if isinstance(results["mempool_hash"], dict) and results["mempool_hash"].get("hashrates") else "MISSING"
    print(f"    → {status}")

    # 7: Mempool difficulty adjustment
    print("  Fetching Mempool difficulty...")
    results["mempool_diff"] = http_get("https://mempool.space/api/v1/difficulty-adjustment")
    status = "OK" if isinstance(results["mempool_diff"], dict) and "progressPercent" in results["mempool_diff"] else "MISSING"
    print(f"    → {status}")

    # 8: CFTC COT - Bitcoin Traders in Financial Futures
    print("  Fetching CFTC COT (Bitcoin)...")
    cot_url = (
        "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
        "?commodity_name=BITCOIN"
        "&$limit=4&$order=report_date_as_yyyy_mm_dd%20DESC"
    )
    cot = http_get(cot_url)
    results["cot"] = cot if isinstance(cot, list) else []
    if results["cot"]:
        latest = results["cot"][0]
        print(f"    → OK: Date {latest.get('report_date_as_yyyy_mm_dd', '?')}, "
              f"OI {latest.get('open_interest_all', '?')}")
    else:
        print("    → MISSING")

    # 9: Blockchain.info hash rate (30 days, backup)
    print("  Fetching Blockchain.info hash rate...")
    bc = http_get("https://api.blockchain.info/charts/hash-rate?timespan=30days&format=json")
    results["blockchain_hash"] = bc if isinstance(bc, dict) else {}
    values = results["blockchain_hash"].get("values", [])
    if values:
        latest_h = values[-1].get("y", 0)
        print(f"    → OK: {latest_h / 1e9:.2f} EH/s")
    else:
        print("    → MISSING")

    # 10: SoSoValue Spot BTC ETF flows
    print("  Fetching SoSoValue ETF flows...")
    etf_current = http_post(
        "https://api.sosovalue.xyz/openapi/v2/etf/currentEtfDataMetrics",
        {"type": "us-btc-spot"},
    )
    etf_history = http_post(
        "https://api.sosovalue.xyz/openapi/v2/etf/historicalInflowChart",
        {"type": "us-btc-spot"},
    )
    results["etf_current"] = etf_current if isinstance(etf_current, dict) and etf_current.get("code") == 0 else {}
    etf_hist_data = etf_history.get("data", []) if isinstance(etf_history, dict) and etf_history.get("code") == 0 else []
    results["etf_history"] = etf_hist_data
    if results["etf_current"].get("data"):
        daily = results["etf_current"]["data"].get("dailyNetInflow", {})
        val = _safe_float(daily.get("value"))
        date = daily.get("lastUpdateDate", "?")
        print(f"    → OK: Daily net inflow ${val / 1e6:+.1f}M ({date})" if val is not None else "    → MISSING")
    else:
        print("    → MISSING")

    # 11-12: OHLCV for Technical Analysis (CryptoCompare daily + 4H)
    print("  Fetching CryptoCompare OHLCV (daily, 365 bars)...")
    cc_daily_url = "https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=365"
    cc_daily_raw = http_get(cc_daily_url)
    daily_candles = _parse_cryptocompare(cc_daily_raw) if isinstance(cc_daily_raw, dict) else []
    print(f"    → {'OK' if daily_candles else 'MISSING'}: {len(daily_candles)} daily candles")

    print("  Fetching CryptoCompare OHLCV (4H, 200 bars)...")
    cc_url = "https://min-api.cryptocompare.com/data/v2/histohour?fsym=BTC&tsym=USD&limit=200&aggregate=4"
    cc_raw = http_get(cc_url)
    h4_candles = _parse_cryptocompare(cc_raw) if isinstance(cc_raw, dict) else []
    print(f"    → {'OK' if h4_candles else 'MISSING'}: {len(h4_candles)} 4H candles")

    # Derive weekly from daily
    weekly_candles = _downsample_to_weekly(daily_candles) if daily_candles else []

    # Compute technical indicators per timeframe
    results["ta_daily"] = compute_ta(daily_candles) if daily_candles else {}
    results["ta_4h"] = compute_ta(h4_candles) if h4_candles else {}
    results["ta_weekly"] = compute_ta(weekly_candles) if weekly_candles else {}
    ta_ok = bool(results["ta_daily"]) or bool(results["ta_4h"])
    print(f"    → TA computed: daily={'OK' if results['ta_daily'] else 'MISSING'}, "
          f"4H={'OK' if results['ta_4h'] else 'MISSING'}, "
          f"weekly={'OK' if results['ta_weekly'] else 'MISSING'}")

    return results


# ---------------------------------------------------------------------------
# CYCLE CALCULATION
# ---------------------------------------------------------------------------
def calculate_cycle() -> dict:
    """Calculate Bitcoin halving cycle position."""
    halving_date = datetime(2024, 4, 19, tzinfo=timezone.utc)
    next_halving = datetime(2028, 4, 7, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)

    days_since = (now - halving_date).days
    days_until = (next_halving - now).days

    if days_since < 180:
        phase = "EARLY_BULL"
    elif days_since < 480:
        phase = "MID_BULL"
    elif days_since < 600:
        phase = "LATE_BULL"
    elif days_since < 900:
        phase = "EARLY_BEAR"
    else:
        phase = "LATE_BEAR"

    return {
        "days_since_halving": days_since,
        "days_until_halving": days_until,
        "halving_date": "2024-04-19",
        "next_halving": "2028-04-07",
        "estimated_phase": phase,
    }


# ---------------------------------------------------------------------------
# DATA AGGREGATION
# ---------------------------------------------------------------------------
def aggregate_data(data: dict, history: list | None = None) -> str:
    """Build text block from all data for the Claude prompt."""
    lines = []

    # Cycle
    cycle = calculate_cycle()
    lines.append("=== CYCLE POSITION ===")
    lines.append(f"  Days since halving (2024-04-19): {cycle['days_since_halving']}")
    lines.append(f"  Days until next halving (~2028-04-07): {cycle['days_until_halving']}")
    lines.append(f"  Estimated phase: {cycle['estimated_phase']}")
    lines.append("")

    # Macro
    lines.append("=== MACRO DATA (FRED) ===")
    for name, key, label in [
        ("dfii10", "DFII10", "10Y Real Yield"),
        ("dollar", "DTWEXBGS", "Dollar Index"),
        ("vix", "VIXCLS", "VIX"),
    ]:
        obs = data.get(name, {})
        val = obs.get("value", "N/A")
        date = obs.get("date", "N/A")
        lines.append(f"  {label} ({key}): {val} (as of {date})")
    lines.append("")

    # Bitcoin price data
    md = data.get("coingecko", {}).get("market_data", {})
    lines.append("=== BITCOIN MARKET DATA (CoinGecko) ===")
    lines.append(f"  Price (USD): ${md.get('current_price', {}).get('usd', 'N/A'):,}" if isinstance(md.get('current_price', {}).get('usd'), (int, float)) else f"  Price (USD): {md.get('current_price', {}).get('usd', 'N/A')}")
    lines.append(f"  Market Cap: ${md.get('market_cap', {}).get('usd', 'N/A'):,.0f}" if isinstance(md.get('market_cap', {}).get('usd'), (int, float)) else f"  Market Cap: {md.get('market_cap', {}).get('usd', 'N/A')}")
    lines.append(f"  24h Volume: ${md.get('total_volume', {}).get('usd', 'N/A'):,.0f}" if isinstance(md.get('total_volume', {}).get('usd'), (int, float)) else f"  24h Volume: {md.get('total_volume', {}).get('usd', 'N/A')}")
    lines.append(f"  24h Change: {md.get('price_change_percentage_24h', 'N/A')}%")
    lines.append(f"  7d Change: {md.get('price_change_percentage_7d', 'N/A')}%")
    lines.append(f"  30d Change: {md.get('price_change_percentage_30d', 'N/A')}%")
    lines.append(f"  ATH: ${md.get('ath', {}).get('usd', 'N/A'):,}" if isinstance(md.get('ath', {}).get('usd'), (int, float)) else f"  ATH: {md.get('ath', {}).get('usd', 'N/A')}")
    lines.append(f"  ATH Change: {md.get('ath_change_percentage', {}).get('usd', 'N/A')}%")
    lines.append(f"  Circulating Supply: {md.get('circulating_supply', 'N/A')}")
    lines.append("")

    # Sentiment
    fng_data = data.get("fng", {}).get("data", [])
    fng_latest = fng_data[0] if fng_data else {}
    fng_7d = fng_data[:7]
    fng_7d_avg = sum(int(d.get("value", 0)) for d in fng_7d) / max(len(fng_7d), 1)

    lines.append("=== SENTIMENT (Fear & Greed Index) ===")
    lines.append(f"  Current: {fng_latest.get('value', 'N/A')} ({fng_latest.get('value_classification', 'N/A')})")
    lines.append(f"  7-day Average: {fng_7d_avg:.0f}")
    if len(fng_data) >= 30:
        fng_30d_avg = sum(int(d.get("value", 0)) for d in fng_data[:30]) / 30
        lines.append(f"  30-day Average: {fng_30d_avg:.0f}")
    lines.append("")

    # Mining
    lines.append("=== MINING DATA ===")
    # Blockchain.info hash rate
    bc_values = data.get("blockchain_hash", {}).get("values", [])
    if bc_values:
        latest_h = bc_values[-1].get("y", 0)
        first_h = bc_values[0].get("y", 0)
        h_change = ((latest_h - first_h) / first_h * 100) if first_h else 0
        lines.append(f"  Current Hash Rate: {latest_h / 1e9:.2f} EH/s")
        lines.append(f"  Hash Rate 30d Ago: {first_h / 1e9:.2f} EH/s")
        lines.append(f"  Hash Rate 30d Change: {h_change:.2f}%")
    else:
        lines.append("  Hash Rate: N/A")

    # Mempool difficulty
    diff = data.get("mempool_diff", {})
    if diff:
        lines.append(f"  Difficulty Progress: {diff.get('progressPercent', 'N/A')}%")
        lines.append(f"  Difficulty Change: {diff.get('difficultyChange', 'N/A')}%")
        retarget_ts = diff.get("estimatedRetargetDate")
        if retarget_ts:
            retarget = datetime.fromtimestamp(retarget_ts / 1000, tz=timezone.utc)
            lines.append(f"  Estimated Retarget: {retarget.strftime('%Y-%m-%d')}")
        lines.append(f"  Remaining Blocks: {diff.get('remainingBlocks', 'N/A')}")
    lines.append("")

    # CFTC COT Positioning (Traders in Financial Futures)
    cot = data.get("cot", [])
    lines.append("=== CFTC COT POSITIONING (Bitcoin Traders in Financial Futures) ===")
    if cot:
        latest = cot[0]
        lines.append(f"  Report Date: {latest.get('report_date_as_yyyy_mm_dd', 'N/A')}")
        lines.append(f"  Asset Manager Long: {latest.get('asset_mgr_positions_long', 'N/A')}")
        lines.append(f"  Asset Manager Short: {latest.get('asset_mgr_positions_short', 'N/A')}")
        lines.append(f"  Leveraged Money Long: {latest.get('lev_money_positions_long', 'N/A')}")
        lines.append(f"  Leveraged Money Short: {latest.get('lev_money_positions_short', 'N/A')}")
        lines.append(f"  Dealer Long: {latest.get('dealer_positions_long_all', 'N/A')}")
        lines.append(f"  Dealer Short: {latest.get('dealer_positions_short_all', 'N/A')}")
        lines.append(f"  Open Interest: {latest.get('open_interest_all', 'N/A')}")

        if len(cot) >= 2:
            prev = cot[1]
            lines.append(f"  Previous Week ({prev.get('report_date_as_yyyy_mm_dd', '?')}):")
            lines.append(f"    Asset Mgr Long: {prev.get('asset_mgr_positions_long', 'N/A')}")
            lines.append(f"    Asset Mgr Short: {prev.get('asset_mgr_positions_short', 'N/A')}")
            lines.append(f"    Lev Money Long: {prev.get('lev_money_positions_long', 'N/A')}")
            lines.append(f"    Lev Money Short: {prev.get('lev_money_positions_short', 'N/A')}")
            lines.append(f"    Open Interest: {prev.get('open_interest_all', 'N/A')}")
    else:
        lines.append("  COT data: N/A (CFTC API may be unavailable)")
    lines.append("")

    # ETF Flows (SoSoValue)
    etf_cur = data.get("etf_current", {}).get("data", {})
    etf_hist = data.get("etf_history", [])
    lines.append("=== SPOT BTC ETF FLOWS (SoSoValue) ===")
    if etf_cur:
        daily_inflow = _safe_float(etf_cur.get("dailyNetInflow", {}).get("value"))
        daily_date = etf_cur.get("dailyNetInflow", {}).get("lastUpdateDate", "N/A")
        cum_inflow = _safe_float(etf_cur.get("cumNetInflow", {}).get("value"))
        total_aum = _safe_float(etf_cur.get("totalNetAssets", {}).get("value"))
        total_btc = _safe_float(etf_cur.get("totalTokenHoldings", {}).get("value"))
        daily_vol = _safe_float(etf_cur.get("dailyTotalValueTraded", {}).get("value"))

        lines.append(f"  Daily Net Inflow: ${daily_inflow / 1e6:+,.1f}M ({daily_date})" if daily_inflow is not None else "  Daily Net Inflow: N/A")
        lines.append(f"  Cumulative Net Inflow: ${cum_inflow / 1e9:,.2f}B" if cum_inflow is not None else "  Cumulative Net Inflow: N/A")
        lines.append(f"  Total AUM: ${total_aum / 1e9:,.2f}B" if total_aum is not None else "  Total AUM: N/A")
        lines.append(f"  Total BTC Holdings: {total_btc:,.0f} BTC" if total_btc is not None else "  Total BTC Holdings: N/A")
        lines.append(f"  Daily Volume: ${daily_vol / 1e9:,.2f}B" if daily_vol is not None else "  Daily Volume: N/A")

        # Top ETF breakdown
        etf_list = etf_cur.get("list", [])
        if etf_list:
            lines.append("  --- Per-ETF Breakdown (top 5 by AUM) ---")
            sorted_etfs = sorted(etf_list, key=lambda e: abs(_safe_float(e.get("netAssets", {}).get("value")) or 0), reverse=True)
            for etf in sorted_etfs[:5]:
                ticker = etf.get("ticker", "?")
                inst = etf.get("institute", "?").strip()
                etf_daily = _safe_float(etf.get("dailyNetInflow", {}).get("value"))
                etf_aum = _safe_float(etf.get("netAssets", {}).get("value"))
                daily_str = f"${etf_daily / 1e6:+,.1f}M" if etf_daily is not None else "N/A"
                aum_str = f"${etf_aum / 1e9:,.2f}B" if etf_aum is not None else "N/A"
                lines.append(f"    {ticker} ({inst}): Daily {daily_str}, AUM {aum_str}")
    else:
        lines.append("  ETF data: N/A (SoSoValue API may be unavailable)")

    # Historical ETF flow summary (7d and 30d)
    if etf_hist:
        recent_7 = etf_hist[:7]
        recent_30 = etf_hist[:30]
        sum_7d = sum(_safe_float(d.get("totalNetInflow")) or 0 for d in recent_7)
        sum_30d = sum(_safe_float(d.get("totalNetInflow")) or 0 for d in recent_30)
        inflow_days_7 = sum(1 for d in recent_7 if (_safe_float(d.get("totalNetInflow")) or 0) > 0)
        outflow_days_7 = sum(1 for d in recent_7 if (_safe_float(d.get("totalNetInflow")) or 0) < 0)
        lines.append(f"  7-Day Net Flow: ${sum_7d / 1e6:+,.1f}M ({inflow_days_7} inflow days, {outflow_days_7} outflow days)")
        lines.append(f"  30-Day Net Flow: ${sum_30d / 1e6:+,.1f}M")

        # Last 7 days detail
        lines.append("  --- Last 7 Trading Days ---")
        for d in recent_7:
            date = d.get("date", "?")
            flow = _safe_float(d.get("totalNetInflow"))
            flow_str = f"${flow / 1e6:+,.1f}M" if flow is not None else "N/A"
            lines.append(f"    {date}: {flow_str}")
    lines.append("")

    # Technical Indicators (compact format, ~80 tokens)
    ta_daily = data.get("ta_daily", {})
    ta_4h = data.get("ta_4h", {})
    ta_weekly = data.get("ta_weekly", {})
    if ta_daily or ta_4h:
        lines.append("=== TECHNICAL INDICATORS (BTCUSDT) ===")

        for label, ta in [("Daily", ta_daily), ("4H", ta_4h), ("Weekly", ta_weekly)]:
            if not ta:
                lines.append(f"  {label}: N/A")
                continue
            parts = []
            if ta.get("ema20"):
                parts.append(f"EMA20=${ta['ema20']:.0f}")
            if ta.get("ema50"):
                parts.append(f"EMA50=${ta['ema50']:.0f}")
            if ta.get("ema200"):
                parts.append(f"EMA200=${ta['ema200']:.0f}")
            if ta.get("ema_stack"):
                parts.append(f"Stack={ta['ema_stack']}")
            if ta.get("ema_slope") is not None:
                parts.append(f"Slope={ta['ema_slope']:+.3f}%/bar")
            if ta.get("rsi14") is not None:
                parts.append(f"RSI14={ta['rsi14']:.0f}")
            macd = ta.get("macd")
            if macd:
                parts.append(f"MACD(line:{macd['line']:+.0f},sig:{macd['signal']:+.0f},hist:{macd['histogram']:+.0f},{macd['trend']})")
            bb = ta.get("bollinger")
            if bb:
                sq = "YES" if bb["squeeze"] else "NO"
                parts.append(f"BB(upper:{bb['upper']:.0f},lower:{bb['lower']:.0f},width:{bb['width_pct']:.1f}%,squeeze:{sq},%B:{bb['pct_b']:.2f})")
            lines.append(f"  {label}: {' '.join(parts)}")

        # MTF alignment summary
        stacks = [ta.get("ema_stack") for ta in [ta_daily, ta_4h, ta_weekly] if ta.get("ema_stack") and ta["ema_stack"] != "UNKNOWN"]
        bullish_count = sum(1 for s in stacks if s == "BULLISH")
        bearish_count = sum(1 for s in stacks if s == "BEARISH")
        if stacks:
            if bullish_count == len(stacks):
                mtf = f"{len(stacks)}/{len(stacks)} BULLISH"
            elif bearish_count == len(stacks):
                mtf = f"{len(stacks)}/{len(stacks)} BEARISH"
            else:
                mtf = f"MIXED ({bullish_count}B/{bearish_count}S/{len(stacks)-bullish_count-bearish_count}M)"
        else:
            mtf = "N/A"

        # Key levels from daily EMAs and Bollinger
        supports = []
        resistances = []
        price = ta_daily.get("price", 0)
        for lbl, val in [("EMA50d", ta_daily.get("ema50")), ("EMA200d", ta_daily.get("ema200"))]:
            if val and val < price:
                supports.append(f"${val:.0f}({lbl})")
            elif val and val > price:
                resistances.append(f"${val:.0f}({lbl})")
        bb_d = ta_daily.get("bollinger")
        if bb_d:
            if bb_d["lower"] < price:
                supports.append(f"${bb_d['lower']:.0f}(BBlow)")
            if bb_d["upper"] > price:
                resistances.append(f"${bb_d['upper']:.0f}(BBup)")

        sup_str = " ".join(supports) if supports else "none identified"
        res_str = " ".join(resistances) if resistances else "none identified"
        lines.append(f"  MTF: {mtf} | Support: {sup_str} | Resistance: {res_str}")
        lines.append("")
    else:
        lines.append("=== TECHNICAL INDICATORS (BTCUSDT) ===")
        lines.append("  UNAVAILABLE (OHLCV fetch failed)")
        lines.append("")

    # Data quality summary
    source_checks = {
        "FRED DFII10": bool(data.get("dfii10", {}).get("value")),
        "FRED Dollar": bool(data.get("dollar", {}).get("value")),
        "FRED VIX": bool(data.get("vix", {}).get("value")),
        "CoinGecko BTC": bool(md.get("current_price", {}).get("usd")),
        "Fear & Greed": bool(fng_data),
        "Mempool Hash": bool(data.get("mempool_hash", {}).get("hashrates")),
        "Mempool Diff": bool(diff.get("progressPercent")),
        "CFTC COT": bool(cot),
        "Blockchain Hash": bool(bc_values),
        "SoSoValue ETF": bool(etf_cur),
        "CryptoCompare Daily": bool(ta_daily),
        "CryptoCompare 4H": bool(ta_4h),
    }
    lines.append("=== DATA QUALITY FLAGS ===")
    for src, ok in source_checks.items():
        lines.append(f"  {src}: {'OK' if ok else 'MISSING'}")

    # Rolling history
    if history:
        lines.append(format_history_for_prompt(history))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLAUDE API
# ---------------------------------------------------------------------------
def _read_sse_stream(resp) -> dict:
    """Read an Anthropic SSE stream, reconstruct a Messages-API-shaped dict.

    The Anthropic streaming format sends events like:
        event: message_start
        data: {"type": "message_start", "message": {...}}

        event: content_block_start
        data: {"type": "content_block_start", "index": 0, ...}

        event: ping
        data: {"type": "ping"}

    We only need the ``data:`` lines; the ``type`` field inside the JSON
    tells us what kind of event it is.
    """
    content_blocks: list[dict] = []
    stop_reason = None
    message_shell: dict = {}
    events_received = 0

    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")

        # SSE lines we care about start with "data: "
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")
        events_received += 1

        # Progress indicator every 50 events
        if events_received % 50 == 0:
            print(f"    ... {events_received} SSE events received")

        if etype == "message_start":
            message_shell = event.get("message", {})

        elif etype == "content_block_start":
            idx = event.get("index", len(content_blocks))
            block = event.get("content_block", {})
            while len(content_blocks) <= idx:
                content_blocks.append({})
            content_blocks[idx] = block

        elif etype == "content_block_delta":
            idx = event.get("index", 0)
            delta = event.get("delta", {})
            if idx < len(content_blocks):
                block = content_blocks[idx]
                dtype = delta.get("type", "")
                if dtype == "text_delta":
                    block["text"] = block.get("text", "") + delta.get("text", "")
                elif dtype == "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
                elif dtype == "input_json_delta":
                    block["_partial_json"] = block.get("_partial_json", "") + delta.get("partial_json", "")

        elif etype == "content_block_stop":
            idx = event.get("index", 0)
            if idx < len(content_blocks):
                block = content_blocks[idx]
                if "_partial_json" in block:
                    try:
                        block["input"] = json.loads(block.pop("_partial_json"))
                    except json.JSONDecodeError:
                        block.pop("_partial_json", None)

        elif etype == "message_delta":
            delta = event.get("delta", {})
            stop_reason = delta.get("stop_reason", stop_reason)

        # ping, error, and other events are silently ignored

    print(f"    Total SSE events: {events_received}")

    result = {**message_shell}
    result["content"] = content_blocks
    result["stop_reason"] = stop_reason
    return result


def call_claude(data_text: str, previous_rec: str) -> dict:
    """Call Claude API with **streaming** to avoid read-timeout on long generations.

    Why streaming?
    Non-streaming requests with large max_tokens + thinking + web_search
    can take 5-10+ minutes of server-side processing.  During that time
    zero bytes are sent to the client.  Networks (including GitHub Actions
    runners) drop idle connections well before that, causing TimeoutError.

    With ``"stream": true`` the server sends SSE events incrementally,
    keeping the TCP connection alive throughout generation.
    """
    print("🤖 Calling Claude API (streaming)...")

    if PROMPT_TEMPLATE.exists():
        template = PROMPT_TEMPLATE.read_text()
    else:
        print("  ⚠ prompt_template.txt not found, using embedded fallback")
        template = (
            "You are a Bitcoin analyst. Today is DATE_PLACEHOLDER.\n\n"
            "DATA_PLACEHOLDER\n\n"
            "Generate an HTML report with BTC allocation recommendation."
        )

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = template.replace("DATE_PLACEHOLDER", today).replace("DATA_PLACEHOLDER", data_text)

    if previous_rec and previous_rec != "NONE":
        prompt += f"\n\nPrevious recommendation was: {previous_rec}. Note any change."

    request_body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 32000,
        "stream": True,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 16000,
        },
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "messages": [{"role": "user", "content": prompt}],
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "interleaved-thinking-2025-05-14",
    }

    max_retries = 2
    # Socket timeout: how long to wait for the NEXT chunk of data.
    # With streaming, chunks arrive every few seconds, so 300s is very
    # generous and only fires if the connection is truly dead.
    socket_timeout = 300
    WALL_LIMIT_SECS = 900   # 15 min budget (workflow timeout is 20 min)
    wall_start = time.monotonic()

    for attempt in range(1, max_retries + 1):
        elapsed = time.monotonic() - wall_start
        remaining = WALL_LIMIT_SECS - elapsed
        if remaining < 120:
            print(f"  ⏱ Only {remaining:.0f}s left in wall-clock budget, aborting retries")
            return {"error": f"Wall-clock budget exhausted after {elapsed:.0f}s"}

        # Build a fresh Request each attempt (urllib can be finicky on reuse)
        req = Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(request_body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            print(f"  Attempt {attempt}/{max_retries} "
                  f"(socket timeout: {socket_timeout}s, "
                  f"wall: {elapsed:.0f}s/{WALL_LIMIT_SECS}s)...")
            with urlopen(req, timeout=socket_timeout) as resp:
                result = _read_sse_stream(resp)

            duration = time.monotonic() - wall_start - elapsed
            print(f"  ✅ Claude responded in {duration:.0f}s "
                  f"(stop_reason: {result.get('stop_reason')}, "
                  f"blocks: {len(result.get('content', []))})")
            return result

        except (URLError, HTTPError) as e:
            body = ""
            if hasattr(e, "read"):
                body = e.read().decode("utf-8", errors="replace")[:500]
            print(f"  ⚠ Attempt {attempt} failed: {e}")
            if body:
                print(f"    Body: {body}")
            if attempt < max_retries:
                wait = 15
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ All {max_retries} attempts failed")
                return {"error": str(e), "body": body}
        except Exception as e:
            print(f"  ⚠ Attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt < max_retries:
                wait = 15
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ All {max_retries} attempts failed")
                return {"error": str(e), "body": ""}


# ---------------------------------------------------------------------------
# PARSE RESPONSE
# ---------------------------------------------------------------------------
def parse_response(claude_response: dict, previous_rec: str) -> dict:
    """Extract HTML report and JSON recommendation from Claude response."""
    content_parts = claude_response.get("content", [])
    text = ""
    for block in content_parts:
        if block.get("type") == "text":
            text += block["text"]

    # Strip any preamble before <!DOCTYPE or <html or <div
    html_start = -1
    for marker in ["<!DOCTYPE", "<!doctype", "<html", "<HTML", "<div", "<DIV"]:
        idx = text.find(marker)
        if idx >= 0 and (html_start < 0 or idx < html_start):
            html_start = idx
    if html_start > 0:
        preamble = text[:html_start].strip()
        if preamble:
            print(f"  ⚠ Stripped {len(preamble)} chars of preamble")
        text = text[html_start:]

    # Extract JSON recommendation
    recommendation = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "asset": "BTC",
        "market_regime": "UNKNOWN",
        "cycle_phase": {"days_since_halving": 0, "estimated_phase": "UNKNOWN"},
        "btc_recommendation": {
            "action": "HOLD",
            "conviction": "LOW",
            "confidence_pct": 0,
            "composite_score": 0,
        },
        "position_guidance": {
            "btc_allocation_pct": 50,
            "stablecoin_allocation_pct": 50,
            "entry_method": "DCA_4WK",
            "sizing_rationale": "Unable to parse recommendation",
        },
        "agent_signals": {},
        "key_drivers": [],
        "key_risks": [],
        "catalyst_calendar": [],
        "conflicts_noted": None,
        "confidence_modifiers_applied": [],
        "next_review": None,
    }

    json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            recommendation = {**recommendation, **parsed}
        except json.JSONDecodeError:
            print("  ⚠ JSON parse failed, using defaults")

    # HTML is everything except the JSON block
    html_report = re.sub(r"```json[\s\S]*?```", "", text).strip()
    if not html_report:
        html_report = "<h1>Error: No report generated</h1>"

    # Build result
    rec = recommendation
    action = rec.get("btc_recommendation", {}).get("action", "HOLD")
    confidence = rec.get("btc_recommendation", {}).get("confidence_pct", 0)
    conviction = rec.get("btc_recommendation", {}).get("conviction", "LOW")
    composite = rec.get("btc_recommendation", {}).get("composite_score", 0)
    regime = rec.get("market_regime", "UNKNOWN")
    phase = rec.get("cycle_phase", {}).get("estimated_phase", "UNKNOWN")
    btc_alloc = rec.get("position_guidance", {}).get("btc_allocation_pct", 50)
    stable_alloc = rec.get("position_guidance", {}).get("stablecoin_allocation_pct", 50)

    rec_changed = previous_rec != action and previous_rec != "NONE"

    emoji = "₿"
    if "BUY" in action:
        emoji = "🟢₿"
    elif "SELL" in action:
        emoji = "🔴₿"
    elif action == "ACCUMULATE":
        emoji = "🟡₿"
    elif action == "REDUCE":
        emoji = "🟠₿"

    alert = (
        f"{emoji} BTC: {action} ({confidence}% {conviction})\n"
        f"Regime: {regime} | Phase: {phase}\n"
        f"Score: {composite:.2f} | Target: {btc_alloc}% BTC / {stable_alloc}% Stable"
    )

    subject = f"Bitcoin Report - {datetime.now(timezone.utc).strftime('%d-%m-%Y')} | {action} | {conviction} CONVICTION"

    return {
        "html_report": html_report,
        "recommendation": action,
        "confidence": confidence,
        "conviction": conviction,
        "composite_score": composite,
        "market_regime": regime,
        "cycle_phase": phase,
        "btc_allocation_pct": btc_alloc,
        "stablecoin_allocation_pct": stable_alloc,
        "recommendation_changed": rec_changed,
        "previous_recommendation": previous_rec,
        "alert_message": alert,
        "email_subject": subject,
        "full_analysis": json.dumps(rec),
    }


# ---------------------------------------------------------------------------
# RECOMMENDATION PERSISTENCE
# ---------------------------------------------------------------------------
def load_previous_recommendation() -> str:
    """Load previous recommendation from file."""
    if PREV_REC_FILE.exists():
        try:
            data = json.loads(PREV_REC_FILE.read_text())
            return data.get("recommendation", "NONE")
        except (json.JSONDecodeError, KeyError):
            pass
    return "NONE"


def save_recommendation(result: dict):
    """Save current recommendation for next run."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "recommendation": result["recommendation"],
        "confidence": result["confidence"],
        "conviction": result["conviction"],
        "composite_score": result["composite_score"],
        "market_regime": result["market_regime"],
        "cycle_phase": result["cycle_phase"],
        "btc_allocation_pct": result["btc_allocation_pct"],
        "stablecoin_allocation_pct": result["stablecoin_allocation_pct"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    PREV_REC_FILE.write_text(json.dumps(payload, indent=2))
    print(f"  💾 Saved recommendation: {result['recommendation']}")


# ---------------------------------------------------------------------------
# ROLLING HISTORY
# ---------------------------------------------------------------------------
def _safe_float(val) -> float | None:
    """Convert to float, return None if invalid."""
    if val is None:
        return None
    try:
        f = float(str(val).replace(",", ""))
        return f if f == f else None  # NaN check
    except (ValueError, TypeError):
        return None


def _safe_int(long_val, short_val) -> int | None:
    """Calculate net speculative position (longs - shorts)."""
    try:
        return int(long_val) - int(short_val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# TECHNICAL INDICATORS (pure Python, no dependencies)
# ---------------------------------------------------------------------------
def _ema(values: list[float], period: int) -> list[float]:
    """EMA with SMA seed for first N values (avoids first-close bias)."""
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    sma_seed = sum(values[:period]) / period
    result = [sma_seed]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _rsi(closes: list[float], period: int = 14) -> list[float]:
    """RSI with Wilder smoothing (matches TradingView)."""
    if len(closes) < period + 1:
        return []
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rs = avg_gain / avg_loss if avg_loss != 0 else 100
    rsi_vals = [100 - 100 / (1 + rs)]
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi_vals.append(100 - 100 / (1 + rs))
    return rsi_vals


def _macd(closes: list[float]) -> dict | None:
    """MACD (12, 26, 9). Returns latest values or None."""
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    if not ema12 or not ema26:
        return None
    # Align: ema12 starts at index 12, ema26 at index 26, so offset ema12
    offset = 26 - 12
    if len(ema12) <= offset:
        return None
    macd_line = [ema12[i + offset] - ema26[i] for i in range(len(ema26))]
    signal = _ema(macd_line, 9)
    if not signal:
        return None
    # Align signal (starts 9 into macd_line)
    sig_offset = 9
    if len(macd_line) <= sig_offset:
        return None
    hist = macd_line[-1] - signal[-1]
    prev_hist = macd_line[-2] - signal[-2] if len(signal) >= 2 and len(macd_line) >= sig_offset + 2 else hist
    return {
        "line": round(macd_line[-1], 2),
        "signal": round(signal[-1], 2),
        "histogram": round(hist, 2),
        "trend": "EXPANDING" if abs(hist) > abs(prev_hist) else "CONTRACTING",
    }


def _bollinger(closes: list[float], period: int = 20, mult: float = 2.0) -> dict | None:
    """Bollinger Bands with relative squeeze detection."""
    if len(closes) < period:
        return None
    sma = sum(closes[-period:]) / period
    variance = sum((c - sma) ** 2 for c in closes[-period:]) / period
    stddev = math.sqrt(variance)
    upper = sma + mult * stddev
    lower = sma - mult * stddev
    width_pct = (mult * stddev * 2) / sma * 100 if sma else 0
    price = closes[-1]
    pct_b = (price - lower) / (upper - lower) if (upper - lower) != 0 else 0.5

    # Relative squeeze: compute BB width over last 100 periods (or available)
    squeeze = False
    lookback = min(len(closes), 100)
    if lookback >= period + 10:
        widths = []
        for i in range(lookback - period + 1):
            sl = closes[-(lookback - i):-(lookback - i - period)] if (lookback - i - period) > 0 else closes[:period]
            s = sum(closes[len(closes) - lookback + i:len(closes) - lookback + i + period]) / period
            v = sum((c - s) ** 2 for c in closes[len(closes) - lookback + i:len(closes) - lookback + i + period]) / period
            w = math.sqrt(v) * mult * 2 / s * 100 if s else 0
            widths.append(w)
        if widths:
            sorted_w = sorted(widths)
            percentile_20 = sorted_w[max(0, len(sorted_w) // 5)]
            squeeze = width_pct <= percentile_20

    return {
        "upper": round(upper, 0),
        "lower": round(lower, 0),
        "middle": round(sma, 0),
        "width_pct": round(width_pct, 1),
        "pct_b": round(pct_b, 2),
        "squeeze": squeeze,
    }


def _ema_slope(ema_values: list[float], lookback: int = 5) -> float | None:
    """Trend strength proxy: % change per bar of EMA over last N bars."""
    if len(ema_values) < lookback + 1:
        return None
    start = ema_values[-(lookback + 1)]
    end = ema_values[-1]
    if start == 0:
        return None
    return round((end - start) / start * 100 / lookback, 3)


def compute_ta(candles: list[dict]) -> dict:
    """Compute all technical indicators from OHLCV candles. Pure stdlib."""
    if not candles or len(candles) < 26:
        return {}
    closes = [c["close"] for c in candles]

    # EMAs
    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)
    ema200 = _ema(closes, 200)
    ema20_val = ema20[-1] if ema20 else None
    ema50_val = ema50[-1] if ema50 else None
    ema200_val = ema200[-1] if ema200 else None

    # EMA alignment
    stack = "UNKNOWN"
    if ema20_val and ema50_val:
        if ema200_val:
            if ema20_val > ema50_val > ema200_val:
                stack = "BULLISH"
            elif ema20_val < ema50_val < ema200_val:
                stack = "BEARISH"
            else:
                stack = "MIXED"
        else:
            if ema20_val > ema50_val:
                stack = "BULLISH"
            elif ema20_val < ema50_val:
                stack = "BEARISH"
            else:
                stack = "MIXED"

    # EMA slope (trend strength proxy)
    slope = _ema_slope(ema20, 5) if ema20 else None

    # RSI
    rsi_vals = _rsi(closes, 14)
    rsi_val = round(rsi_vals[-1], 1) if rsi_vals else None

    # MACD
    macd = _macd(closes)

    # Bollinger Bands
    bb = _bollinger(closes)

    # Sanity checks
    result = {
        "price": closes[-1],
        "ema20": round(ema20_val, 0) if ema20_val else None,
        "ema50": round(ema50_val, 0) if ema50_val else None,
        "ema200": round(ema200_val, 0) if ema200_val else None,
        "ema_stack": stack,
        "ema_slope": slope,
        "rsi14": rsi_val,
        "macd": macd,
        "bollinger": bb,
    }

    # Sanity validation
    if rsi_val is not None and not (0 <= rsi_val <= 100):
        print(f"  ⚠ RSI sanity check failed: {rsi_val}, discarding")
        result["rsi14"] = None
    if bb and bb["upper"] <= bb["lower"]:
        print(f"  ⚠ Bollinger sanity check failed: upper <= lower, discarding")
        result["bollinger"] = None

    return result


def _parse_cryptocompare(raw: dict) -> list[dict]:
    """Parse CryptoCompare histohour response."""
    data = raw.get("Data", {}).get("Data", [])
    candles = []
    for item in data:
        if isinstance(item, dict) and item.get("close"):
            candles.append({
                "open": float(item["open"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "close": float(item["close"]),
            })
    return candles


def _downsample_to_weekly(daily_candles: list[dict]) -> list[dict]:
    """Downsample daily candles to weekly (groups of 5 trading days)."""
    weekly = []
    for i in range(0, len(daily_candles) - 4, 5):
        chunk = daily_candles[i:i + 5]
        weekly.append({
            "open": chunk[0]["open"],
            "high": max(c["high"] for c in chunk),
            "low": min(c["low"] for c in chunk),
            "close": chunk[-1]["close"],
        })
    return weekly


def load_history() -> list:
    """Load historical data snapshots."""
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return []


def save_daily_snapshot(data: dict, recommendation: str = "PENDING"):
    """Append today's data point to rolling history."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    md = data.get("coingecko", {}).get("market_data", {})
    fng_data = data.get("fng", {}).get("data", [])
    fng_val = int(fng_data[0].get("value", 0)) if fng_data else None
    bc_values = data.get("blockchain_hash", {}).get("values", [])
    hash_eh = bc_values[-1].get("y", 0) / 1e9 if bc_values else None
    cot = data.get("cot", [])
    cot_latest = cot[0] if cot else {}
    etf_cur = data.get("etf_current", {}).get("data", {})
    etf_daily_raw = _safe_float(etf_cur.get("dailyNetInflow", {}).get("value"))
    etf_daily_m = round(etf_daily_raw / 1e6, 1) if etf_daily_raw is not None else None

    snapshot = {
        "date": today,
        "btc_price": _safe_float(md.get("current_price", {}).get("usd")),
        "dfii10": _safe_float(data.get("dfii10", {}).get("value")),
        "dollar_idx": _safe_float(data.get("dollar", {}).get("value")),
        "vix": _safe_float(data.get("vix", {}).get("value")),
        "fng": fng_val,
        "hash_eh": round(hash_eh, 2) if hash_eh else None,
        "cot_net_spec": _safe_int(
            cot_latest.get("asset_mgr_positions_long"),
            cot_latest.get("asset_mgr_positions_short"),
        ),
        "cot_oi": _safe_float(cot_latest.get("open_interest_all")),
        "etf_flow_m": etf_daily_m,
        "recommendation": recommendation,
    }

    # Replace today if re-running
    history = [h for h in history if h.get("date") != today]
    history.append(snapshot)
    history = history[-MAX_HISTORY_DAYS:]

    HISTORY_FILE.write_text(json.dumps(history, indent=2))
    print(f"  📈 Saved daily snapshot ({len(history)} days in history)")


def format_history_for_prompt(history: list) -> str:
    """Format rolling history as a compact table for Claude's context."""
    if not history:
        return ""

    lines = [
        "",
        "=== ROLLING DATA HISTORY (last {} days, for trend analysis) ===".format(len(history)),
        "",
        "Date       | BTC($)  | Real10Y | DXY    | VIX   | F&G | Hash(EH) | NetSpec | OI     | ETF$M  | Rec",
        "-----------|---------|---------|--------|-------|-----|----------|---------|--------|--------|----",
    ]

    for h in history:
        btc = f"{h.get('btc_price', 0):>7.0f}" if h.get("btc_price") else "    N/A"
        dfii = f"{h.get('dfii10', 0):>7.2f}" if h.get("dfii10") is not None else "    N/A"
        dxy = f"{h.get('dollar_idx', 0):>6.1f}" if h.get("dollar_idx") is not None else "   N/A"
        vix = f"{h.get('vix', 0):>5.1f}" if h.get("vix") is not None else "  N/A"
        fng = f"{h.get('fng', 0):>3d}" if h.get("fng") is not None else "N/A"
        hsh = f"{h.get('hash_eh', 0):>8.2f}" if h.get("hash_eh") is not None else "     N/A"
        net = f"{h.get('cot_net_spec', 0):>7d}" if h.get("cot_net_spec") is not None else "    N/A"
        oi = f"{h.get('cot_oi', 0):>6.0f}" if h.get("cot_oi") is not None else "   N/A"
        etf = f"{h.get('etf_flow_m', 0):>+6.0f}" if h.get("etf_flow_m") is not None else "   N/A"
        rec = h.get("recommendation", "?")[:4]

        lines.append(
            f"{h.get('date', '?'):10s} | {btc} | {dfii} | {dxy} | {vix} | {fng} | {hsh} | {net} | {oi} | {etf} | {rec}"
        )

    lines.append("")
    lines.append("KEY: F&G = Fear & Greed (0=Extreme Fear, 100=Extreme Greed)")
    lines.append("     NetSpec = Non-Commercial Longs minus Shorts | OI = Open Interest")
    lines.append("     Hash(EH) = Hash Rate in Exahash/s")
    lines.append("     ETF$M = Daily Spot BTC ETF Net Flow in USD Millions")
    lines.append("")
    lines.append("USE THIS HISTORY TO: identify trends in price momentum, macro conditions,")
    lines.append("sentiment shifts, hash rate growth, and positioning changes.")
    lines.append("Compare today's values against the recent trajectory, not just static levels.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EMAIL
# ---------------------------------------------------------------------------
def send_email(result: dict):
    """Send report via Gmail SMTP using App Password."""
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        print("  ⚠ Gmail credentials not set, skipping email")
        report_file = DATA_DIR / f"report_{datetime.now().strftime('%Y%m%d')}.html"
        report_file.write_text(result["html_report"])
        print(f"  📄 Report saved to {report_file}")
        return

    print(f"📧 Sending email to {GMAIL_ADDRESS}...")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = result["email_subject"]
    msg["From"] = f"BTC Analyst <{GMAIL_ADDRESS}>"
    msg["To"] = GMAIL_ADDRESS

    plain = f"{result['alert_message']}\n\nFull HTML report attached."
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(result["html_report"], "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, [GMAIL_ADDRESS], msg.as_string())
        print("  ✅ Email sent successfully")
    except Exception as e:
        print(f"  ❌ Email error: {e}")
        report_file = DATA_DIR / f"report_{datetime.now().strftime('%Y%m%d')}.html"
        report_file.write_text(result["html_report"])
        print(f"  📄 Report saved to {report_file}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(f"₿ Bitcoin Analyst Daily Report — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # Validate config
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not FRED_API_KEY:
        missing.append("FRED_API_KEY")
    if missing:
        print(f"❌ Missing required secrets: {', '.join(missing)}")
        sys.exit(1)

    # 1. Fetch all data
    data = fetch_all_data()

    # 2. Load history and aggregate
    history = load_history()
    print(f"  📈 Loaded {len(history)} days of history")
    data_text = aggregate_data(data, history)
    print("\n📊 Aggregated data block:")
    print(data_text)
    print()

    # Check data quality
    missing_count = data_text.count("MISSING")
    ok_count = data_text.count(": OK")
    print(f"  Data quality: {ok_count}/12 OK, {missing_count}/12 MISSING")
    if missing_count > 5:
        print("  ⚠ WARNING: Most data sources failed. Report will rely on estimates.")

    # 3. Load previous recommendation
    prev_rec = load_previous_recommendation()
    print(f"  Previous recommendation: {prev_rec}")

    # 4. Call Claude
    claude_response = call_claude(data_text, prev_rec)
    if "error" in claude_response:
        print(f"❌ Claude API failed: {claude_response['error']}")
        sys.exit(1)

    # 5. Parse response
    result = parse_response(claude_response, prev_rec)
    print(f"\n📋 Result:")
    print(f"  {result['alert_message']}")
    if result.get("recommendation_changed"):
        print(f"  ⚡ RECOMMENDATION CHANGED from {prev_rec} → {result['recommendation']}")

    # 6. Save recommendation + daily snapshot
    save_recommendation(result)
    save_daily_snapshot(data, result["recommendation"])

    # 7. Send email
    send_email(result)

    # 8. Write summary for GitHub Actions
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(f"## Bitcoin Analyst Report — {datetime.now().strftime('%Y-%m-%d')}\n\n")
            f.write(f"**{result['alert_message']}**\n\n")
            if result.get("recommendation_changed"):
                f.write(f"⚡ Changed from `{prev_rec}` → `{result['recommendation']}`\n\n")
            f.write(f"Data quality: {ok_count}/12 sources OK\n")

    print("\n✅ Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n💥 Unhandled error: {e}")
        traceback.print_exc()
        sys.exit(1)
