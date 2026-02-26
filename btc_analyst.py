#!/usr/bin/env python3
"""
Bitcoin Analyst Daily Report - GitHub Actions Version
Fetches market data from 9 API sources, runs multi-agent analysis via Claude,
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

Secrets required (set in GitHub repo settings):
  ANTHROPIC_API_KEY  - Claude API key
  FRED_API_KEY       - FRED (St. Louis Fed) API key
  GMAIL_ADDRESS      - Gmail address to send from/to
  GMAIL_APP_PASSWORD - Gmail App Password (not your normal password)
"""

import json
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
    print("📡 Fetching data from 9 sources...")

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

    # 8: CFTC COT - Bitcoin Financial Futures
    print("  Fetching CFTC COT (Bitcoin)...")
    cot_url = (
        "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
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

    # CFTC COT Positioning
    cot = data.get("cot", [])
    lines.append("=== CFTC COT POSITIONING (Bitcoin Financial Futures) ===")
    if cot:
        latest = cot[0]
        lines.append(f"  Report Date: {latest.get('report_date_as_yyyy_mm_dd', 'N/A')}")
        lines.append(f"  Non-Commercial Long: {latest.get('noncomm_positions_long_all', 'N/A')}")
        lines.append(f"  Non-Commercial Short: {latest.get('noncomm_positions_short_all', 'N/A')}")
        lines.append(f"  Commercial Long: {latest.get('comm_positions_long_all', 'N/A')}")
        lines.append(f"  Commercial Short: {latest.get('comm_positions_short_all', 'N/A')}")
        lines.append(f"  Open Interest: {latest.get('open_interest_all', 'N/A')}")

        if len(cot) >= 2:
            prev = cot[1]
            lines.append(f"  Previous Week ({prev.get('report_date_as_yyyy_mm_dd', '?')}):")
            lines.append(f"    Non-Comm Long: {prev.get('noncomm_positions_long_all', 'N/A')}")
            lines.append(f"    Non-Comm Short: {prev.get('noncomm_positions_short_all', 'N/A')}")
            lines.append(f"    Open Interest: {prev.get('open_interest_all', 'N/A')}")
    else:
        lines.append("  COT data: N/A (CFTC API may be unavailable)")
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
def call_claude(data_text: str, previous_rec: str) -> dict:
    """Call Claude API with aggregated data. Retries on timeout.

    Uses a total time budget (WALL_LIMIT_SECS) to guarantee the function
    never exceeds the GitHub Actions job timeout, even in the worst-case
    retry path.
    """
    print("🤖 Calling Claude API...")

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
        "thinking": {
            "type": "adaptive",
        },
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "messages": [{"role": "user", "content": prompt}],
    }

    req = Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    max_retries = 3
    timeout_secs = 240       # 4 min per attempt (adaptive thinking is faster)
    WALL_LIMIT_SECS = 660    # 11 min total wall-clock budget for API calls
    wall_start = time.monotonic()

    for attempt in range(1, max_retries + 1):
        elapsed = time.monotonic() - wall_start
        remaining = WALL_LIMIT_SECS - elapsed
        if remaining < 60:
            print(f"  ⏱ Only {remaining:.0f}s left in wall-clock budget, aborting retries")
            return {"error": f"Wall-clock budget exhausted after {elapsed:.0f}s"}

        # Use the smaller of per-attempt timeout and remaining budget
        effective_timeout = min(timeout_secs, int(remaining))

        try:
            print(f"  Attempt {attempt}/{max_retries} (timeout: {effective_timeout}s, "
                  f"wall: {elapsed:.0f}s/{WALL_LIMIT_SECS}s)...")
            with urlopen(req, timeout=effective_timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                print(f"  ✅ Claude responded (stop_reason: {result.get('stop_reason')})")
                return result
        except (URLError, HTTPError) as e:
            body = ""
            if hasattr(e, "read"):
                body = e.read().decode("utf-8", errors="replace")[:500]
            print(f"  ⚠ Attempt {attempt} failed: {e}")
            if body:
                print(f"    Body: {body}")
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ All {max_retries} attempts failed")
                return {"error": str(e), "body": body}
        except Exception as e:
            print(f"  ⚠ Attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt < max_retries:
                wait = 10 * attempt
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

    subject = f"₿ Bitcoin Report - {datetime.now(timezone.utc).strftime('%Y-%m-%d')} | {action} | {conviction}"

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

    snapshot = {
        "date": today,
        "btc_price": _safe_float(md.get("current_price", {}).get("usd")),
        "dfii10": _safe_float(data.get("dfii10", {}).get("value")),
        "dollar_idx": _safe_float(data.get("dollar", {}).get("value")),
        "vix": _safe_float(data.get("vix", {}).get("value")),
        "fng": fng_val,
        "hash_eh": round(hash_eh, 2) if hash_eh else None,
        "cot_net_spec": _safe_int(
            cot_latest.get("noncomm_positions_long_all"),
            cot_latest.get("noncomm_positions_short_all"),
        ),
        "cot_oi": _safe_float(cot_latest.get("open_interest_all")),
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
        "Date       | BTC($)  | Real10Y | DXY    | VIX   | F&G | Hash(EH) | NetSpec | OI     | Rec",
        "-----------|---------|---------|--------|-------|-----|----------|---------|--------|----",
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
        rec = h.get("recommendation", "?")[:4]

        lines.append(
            f"{h.get('date', '?'):10s} | {btc} | {dfii} | {dxy} | {vix} | {fng} | {hsh} | {net} | {oi} | {rec}"
        )

    lines.append("")
    lines.append("KEY: F&G = Fear & Greed (0=Extreme Fear, 100=Extreme Greed)")
    lines.append("     NetSpec = Non-Commercial Longs minus Shorts | OI = Open Interest")
    lines.append("     Hash(EH) = Hash Rate in Exahash/s")
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
    print(f"  Data quality: {ok_count}/9 OK, {missing_count}/9 MISSING")
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
            f.write(f"Data quality: {ok_count}/9 sources OK\n")

    print("\n✅ Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n💥 Unhandled error: {e}")
        traceback.print_exc()
        sys.exit(1)
