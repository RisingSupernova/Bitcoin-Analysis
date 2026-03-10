# ₿ Bitcoin Analyst Daily Report

Automated daily Bitcoin analysis using Claude AI with multi-agent framework.

## Data Sources (12 APIs)

| # | Source | Data |
|---|--------|------|
| 1 | FRED DFII10 | 10Y Real Yields |
| 2 | FRED DTWEXBGS | Trade-Weighted Dollar Index |
| 3 | FRED VIXCLS | VIX Volatility |
| 4 | CoinGecko | BTC price, market cap, ATH, supply |
| 5 | Alternative.me | Fear & Greed Index (30-day) |
| 6 | Mempool.space | Hash rate (1 month) |
| 7 | Mempool.space | Difficulty adjustment |
| 8 | CFTC | Bitcoin Financial Futures COT positioning |
| 9 | Blockchain.info | Hash rate backup (30 days) |
| 10 | SoSoValue | Spot BTC ETF flows (daily + historical) |
| 11 | CryptoCompare | BTC OHLCV daily candles (365 bars) |
| 12 | CryptoCompare | BTC OHLCV 4-hour candles (200 bars) |

## Analysis Framework

6-agent analysis + CIO Orchestrator synthesis:

1. **Macro Analyst** — Real rates, DXY, VIX
2. **On-Chain Analyst** — Price vs ATH, cycle position, supply
3. **Mining Analyst** — Hash rate, difficulty, miner health
4. **ETF Flow Analyst** — Spot BTC ETF flows (SoSoValue hard data)
5. **Derivatives/Positioning** — CFTC COT, Fear & Greed sentiment
6. **Technical Analyst** — EMA structure, RSI, MACD, Bollinger Bands across daily/4H/weekly timeframes

### Technical Indicators (Agent 6)

Pure Python computation (no pip dependencies) of:
- **EMA 20/50/200** with SMA seed — alignment detection (bullish/bearish/mixed stack)
- **RSI-14** with Wilder smoothing — overbought/oversold detection
- **MACD (12, 26, 9)** — histogram trend (expanding/contracting)
- **Bollinger Bands (20, 2)** — relative squeeze detection (percentile-based, not fixed threshold)
- **EMA Slope** — trend strength proxy (replaces ADX)
- **Multi-timeframe alignment** — daily, 4H, weekly consensus

## Features

- **Rolling History**: 45-day data log for trend analysis (committed to git)
- **Halving Cycle**: Automatic phase detection (Early/Mid/Late Bull, Early/Late Bear)
- **Extended Thinking**: Claude Sonnet 4.6 with 16k thinking budget
- **Web Search**: Live market conditions augmentation
- **Hard Data ETF Flows**: Exact SoSoValue API numbers, not web search estimates
- **Technical Analysis**: Pre-computed indicators across 3 timeframes with MTF alignment
- **Mobile-Friendly**: Gmail-compatible HTML with inline CSS, card layout
- **Preamble Stripping**: Clean HTML output, no AI "thinking out loud"
- **Graceful Degradation**: Missing data sources default to NEUTRAL signals

## Setup

### 1. Create GitHub Repository

Create a new **private** repository and push these files.

### 2. Add Secrets

Go to **Settings → Secrets and variables → Actions** and add:

| Secret | Description |
|--------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key |
| `FRED_API_KEY` | [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `GMAIL_ADDRESS` | Gmail address (sender + recipient) |
| `GMAIL_APP_PASSWORD` | [Gmail App Password](https://myaccount.google.com/apppasswords) |

### 3. Run

- **Automatic**: Runs Mon-Fri at 06:00 Malta time (summer)
- **Manual**: Actions tab → Bitcoin Analyst Daily Report → Run workflow

## Schedule

Cron: `0 4 * * 1-5` (04:00 UTC)
- Summer (CEST): 06:00 Malta
- Winter (CET): 05:00 Malta

## File Structure

```
├── btc_analyst.py              # Main script (stdlib only, no pip)
├── prompt_template.txt         # Claude analysis prompt (6-agent framework)
├── .github/workflows/
│   └── btc_analyst.yml         # GitHub Actions workflow
├── data/
│   ├── previous_recommendation.json
│   ├── history.json            # 45-day rolling data
│   └── report_YYYYMMDD.html   # Saved reports (if email fails)
└── README.md
```
