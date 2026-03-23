# -*- coding: utf-8 -*-
"""
stock_engine.py  –  Single source of truth for all data fetching & metric calculations.

Refactored from the original stock_researcher.py (Colab notebook) into a clean,
importable module.  Every public function accepts a ticker string and returns
structured data (dicts / DataFrames) so the dashboard and video layers never
touch yfinance directly.

Customisation tips
------------------
* SECTOR_PEER_MAP  – add / remove sectors & peers as your coverage grows.
* GROWTH_GATE      – the minimum CAGR to count as "BUY" on revenue growth.
* Thresholds in `predict_stock_action` control the Buy / Hold / Sell logic.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuration – tweak these to adjust sensitivity
# ---------------------------------------------------------------------------

# Maps sectors → representative peer tickers for live benchmarking
SECTOR_PEER_MAP = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "Communication Services": ["META", "NFLX", "DIS", "TMUS", "VZ"],
    "Energy": ["XOM", "CVX", "SHEL", "COP", "BP"],
    "Financial Services": ["JPM", "BAC", "WFC", "C", "GS"],
    "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "LLY"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST"],
    "Consumer Cyclical": ["TSLA", "HD", "ORCL", "NKE", "SBUX"],
    "Industrials": ["CAT", "HON", "UNP", "GE", "RTX"],
    "Real Estate": ["PLD", "AMT", "CCI", "SPG", "EQIX"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "Basic Materials": ["LIN", "APD", "ECL", "SHW", "NEM"],
}

# Minimum CAGR to qualify as a "BUY" signal on revenue growth
GROWTH_GATE = 0.10  # 10 %

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_number(number):
    """Format a number into human-readable units (B / M / K)."""
    if number is None or (isinstance(number, float) and np.isnan(number)):
        return "N/A"
    number = float(number)
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}B"
    elif abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    elif abs(number) >= 1_000:
        return f"{number / 1_000:.2f}K"
    else:
        return f"{number:.2f}"


def _safe_get(series_or_df, key, default=None):
    """Safely index into a pandas object, returning *default* on any error."""
    try:
        val = series_or_df.loc[key]
        if isinstance(val, pd.Series):
            val = val.dropna()
            if val.empty:
                return default
            return val.iloc[0]
        return val
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Core data-fetching functions
# ---------------------------------------------------------------------------

def get_company_info(ticker: str) -> dict:
    """Return the full yfinance info dict for *ticker*.
    
    Raises ValueError with a user-friendly message on invalid ticker.
    """
    try:
        obj = yf.Ticker(ticker)
        info = obj.info
        # yfinance returns a nearly-empty dict for bogus tickers
        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            # Try fetching history as a second check
            hist = obj.history(period="5d")
            if hist.empty:
                raise ValueError(f"'{ticker}' does not appear to be a valid ticker symbol.")
        return info
    except Exception as exc:
        if "ValueError" in type(exc).__name__:
            raise
        raise ValueError(f"Could not fetch data for '{ticker}'. Check the symbol and try again.") from exc


def get_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Return OHLCV DataFrame for the given period."""
    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        raise ValueError(f"No price data available for '{ticker}'.")
    return df


def get_financials(ticker: str) -> dict:
    """Return annual income statement, balance sheet, and cash-flow statement."""
    obj = yf.Ticker(ticker)
    return {
        "income": obj.income_stmt,
        "balance": obj.balance_sheet,
        "cashflow": obj.cashflow,
    }


def get_quarterly_financials(ticker: str) -> dict:
    """Return quarterly income statement and earnings data for trending."""
    obj = yf.Ticker(ticker)
    return {
        "income": obj.quarterly_income_stmt,
        "balance": obj.quarterly_balance_sheet,
    }


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Relative Strength Index on a closing-price Series."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices: pd.Series,
                 fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram) tuple of Series."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Fundamental metric calculations  (mirrors original notebook logic)
# ---------------------------------------------------------------------------

def compute_fundamental_metrics(ticker: str) -> dict:
    """
    Compute all fundamental metrics for *ticker*.

    Returns a dict with keys:
        cagr, gross_margin, operating_margin, net_margin,
        current_ratio, debt_to_equity, net_debt, debt_to_asset,
        interest_coverage, pe_ratio, pfcf_ratio, roe,
        market_cap, current_price, shares_outstanding,
        eps, fifty_two_week_high, fifty_two_week_low, dividend_yield,
        sector, company_name
    """
    info = get_company_info(ticker)
    fins = get_financials(ticker)
    inc = fins["income"]
    bal = fins["balance"]
    cf = fins["cashflow"]

    result = {}
    result["company_name"] = info.get("longName", info.get("shortName", ticker))
    result["sector"] = info.get("sector", "Unknown")

    # ── Price & market data ──────────────────────────────────────────
    result["current_price"] = info.get("currentPrice") or info.get("regularMarketPrice")
    result["market_cap"] = info.get("marketCap")
    result["shares_outstanding"] = info.get("sharesOutstanding")
    result["eps"] = info.get("trailingEps")
    result["fifty_two_week_high"] = info.get("fiftyTwoWeekHigh")
    result["fifty_two_week_low"] = info.get("fiftyTwoWeekLow")
    result["dividend_yield"] = info.get("dividendYield")
    result["pe_ratio_info"] = info.get("trailingPE")  # from info, as fallback
    result["ev_to_ebitda"] = info.get("enterpriseToEbitda")
    result["beta"] = info.get("beta")

    # ── Revenue CAGR ─────────────────────────────────────────────────
    try:
        rev = inc.loc["Total Revenue"].dropna()
        n = len(rev) - 1
        if n > 0:
            result["cagr"] = ((rev.iloc[0] / rev.iloc[-1]) ** (1 / n)) - 1
        else:
            result["cagr"] = None
    except Exception:
        result["cagr"] = None

    # ── Profit margins ───────────────────────────────────────────────
    try:
        gp = _safe_get(inc, "Gross Profit")
        tr = _safe_get(inc, "Total Revenue")
        result["gross_margin"] = gp / tr if gp and tr else None
    except Exception:
        result["gross_margin"] = None

    try:
        gp = _safe_get(inc, "Gross Profit")
        oe = _safe_get(inc, "Operating Expense")
        tr = _safe_get(inc, "Total Revenue")
        if gp is not None and oe is not None and tr:
            result["operating_margin"] = (gp - oe) / tr
        else:
            result["operating_margin"] = None
    except Exception:
        result["operating_margin"] = None

    try:
        ni = _safe_get(inc, "Net Income")
        tr = _safe_get(inc, "Total Revenue")
        result["net_margin"] = ni / tr if ni and tr else None
    except Exception:
        result["net_margin"] = None

    # ── Balance-sheet ratios ─────────────────────────────────────────
    try:
        ca = _safe_get(bal, "Current Assets")
        cl = _safe_get(bal, "Current Liabilities")
        result["current_ratio"] = ca / cl if ca and cl else None
    except Exception:
        result["current_ratio"] = None

    try:
        td = _safe_get(bal, "Total Debt")
        se = _safe_get(bal, "Stockholders Equity")
        result["debt_to_equity"] = td / se if td and se else None
    except Exception:
        result["debt_to_equity"] = None

    try:
        result["net_debt"] = _safe_get(bal, "Net Debt")
    except Exception:
        result["net_debt"] = None

    try:
        td = _safe_get(bal, "Total Debt")
        ta = _safe_get(bal, "Total Assets")
        result["debt_to_asset"] = td / ta if td and ta else None
    except Exception:
        result["debt_to_asset"] = None

    # ── Interest coverage ────────────────────────────────────────────
    try:
        oi = _safe_get(inc, "Operating Income")
        ie = _safe_get(inc, "Interest Expense")
        result["interest_coverage"] = oi / ie if oi and ie else None
    except Exception:
        result["interest_coverage"] = None

    # ── Valuation ────────────────────────────────────────────────────
    try:
        ni = _safe_get(inc, "Net Income")
        mc = result["market_cap"]
        result["pe_ratio"] = mc / ni if mc and ni and ni != 0 else result["pe_ratio_info"]
    except Exception:
        result["pe_ratio"] = result.get("pe_ratio_info")

    try:
        fcf = _safe_get(cf, "Free Cash Flow")
        mc = result["market_cap"]
        result["pfcf_ratio"] = mc / fcf if mc and fcf and fcf != 0 else None
    except Exception:
        result["pfcf_ratio"] = None

    # ── ROE ──────────────────────────────────────────────────────────
    try:
        ni = _safe_get(inc, "Net Income")
        se = _safe_get(bal, "Stockholders Equity")
        result["roe"] = ni / se if ni and se and se != 0 else None
    except Exception:
        result["roe"] = None

    return result


# ---------------------------------------------------------------------------
# Sector benchmark engine  (mirrors original get_auto_benchmarks)
# ---------------------------------------------------------------------------

def get_sector_benchmarks(ticker: str) -> dict:
    """
    Fetch live sector-median P/E, net margin, and ROE from representative peers.

    Returns dict with keys: sector, pe, net_margin, roe, peers
    """
    info = get_company_info(ticker)
    sector = info.get("sector", "Technology")
    peers = SECTOR_PEER_MAP.get(sector, SECTOR_PEER_MAP["Technology"])

    pe_list, margin_list, roe_list = [], [], []
    for p in peers:
        try:
            p_info = yf.Ticker(p).info
            pe = p_info.get("trailingPE")
            margin = p_info.get("profitMargins")
            roe = p_info.get("returnOnEquity")
            if pe:
                pe_list.append(pe)
            if margin:
                margin_list.append(margin)
            if roe:
                roe_list.append(roe)
        except Exception:
            continue

    return {
        "sector": sector,
        "pe": float(np.median(pe_list)) if pe_list else 25.0,
        "net_margin": float(np.median(margin_list)) if margin_list else 0.15,
        "roe": float(np.median(roe_list)) if roe_list else 0.15,
        "peers": peers,
    }


# ---------------------------------------------------------------------------
# Buy / Hold / Sell verdict  (mirrors original predict_stock_action)
# ---------------------------------------------------------------------------

def predict_stock_action(ticker: str) -> dict:
    """
    Run the full multi-metric analysis and return a structured verdict.

    Returns dict with keys:
        verdict        – "BUY" / "HOLD" / "SELL"
        points         – int, number of metrics that passed
        total          – int, total metrics evaluated
        confidence     – float 0-100, percentage of metrics aligned
        analysis       – list of dicts [{metric, value, target, signal}, ...]
        metrics        – full dict from compute_fundamental_metrics
        benchmarks     – dict from get_sector_benchmarks
    """
    metrics = compute_fundamental_metrics(ticker)
    bench = get_sector_benchmarks(ticker)

    analysis = []
    points = 0
    total = 0

    # 1. P/E vs sector
    if metrics["pe_ratio"] is not None:
        total += 1
        passed = metrics["pe_ratio"] < bench["pe"]
        signal = "BUY" if passed else "OVERVALUED"
        analysis.append({
            "metric": "P/E Ratio",
            "value": f'{metrics["pe_ratio"]:.2f}',
            "target": f'< {bench["pe"]:.1f}x',
            "signal": signal,
        })
        if passed:
            points += 1

    # 2. P/FCF vs sector
    if metrics["pfcf_ratio"] is not None:
        total += 1
        pfcf_target = bench["pe"] * 0.9
        passed = metrics["pfcf_ratio"] < pfcf_target
        signal = "BUY" if passed else "EXPENSIVE"
        analysis.append({
            "metric": "P/FCF Ratio",
            "value": f'{metrics["pfcf_ratio"]:.2f}',
            "target": f'< {pfcf_target:.1f}x',
            "signal": signal,
        })
        if passed:
            points += 1

    # 3. ROE vs sector
    if metrics["roe"] is not None:
        total += 1
        passed = metrics["roe"] > bench["roe"]
        signal = "BUY" if passed else "WEAK"
        analysis.append({
            "metric": "ROE",
            "value": f'{metrics["roe"]:.2%}',
            "target": f'> {bench["roe"]:.1%}',
            "signal": signal,
        })
        if passed:
            points += 1

    # 4. Revenue CAGR
    if metrics["cagr"] is not None:
        total += 1
        passed = metrics["cagr"] > GROWTH_GATE
        signal = "BUY" if passed else "SLOW"
        analysis.append({
            "metric": "Revenue CAGR",
            "value": f'{metrics["cagr"]:.2%}',
            "target": f'> {GROWTH_GATE:.0%}',
            "signal": signal,
        })
        if passed:
            points += 1

    # 5. Net margin vs sector
    if metrics["net_margin"] is not None:
        total += 1
        passed = metrics["net_margin"] > bench["net_margin"]
        signal = "BUY" if passed else "THIN"
        analysis.append({
            "metric": "Net Margin",
            "value": f'{metrics["net_margin"]:.2%}',
            "target": f'> {bench["net_margin"]:.1%}',
            "signal": signal,
        })
        if passed:
            points += 1

    # ── Verdict logic ────────────────────────────────────────────────
    # Same thresholds as original: ≥4 BUY, 3 HOLD, ≤2 SELL
    confidence = (points / total * 100) if total > 0 else 0
    if points >= 4:
        verdict = "BUY"
    elif points == 3:
        verdict = "HOLD"
    else:
        verdict = "SELL"

    return {
        "verdict": verdict,
        "points": points,
        "total": total,
        "confidence": round(confidence, 1),
        "analysis": analysis,
        "metrics": metrics,
        "benchmarks": bench,
    }


# ---------------------------------------------------------------------------
# Peer comparison
# ---------------------------------------------------------------------------

def get_peer_comparison(ticker: str, max_peers: int = 3) -> pd.DataFrame:
    """
    Return a DataFrame comparing the target ticker against its sector peers
    on core metrics: P/E, EPS, Market Cap, Net Margin, ROE, Revenue CAGR.
    """
    bench = get_sector_benchmarks(ticker)
    # Remove the ticker itself from peers if present
    peers = [p for p in bench["peers"] if p.upper() != ticker.upper()][:max_peers]
    tickers_to_compare = [ticker.upper()] + peers

    rows = []
    for t in tickers_to_compare:
        try:
            info = yf.Ticker(t).info
            rows.append({
                "Ticker": t,
                "Company": info.get("shortName", t),
                "Market Cap": format_number(info.get("marketCap")),
                "P/E": round(info.get("trailingPE") or 0, 2),
                "EPS": round(info.get("trailingEps") or 0, 2),
                "Net Margin": f'{(info.get("profitMargins") or 0):.1%}',
                "ROE": f'{(info.get("returnOnEquity") or 0):.1%}',
                "Div Yield": f'{(info.get("dividendYield") or 0):.2%}',
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# News headlines (basic – uses yfinance .news)
# ---------------------------------------------------------------------------

def get_news_headlines(ticker: str, count: int = 5) -> list[dict]:
    """
    Return the most recent *count* news items for *ticker*.

    Each item: {title, publisher, link, published}
    Sentiment is assigned with a simple keyword heuristic.
    """
    try:
        news_items = yf.Ticker(ticker).news or []
    except Exception:
        return []

    positive_kw = {
        "surge", "soar", "rally", "gain", "beat", "record", "upgrade",
        "profit", "growth", "boom", "bullish", "buy", "outperform", "strong",
        "up", "high", "positive", "optimistic", "recover",
    }
    negative_kw = {
        "drop", "fall", "crash", "plunge", "miss", "loss", "downgrade",
        "decline", "risk", "concern", "sell", "bearish", "weak", "cut",
        "down", "low", "negative", "fear", "recession", "warning", "layoff",
    }

    results = []
    for item in news_items[:count]:
        # yfinance ≥0.2.31 nests data under "content"; older versions use flat keys
        content = item.get("content", item)  # unwrap if nested
        title = content.get("title", item.get("title", ""))
        
        # Publisher may be nested under provider.displayName
        provider = content.get("provider", {})
        publisher = (
            provider.get("displayName", "")
            if isinstance(provider, dict)
            else item.get("publisher", "")
        )
        
        # Link: try canonicalUrl.url, then clickThroughUrl.url, then flat "link"
        canon = content.get("canonicalUrl", {})
        click = content.get("clickThroughUrl", {})
        link = (
            (canon.get("url") if isinstance(canon, dict) else None)
            or (click.get("url") if isinstance(click, dict) else None)
            or item.get("link", "")
        )
        
        # Handle different timestamp formats
        pub_ts = (
            content.get("pubDate")
            or item.get("providerPublishTime")
            or item.get("publishedDate", "")
        )
        if isinstance(pub_ts, (int, float)):
            published = datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M")
        elif isinstance(pub_ts, str) and pub_ts:
            # ISO format from newer API
            try:
                published = datetime.fromisoformat(pub_ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
            except Exception:
                published = pub_ts
        else:
            published = str(pub_ts)

        # Simple keyword sentiment
        title_lower = title.lower()
        pos_count = sum(1 for w in positive_kw if w in title_lower)
        neg_count = sum(1 for w in negative_kw if w in title_lower)
        if pos_count > neg_count:
            sentiment = "Positive"
        elif neg_count > pos_count:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        results.append({
            "title": title,
            "publisher": publisher,
            "link": link,
            "published": published,
            "sentiment": sentiment,
        })

    return results


# ---------------------------------------------------------------------------
# Quarterly revenue & EPS for the Revenue & Earnings Trends panel
# ---------------------------------------------------------------------------

def get_quarterly_trends(ticker: str, quarters: int = 8) -> pd.DataFrame:
    """
    Return a DataFrame with quarterly Revenue and EPS for the last *quarters*.

    Columns: Quarter, Revenue, EPS
    """
    obj = yf.Ticker(ticker)
    q_inc = obj.quarterly_income_stmt

    rows = []
    try:
        revenues = q_inc.loc["Total Revenue"].dropna().head(quarters)
        net_incomes = q_inc.loc["Net Income"].dropna().head(quarters)
        shares = obj.info.get("sharesOutstanding", 1)

        for date in revenues.index:
            rev = revenues.get(date, None)
            ni = net_incomes.get(date, None) if date in net_incomes.index else None
            eps = ni / shares if ni is not None and shares else None
            rows.append({
                "Quarter": date.strftime("%Y Q") + str((date.month - 1) // 3 + 1),
                "Revenue": rev,
                "EPS": round(eps, 2) if eps is not None else None,
                "Date": date,
            })
    except Exception:
        pass

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Generate narration script for video
# ---------------------------------------------------------------------------

def generate_narration_script(ticker: str, result: dict) -> str:
    """
    Build a ~30-second narration script following the required structure:
      1. Current price & recent trend  (5 sec  ≈ 15 words)
      2. Top 2–3 supporting metrics     (10 sec ≈ 30 words)
      3. Key risk to watch              (8 sec  ≈ 24 words)
      4. Final verdict + confidence     (7 sec  ≈ 21 words)
    """
    m = result["metrics"]
    v = result["verdict"]
    c = result["confidence"]
    analysis = result["analysis"]

    name = m.get("company_name", ticker)
    price = m.get("current_price")
    price_str = f"${price:,.2f}" if price else "an undisclosed price"

    # ── Section 1: Price & trend ─────────────────────────────────────
    try:
        hist = get_price_history(ticker, period="1mo")
        month_ago_price = hist["Close"].iloc[0]
        change_pct = ((price - month_ago_price) / month_ago_price) * 100
        if change_pct >= 0:
            trend = f"up {abs(change_pct):.1f}% over the past month"
        else:
            trend = f"down {abs(change_pct):.1f}% over the past month"
    except Exception:
        trend = "with mixed recent performance"

    sec1 = f"{name} is currently trading at {price_str}, {trend}."

    # ── Section 2: Top supporting metrics ────────────────────────────
    buy_metrics = [a for a in analysis if a["signal"] == "BUY"]
    if len(buy_metrics) >= 2:
        m1, m2 = buy_metrics[0], buy_metrics[1]
        sec2 = (
            f"Key strengths include a {m1['metric']} of {m1['value']}, "
            f"beating the sector target of {m1['target']}, "
            f"and a {m2['metric']} of {m2['value']} versus the sector benchmark of {m2['target']}."
        )
    elif len(buy_metrics) == 1:
        m1 = buy_metrics[0]
        sec2 = (
            f"A notable strength is the {m1['metric']} at {m1['value']}, "
            f"which beats the sector target of {m1['target']}."
        )
    else:
        sec2 = "Currently, none of the core metrics are outperforming sector benchmarks."

    # ── Section 3: Key risk ──────────────────────────────────────────
    fail_metrics = [a for a in analysis if a["signal"] != "BUY"]
    if fail_metrics:
        risk = fail_metrics[0]
        sec3 = (
            f"The key risk to watch is {risk['metric']}, currently at {risk['value']}, "
            f"which falls short of the sector target of {risk['target']}. "
            f"This could pressure future returns."
        )
    else:
        sec3 = "No major risks stand out at this time, though macro conditions should always be monitored."

    # ── Section 4: Verdict ───────────────────────────────────────────
    sec4 = (
        f"Our overall assessment: {v}, with a confidence score of {c:.0f}%. "
        f"{name} passes {result['points']} out of {result['total']} sector gates."
    )

    return f"{sec1}\n\n{sec2}\n\n{sec3}\n\n{sec4}"
