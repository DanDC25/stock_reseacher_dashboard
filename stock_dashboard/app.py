# -*- coding: utf-8 -*-
"""
app.py  –  Interactive Client-Facing Stock Research Dashboard

Run with:  streamlit run app.py

Uses stock_engine.py as the single source of truth for all data & metrics.
Uses Plotly for every chart.  Streamlit handles layout, caching, and interactivity.

Customisation tips
------------------
* Colours & fonts        → search "STYLE" comments below.
* Chart heights / widths → tweak the `height=` arg on each Plotly figure.
* Dark/light defaults    → change `st.session_state.dark_mode` init.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os, sys, time

# ── Ensure the engine module is importable ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stock_engine as engine

# ---------------------------------------------------------------------------
# Page config  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"

# ---------------------------------------------------------------------------
# Dark / light theme  (STYLE – tweak colours here)
# ---------------------------------------------------------------------------
dark_mode = st.session_state.dark_mode

# Plotly template matching the toggle
PLOTLY_TEMPLATE = "plotly_dark" if dark_mode else "plotly_white"
BG_COLOR = "#0E1117" if dark_mode else "#FFFFFF"
CARD_BG = "#1E1E2F" if dark_mode else "#F7F7FA"
TEXT_COLOR = "#FAFAFA" if dark_mode else "#1E1E2F"
ACCENT_GREEN = "#00C853"
ACCENT_RED = "#FF1744"
ACCENT_YELLOW = "#FFD600"
GRID_COLOR = "#2A2A3C" if dark_mode else "#E0E0E0"

# Inject custom CSS for card styling & responsiveness
st.markdown(f"""
<style>
    /* Main background */
    .stApp {{
        background-color: {BG_COLOR};
    }}
    /* Metric cards */
    .metric-card {{
        background: {CARD_BG};
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        border: 1px solid {'#2A2A3C' if dark_mode else '#E0E0E0'};
    }}
    .metric-card h4 {{
        margin: 0 0 4px 0;
        font-size: 0.85rem;
        color: {'#888' if dark_mode else '#666'};
    }}
    .metric-card .value {{
        font-size: 1.35rem;
        font-weight: 700;
        color: {TEXT_COLOR};
    }}
    /* Verdict badge */
    .verdict-badge {{
        display: inline-block;
        padding: 14px 32px;
        border-radius: 14px;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 1px;
        text-align: center;
    }}
    .verdict-BUY  {{ background: {ACCENT_GREEN}22; color: {ACCENT_GREEN}; border: 2px solid {ACCENT_GREEN}; }}
    .verdict-HOLD {{ background: {ACCENT_YELLOW}22; color: {ACCENT_YELLOW}; border: 2px solid {ACCENT_YELLOW}; }}
    .verdict-SELL {{ background: {ACCENT_RED}22; color: {ACCENT_RED}; border: 2px solid {ACCENT_RED}; }}
    /* Sentiment pills */
    .sentiment-Positive {{ color: {ACCENT_GREEN}; font-weight: 600; }}
    .sentiment-Negative {{ color: {ACCENT_RED}; font-weight: 600; }}
    .sentiment-Neutral  {{ color: {ACCENT_YELLOW}; font-weight: 600; }}
    /* Hide default Streamlit hamburger for cleaner look */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    /* Responsive tweaks */
    @media (max-width: 768px) {{
        .verdict-badge {{ font-size: 1.1rem; padding: 10px 18px; }}
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: tooltip-enabled metric card (HTML)
# ---------------------------------------------------------------------------
# METRIC_TOOLTIPS – plain-English explanations for non-technical clients
METRIC_TOOLTIPS = {
    "Market Cap": "Total market value of all outstanding shares. Bigger = larger company.",
    "P/E Ratio": "Price-to-Earnings ratio. Compares the stock price to earnings per share. Lower may indicate better value.",
    "EPS": "Earnings Per Share. The portion of profit allocated to each share of stock.",
    "52-Wk High": "The highest price the stock has traded at over the past 52 weeks.",
    "52-Wk Low": "The lowest price the stock has traded at over the past 52 weeks.",
    "Div Yield": "Annual dividend as a percentage of the stock price. Higher = more income per dollar invested.",
    "Beta": "Measures volatility relative to the market. >1 means more volatile, <1 means less.",
    "Net Margin": "What percentage of revenue becomes profit after all expenses.",
}


def metric_card_html(label: str, value: str) -> str:
    """Return an HTML snippet for a styled metric card with a tooltip."""
    tooltip = METRIC_TOOLTIPS.get(label, "")
    return f"""
    <div class="metric-card" title="{tooltip}">
        <h4>{label} ℹ️</h4>
        <div class="value">{value}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Top bar: ticker search + dark/light toggle
# ---------------------------------------------------------------------------
top_left, top_right = st.columns([4, 1])

with top_left:
    ticker_input = st.text_input(
        "🔍  Enter a stock ticker",
        value=st.session_state.ticker,
        placeholder="e.g. AAPL, MSFT, TSLA, NVDA …",
        key="ticker_input",
    )
    # Normalise to uppercase
    ticker = ticker_input.strip().upper()

with top_right:
    # Dark / light toggle
    toggle_label = "🌙 Dark" if dark_mode else "☀️ Light"
    if st.button(toggle_label, use_container_width=True):
        st.session_state.dark_mode = not dark_mode
        st.rerun()

# Store normalised ticker back
st.session_state.ticker = ticker

# ---------------------------------------------------------------------------
# Main data loading  (cached for speed; clears on ticker change)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Fetching market data …", ttl=300)
def load_all_data(tkr: str):
    """Fetch everything we need in one cached call."""
    result = engine.predict_stock_action(tkr)
    hist = engine.get_price_history(tkr, period="1y")
    q_trends = engine.get_quarterly_trends(tkr, quarters=8)
    peers = engine.get_peer_comparison(tkr, max_peers=3)
    news = engine.get_news_headlines(tkr, count=5)
    narration = engine.generate_narration_script(tkr, result)
    return result, hist, q_trends, peers, news, narration


# ── Load data with graceful error handling ───────────────────────────────
try:
    result, hist, q_trends, peers, news, narration = load_all_data(ticker)
except ValueError as ve:
    st.error(f"⚠️  {ve}")
    st.info("Please enter a valid stock ticker symbol (e.g. AAPL, MSFT, TSLA).")
    st.stop()
except Exception as exc:
    st.error(f"⚠️  Something went wrong: {exc}")
    st.info(
        "This is likely a temporary issue with the data provider. "
        "Please wait a moment and refresh."
    )
    st.stop()

metrics = result["metrics"]
bench = result["benchmarks"]

# ---------------------------------------------------------------------------
# Header: company name + verdict badge
# ---------------------------------------------------------------------------
st.markdown(f"## {metrics['company_name']}  ({ticker})")

v_col1, v_col2, v_col3 = st.columns([2, 1, 1])
with v_col1:
    verdict = result["verdict"]
    conf = result["confidence"]
    st.markdown(
        f'<div class="verdict-badge verdict-{verdict}">'
        f'{verdict} &nbsp;·&nbsp; {conf:.0f}% Confidence'
        f'</div>',
        unsafe_allow_html=True,
    )
with v_col2:
    price_str = f"${metrics['current_price']:,.2f}" if metrics["current_price"] else "N/A"
    st.metric("Current Price", price_str)
with v_col3:
    st.metric("Sector", bench["sector"])

st.divider()

# ---------------------------------------------------------------------------
# Key Metrics summary cards  (with tooltips)
# ---------------------------------------------------------------------------
st.subheader("Key Metrics")
mc1, mc2, mc3, mc4 = st.columns(4)
mc5, mc6, mc7, mc8 = st.columns(4)

with mc1:
    st.markdown(metric_card_html("Market Cap", engine.format_number(metrics.get("market_cap"))), unsafe_allow_html=True)
with mc2:
    pe_val = f'{metrics["pe_ratio"]:.2f}' if metrics.get("pe_ratio") else "N/A"
    st.markdown(metric_card_html("P/E Ratio", pe_val), unsafe_allow_html=True)
with mc3:
    eps_val = f'${metrics["eps"]:.2f}' if metrics.get("eps") else "N/A"
    st.markdown(metric_card_html("EPS", eps_val), unsafe_allow_html=True)
with mc4:
    high_val = f'${metrics["fifty_two_week_high"]:,.2f}' if metrics.get("fifty_two_week_high") else "N/A"
    st.markdown(metric_card_html("52-Wk High", high_val), unsafe_allow_html=True)
with mc5:
    low_val = f'${metrics["fifty_two_week_low"]:,.2f}' if metrics.get("fifty_two_week_low") else "N/A"
    st.markdown(metric_card_html("52-Wk Low", low_val), unsafe_allow_html=True)
with mc6:
    div_val = f'{metrics["dividend_yield"]:.2%}' if metrics.get("dividend_yield") else "0.00%"
    st.markdown(metric_card_html("Div Yield", div_val), unsafe_allow_html=True)
with mc7:
    beta_val = f'{metrics["beta"]:.2f}' if metrics.get("beta") else "N/A"
    st.markdown(metric_card_html("Beta", beta_val), unsafe_allow_html=True)
with mc8:
    nm_val = f'{metrics["net_margin"]:.1%}' if metrics.get("net_margin") else "N/A"
    st.markdown(metric_card_html("Net Margin", nm_val), unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Chart Panel 1 & 2:  Candlestick with Volume  +  RSI & MACD
# ---------------------------------------------------------------------------
st.subheader("Price History & Technical Indicators")

# ── Candlestick + Volume ─────────────────────────────────────────────────
fig_candle = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.75, 0.25],
    subplot_titles=("", "Volume"),
)

fig_candle.add_trace(
    go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"],
        increasing_line_color=ACCENT_GREEN,
        decreasing_line_color=ACCENT_RED,
        name="Price",
    ),
    row=1, col=1,
)

# Volume bars coloured by direction
colors = [
    ACCENT_GREEN if c >= o else ACCENT_RED
    for c, o in zip(hist["Close"], hist["Open"])
]
fig_candle.add_trace(
    go.Bar(x=hist.index, y=hist["Volume"], marker_color=colors,
           opacity=0.5, name="Volume", showlegend=False),
    row=2, col=1,
)

fig_candle.update_layout(
    template=PLOTLY_TEMPLATE,
    height=520,
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=30, b=10),
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=BG_COLOR,
    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
)
fig_candle.update_xaxes(gridcolor=GRID_COLOR)
fig_candle.update_yaxes(gridcolor=GRID_COLOR)

st.plotly_chart(fig_candle, use_container_width=True)

# ── RSI & MACD ──────────────────────────────────────────────────────────
rsi = engine.compute_rsi(hist["Close"])
macd_line, signal_line, macd_hist = engine.compute_macd(hist["Close"])

fig_tech = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("RSI (14)", "MACD (12, 26, 9)"),
)

# RSI
fig_tech.add_trace(
    go.Scatter(x=rsi.index, y=rsi, line=dict(color="#7C4DFF", width=1.5), name="RSI"),
    row=1, col=1,
)
fig_tech.add_hline(y=70, line_dash="dash", line_color=ACCENT_RED, opacity=0.5, row=1, col=1)
fig_tech.add_hline(y=30, line_dash="dash", line_color=ACCENT_GREEN, opacity=0.5, row=1, col=1)

# MACD
fig_tech.add_trace(
    go.Scatter(x=macd_line.index, y=macd_line, line=dict(color="#29B6F6", width=1.5), name="MACD"),
    row=2, col=1,
)
fig_tech.add_trace(
    go.Scatter(x=signal_line.index, y=signal_line, line=dict(color="#FF7043", width=1.5), name="Signal"),
    row=2, col=1,
)
hist_colors = [ACCENT_GREEN if v >= 0 else ACCENT_RED for v in macd_hist]
fig_tech.add_trace(
    go.Bar(x=macd_hist.index, y=macd_hist, marker_color=hist_colors, name="Histogram", opacity=0.6),
    row=2, col=1,
)

fig_tech.update_layout(
    template=PLOTLY_TEMPLATE, height=420,
    margin=dict(l=10, r=10, t=40, b=10),
    paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
    legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
)
fig_tech.update_xaxes(gridcolor=GRID_COLOR)
fig_tech.update_yaxes(gridcolor=GRID_COLOR)

st.plotly_chart(fig_tech, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Chart Panel 3 & 4 side-by-side: Valuation  |  Revenue & Earnings
# ---------------------------------------------------------------------------
val_col, earn_col = st.columns(2)

# ── Valuation: P/E and EV/EBITDA vs sector ───────────────────────────────
with val_col:
    st.subheader("Valuation vs Sector")

    pe_val = metrics.get("pe_ratio") or 0
    ev_val = metrics.get("ev_to_ebitda") or 0
    # sector EV/EBITDA approximation (typically ~60-70% of sector P/E)
    sector_ev = bench["pe"] * 0.65

    fig_val = go.Figure()
    categories = ["P/E Ratio", "EV/EBITDA"]
    company_vals = [pe_val, ev_val]
    sector_vals = [bench["pe"], sector_ev]

    fig_val.add_trace(go.Bar(
        x=categories, y=company_vals,
        name=ticker, marker_color="#7C4DFF",
        text=[f"{v:.1f}" for v in company_vals], textposition="outside",
    ))
    fig_val.add_trace(go.Bar(
        x=categories, y=sector_vals,
        name="Sector Median", marker_color="#546E7A",
        text=[f"{v:.1f}" for v in sector_vals], textposition="outside",
    ))

    fig_val.update_layout(
        template=PLOTLY_TEMPLATE, height=380,
        barmode="group",
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    fig_val.update_yaxes(gridcolor=GRID_COLOR)
    st.plotly_chart(fig_val, use_container_width=True)

# ── Revenue & Earnings Trends ────────────────────────────────────────────
with earn_col:
    st.subheader("Revenue & Earnings Trends")

    if not q_trends.empty:
        fig_re = make_subplots(specs=[[{"secondary_y": True}]])

        fig_re.add_trace(
            go.Bar(
                x=q_trends["Quarter"], y=q_trends["Revenue"],
                name="Revenue", marker_color="#29B6F6", opacity=0.7,
            ),
            secondary_y=False,
        )
        fig_re.add_trace(
            go.Scatter(
                x=q_trends["Quarter"], y=q_trends["EPS"],
                name="EPS", line=dict(color=ACCENT_GREEN, width=2.5),
                mode="lines+markers",
            ),
            secondary_y=True,
        )

        fig_re.update_layout(
            template=PLOTLY_TEMPLATE, height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        )
        fig_re.update_yaxes(title_text="Revenue", gridcolor=GRID_COLOR, secondary_y=False)
        fig_re.update_yaxes(title_text="EPS ($)", gridcolor=GRID_COLOR, secondary_y=True)
        st.plotly_chart(fig_re, use_container_width=True)
    else:
        st.info("Quarterly data not available for this ticker.")

st.divider()

# ---------------------------------------------------------------------------
# Sector Analysis Table
# ---------------------------------------------------------------------------
st.subheader("Sector Analysis")

analysis_df = pd.DataFrame(result["analysis"])
if not analysis_df.empty:
    # Colour-code the Signal column
    def colour_signal(val):
        if val == "BUY":
            return f"color: {ACCENT_GREEN}; font-weight: 700"
        else:
            return f"color: {ACCENT_RED}; font-weight: 700"

    styled = analysis_df.rename(columns={
        "metric": "Metric", "value": "Current Value",
        "target": "Sector Target", "signal": "Signal",
    }).style.map(colour_signal, subset=["Signal"])  # pandas >=2.1 uses .map()

    st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Peer Comparison
# ---------------------------------------------------------------------------
st.subheader("Peer Comparison")

if not peers.empty:
    st.dataframe(peers, use_container_width=True, hide_index=True)
else:
    st.info("Peer data unavailable.")

st.divider()

# ---------------------------------------------------------------------------
# News Sentiment Feed
# ---------------------------------------------------------------------------
st.subheader("News Sentiment")

if news:
    for item in news:
        sent_class = f'sentiment-{item["sentiment"]}'
        st.markdown(
            f'<div class="metric-card">'
            f'<span class="{sent_class}">[{item["sentiment"]}]</span> &nbsp;'
            f'<a href="{item["link"]}" target="_blank" '
            f'style="color:{TEXT_COLOR}; text-decoration:none; font-weight:600;">'
            f'{item["title"]}</a><br>'
            f'<small style="color:#888;">{item["publisher"]} · {item["published"]}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )
else:
    st.info("No recent news found for this ticker.")

st.divider()

# ---------------------------------------------------------------------------
# Video Generation Section
# ---------------------------------------------------------------------------
st.subheader("🎬 Generate AI-Narrated Video Report")
st.caption(
    "Creates a ~30-second video summarising the analysis with AI voiceover. "
    "This will take 2–4 minutes to render."
)

if st.button("Generate Video", type="primary", use_container_width=False):
    with st.spinner("Rendering video … this takes 2–4 minutes"):
        try:
            from video_generator import generate_report_video

            video_path = generate_report_video(
                ticker=ticker,
                result=result,
                hist=hist,
                narration_text=narration,
                dark_mode=dark_mode,
            )

            if os.path.exists(video_path):
                st.success("Video generated successfully.")
                with open(video_path, "rb") as f:
                    st.download_button(
                        label="⬇ Download Video Report",
                        data=f,
                        file_name=f"{ticker}_report_{pd.Timestamp.now().strftime('%Y%m%d')}.mp4",
                        mime="video/mp4",
                    )
                st.video(video_path)
            else:
                st.error("Video file was not created. Check logs for details.")
        except ImportError:
            st.error(
                "Video generator module not found. "
                "Make sure video_generator.py is in the same directory."
            )
        except Exception as exc:
            st.error(f"Video generation failed: {exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Data sourced from Yahoo Finance. This dashboard is for informational purposes only "
    "and does not constitute financial advice. Past performance is not indicative of future results."
)
