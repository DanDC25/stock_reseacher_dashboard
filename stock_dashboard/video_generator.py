# -*- coding: utf-8 -*-
"""
video_generator.py  –  Auto-generate a ~30-second AI-narrated stock analysis video.

Creates:
  1. Title card (company name + ticker + date)
  2. Animated chart frames (candlestick, RSI, valuation)
  3. Verdict overlay
  4. AI voiceover (gTTS as default fallback; ElevenLabs if API key set)
  5. Composites everything with MoviePy

The `generate_report_video()` function is the single entry point called by
the dashboard.  It returns the path to the finished .mp4 file.

Customisation tips
------------------
* FONT_PATH          – change to a custom .ttf for branding.
* TITLE_DURATION     – seconds for the opening title card.
* FRAME_DURATION     – seconds per chart slide.
* VIDEO_FPS          – frames per second for the final render.
* BG_DARK / BG_LIGHT – background colour for chart images.
"""

import os
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime

# Chart rendering (static images for video frames)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Video compositing  (MoviePy v2 API – no more moviepy.editor)
from moviepy import (
    ImageClip, AudioFileClip, CompositeVideoClip,
    concatenate_videoclips, ColorClip, TextClip,
)

# TTS fallback
from gtts import gTTS

# Engine for metric helpers
import stock_engine as engine

# ---------------------------------------------------------------------------
# Configuration  (STYLE – tweak these)
# ---------------------------------------------------------------------------
TITLE_DURATION = 4        # seconds for opening title card
FRAME_DURATION = 7        # seconds per chart frame (fits narration pacing)
VERDICT_DURATION = 7      # seconds for closing verdict card
VIDEO_FPS = 24            # output video framerate
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# Colours
BG_DARK = "#0E1117"
BG_LIGHT = "#FFFFFF"
ACCENT_GREEN = "#00C853"
ACCENT_RED = "#FF1744"
ACCENT_YELLOW = "#FFD600"

# ---------------------------------------------------------------------------
# TTS: generate voiceover audio file
# ---------------------------------------------------------------------------

def _generate_voiceover(text: str, output_path: str) -> str:
    """
    Generate an MP3 voiceover file from *text*.

    Tries ElevenLabs first (if ELEVENLABS_API_KEY env var is set),
    otherwise falls back to Google TTS (gTTS).

    Returns the path to the generated audio file.
    """
    elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY")

    if elevenlabs_key:
        try:
            import requests
            # ElevenLabs API – using the "Rachel" voice (calm, narration)
            voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": elevenlabs_key,
                "Content-Type": "application/json",
            }
            payload = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.6,
                    "similarity_boost": 0.75,
                },
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                return output_path
        except Exception:
            pass  # fall through to gTTS

    # Fallback: gTTS
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Chart frame generators  (Plotly → static PNG)
# ---------------------------------------------------------------------------

def _save_plotly_fig(fig, path: str, width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
    """Write a Plotly figure to a PNG file."""
    fig.write_image(path, width=width, height=height, scale=2)


def _make_candlestick_frame(hist: pd.DataFrame, ticker: str,
                             dark: bool, out_path: str):
    """Generate a candlestick + volume chart image."""
    bg = BG_DARK if dark else BG_LIGHT
    text_c = "#FAFAFA" if dark else "#1E1E2F"
    template = "plotly_dark" if dark else "plotly_white"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"],
        increasing_line_color=ACCENT_GREEN,
        decreasing_line_color=ACCENT_RED,
        name="Price",
    ), row=1, col=1)

    colors = [ACCENT_GREEN if c >= o else ACCENT_RED
              for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"],
                         marker_color=colors, opacity=0.5,
                         showlegend=False), row=2, col=1)

    fig.update_layout(
        template=template, paper_bgcolor=bg, plot_bgcolor=bg,
        title=dict(text=f"{ticker} — Price History (1Y)",
                   font=dict(size=28, color=text_c)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=40, t=80, b=40),
        font=dict(color=text_c),
    )
    _save_plotly_fig(fig, out_path)


def _make_rsi_frame(hist: pd.DataFrame, ticker: str,
                    dark: bool, out_path: str):
    """Generate an RSI chart image."""
    bg = BG_DARK if dark else BG_LIGHT
    text_c = "#FAFAFA" if dark else "#1E1E2F"
    template = "plotly_dark" if dark else "plotly_white"

    rsi = engine.compute_rsi(hist["Close"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi,
                             line=dict(color="#7C4DFF", width=2.5),
                             name="RSI (14)"))
    fig.add_hline(y=70, line_dash="dash", line_color=ACCENT_RED, opacity=0.6)
    fig.add_hline(y=30, line_dash="dash", line_color=ACCENT_GREEN, opacity=0.6)
    fig.add_hrect(y0=30, y1=70, fillcolor="#7C4DFF", opacity=0.05)

    fig.update_layout(
        template=template, paper_bgcolor=bg, plot_bgcolor=bg,
        title=dict(text=f"{ticker} — RSI (Relative Strength Index)",
                   font=dict(size=28, color=text_c)),
        yaxis=dict(range=[0, 100]),
        margin=dict(l=60, r=40, t=80, b=40),
        font=dict(color=text_c),
    )
    _save_plotly_fig(fig, out_path)


def _make_valuation_frame(metrics: dict, bench: dict, ticker: str,
                          dark: bool, out_path: str):
    """Generate a valuation comparison bar chart image."""
    bg = BG_DARK if dark else BG_LIGHT
    text_c = "#FAFAFA" if dark else "#1E1E2F"
    template = "plotly_dark" if dark else "plotly_white"

    pe_val = metrics.get("pe_ratio") or 0
    ev_val = metrics.get("ev_to_ebitda") or 0
    sector_ev = bench["pe"] * 0.65

    fig = go.Figure()
    cats = ["P/E Ratio", "EV/EBITDA"]
    fig.add_trace(go.Bar(x=cats, y=[pe_val, ev_val], name=ticker,
                         marker_color="#7C4DFF",
                         text=[f"{v:.1f}" for v in [pe_val, ev_val]],
                         textposition="outside", textfont=dict(size=20)))
    fig.add_trace(go.Bar(x=cats, y=[bench["pe"], sector_ev],
                         name="Sector Median", marker_color="#546E7A",
                         text=[f"{v:.1f}" for v in [bench["pe"], sector_ev]],
                         textposition="outside", textfont=dict(size=20)))

    fig.update_layout(
        template=template, paper_bgcolor=bg, plot_bgcolor=bg,
        barmode="group",
        title=dict(text=f"{ticker} — Valuation vs Sector",
                   font=dict(size=28, color=text_c)),
        margin=dict(l=60, r=40, t=80, b=40),
        font=dict(color=text_c, size=16),
    )
    _save_plotly_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Title & verdict card generators
# ---------------------------------------------------------------------------

def _make_title_card(company_name: str, ticker: str,
                     dark: bool, out_path: str):
    """Generate a branded title card image using Plotly (no external fonts needed)."""
    bg = BG_DARK if dark else BG_LIGHT
    text_c = "#FAFAFA" if dark else "#1E1E2F"

    date_str = datetime.now().strftime("%B %d, %Y")

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=bg, plot_bgcolor=bg,
        xaxis=dict(visible=False, range=[0, 10]),
        yaxis=dict(visible=False, range=[0, 10]),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    # Company name
    fig.add_annotation(
        x=5, y=6, text=f"<b>{company_name}</b>",
        showarrow=False,
        font=dict(size=52, color=text_c),
        xanchor="center", yanchor="middle",
    )
    # Ticker + date
    fig.add_annotation(
        x=5, y=4, text=f"{ticker}  ·  {date_str}",
        showarrow=False,
        font=dict(size=28, color="#888888"),
        xanchor="center", yanchor="middle",
    )
    # Subtitle
    fig.add_annotation(
        x=5, y=2.5, text="AI Stock Analysis Report",
        showarrow=False,
        font=dict(size=22, color="#7C4DFF"),
        xanchor="center", yanchor="middle",
    )
    _save_plotly_fig(fig, out_path)


def _make_verdict_card(verdict: str, confidence: float,
                       points: int, total: int,
                       company_name: str,
                       dark: bool, out_path: str):
    """Generate a verdict summary card image."""
    bg = BG_DARK if dark else BG_LIGHT
    text_c = "#FAFAFA" if dark else "#1E1E2F"

    color_map = {"BUY": ACCENT_GREEN, "HOLD": ACCENT_YELLOW, "SELL": ACCENT_RED}
    v_color = color_map.get(verdict, "#888888")

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=bg, plot_bgcolor=bg,
        xaxis=dict(visible=False, range=[0, 10]),
        yaxis=dict(visible=False, range=[0, 10]),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.add_annotation(
        x=5, y=6.5,
        text=f"<b>{verdict}</b>",
        showarrow=False,
        font=dict(size=72, color=v_color),
        xanchor="center", yanchor="middle",
    )
    fig.add_annotation(
        x=5, y=4.5,
        text=f"{confidence:.0f}% Confidence  ·  {points}/{total} Metrics Passed",
        showarrow=False,
        font=dict(size=30, color=text_c),
        xanchor="center", yanchor="middle",
    )
    fig.add_annotation(
        x=5, y=3,
        text=company_name,
        showarrow=False,
        font=dict(size=24, color="#888888"),
        xanchor="center", yanchor="middle",
    )
    _save_plotly_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report_video(
    ticker: str,
    result: dict,
    hist: pd.DataFrame,
    narration_text: str,
    dark_mode: bool = True,
    output_dir: str | None = None,
) -> str:
    """
    Generate a complete ~30-second video report for *ticker*.

    Parameters
    ----------
    ticker         : stock ticker symbol
    result         : dict returned by stock_engine.predict_stock_action()
    hist           : OHLCV DataFrame from stock_engine.get_price_history()
    narration_text : script text from stock_engine.generate_narration_script()
    dark_mode      : use dark theme for chart frames
    output_dir     : directory to save the output; defaults to a temp dir

    Returns
    -------
    str – absolute path to the generated .mp4 file
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="stock_video_")
    os.makedirs(output_dir, exist_ok=True)

    metrics = result["metrics"]
    bench = result["benchmarks"]

    # ── Step 1: Generate voiceover audio ─────────────────────────────
    audio_path = os.path.join(output_dir, "voiceover.mp3")
    _generate_voiceover(narration_text, audio_path)
    audio_clip = AudioFileClip(audio_path)
    total_audio_duration = audio_clip.duration

    # ── Step 2: Calculate frame durations to match audio ─────────────
    # 4 visual segments: title(~5s), charts(~10s), risk chart(~8s), verdict(~7s)
    # We'll distribute proportionally to the audio length
    proportions = [5/30, 10/30, 8/30, 7/30]
    durations = [p * total_audio_duration for p in proportions]

    # Title card gets first proportion
    title_dur = durations[0]
    # Candlestick gets half of the charts proportion, RSI the other half
    candle_dur = durations[1] * 0.5
    rsi_dur = durations[1] * 0.5
    # Valuation gets the risk proportion
    val_dur = durations[2]
    # Verdict gets the final proportion
    verdict_dur = durations[3]

    # ── Step 3: Generate chart frame images ──────────────────────────
    title_img = os.path.join(output_dir, "title.png")
    candle_img = os.path.join(output_dir, "candlestick.png")
    rsi_img = os.path.join(output_dir, "rsi.png")
    val_img = os.path.join(output_dir, "valuation.png")
    verdict_img = os.path.join(output_dir, "verdict.png")

    _make_title_card(metrics["company_name"], ticker, dark_mode, title_img)
    _make_candlestick_frame(hist, ticker, dark_mode, candle_img)
    _make_rsi_frame(hist, ticker, dark_mode, rsi_img)
    _make_valuation_frame(metrics, bench, ticker, dark_mode, val_img)
    _make_verdict_card(
        result["verdict"], result["confidence"],
        result["points"], result["total"],
        metrics["company_name"], dark_mode, verdict_img,
    )

    # ── Step 4: Build video clips from images ────────────────────────
    #    MoviePy v2 uses with_duration() / resized() / with_audio()
    clips = [
        ImageClip(title_img).with_duration(title_dur),
        ImageClip(candle_img).with_duration(candle_dur),
        ImageClip(rsi_img).with_duration(rsi_dur),
        ImageClip(val_img).with_duration(val_dur),
        ImageClip(verdict_img).with_duration(verdict_dur),
    ]

    # Resize all clips to target resolution
    clips = [c.resized((VIDEO_WIDTH, VIDEO_HEIGHT)) for c in clips]

    # ── Step 5: Concatenate and add audio ────────────────────────────
    video = concatenate_videoclips(clips, method="compose")
    video = video.with_audio(audio_clip)

    # ── Step 6: Render final output ──────────────────────────────────
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join(output_dir, f"{ticker}_report_{date_str}.mp4")

    video.write_videofile(
        output_path,
        fps=VIDEO_FPS,
        codec="libx264",
        audio_codec="aac",
        preset="fast",          # fast encode for 2-vCPU sandbox
        threads=2,
        logger=None,            # suppress MoviePy progress bars in Streamlit
    )

    # Cleanup audio clip reference
    audio_clip.close()

    return output_path
