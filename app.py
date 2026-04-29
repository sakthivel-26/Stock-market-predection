import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib
import json
import re
import requests
from bs4 import BeautifulSoup
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
try:
    from nselib import capital_market, derivatives
except (ImportError, ModuleNotFoundError):
    capital_market = None
    derivatives = None
from concurrent.futures import ThreadPoolExecutor, as_completed
import time



from dotenv import load_dotenv
import os

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def get_telegram_chat_id(bot_token):
    if not bot_token:
        return None

    try:
        url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("ok") and data.get("result"):
            return data["result"][-1]["message"]["chat"]["id"]

    except requests.RequestException:
        return None
    except (KeyError, IndexError, TypeError, ValueError):
        return None

    return None


def resolve_telegram_chat_id(bot_token):
    if TELEGRAM_CHAT_ID:
        return TELEGRAM_CHAT_ID
    return get_telegram_chat_id(bot_token)


# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="AI Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .shakthi-badge-wrap {
        position: fixed;
        right: 18px;
        bottom: 18px;
        z-index: 9999;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .shakthi-powered-badge {
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(12, 17, 28, 0.88);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #f5f7fb;
        font-size: 13px;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
    }

    .shakthi-github-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        text-decoration: none;
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(12, 17, 28, 0.88);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #f5f7fb;
        font-size: 13px;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        transition: transform 0.18s ease, border-color 0.18s ease;
    }

    .shakthi-github-badge:hover {
        transform: translateY(-1px);
        border-color: rgba(255, 255, 255, 0.3);
    }

    .shakthi-github-badge svg {
        width: 16px;
        height: 16px;
        fill: currentColor;
    }

    @media (max-width: 640px) {
        .shakthi-badge-wrap {
            right: 12px;
            bottom: 12px;
            gap: 8px;
        }

        .shakthi-powered-badge {
            font-size: 12px;
            padding: 8px 12px;
        }

        .shakthi-github-badge {
            font-size: 12px;
            padding: 8px 12px;
        }
    }
    </style>
    <div class="shakthi-badge-wrap">
        <div class="shakthi-powered-badge">Powered by S H A K T H I ❤️</div>
        <a class="shakthi-github-badge" href="https://github.com/sakthivel-26" target="_blank" rel="noopener noreferrer" title="Connect on GitHub">
            <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.65 7.65 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 AI Stock Analyzer")

# ========================
# CONFIG
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "stock_model.pkl")

SCANNER_STOCKS = [
    # Nifty 50 - Large Cap
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "LT.NS",
    "WIPRO.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "KOTAKBANK.NS",
    "HINDUNILVR.NS",
    "AXISBANK.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TATAMOTORS.NS",
    "NTPC.NS",
    "TITAN.NS",
    "BAJFINANCE.NS",
    "ASIANPAINT.NS",
    "HCLTECH.NS",
    "ULTRACEMCO.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "NESTLEIND.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "M&M.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "COALINDIA.NS",
    "TECHM.NS",
    "INDUSINDBK.NS",
    "HINDALCO.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "BAJAJFINSV.NS",
    "DIVISLAB.NS",
    "BRITANNIA.NS",
    "EICHERMOT.NS",
    # Mid Cap
    "TATAPOWER.NS",
    "IRCTC.NS",
    "VEDL.NS",
    "BANKBARODA.NS",
    "PNB.NS",
    "ZOMATO.NS",
    "JIOFIN.NS",
    "DLF.NS",
    "HAL.NS",
    "BEL.NS",
]

TICKER_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
    "TATAMOTORS.NS", "BAJFINANCE.NS"
]

BSE_STOCKS = {
    "Reliance Industries": "500325.BO",
    "TCS": "532540.BO",
    "HDFC Bank": "500180.BO",
    "Bharti Airtel": "532454.BO",
    "ICICI Bank": "532174.BO",
    "Infosys": "500209.BO",
    "State Bank of India": "500112.BO",
    "Hindustan Unilever": "500696.BO",
    "ITC": "500875.BO",
    "LIC": "543526.BO",
    "Larsen & Toubro": "500510.BO",
    "Sun Pharma": "524715.BO",
    "Axis Bank": "532215.BO",
    "Kotak Mahindra Bank": "500247.BO",
    "Bajaj Finance": "500034.BO",
    "Maruti Suzuki": "532500.BO",
    "NTPC": "532555.BO",
    "UltraTech Cement": "532538.BO",
    "Titan": "500114.BO",
    "Asian Paints": "500820.BO",
    "Nestle India": "500790.BO",
    "Mahindra & Mahindra": "500520.BO",
    "Power Grid": "532898.BO",
    "Adani Ports": "532921.BO",
    "Tata Motors": "500570.BO",
    "Wipro": "507685.BO",
    "JSW Steel": "500228.BO",
    "Coal India": "533278.BO",
    "HCL Technologies": "532281.BO",
    "Bajaj Finserv": "532978.BO",
}

GAINERS = ["RELIANCE.NS", "ICICIBANK.NS", "LT.NS", "BHARTIARTL.NS", "ITC.NS"]
LOSERS = ["TCS.NS", "INFY.NS", "WIPRO.NS", "TATASTEEL.NS", "HINDALCO.NS"]

# Yahoo tickers occasionally change; keep aliases for resilient downloads.
SYMBOL_ALIASES = {
    "ZOMATO.NS": ["ETERNAL.NS"],
    "SUZLONENERGY.NS": ["SUZLON.NS"],
}

# Fallback map when some BSE Yahoo tickers are temporarily unavailable.
BSE_TO_NSE_ALIASES = {
    "500325.BO": "RELIANCE.NS",
    "532540.BO": "TCS.NS",
    "500180.BO": "HDFCBANK.NS",
    "532454.BO": "BHARTIARTL.NS",
    "532174.BO": "ICICIBANK.NS",
    "500209.BO": "INFY.NS",
    "500112.BO": "SBIN.NS",
    "500696.BO": "HINDUNILVR.NS",
    "500875.BO": "ITC.NS",
    "543526.BO": "LICI.NS",
    "500510.BO": "LT.NS",
    "524715.BO": "SUNPHARMA.NS",
    "532215.BO": "AXISBANK.NS",
    "500247.BO": "KOTAKBANK.NS",
    "500034.BO": "BAJFINANCE.NS",
    "532500.BO": "MARUTI.NS",
    "532555.BO": "NTPC.NS",
    "532538.BO": "ULTRACEMCO.NS",
    "500114.BO": "TITAN.NS",
    "500820.BO": "ASIANPAINT.NS",
    "500790.BO": "NESTLEIND.NS",
    "500520.BO": "M&M.NS",
    "532898.BO": "POWERGRID.NS",
    "532921.BO": "ADANIPORTS.NS",
    "500570.BO": "TATAMOTORS.NS",
    "507685.BO": "WIPRO.NS",
    "500228.BO": "JSWSTEEL.NS",
    "533278.BO": "COALINDIA.NS",
    "532281.BO": "HCLTECH.NS",
    "532978.BO": "BAJAJFINSV.NS",
}

# ========================
# LOAD MODEL
# ========================
@st.cache_resource
def load_model(model_path, model_mtime):
    """Load pre-trained model"""
    if not os.path.exists(model_path):
        return None
    try:
        m = joblib.load(model_path)
        # Verify model matches the current training feature set.
        expected_features = 10
        if hasattr(m, 'n_features_in_') and m.n_features_in_ != expected_features:
            return None
        return m
    except Exception:
        return None

model_mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
model = load_model(MODEL_PATH, model_mtime)

# ========================
# HELPER FUNCTIONS
# ========================

def get_ai_explanation(recommendation, pro_score, rsi_val, ma20_val, ma50_val, current_price, macd_hist, bb_lower, bb_upper):
    """Generate a context-aware AI explanation from the actual analysis data."""
    reasons = []

    in_downtrend = current_price < ma20_val and current_price < ma50_val
    in_uptrend = current_price > ma20_val and current_price > ma50_val

    # Trend context
    if in_downtrend:
        ma20_gap = round(((current_price - ma20_val) / ma20_val) * 100, 1)
        reasons.append(f"price is {abs(ma20_gap)}% below MA20 in a downtrend")
    elif in_uptrend:
        ma20_gap = round(((current_price - ma20_val) / ma20_val) * 100, 1)
        reasons.append(f"price is {ma20_gap}% above MA20 in an uptrend")
    else:
        reasons.append("price is between key moving averages")

    # RSI
    if rsi_val < 30:
        if in_downtrend:
            reasons.append(f"RSI at {rsi_val:.1f} is oversold but confirms bearish momentum")
        else:
            reasons.append(f"RSI at {rsi_val:.1f} is oversold — potential bounce")
    elif rsi_val > 70:
        if in_uptrend:
            reasons.append(f"RSI at {rsi_val:.1f} shows strong momentum")
        else:
            reasons.append(f"RSI at {rsi_val:.1f} is overbought — potential pullback")
    elif rsi_val <= 40:
        reasons.append(f"RSI at {rsi_val:.1f} is weak")
    elif rsi_val >= 60:
        reasons.append(f"RSI at {rsi_val:.1f} shows decent momentum")
    else:
        reasons.append(f"RSI at {rsi_val:.1f} is neutral")

    # MACD
    if macd_hist > 0:
        reasons.append("MACD histogram is positive")
    elif macd_hist < 0:
        reasons.append("MACD histogram is negative")

    # Bollinger
    if current_price <= bb_lower:
        if in_downtrend:
            reasons.append("price is walking the lower Bollinger Band")
        else:
            reasons.append("price is at the lower Bollinger Band")
    elif current_price >= bb_upper:
        if in_uptrend:
            reasons.append("price is riding the upper Bollinger Band")
        else:
            reasons.append("price is at the upper Bollinger Band")

    # Build message
    action = recommendation.replace("STRONG ", "")
    strength = "strongly " if "STRONG" in recommendation else ""
    reason_str = ", ".join(reasons[:3])  # cap at 3 reasons for readability

    return f"AI {strength}suggests {action} (score {pro_score:+d}/10) because {reason_str}."


def _get_llm_stock_description(symbol, context):
    """LLM descriptions removed — return None to keep technical-only flow."""
    return None


def tradingview_mini_chart(symbol, height=220):
    """Render a TradingView mini chart widget for an NSE or BSE stock."""
    symbol = symbol.upper()
    if symbol.endswith(".BO"):
        mapped_nse = BSE_TO_NSE_ALIASES.get(symbol)
        if mapped_nse:
            tv_symbol = "NSE:" + mapped_nse.replace(".NS", "")
        else:
            tv_symbol = "BSE:" + symbol.replace(".BO", "")
    else:
        tv_symbol = "NSE:" + symbol.replace(".NS", "")

    html = f"""
    <div class="tradingview-widget-container" style="width:100%;height:{height}px;">
      <div class="tradingview-widget-container__widget" style="width:100%;height:100%;"></div>
      <script type="text/javascript"
              src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js"
              async>
      {{
        "symbol": "{tv_symbol}",
        "width": "100%",
        "height": "{height}",
        "locale": "en",
        "dateRange": "3M",
        "colorTheme": "dark",
        "isTransparent": true,
        "autosize": true,
        "largeChartUrl": "",
        "chartOnly": false,
        "noTimeScale": true
      }}
      </script>
    </div>
    """
    components.html(html, height=height + 10)

def calculate_confidence_gauge(confidence):
    """Calculate confidence percentage"""
    if pd.isna(confidence):
        return 0
    return round(confidence * 100)


def format_crore(value):
    """Format a numeric value in crore notation for UI display."""
    return f"{value / 10000000:.2f} Cr"


def get_llm_stock_context(symbol, context):
    """Generate a concise stock verdict prompt for the configured LLM backend."""
    trend = context.get("trend", "Unknown")
    trigger = context.get("trigger", "None")
    confidence = context.get("confidence", "Low")
    risk_note = context.get("risk_note", "")
    forecast = context.get("forecast", "")
    entry = context.get("entry", "")

    return f"""
You are an expert Indian stock swing trading analyst.
Give a concise verdict on whether this stock is a BUY or NOT BUY.
Return 3-5 short sentences.
Be strict, avoid hype, and mention the key reason.

Required format:
Verdict: BUY or NOT BUY
Reason: one short sentence
Risk: one short sentence

Stock: {symbol}
Trend: {trend}
Trigger: {trigger}
Confidence: {confidence}
Forecast: {forecast}
Entry: {entry}
Risk Note: {risk_note}
""".strip()


def get_ai_swing_trading_verdict(symbol, swing_data):
    """
    Generate AI-powered buy/not buy suggestion for swing trading using Qwen LLM.
    Falls back to technical analysis if LLM is unavailable.
    
    Args:
        symbol: Stock symbol (e.g., 'TCS.NS')
        swing_data: Dict with swing trading analysis (signals, rsi, ma20, score, etc.)
    
    Returns:
        str: AI verdict with buy/not buy suggestion
    """
    rsi = swing_data.get("rsi", 50)
    ma20 = swing_data.get("ma20", 0)
    ma50 = swing_data.get("ma50", 0)
    close = swing_data.get("close", 0)
    signals = swing_data.get("signals", [])
    score = swing_data.get("score", 0)
    atr = swing_data.get("atr", 0)
    volume_ratio = swing_data.get("volume_ratio", 1.0)
    
    # Determine trend
    if close > ma20 > ma50:
        trend_text = "Bullish uptrend"
    elif close < ma20 < ma50:
        trend_text = "Bearish downtrend"
    else:
        trend_text = "Sideways/Mixed"
    
    # RSI context
    if rsi > 70:
        rsi_text = "Overbought"
    elif rsi > 60:
        rsi_text = "Strong bullish momentum"
    elif rsi > 50:
        rsi_text = "Bullish momentum"
    elif rsi > 40:
        rsi_text = "Neutral momentum"
    elif rsi < 30:
        rsi_text = "Oversold"
    else:
        rsi_text = "Bearish momentum"
    
    signals_text = " | ".join(signals[:3]) if signals else "No specific signals"
    
    prompt = f"""
You are a strict swing trading analyst for Indian stocks (NSE).
Analyze this stock and give a BUY or NOT BUY verdict in 2-3 short sentences.
Be objective, avoid hype, and focus on risk-reward.

Stock: {symbol}
Trend: {trend_text}
RSI: {rsi:.1f} ({rsi_text})
Price vs MA20: {close:.2f} vs {ma20:.2f}
Score: {score:+d}/10
Volume Ratio: {volume_ratio:.2f}x
ATR: {atr:.2f}
Key Signals: {signals_text}

Response format:
Verdict: BUY or NOT BUY
Why: [one compelling reason]
""".strip()
    
    # LLM support removed — use deterministic technical verdict only
    return _generate_technical_verdict(swing_data)


def _generate_technical_verdict(swing_data):
    """
    Generate a verdict based purely on technical analysis when LLM is unavailable.
    Fallback function for when OpenRouter/Gemini are down.
    """
    rsi = swing_data.get("rsi", 50)
    close = swing_data.get("close", 0)
    ma20 = swing_data.get("ma20", 0)
    ma50 = swing_data.get("ma50", 0)
    score = swing_data.get("score", 0)
    volume_ratio = swing_data.get("volume_ratio", 1.0)
    signals = swing_data.get("signals", [])
    
    # Build technical verdict
    reason_parts = []
    
    # Trend assessment
    if close > ma20 > ma50:
        trend_ok = True
        reason_parts.append("Bullish trend confirmed")
    elif close < ma20 < ma50:
        trend_ok = False
        reason_parts.append("Bearish trend")
    else:
        trend_ok = False
        reason_parts.append("Mixed trend")
    
    # RSI assessment
    if 40 < rsi < 70:
        rsi_ok = True
        reason_parts.append("RSI in healthy zone")
    elif rsi >= 70:
        rsi_ok = False
        reason_parts.append("RSI overbought")
    elif rsi <= 30:
        rsi_ok = True
        reason_parts.append("RSI oversold (bounce potential)")
    else:
        rsi_ok = False
        reason_parts.append("RSI weak")
    
    # Volume assessment
    volume_ok = volume_ratio > 1.2
    if volume_ok:
        reason_parts.append(f"Volume {volume_ratio:.1f}x confirmed")
    else:
        reason_parts.append(f"Volume {volume_ratio:.1f}x weak")
    
    # Score assessment
    if score >= 6:
        score_ok = True
        reason_parts.append(f"Tech score strong ({score:+d})")
    elif score >= 0:
        score_ok = "mixed"
        reason_parts.append(f"Tech score neutral ({score:+d})")
    else:
        score_ok = False
        reason_parts.append(f"Tech score weak ({score:+d})")
    
    # Determine verdict
    buy_factors = sum([trend_ok, rsi_ok, volume_ok, score_ok == True])
    negative_factors = sum([not trend_ok, not rsi_ok, not volume_ok, score_ok == False])
    
    if buy_factors >= 3 and negative_factors == 0:
        verdict = "BUY"
        reason = "Multiple technical factors aligned bullishly"
    elif buy_factors >= 2 and volume_ok and trend_ok:
        verdict = "BUY"
        reason = "Trend + volume + score supportive"
    elif score >= 5 and trend_ok and rsi_ok:
        verdict = "BUY"
        reason = "Technicals suggest upside setup"
    elif negative_factors >= 2 or (not trend_ok and rsi >= 70):
        verdict = "NOT BUY"
        reason = "Trend weakness or overbought conditions"
    elif score < 0 and rsi > 60:
        verdict = "NOT BUY"
        reason = "Weakening technicals at high RSI"
    else:
        verdict = "HOLD/WAIT"
        reason = "Mixed signals, wait for clearer setup"
    
    risk = "Monitor trend break below MA20" if trend_ok else "Wait for reversal below MA50"
    
    return f"Verdict: {verdict}\nReason: {reason}\nRisk: {risk}\n[⚠️ Technical Fallback - LLM unavailable]"


def _call_llm_text(prompt):
    # LLM support removed — always return None so callers use technical fallback
    return None


# ========================
# ADVANCED SWING TRADING ANALYSIS
# ========================

def calculate_ema(data_series, period):
    """Calculate Exponential Moving Average."""
    return data_series.ewm(span=period, adjust=False).mean()


def detect_ema_crossover(data):
    """Detect 9 EMA and 21 EMA crossover signal."""
    if data is None or len(data) < 21:
        return None, None, None
    
    close = data["Close"].squeeze()
    ema9 = calculate_ema(close, 9)
    ema21 = calculate_ema(close, 21)
    
    latest_ema9 = float(ema9.iloc[-1])
    latest_ema21 = float(ema21.iloc[-1])
    prev_ema9 = float(ema9.iloc[-2]) if len(ema9) > 1 else latest_ema9
    prev_ema21 = float(ema21.iloc[-2]) if len(ema21) > 1 else latest_ema21
    
    # Bullish crossover: 9 EMA crosses above 21 EMA
    bullish_cross = (prev_ema9 <= prev_ema21) and (latest_ema9 > latest_ema21)
    # Bearish crossover: 9 EMA crosses below 21 EMA
    bearish_cross = (prev_ema9 >= prev_ema21) and (latest_ema9 < latest_ema21)
    
    return {
        "ema9": latest_ema9,
        "ema21": latest_ema21,
        "bullish_cross": bullish_cross,
        "bearish_cross": bearish_cross,
        "ema9_above_ema21": latest_ema9 > latest_ema21
    }


def detect_advanced_candle_patterns(data):
    """Detect advanced candlestick patterns: Hammer, Morning Star, Shooting Star, Evening Star."""
    if data is None or len(data) < 3:
        return []
    
    patterns = []
    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest
    prev_prev = data.iloc[-3] if len(data) > 2 else None
    
    latest_open = float(latest["Open"])
    latest_close = float(latest["Close"])
    latest_high = float(latest["High"])
    latest_low = float(latest["Low"])
    
    prev_open = float(prev["Open"])
    prev_close = float(prev["Close"])
    prev_high = float(prev["High"])
    prev_low = float(prev["Low"])
    
    body_high = max(latest_open, latest_close)
    body_low = min(latest_open, latest_close)
    body_size = abs(latest_close - latest_open)
    range_size = latest_high - latest_low
    
    prev_body_high = max(prev_open, prev_close)
    prev_body_low = min(prev_open, prev_close)
    prev_body_size = abs(prev_close - prev_open)
    
    # Hammer: small body at top, long lower wick, used at support
    lower_wick = body_low - latest_low
    upper_wick = latest_high - body_high
    if body_size > 0 and lower_wick > (2 * body_size) and upper_wick < body_size and latest_close > latest_open:
        patterns.append(("Hammer", "Support"))
    
    # Shooting Star: small body at bottom, long upper wick, used at resistance
    if body_size > 0 and upper_wick > (2 * body_size) and lower_wick < body_size and latest_close < latest_open:
        patterns.append(("Shooting Star", "Resistance"))
    
    # Morning Star: bearish candle, small gap down, bullish candle closing above mid of first candle
    if prev_prev is not None:
        prev_prev_close = float(prev_prev["Close"])
        prev_prev_open = float(prev_prev["Open"])
        prev_prev_body_high = max(prev_prev_open, prev_prev_close)
        if (prev_close < prev_open and  # First candle bearish
            prev_close > body_high and  # Gap down
            latest_close > latest_open and  # Last candle bullish
            latest_close > (prev_prev_body_high + prev_prev_open) / 2):  # Closes above mid
            patterns.append(("Morning Star", "Support"))
    
    # Evening Star: bullish candle, small gap up, bearish candle closing below mid of first candle
    if prev_prev is not None:
        prev_prev_close = float(prev_prev["Close"])
        prev_prev_open = float(prev_prev["Open"])
        prev_prev_body_low = min(prev_prev_open, prev_prev_close)
        if (prev_close > prev_open and  # First candle bullish
            latest_open > prev_close and  # Gap up
            latest_close < latest_open and  # Last candle bearish
            latest_close < (prev_prev_body_low + prev_prev_open) / 2):  # Closes below mid
            patterns.append(("Evening Star", "Resistance"))
    
    return patterns


def check_support_resistance_levels(data):
    """Identify recent support and resistance levels."""
    if data is None or len(data) < 20:
        return None, None
    
    lookback = min(20, len(data))
    recent_high = float(pd.to_numeric(data["High"].tail(lookback), errors="coerce").max())
    recent_low = float(pd.to_numeric(data["Low"].tail(lookback), errors="coerce").min())
    
    return recent_low, recent_high


def check_nifty_50_trend():
    """Check if Nifty 50 is in bullish or bearish trend using 50 EMA."""
    try:
        nifty_data = yf.download("^NSEI", period="100d", progress=False)
        if nifty_data is None or nifty_data.empty or len(nifty_data) < 50:
            return None
        
        close = nifty_data["Close"].squeeze()
        ema50 = calculate_ema(close, 50)
        latest_close = float(close.iloc[-1])
        latest_ema50 = float(ema50.iloc[-1])
        
        if latest_close > latest_ema50:
            return "BULLISH"
        elif latest_close < latest_ema50:
            return "BEARISH"
        else:
            return "SIDEWAYS"
    except Exception:
        return None


def comprehensive_swing_analysis(symbol, data):
    """Comprehensive swing trading analysis with all rules applied."""
    if data is None or len(data) < 50:
        return None
    
    try:
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        close_series = data["Close"]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.squeeze()
        high_series = data["High"]
        if isinstance(high_series, pd.DataFrame):
            high_series = high_series.squeeze()
        low_series = data["Low"]
        if isinstance(low_series, pd.DataFrame):
            low_series = low_series.squeeze()
        volume_series = data["Volume"]
        if isinstance(volume_series, pd.DataFrame):
            volume_series = volume_series.squeeze()
        
        close = close_series
        high = high_series
        low = low_series
        volume = volume_series
        
        # 1. Calculate all EMAs
        ema9 = calculate_ema(close, 9)
        ema21 = calculate_ema(close, 21)
        ema50 = calculate_ema(close, 50)
        
        latest_close = float(close.iloc[-1])
        latest_ema9 = float(ema9.iloc[-1])
        latest_ema21 = float(ema21.iloc[-1])
        latest_ema50 = float(ema50.iloc[-1])
        
        # 2. Calculate RSI (14)
        rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        latest_rsi = float(rsi.iloc[-1])
        
        # 3. Calculate MACD (12, 26, 9)
        macd = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
        latest_macd = float(macd.macd().iloc[-1])
        latest_macd_signal = float(macd.macd_signal().iloc[-1])
        latest_macd_hist = float(macd.macd_diff().iloc[-1])
        prev_macd_hist = float(macd.macd_diff().iloc[-2]) if len(macd.macd_diff()) > 1 else latest_macd_hist
        
        # 4. Volume analysis
        vol_avg20 = volume.rolling(20).mean()
        latest_volume = float(volume.iloc[-1])
        latest_vol_avg = float(vol_avg20.iloc[-1])
        volume_ratio = latest_volume / latest_vol_avg if latest_vol_avg > 0 else 0
        
        # 5. EMA crossover detection
        ema_cross = detect_ema_crossover(data)
        
        # 6. Candlestick patterns
        patterns = detect_advanced_candle_patterns(data)
        
        # 7. Support and resistance
        lookback = min(20, len(data))
        recent_lows = pd.to_numeric(low.tail(lookback), errors="coerce").dropna()
        recent_highs = pd.to_numeric(high.tail(lookback), errors="coerce").dropna()
        support_level = float(recent_lows.min()) if len(recent_lows) > 0 else latest_close
        resistance_level = float(recent_highs.max()) if len(recent_highs) > 0 else latest_close
        
        # 8. Nifty 50 filter
        nifty_trend = check_nifty_50_trend()
        
        # ===== CHECK BUY SIGNAL RULES =====
        buy_rules = {
            "ema9_above_ema21": ema_cross["ema9_above_ema21"],
            "price_above_ema50": latest_close > latest_ema50,
            "macd_above_signal": latest_macd > latest_macd_signal,
            "macd_hist_green": latest_macd_hist > 0,
            "macd_hist_rising": latest_macd_hist > prev_macd_hist,
            "volume_confirmed": volume_ratio >= 1.5,
            "rsi_in_zone": 50 <= latest_rsi <= 70,
            "bullish_pattern": any("Morning" in p[0] or "Hammer" in p[0] for p in patterns if p[1] == "Support"),
            "ema9_crossed_above": ema_cross["bullish_cross"],
            "nifty_bullish": nifty_trend == "BULLISH"
        }
        
        # ===== CHECK SELL SIGNAL RULES =====
        sell_rules = {
            "ema9_below_ema21": not ema_cross["ema9_above_ema21"],
            "price_below_ema50": latest_close < latest_ema50,
            "macd_below_signal": latest_macd < latest_macd_signal,
            "macd_hist_red": latest_macd_hist < 0,
            "macd_hist_falling": latest_macd_hist < prev_macd_hist,
            "volume_confirmed": volume_ratio >= 1.5,
            "rsi_in_zone": 30 <= latest_rsi <= 50,
            "bearish_pattern": any("Evening" in p[0] or "Shooting" in p[0] for p in patterns if p[1] == "Resistance"),
            "ema9_crossed_below": ema_cross["bearish_cross"],
            "nifty_bearish": nifty_trend == "BEARISH"
        }
        
        # Calculate signal strength
        buy_score = sum([1 for v in buy_rules.values() if v]) if any(buy_rules.values()) else 0
        sell_score = sum([1 for v in sell_rules.values() if v]) if any(sell_rules.values()) else 0
        
        # Determine signal
        buy_signal = buy_score >= 7  # Require at least 7/10 rules for strong signal
        sell_signal = sell_score >= 7
        
        # Calculate entry, SL, targets
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        latest_atr = float(atr.iloc[-1]) if len(atr) > 0 else 0
        
        if buy_signal:
            entry_price = round(latest_close, 2)
            stop_loss = round(support_level - (latest_atr * 0.5), 2)
            risk_per_trade = entry_price - stop_loss
            target1 = round(entry_price + (entry_price * 0.05), 2)  # 5%
            target2 = round(entry_price + (entry_price * 0.08), 2)  # 8%
            r_r_ratio = ((target1 - entry_price) / risk_per_trade) if risk_per_trade > 0 else 0
        elif sell_signal:
            entry_price = round(latest_close, 2)
            stop_loss = round(resistance_level + (latest_atr * 0.5), 2)
            risk_per_trade = stop_loss - entry_price
            target1 = round(entry_price - (entry_price * 0.05), 2)  # 5%
            target2 = round(entry_price - (entry_price * 0.08), 2)  # 8%
            r_r_ratio = ((entry_price - target1) / risk_per_trade) if risk_per_trade > 0 else 0
        else:
            entry_price = round(latest_close, 2)
            stop_loss = round(support_level, 2)
            target1 = round(latest_close, 2)
            target2 = round(latest_close, 2)
            r_r_ratio = 0
        
        # Trend direction
        if latest_close > latest_ema50:
            trend_direction = "BULLISH"
        elif latest_close < latest_ema50:
            trend_direction = "BEARISH"
        else:
            trend_direction = "SIDEWAYS"
        
        # Confidence level
        if buy_signal or sell_signal:
            confidence = "HIGH" if (buy_score >= 9 or sell_score >= 9) else "MEDIUM"
        else:
            confidence = "LOW"
        
        # Build reason
        reason_parts = []
        if buy_signal:
            reason_parts.append(f"✅ 9 EMA > 21 EMA")
            reason_parts.append(f"✅ Price > 50 EMA (₹{latest_ema50:.2f})")
            reason_parts.append(f"✅ MACD Histogram GREEN")
            reason_parts.append(f"✅ Volume {volume_ratio:.2f}x (>1.5x)")
            reason_parts.append(f"✅ RSI {latest_rsi:.1f} (50-70 zone)")
            if buy_rules["bullish_pattern"]:
                reason_parts.append(f"✅ Bullish Pattern detected at support")
            reason_parts.append(f"✅ Nifty Bullish trend confirmed")
        elif sell_signal:
            reason_parts.append(f"❌ 9 EMA < 21 EMA")
            reason_parts.append(f"❌ Price < 50 EMA (₹{latest_ema50:.2f})")
            reason_parts.append(f"❌ MACD Histogram RED")
            reason_parts.append(f"❌ Volume {volume_ratio:.2f}x (>1.5x)")
            reason_parts.append(f"❌ RSI {latest_rsi:.1f} (30-50 zone)")
            if sell_rules["bearish_pattern"]:
                reason_parts.append(f"❌ Bearish Pattern detected at resistance")
            reason_parts.append(f"❌ Nifty Bearish trend confirmed")
        else:
            reason_parts.append(f"No clear signal - Insufficient rules met")
            reason_parts.append(f"Buy score: {buy_score}/10 | Sell score: {sell_score}/10")
        
        return {
            "symbol": symbol,
            "trend_direction": trend_direction,
            "buy_signal": buy_signal and nifty_trend == "BULLISH",
            "sell_signal": sell_signal and nifty_trend == "BEARISH",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target1": target1,
            "target2": target2,
            "risk_reward_ratio": round(r_r_ratio, 2),
            "confidence": confidence,
            "reasons": reason_parts,
            "ema9": round(latest_ema9, 2),
            "ema21": round(latest_ema21, 2),
            "ema50": round(latest_ema50, 2),
            "rsi": round(latest_rsi, 2),
            "macd": round(latest_macd, 4),
            "macd_signal": round(latest_macd_signal, 4),
            "macd_hist": round(latest_macd_hist, 4),
            "volume_ratio": round(volume_ratio, 2),
            "patterns": patterns,
            "support_level": round(support_level, 2),
            "resistance_level": round(resistance_level, 2),
            "nifty_trend": nifty_trend,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "atr": round(latest_atr, 2)
        }
    
    except Exception as e:
        return None


def train_model_func():
    """
    Train ML model using historical stock data.
    """

    progress_bar = st.progress(0)
    status_text = st.empty()

    all_data = []

    total_stocks = len(SCANNER_STOCKS)

    for idx, symbol in enumerate(SCANNER_STOCKS):

        time.sleep(0.3)  # prevent Yahoo rate limit

        try:
            status_text.text(f"Downloading {symbol}...")

            data = safe_download(symbol, "5y")

            if data is None:
                st.warning(f"⚠️ No data for {symbol}, skipping...")
                continue

            # Technical Indicators
            close  = data["Close"].squeeze()
            high   = data["High"].squeeze()
            low    = data["Low"].squeeze()
            volume = data["Volume"].squeeze()

            data["MA20"] = close.rolling(20).mean()
            data["MA50"] = close.rolling(50).mean()

            data["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

            data["MACD_Hist"] = ta.trend.MACD(close=close).macd_diff()

            bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            bb_range = bb.bollinger_hband() - bb.bollinger_lband()
            data["BB_pband"] = (close - bb.bollinger_lband()) / bb_range.replace(0, np.nan)

            data["ATR_ratio"] = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range() / close.replace(0, np.nan)

            data["Volume_Avg20"] = volume.rolling(20).mean()
            data["Volume_ratio"] = volume / data["Volume_Avg20"].replace(0, np.nan)

            data["Return"]    = close.pct_change()
            data["Return_5d"] = close.pct_change(5)

            obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
            obv_ma10 = obv.rolling(10).mean()
            data["OBV_ratio"] = obv / obv_ma10.abs().replace(0, np.nan)

            # Target: 5-day forward direction (less noise than next-day)
            data["Target"] = (close.shift(-5) > close).astype(int)

            data = data.dropna()

            if len(data) < 50:
                st.warning(f"⚠️ Not enough training data for {symbol}")
                continue

            all_data.append(data)

            progress_bar.progress((idx + 1) / total_stocks)

        except Exception as e:
            st.warning(f"❌ Error downloading {symbol}: {e}")

    status_text.text("Preparing dataset...")

    if len(all_data) == 0:
        st.error("❌ No data downloaded. Model cannot be trained.")
        return None

    dataset = pd.concat(all_data, ignore_index=True)

    # Features used by model
    FEATURE_COLS = [
        "MA20", "MA50", "RSI", "MACD_Hist",
        "BB_pband", "ATR_ratio", "Volume_ratio",
        "Return", "Return_5d", "OBV_ratio"
    ]
    features_df = dataset[FEATURE_COLS]

    # Keep only finite rows to avoid unstable training behavior.
    features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()
    target_series = dataset.loc[features_df.index, "Target"]

    features = features_df.values
    target = target_series.values

    if len(features) < 200:
        st.error("❌ Not enough clean training rows after preprocessing.")
        return None

    status_text.text("Training RandomForest model...")

    X_train, X_val, y_train, y_val = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=12,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    joblib.dump(model, MODEL_PATH)

    status_text.text("Model trained successfully! ✅")
    st.caption(
        f"Validation Metrics → Accuracy: {acc:.2%} | Precision: {prec:.2%} | Recall: {rec:.2%} | F1: {f1:.2%}"
    )
    progress_bar.progress(1.0)

    return {
        "model": model,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        },
    }


def auto_train_model_if_missing():
    """Train model automatically once per session if no model is loaded."""
    if st.session_state.get("auto_train_attempted", False):
        return

    if model is not None:
        return

    st.session_state["auto_train_attempted"] = True
    with st.spinner("No model found. Training automatically..."):
        try:
            result = train_model_func()
            if result is not None and result.get("model") is not None:
                st.session_state["last_train_metrics"] = result.get("metrics", {})
                st.session_state["train_success"] = True
                load_model.clear()
                st.rerun()
            else:
                st.warning("⚠️ Auto-training could not complete. Use Train & Load Model to retry.")
        except Exception as e:
            st.warning(f"⚠️ Auto-training failed: {e}")


def render_market_summary_block():
    """Render a compact live market summary for use inside stock analysis pages."""
    st.subheader("📡 Market Summary")

    breadth_data = get_market_breadth()
    vix_data = get_india_vix()
    oi_data = get_participant_oi()

    c1, c2, c3 = st.columns(3)

    with c1:
        if breadth_data:
            st.metric("NIFTY 50", f"₹{breadth_data['nifty_last']:,.0f}", f"{breadth_data['nifty_change_pct']:+.2f}%")
            st.metric("A/D Ratio", breadth_data["ad_ratio"], f"{breadth_data['advances']}A / {breadth_data['declines']}D")
        else:
            st.info("Market breadth unavailable")

    with c2:
        if vix_data:
            st.metric("India VIX", vix_data["vix_close"], f"{vix_data['vix_change_pct']:+.1f}%")
            st.metric("VIX Range", f"{vix_data['vix_low']} - {vix_data['vix_high']}")
        else:
            st.info("VIX data unavailable")

    with c3:
        if oi_data:
            fii_oi = oi_data.get("FII")
            dii_oi = oi_data.get("DII")
            if fii_oi:
                st.metric("FII Net OI", format_crore(fii_oi["net"]), "Long" if fii_oi["net"] > 0 else "Short")
            if dii_oi:
                st.metric("DII Net OI", format_crore(dii_oi["net"]), "Long" if dii_oi["net"] > 0 else "Short")
        else:
            st.info("OI data unavailable")

    movers = get_nse_top_movers()
    if movers:
        top_col1, top_col2 = st.columns(2)
        with top_col1:
            st.markdown("**Top Gainers**")
            if movers.get("gainers"):
                for g in movers["gainers"][:5]:
                    st.write(f"• **{g['symbol']}** — ₹{g['ltp']} ({g['change_pct']:+.2f}%)")
        with top_col2:
            st.markdown("**Top Losers**")
            if movers.get("losers"):
                for l in movers["losers"][:5]:
                    st.write(f"• **{l['symbol']}** — ₹{l['ltp']} ({l['change_pct']:+.2f}%)")

# ========================
# NSE INDIA DIRECT API
# Mirrors the stock-nse-india TypeScript library (NseIndia class)
# ========================

class NseIndiaAPI:
    """
    Python equivalent of the stock-nse-india TypeScript NseIndia class.
    Manages session cookies and exposes the same API surface.
    """
    BASE_URL = "https://www.nseindia.com"
    _session = None
    _cookie_ts: float = 0
    COOKIE_TTL = 300  # seconds

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.nseindia.com/",
        "X-Requested-With": "XMLHttpRequest",
    }

    def _refresh_session(self):
        s = requests.Session()
        s.headers.update(self.HEADERS)
        try:
            s.get(self.BASE_URL, timeout=15)
            s.get(f"{self.BASE_URL}/market-data/live-equity-market", timeout=10)
        except Exception:
            pass
        self._session = s
        self._cookie_ts = time.time()

    def _get_session(self):
        if self._session is None or (time.time() - self._cookie_ts > self.COOKIE_TTL):
            self._refresh_session()
        return self._session

    def get_data(self, url: str):
        """getData: Generic GET with auto-retry on auth failure."""
        for attempt in range(2):
            try:
                session = self._get_session()
                resp = session.get(url, timeout=15)
                if resp.status_code in (401, 403):
                    self._refresh_session()
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception:
                if attempt == 0:
                    self._refresh_session()
        return None

    def get_data_by_endpoint(self, endpoint: str):
        """getDataByEndpoint."""
        return self.get_data(f"{self.BASE_URL}/api/{endpoint}")

    # ---- Equity methods (match NseIndia TypeScript class) ----

    def get_equity_details(self, symbol: str):
        """getEquityDetails: live quote for a symbol."""
        return self.get_data_by_endpoint(f"quote-equity?symbol={symbol.upper()}")

    def get_equity_trade_info(self, symbol: str):
        """getEquityTradeInfo: trade info including delivery & circuit limits."""
        return self.get_data_by_endpoint(
            f"quote-equity?symbol={symbol.upper()}&section=trade_info"
        )

    def get_equity_intraday_data(self, symbol: str):
        """getEquityIntradayData: intraday OHLC candles."""
        return self.get_data_by_endpoint(
            f"chart-databyindex?index={symbol.upper()}&indices=false"
        )

    def get_equity_historical_data(self, symbol: str, from_date: str, to_date: str):
        """
        getEquityHistoricalData: daily OHLCV history.
        Dates must be in DD-MM-YYYY format.
        """
        ep = (
            f"historical/cm/equity?symbol={symbol.upper()}"
            f"&series[]=EQ&from={from_date}&to={to_date}"
        )
        return self.get_data_by_endpoint(ep)

    def get_equity_option_chain(self, symbol: str):
        """getEquityOptionChain."""
        return self.get_data_by_endpoint(f"option-chain-equities?symbol={symbol.upper()}")

    def get_equity_stock_indices(self, index: str):
        """getEquityStockIndices."""
        return self.get_data_by_endpoint(f"equity-stockIndices?index={index}")

    def get_equity_series(self, symbol: str):
        """getEquitySeries."""
        return self.get_data_by_endpoint(f"equity-meta-info?symbol={symbol.upper()}")

    def get_all_indices(self):
        """getAllIndices."""
        return self.get_data_by_endpoint("allIndices")

    def get_index_names(self):
        """getIndexNames."""
        return self.get_data_by_endpoint("index-names")

    def get_index_intraday_data(self, index: str):
        """getIndexIntradayData."""
        return self.get_data_by_endpoint(f"chart-databyindex?index={index}&indices=true")

    def get_index_option_chain(self, index_symbol: str, expiry: str = None):
        """getIndexOptionChain. Optional expiry in DD-MMM-YYYY format."""
        ep = f"option-chain-indices?symbol={index_symbol}"
        if expiry:
            ep += f"&expiry={expiry}"
        return self.get_data_by_endpoint(ep)

    def get_index_option_chain_contract_info(self, index_symbol: str):
        """getIndexOptionChainContractInfo."""
        return self.get_data_by_endpoint(
            f"option-chain-indices?symbol={index_symbol}"
        )

    def get_market_status(self):
        """getMarketStatus."""
        return self.get_data_by_endpoint("marketStatus")

    def get_market_turnover(self):
        """getMarketTurnover."""
        return self.get_data_by_endpoint("market-turnover")

    def get_pre_open_market_data(self):
        """getPreOpenMarketData."""
        return self.get_data_by_endpoint("market-data-pre-open?key=ALL")

    def get_all_stock_symbols(self):
        """getAllStockSymbols."""
        return self.get_data_by_endpoint("equity-master")

    def get_equity_master(self):
        """getEquityMaster."""
        return self.get_data_by_endpoint("equity-master")

    def get_trading_holidays(self):
        """getTradingHolidays."""
        return self.get_data_by_endpoint("holiday-master?type=trading")

    def get_clearing_holidays(self):
        """getClearingHolidays."""
        return self.get_data_by_endpoint("holiday-master?type=clearing")

    def get_glossary(self):
        """getGlossary."""
        return self.get_data_by_endpoint("cmsContent?url=/glossary")

    def get_circulars(self):
        """getCirculars."""
        return self.get_data_by_endpoint("circulars")

    def get_latest_circulars(self):
        """getLatestCirculars."""
        return self.get_data_by_endpoint("latest-circular")


# Singleton — one session shared across Streamlit reruns
_nse_api = NseIndiaAPI()


@st.cache_data(ttl=60)
def nse_get_equity_details(symbol_clean: str):
    """
    getEquityDetails wrapper — returns live price info dict.
    Falls back gracefully to None if the API is unavailable.
    """
    try:
        raw = _nse_api.get_equity_details(symbol_clean)
        if not raw:
            return None
        pi = raw.get("priceInfo", {})
        intra = pi.get("intraDayHighLow", {})
        wk52 = pi.get("weekHighLow", {})
        return {
            "lastPrice": float(pi.get("lastPrice", 0) or 0),
            "open": float(pi.get("open", 0) or 0),
            "prevClose": float(pi.get("close", 0) or 0),
            "high": float(intra.get("max", 0) or 0),
            "low": float(intra.get("min", 0) or 0),
            "change": float(pi.get("change", 0) or 0),
            "pChange": float(pi.get("pChange", 0) or 0),
            "vwap": float(pi.get("vwap", 0) or 0),
            "week52High": float(wk52.get("max", 0) or 0),
            "week52Low": float(wk52.get("min", 0) or 0),
        }
    except Exception:
        return None


@st.cache_data(ttl=20)
def get_live_market_quote(symbol_clean: str, yahoo_symbol: str = None):
    """Fetch the freshest available quote for a stock without using Yahoo."""
    is_bse_symbol = bool(yahoo_symbol and yahoo_symbol.upper().endswith(".BO"))

    if is_bse_symbol:
        # BSE quote path for SENSEX/BSE stocks (numeric scrip code).
        try:
            scrip_code = str(symbol_clean).strip()
            if scrip_code.isdigit():
                bse_headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36"
                    ),
                    "Accept": "application/json, text/plain, */*",
                    "Referer": "https://www.bseindia.com/",
                }
                bse_url = f"https://api.bseindia.com/BseIndiaAPI/api/GetQuoteHeader/w?quotetype=EQ&scripcode={scrip_code}&seriesid="
                bse_resp = requests.get(bse_url, headers=bse_headers, timeout=12)
                if bse_resp.ok:
                    raw = bse_resp.json()
                    lp = float(raw.get("currentValue") or raw.get("LTP") or 0)
                    chg_pct = float(raw.get("percentChange") or raw.get("ChangePercent") or 0)
                    quote_dt = raw.get("updatedOn") or raw.get("UpdTime") or ""
                    if lp > 0:
                        return {
                            "price": lp,
                            "change_pct": chg_pct,
                            "source": "SENSEX Live",
                            "quote_time": quote_dt if quote_dt else datetime.now().strftime("%d %b %Y %I:%M %p"),
                        }
        except Exception:
            pass

        return None

    # Map BSE symbols to an NSE tradable symbol when possible.
    candidate_symbol = symbol_clean
    if yahoo_symbol:
        upper_hint = yahoo_symbol.upper()
        if upper_hint.endswith(".BO"):
            mapped = BSE_TO_NSE_ALIASES.get(upper_hint)
            if mapped:
                candidate_symbol = mapped.replace(".NS", "")

    # --- Direct NSE quote-equity API endpoint (most reliable) ---
    try:
        nse_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        }
        nse_session = requests.Session()
        nse_session.get("https://www.nseindia.com", headers=nse_headers, timeout=10)
        
        quote_url = f"https://www.nseindia.com/api/quote-equity?symbol={candidate_symbol.upper()}"
        quote_resp = nse_session.get(quote_url, headers=nse_headers, timeout=12)
        if quote_resp.ok:
            quote_data = quote_resp.json()
            if quote_data.get("data") and len(quote_data["data"]) > 0:
                info = quote_data["data"][0]
                ltp = float(info.get("lastPrice") or info.get("LTP") or 0)
                chg = float(info.get("pChange") or info.get("change") or 0)
                if ltp > 0:
                    return {
                        "price": ltp,
                        "change_pct": chg,
                        "source": "NSE Live (Direct)",
                        "quote_time": datetime.now().strftime("%d %b %Y %I:%M %p"),
                    }
    except Exception:
        pass

    # --- Fallback: Use NseLib API ---
    try:
        raw = _nse_api.get_equity_details(candidate_symbol)
        if raw:
            pi = raw.get("priceInfo", {})
            last_price = float(pi.get("lastPrice", 0) or 0)
            if last_price > 0:
                return {
                    "price": last_price,
                    "change_pct": float(pi.get("pChange", 0) or 0),
                    "source": "NSE Live",
                    "quote_time": datetime.now().strftime("%d %b %Y %I:%M %p"),
                }
    except Exception:
        pass

    # Fallback to NSE intraday endpoint (chart-databyindex).
    try:
        raw_intraday = _nse_api.get_equity_intraday_data(candidate_symbol)
        if raw_intraday:
            graph_data = raw_intraday.get("grapthData") or raw_intraday.get("graphData") or []
            if graph_data:
                numeric_points = []
                for pt in graph_data:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        ts_val, price_val = pt[0], pt[1]
                        try:
                            p = float(price_val)
                            t = int(ts_val)
                            if p > 0:
                                numeric_points.append((t, p))
                        except Exception:
                            continue

                if numeric_points:
                    first_ts, first_price = numeric_points[0]
                    last_ts, last_price = numeric_points[-1]
                    change_pct = ((last_price - first_price) / first_price) * 100 if first_price > 0 else 0

                    try:
                        quote_dt = datetime.fromtimestamp(last_ts / 1000)
                        quote_time = quote_dt.strftime("%d %b %Y %I:%M %p")
                    except Exception:
                        quote_time = datetime.now().strftime("%d %b %Y %I:%M %p")

                    return {
                        "price": round(last_price, 2),
                        "change_pct": round(change_pct, 2),
                        "source": "NSE Intraday",
                        "quote_time": quote_time,
                    }
    except Exception:
        pass

    return None


@st.cache_data(ttl=120)
def get_live_index_data(index_symbol: str):
    """Fetch current market data for a broad index like NIFTY 50 or SENSEX."""
    try:
        index_data = yf.download(index_symbol, period="1d", interval="5m", progress=False)
        if index_data is None or index_data.empty:
            index_data = yf.download(index_symbol, period="5d", interval="1d", progress=False)
        if index_data is None or index_data.empty:
            return None

        if isinstance(index_data.columns, pd.MultiIndex):
            index_data.columns = index_data.columns.droplevel(1)

        close_series = pd.to_numeric(index_data["Close"], errors="coerce").dropna()
        if close_series.empty:
            return None

        current_price = float(close_series.iloc[-1])
        previous_price = float(close_series.iloc[-2]) if len(close_series) > 1 else current_price
        change_pct = ((current_price - previous_price) / previous_price) * 100 if previous_price else 0

        open_series = pd.to_numeric(index_data["Open"], errors="coerce").dropna()
        high_series = pd.to_numeric(index_data["High"], errors="coerce").dropna()
        low_series = pd.to_numeric(index_data["Low"], errors="coerce").dropna()

        return {
            "current_price": round(current_price, 2),
            "change_pct": round(change_pct, 2),
            "high": round(float(high_series.max()), 2) if not high_series.empty else round(current_price, 2),
            "low": round(float(low_series.min()), 2) if not low_series.empty else round(current_price, 2),
            "open": round(float(open_series.iloc[0]), 2) if not open_series.empty else round(current_price, 2),
            "previous_close": round(previous_price, 2),
            "points_change": round(current_price - previous_price, 2),
        }
    except Exception:
        return None


@st.cache_data(ttl=300)
def nse_get_equity_history(symbol_clean: str, days: int = 200):
    """
    getEquityHistoricalData wrapper — returns a DataFrame with OHLCV columns.
    Date index is sorted ascending. Returns None on failure (caller falls back to yfinance).
    """
    try:
        to_date = datetime.now().strftime("%d-%m-%Y")
        buffer_days = int(days * 1.6)  # account for weekends/holidays
        from_date = (datetime.now() - timedelta(days=buffer_days)).strftime("%d-%m-%Y")

        raw = _nse_api.get_equity_historical_data(symbol_clean, from_date, to_date)
        if not raw or "data" not in raw:
            return None

        rows = []
        for rec in raw["data"]:
            try:
                date_str = rec.get("CH_TIMESTAMP") or rec.get("mTIMESTAMP")
                rows.append({
                    "Date": pd.to_datetime(date_str),
                    "Open": float(rec.get("CH_OPENING_PRICE") or 0),
                    "High": float(rec.get("CH_TRADE_HIGH_PRICE") or 0),
                    "Low": float(rec.get("CH_TRADE_LOW_PRICE") or 0),
                    "Close": float(rec.get("CH_CLOSING_PRICE") or 0),
                    "Volume": float(rec.get("CH_TOTAL_TRADED_QUANTITY") or 0),
                })
            except Exception:
                continue

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df = df.sort_values("Date").set_index("Date")
        df = df[df["Close"] > 0].tail(days)
        return df
    except Exception:
        return None


@st.cache_data(ttl=60)
def nse_get_market_status():
    """getMarketStatus wrapper — returns dict with isOpen, status, tradeDate."""
    try:
        raw = _nse_api.get_market_status()
        if not raw:
            return None
        for m in raw.get("marketState", []):
            if m.get("market") == "Capital Market":
                return {
                    "isOpen": m.get("marketStatus") == "Open",
                    "status": m.get("marketStatus", "Unknown"),
                    "tradeDate": m.get("tradeDate", ""),
                }
        return None
    except Exception:
        return None


@st.cache_data(ttl=60)
def nse_get_all_indices():
    """getAllIndices wrapper — returns list of index dicts."""
    try:
        raw = _nse_api.get_all_indices()
        return raw.get("data") if raw else None
    except Exception:
        return None


def format_nse_index_row(row):
    """Normalize a raw NSE index row into a display-friendly dict.
    Supports stock-nse-india AllIndicesData keys.
    """
    index_name = (
        row.get("indexName")
        or row.get("index")
        or row.get("indexSymbol")
        or row.get("symbol")
        or "N/A"
    )

    last_value = float(row.get("last", row.get("lastPrice", 0)) or 0)
    change_value = float(row.get("change", row.get("changeValue", 0)) or 0)
    percent_change = float(row.get("percChange", row.get("percentChange", row.get("pChange", 0))) or 0)
    open_value = float(row.get("open", 0) or 0)
    high_value = float(row.get("high", 0) or 0)
    low_value = float(row.get("low", 0) or 0)
    prev_close = float(row.get("previousClose", 0) or 0)
    time_val = row.get("timeVal", "")

    return {
        "Index": index_name,
        "Last": round(last_value, 2),
        "Change": round(change_value, 2),
        "Change %": round(percent_change, 2),
        "Open": round(open_value, 2),
        "High": round(high_value, 2),
        "Low": round(low_value, 2),
        "Prev Close": round(prev_close, 2),
        "Time": str(time_val),
    }


@st.cache_data(ttl=3600)
def get_nse_stock_universe():
    """Return a broad NSE equity universe from the NSE equity master."""
    try:
        raw = _nse_api.get_equity_master()
        records = raw.get("data") if isinstance(raw, dict) else raw
        if not records:
            return SCANNER_STOCKS

        symbols = []
        for rec in records:
            try:
                symbol = (
                    rec.get("symbol")
                    or rec.get("SYMBOL")
                    or rec.get("tradingsymbol")
                    or rec.get("tradingSymbol")
                )
                series = (rec.get("series") or rec.get("SERIES") or "").upper()
                if symbol and (series in ("EQ", "") or series == "EQ"):
                    symbols.append(f"{str(symbol).upper()}.NS")
            except Exception:
                continue

        unique_symbols = list(dict.fromkeys(symbols))
        return unique_symbols if unique_symbols else SCANNER_STOCKS
    except Exception:
        return SCANNER_STOCKS


@st.cache_data(ttl=60)
def nse_get_pre_open_data():
    """getPreOpenMarketData wrapper."""
    try:
        raw = _nse_api.get_pre_open_market_data()
        return raw.get("data") if raw else None
    except Exception:
        return None


# ========================
# NSE INDIA DATA HELPERS
# ========================

def _get_last_trading_date():
    """Get the most recent trading date (skip weekends)."""
    dt = datetime.now()
    while dt.weekday() >= 5:  # Saturday=5, Sunday=6
        dt -= timedelta(days=1)
    # if market hasn't closed yet today, use yesterday
    if dt.date() == datetime.now().date() and datetime.now().hour < 16:
        dt -= timedelta(days=1)
        while dt.weekday() >= 5:
            dt -= timedelta(days=1)
    return dt


@st.cache_data(ttl=300)
def get_delivery_data(symbol_clean):
    """Fetch delivery % from NSE for a stock (last 5 trading days)."""
    try:
        if capital_market is None:
            return None
        today = datetime.now().strftime("%d-%m-%Y")
        from_date = (datetime.now() - timedelta(days=15)).strftime("%d-%m-%Y")
        pvd = capital_market.price_volume_and_deliverable_position_data(
            symbol_clean, from_date, today
        )
        if pvd is None or pvd.empty:
            return None
        # Clean column names (BOM character)
        pvd.columns = [c.strip().strip('\ufeff').strip('"') for c in pvd.columns]
        if "Symbol" in pvd.columns:
            pvd = pvd.rename(columns={"Symbol": "symbol"})
        # Get delivery %
        col = "%DlyQttoTradedQty"
        if col not in pvd.columns:
            for c in pvd.columns:
                if "Dly" in c and "Traded" in c:
                    col = c
                    break
        if col in pvd.columns:
            pvd[col] = pd.to_numeric(pvd[col], errors="coerce")
            latest_delivery = pvd[col].iloc[0]
            avg_delivery = pvd[col].mean()
            return {
                "latest_delivery_pct": round(float(latest_delivery), 2),
                "avg_delivery_pct": round(float(avg_delivery), 2),
            }
    except Exception:
        pass
    return None


@st.cache_data(ttl=300)
def get_market_breadth():
    """Fetch market breadth (advances/declines) from NSE indices."""
    try:
        if capital_market is None:
            return None
        idx = capital_market.market_watch_all_indices()
        if idx is None or idx.empty:
            return None
        # Find NIFTY 50 row
        nifty_row = idx[idx["indexSymbol"] == "NIFTY 50"]
        if nifty_row.empty:
            nifty_row = idx.iloc[0:1]
        row = nifty_row.iloc[0]
        advances = int(row.get("advances", 0))
        declines = int(row.get("declines", 0))
        unchanged = int(row.get("unchanged", 0))
        ad_ratio = round(advances / max(declines, 1), 2)
        return {
            "advances": advances,
            "declines": declines,
            "unchanged": unchanged,
            "ad_ratio": ad_ratio,
            "nifty_change_pct": round(float(row.get("percentChange", 0)), 2),
            "nifty_last": float(row.get("last", 0)),
        }
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_india_vix():
    """Fetch latest India VIX data."""
    try:
        if capital_market is None:
            return None
        today = datetime.now().strftime("%d-%m-%Y")
        from_date = (datetime.now() - timedelta(days=10)).strftime("%d-%m-%Y")
        vix = capital_market.india_vix_data(from_date, today)
        if vix is None or vix.empty:
            return None
        latest = vix.iloc[-1]
        return {
            "vix_close": round(float(latest.get("CLOSE_INDEX_VAL", 0)), 2),
            "vix_change_pct": round(float(latest.get("VIX_PERC_CHG", 0)), 2),
            "vix_high": round(float(latest.get("HIGH_INDEX_VAL", 0)), 2),
            "vix_low": round(float(latest.get("LOW_INDEX_VAL", 0)), 2),
        }
    except Exception:
        return None



@st.cache_data(ttl=300)
def get_participant_oi():
    """Fetch participant-wise OI (FII/DII/Pro/Client)."""
    try:
        if derivatives is None:
            return None
        dt = _get_last_trading_date()
        trade_date = dt.strftime("%d-%m-%Y")
        poi = derivatives.participant_wise_open_interest(trade_date)
        if poi is None or poi.empty:
            return None
        result = {}
        for _, row in poi.iterrows():
            ct = str(row.get("Client Type", "")).strip()
            if ct in ("FII", "DII", "Client", "Pro"):
                long_col = [c for c in poi.columns if "Total Long" in c]
                short_col = [c for c in poi.columns if "Total Short" in c]
                total_long = int(str(row[long_col[0]]).strip()) if long_col else 0
                total_short = int(str(row[short_col[0]]).strip()) if short_col else 0
                result[ct] = {
                    "long": total_long,
                    "short": total_short,
                    "net": total_long - total_short,
                }
        return result if result else None
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_fii_dii_today_activity():
    """Fetch latest available FII/DII cash activity from Groww's embedded page data."""
    url = "https://groww.in/fii-dii-data"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://groww.in/",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()

        match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', resp.text, re.S)
        if not match:
            return None

        payload = json.loads(match.group(1))
        rows = payload.get("props", {}).get("pageProps", {}).get("initialData", [])
        if not rows:
            return None

        latest = rows[0]
        fii = latest.get("fii") or {}
        dii = latest.get("dii") or {}
        as_of = latest.get("date")

        if not fii or not dii or not as_of:
            return None

        return {
            "date": str(as_of),
            "fii": {
                "buy": float(fii.get("grossBuy", 0) or 0),
                "sell": float(fii.get("grossSell", 0) or 0),
                "net": float(fii.get("netBuySell", 0) or 0),
            },
            "dii": {
                "buy": float(dii.get("grossBuy", 0) or 0),
                "sell": float(dii.get("grossSell", 0) or 0),
                "net": float(dii.get("netBuySell", 0) or 0),
            },
            "source": "Groww",
        }
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_nse_top_movers():
    """Fetch real-time top gainers and losers from NSE."""
    try:
        if capital_market is None:
            return None
        gainers = capital_market.top_gainers_or_losers("gainers")
        losers = capital_market.top_gainers_or_losers("loosers")
        result = {"gainers": [], "losers": []}
        if gainers is not None and not gainers.empty:
            for _, row in gainers.head(10).iterrows():
                result["gainers"].append({
                    "symbol": str(row.get("symbol", "")).strip(),
                    "ltp": float(row.get("ltp", 0)),
                    "change_pct": round(float(row.get("perChange", row.get("net_price", 0))), 2),
                })
        if losers is not None and not losers.empty:
            for _, row in losers.head(10).iterrows():
                result["losers"].append({
                    "symbol": str(row.get("symbol", "")).strip(),
                    "ltp": float(row.get("ltp", 0)),
                    "change_pct": round(float(row.get("perChange", row.get("net_price", 0))), 2),
                })
        return result
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_bse_market_snapshot():
    """Fetch a BSE snapshot using SENSEX and the tracked BSE stock list."""
    try:
        sensex_live = get_live_index_data("^BSESN")
        if sensex_live:
            sensex_last = float(sensex_live["current_price"])
            sensex_change_pct = float(sensex_live["change_pct"])
        else:
            sensex_data = safe_download("^BSESN", period="5d")
            if sensex_data is None or sensex_data.empty or "Close" not in sensex_data:
                return None
            sensex_series = pd.to_numeric(sensex_data["Close"], errors="coerce").dropna()
            if len(sensex_series) < 2:
                return None
            sensex_last = float(sensex_series.iloc[-1])
            sensex_prev = float(sensex_series.iloc[-2])
            sensex_change_pct = ((sensex_last - sensex_prev) / sensex_prev) * 100 if sensex_prev else 0

        def fetch_mover(item):
            name, ticker = item
            quote_symbol = BSE_TO_NSE_ALIASES.get(ticker, ticker)
            live_quote = get_live_market_quote(quote_symbol.replace(".NS", ""), quote_symbol)

            if live_quote and live_quote.get("price", 0) > 0:
                latest = float(live_quote["price"])
                change_pct = float(live_quote.get("change_pct", 0))
            else:
                intraday = safe_download(quote_symbol, period="5d")
                if intraday is None or intraday.empty or "Close" not in intraday:
                    return None
                close_series = pd.to_numeric(intraday["Close"], errors="coerce").dropna()
                if len(close_series) < 2:
                    return None
                latest = float(close_series.iloc[-1])
                previous = float(close_series.iloc[-2])
                change_pct = ((latest - previous) / previous) * 100 if previous else 0

            return {
                "name": name,
                "symbol": ticker.replace(".BO", ""),
                "ltp": round(latest, 2),
                "change_pct": round(change_pct, 2),
            }

        movers = []
        with ThreadPoolExecutor(max_workers=min(8, max(4, len(BSE_STOCKS) // 3))) as executor:
            for item in executor.map(fetch_mover, BSE_STOCKS.items()):
                if item is not None:
                    movers.append(item)

        if not movers:
            return None

        gainers = sorted(movers, key=lambda item: item["change_pct"], reverse=True)
        losers = sorted(movers, key=lambda item: item["change_pct"])
        advances = sum(1 for item in movers if item["change_pct"] > 0)
        declines = sum(1 for item in movers if item["change_pct"] < 0)
        unchanged = len(movers) - advances - declines
        ad_ratio = round(advances / max(declines, 1), 2)

        return {
            "sensex_last": round(sensex_last, 2),
            "sensex_change_pct": round(sensex_change_pct, 2),
            "advances": advances,
            "declines": declines,
            "unchanged": unchanged,
            "ad_ratio": ad_ratio,
            "gainers": gainers[:10],
            "losers": losers[:10],
        }
    except Exception:
        return None


def get_fetch_period(period):
    """Fetch enough history for indicators even when the requested view is short."""
    period_map = {
        "1wk": "6mo",
        "1mo": "6mo",
        "2mo": "6mo",
        "3mo": "6mo",
        "6mo": "6mo",
        "1y": "1y",
        "2y": "2y",
        "5y": "5y",
    }
    return period_map.get(period, period)


def trim_data_to_period(data, period):
    """Trim processed data to the UI-selected view window."""
    row_map = {
        "1wk": 5,
        "1mo": 22,
        "2mo": 44,
        "3mo": 66,
        "6mo": 132,
        "1y": 252,
        "2y": 504,
        "5y": 1260,
    }
    rows = row_map.get(period)
    if rows is None:
        return data
    return data.tail(rows)


def get_swing_expiry_date(period):
    """Return an approximate swing setup expiry date based on selected scan period."""
    day_map = {
        "1wk": 7,
        "1mo": 30,
        "2mo": 60,
        "3mo": 30,
        "6mo": 180,
        "1y": 365,
    }
    days = day_map.get(period, 30)
    expiry_dt = datetime.now() + timedelta(days=days)
    return expiry_dt.strftime("%d-%b-%Y")


def detect_candlestick_patterns(data):
    """Detect recent two-candle reversal patterns used in swing trading."""
    if data is None or len(data) < 2:
        return []

    latest = data.iloc[-1]
    prev = data.iloc[-2]

    prev_open = float(prev["Open"])
    prev_close = float(prev["Close"])
    latest_open = float(latest["Open"])
    latest_close = float(latest["Close"])

    prev_body = abs(prev_close - prev_open)
    latest_body = abs(latest_close - latest_open)

    prev_bull = prev_close > prev_open
    prev_bear = prev_close < prev_open
    latest_bull = latest_close > latest_open
    latest_bear = latest_close < latest_open

    patterns = []

    # Bullish engulfing: bearish candle followed by a larger bullish body that engulfs it.
    if (
        prev_bear and latest_bull
        and latest_open <= prev_close
        and latest_close >= prev_open
        and latest_body >= prev_body * 0.9
    ):
        patterns.append("Bullish Engulfing")

    # Bearish engulfing: bullish candle followed by a larger bearish body that engulfs it.
    if (
        prev_bull and latest_bear
        and latest_open >= prev_close
        and latest_close <= prev_open
        and latest_body >= prev_body * 0.9
    ):
        patterns.append("Bearish Engulfing")

    # Bullish harami: small bullish body contained within prior bearish body.
    if (
        prev_bear and latest_bull
        and latest_open >= min(prev_open, prev_close)
        and latest_close <= max(prev_open, prev_close)
        and latest_body <= prev_body * 0.7
    ):
        patterns.append("Bullish Harami")

    # Bearish harami: small bearish body contained within prior bullish body.
    if (
        prev_bull and latest_bear
        and latest_open <= max(prev_open, prev_close)
        and latest_close >= min(prev_open, prev_close)
        and latest_body <= prev_body * 0.7
    ):
        patterns.append("Bearish Harami")

    return patterns


def detect_price_action_setups(data, ma20_val=None):
    """Detect practical breakout/breakdown and rejection-style price action setups."""
    if data is None or len(data) < 22:
        return {
            "signals": [],
            "bias": "Neutral",
            "recent_high": None,
            "recent_low": None,
        }

    latest = data.iloc[-1]
    prev = data.iloc[-2]

    latest_close = float(latest["Close"])
    latest_open = float(latest["Open"])
    latest_high = float(latest["High"])
    latest_low = float(latest["Low"])
    prev_close = float(prev["Close"])

    lookback_high = pd.to_numeric(data["High"].tail(21).iloc[:-1], errors="coerce").dropna()
    lookback_low = pd.to_numeric(data["Low"].tail(21).iloc[:-1], errors="coerce").dropna()
    recent_high = float(lookback_high.max()) if len(lookback_high) > 0 else latest_high
    recent_low = float(lookback_low.min()) if len(lookback_low) > 0 else latest_low

    signals = []
    pa_score = 0

    if latest_close > recent_high and prev_close <= recent_high:
        signals.append("Bullish Breakout")
        pa_score += 2

    if latest_close < recent_low and prev_close >= recent_low:
        signals.append("Bearish Breakdown")
        pa_score -= 2

    if latest_low <= recent_low * 1.01 and latest_close > latest_open:
        signals.append("Support Rejection Bounce")
        pa_score += 1

    if latest_high >= recent_high * 0.99 and latest_close < latest_open:
        signals.append("Resistance Rejection")
        pa_score -= 1

    if ma20_val is not None:
        if prev_close < ma20_val <= latest_close:
            signals.append("MA20 Reclaim")
            pa_score += 1
        elif prev_close > ma20_val >= latest_close:
            signals.append("MA20 Loss")
            pa_score -= 1

    if pa_score >= 2:
        bias = "Bullish"
    elif pa_score <= -2:
        bias = "Bearish"
    else:
        bias = "Neutral"

    return {
        "signals": signals,
        "bias": bias,
        "recent_high": round(recent_high, 2),
        "recent_low": round(recent_low, 2),
    }


def normalize_symbol_input(raw_symbol, is_bse=False):
    """Normalize manual symbol input for reliable Yahoo/TradingView lookups."""
    symbol = raw_symbol.strip().upper().replace(" ", "")

    if not symbol:
        return symbol

    if is_bse:
        if symbol.isdigit():
            return f"{symbol}.BO"
        if not symbol.endswith(".BO") and not symbol.endswith(".NS"):
            return f"{symbol}.BO"
        return symbol

    if symbol.isdigit():
        # Users sometimes paste BSE scrip codes while NSE is selected.
        return f"{symbol}.BO"
    if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
        return f"{symbol}.NS"
    return symbol


@st.cache_data(ttl=1800)
def get_nse_nearest_option_expiry(symbol_clean):
    """Fetch nearest valid NSE option expiry for an equity symbol.

    Returns date string (e.g. 27-Mar-2026) or None if unavailable.
    """
    try:
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/",
        }

        # Prime cookies required by NSE API
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol_clean.upper()}"
        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None

        payload = resp.json()
        expiry_dates = payload.get("records", {}).get("expiryDates", [])
        if not expiry_dates:
            return None

        today = datetime.now().date()
        valid = []
        for d in expiry_dates:
            try:
                dt = datetime.strptime(d, "%d-%b-%Y").date()
                if dt >= today:
                    valid.append((dt, d))
            except Exception:
                continue

        if not valid:
            return None

        valid.sort(key=lambda x: x[0])
        return valid[0][1]
    except Exception:
        return None


@st.cache_data(ttl=120)
def get_nse_equity_option_chain_data(symbol_clean, expiry_date=None):
    """Fetch EquityOptionChainData-like payload with timestamp and data rows."""
    try:
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/",
        }

        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol_clean.upper()}"
        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None

        payload = resp.json()
        records = payload.get("records", {})
        timestamp = records.get("timestamp", "")
        data_rows = records.get("data", []) or []

        if expiry_date:
            filtered = []
            for row in data_rows:
                if str(row.get("expiryDate", "")).strip() == str(expiry_date).strip():
                    filtered.append(row)
            data_rows = filtered

        return {
            "timestamp": timestamp,
            "data": data_rows,
        }
    except Exception:
        return None


# ========================
# SWING TRADING DETECTION
# ========================
# ====================================
# ADVANCED SWING TRADING SCORING SYSTEM (v2.0)
# ====================================

def calculate_advanced_indicators(data):
    """Calculate all advanced indicators needed for improved scoring."""
    if data is None or len(data) < 50:
        return None
    
    try:
        close = data["Close"].squeeze() if isinstance(data["Close"], pd.DataFrame) else data["Close"]
        high = data["High"].squeeze() if isinstance(data["High"], pd.DataFrame) else data["High"]
        low = data["Low"].squeeze() if isinstance(data["Low"], pd.DataFrame) else data["Low"]
        
        # 9 EMA and 21 EMA
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema_signal = "Bullish" if ema9.iloc[-1] > ema21.iloc[-1] else "Bearish"
        ema_cross = (ema9.iloc[-2] <= ema21.iloc[-2] and ema9.iloc[-1] > ema21.iloc[-1]) if len(ema9) > 1 else False
        
        # ADX (Average Directional Index)
        try:
            adx_indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
            adx_value = float(adx_indicator.adx().iloc[-1]) if adx_indicator.adx() is not None else 0
        except:
            adx_value = 0
        
        # Stochastic RSI
        try:
            stoch_rsi = ta.momentum.stochrsi(close=close, length=14, rsi_length=14, k=3, d=3)
            stoch_k = float(stoch_rsi.iloc[-1, 0]) if stoch_rsi is not None and len(stoch_rsi) > 0 else 50
        except:
            stoch_k = 50
        
        # Regular RSI
        rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        rsi_value = float(rsi.iloc[-1])
        
        # Breakout distance
        breakout_high = high.rolling(20).max().iloc[-2]
        current_price = close.iloc[-1]
        breakout_distance = ((current_price - breakout_high) / breakout_high * 100) if breakout_high > 0 else 0
        
        return {
            "ema9": float(ema9.iloc[-1]),
            "ema21": float(ema21.iloc[-1]),
            "ema_signal": ema_signal,
            "ema_cross": ema_cross,
            "adx": adx_value,
            "stoch_rsi": stoch_k,
            "rsi": rsi_value,
            "breakout_distance": breakout_distance,
            "current_price": current_price
        }
    except Exception as e:
        return None


def generate_swing_signal_report(signal_data):
    """Generate a detailed swing trading signal scorecard report."""
    if signal_data is None:
        return None
    
    try:
        report = f"""
════════════════════════════════════════════════════════════════════
📊 SWING TRADE SIGNAL REPORT
════════════════════════════════════════════════════════════════════
Stock        : {signal_data.get('stock', 'N/A')}
Date         : {signal_data.get('data_date', 'N/A')}
Signal       : {signal_data.get('swing_type', 'NEUTRAL')}
Confidence   : {signal_data.get('confidence', 'N/A')}

────────────────────────────────────────────────────────────────────
📈 INDICATOR SCORECARD
────────────────────────────────────────────────────────────────────
Price vs MA20    : {'✅' if signal_data.get('close', 0) > signal_data.get('ma20', 0) else '❌'} (₹{signal_data.get('close')})
Price vs MA50    : {'✅' if signal_data.get('close', 0) > signal_data.get('ma50', 0) else '❌'} (₹{signal_data.get('ma50')})
9EMA vs 21EMA    : ✅ {signal_data.get('ema_signal', 'N/A')} (9:{signal_data.get('ema9')}, 21:{signal_data.get('ema21')})
RSI (14)         : {signal_data.get('rsi', 0):.1f} {'✅ Healthy' if 40 <= signal_data.get('rsi', 0) <= 60 else '⚠️'}
Stoch RSI        : {signal_data.get('stoch_rsi', 0):.0f} 
MACD             : {signal_data.get('macd_status', 'N/A')}
Volume           : {'✅' if signal_data.get('volume_confirmed') else '❌'} {signal_data.get('volume_ratio', 0):.2f}x avg
ADX              : {'✅' if signal_data.get('adx', 0) > 25 else ('⚠️' if signal_data.get('adx', 0) > 20 else '❌')} {signal_data.get('adx', 0):.0f}
Weekly Trend     : {signal_data.get('weekly_trend', 'N/A')}
Candlestick      : {signal_data.get('candlestick_pattern', 'No Pattern')}

TECH SCORE       : {signal_data.get('tech_score', 0)}/10

────────────────────────────────────────────────────────────────────
⚠️  PENALTY SCORECARD
────────────────────────────────────────────────────────────────────
"""
        
        penalties = signal_data.get('penalty_reasons', [])
        if penalties:
            for penalty in penalties:
                report += f"{penalty}\n"
        else:
            report += "✅ No penalties\n"
        
        report += f"""
PENALTY SCORE    : -{signal_data.get('penalty_score', 0)}

────────────────────────────────────────────────────────────────────
🎯 FINAL SCORE
────────────────────────────────────────────────────────────────────
Tech Score       : {signal_data.get('tech_score', 0)}
Penalty          : -{signal_data.get('penalty_score', 0)}
Market Score     : +{signal_data.get('market_score', 0)}
FINAL SCORE      : {signal_data.get('score', 0)}/10
SIGNAL           : {signal_data.get('swing_type', 'NEUTRAL')} ({signal_data.get('confidence', 'N/A')})

────────────────────────────────────────────────────────────────────
📋 TRADE PLAN
────────────────────────────────────────────────────────────────────
Entry Price      : ₹{signal_data.get('entry_price', 0):.2f}
Stop Loss        : ₹{signal_data.get('stop_loss_long', signal_data.get('stop_loss_short', 0)):.2f}
Target 1 (5%)    : ₹{signal_data.get('swing_target_up', 0):.2f}
Target 2 (8%)    : ₹{signal_data.get('swing_target_up_2', 0):.2f}
Risk/Share       : ₹{signal_data.get('risk_per_share', 0):.2f}
Shares           : {signal_data.get('shares_to_buy', 0)}
Total Risk       : ₹{signal_data.get('risk_amount', 0):.2f}
Position Size    : {signal_data.get('position_sizing', 'N/A')}
Hold Period      : {signal_data.get('hold_period_days', 'N/A')} days

────────────────────────────────────────────────────────────────────
🚨 RISK WARNINGS
────────────────────────────────────────────────────────────────────
"""
        
        avoid_reasons = signal_data.get('avoid_reasons', [])
        if avoid_reasons:
            for i, reason in enumerate(avoid_reasons, 1):
                report += f"{i}. {reason}\n"
        else:
            report += "✅ No hard stop conditions triggered\n"
        
        report += """────────────────────────────────────────────────────────────────────
✅ EXIT PLAN
────────────────────────────────────────────────────────────────────
- Book 50% profit at Target 1
- Trail SL to breakeven after T1
- Full exit if RSI > 75
- Full exit if daily close below 20 EMA
- Full exit if 9EMA crosses below 21EMA
════════════════════════════════════════════════════════════════════
"""
        return report
    except:
        return None


def analyze_single_stock(symbol, period="1mo"):
    """Enhanced single stock analysis with improved accuracy."""
    try:
        symbol_clean = symbol.upper().replace(".NS", "").replace(".BO", "")
        
        # Fetch data
        is_bse = symbol.upper().endswith(".BO")
        if is_bse:
            fetch_period = get_fetch_period(period)
            data = safe_download(symbol.upper(), fetch_period)
            if data is None or data.empty:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
        else:
            data = nse_get_equity_history(symbol_clean, days=200)
            if data is None or data.empty:
                fetch_period = get_fetch_period(period)
                data = yf.download(symbol.upper(), period=fetch_period, progress=False)
                if data is None or data.empty:
                    return None
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
        
        if len(data) < 50:
            return None
        
        # Calculate indicators
        close = data["Close"].squeeze() if isinstance(data["Close"], pd.DataFrame) else data["Close"]
        high = data["High"].squeeze() if isinstance(data["High"], pd.DataFrame) else data["High"]
        low = data["Low"].squeeze() if isinstance(data["Low"], pd.DataFrame) else data["Low"]
        volume = data["Volume"].squeeze() if isinstance(data["Volume"], pd.DataFrame) else data["Volume"]
        
        # Get advanced indicators
        adv_ind = calculate_advanced_indicators(data)
        if adv_ind is None:
            return None
        
        # Get candlestick pattern
        pattern, pattern_signal = detect_candlestick_pattern_improved(data)
        
        # Get sector strength
        sector_strength, sector_info = calculate_sector_strength(symbol_clean, {})
        
        # Get market conditions
        market_cond = get_market_conditions()
        
        # Calculate all key metrics
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1]
        
        macd = ta.trend.MACD(close=close)
        macd_val = macd.macd().iloc[-1]
        macd_signal = macd.macd_signal().iloc[-1]
        
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range().iloc[-1]
        
        vol_avg = volume.rolling(20).mean().iloc[-1]
        vol_ratio = close.iloc[-1] / vol_avg if vol_avg > 0 else 0
        
        return {
            "symbol": symbol.upper(),
            "current_price": close.iloc[-1],
            "ma20": ma20,
            "ma50": ma50,
            "ema9": adv_ind["ema9"],
            "ema21": adv_ind["ema21"],
            "rsi": rsi,
            "stoch_rsi": adv_ind["stoch_rsi"],
            "adx": adv_ind["adx"],
            "macd": macd_val,
            "macd_signal": macd_signal,
            "atr": atr,
            "volume_ratio": vol_ratio,
            "pattern": pattern,
            "pattern_signal": pattern_signal,
            "sector_strength": sector_strength,
            "nifty_trend": market_cond["nifty_trend"],
            "vix": market_cond["vix_level"],
            "vix_category": market_cond["vix_category"],
            "market_warnings": market_cond["warnings"]
        }
    except:
        return None


def detect_candlestick_pattern_improved(data):
    """Detect candlestick patterns with improved accuracy."""
    if data is None or len(data) < 3:
        return "No Pattern", None
    
    try:
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else latest
        prev_prev = data.iloc[-3] if len(data) > 2 else None
        
        open_price = float(latest["Open"])
        close_price = float(latest["Close"])
        high_price = float(latest["High"])
        low_price = float(latest["Low"])
        
        prev_open = float(prev["Open"])
        prev_close = float(prev["Close"])
        prev_high = float(prev["High"])
        prev_low = float(prev["Low"])
        
        body = abs(close_price - open_price)
        prev_body = abs(prev_close - prev_open)
        upper_shadow = high_price - max(close_price, open_price)
        lower_shadow = min(close_price, open_price) - low_price
        range_size = high_price - low_price
        
        # Bullish Engulfing
        if (close_price > open_price and
            prev_close < prev_open and
            body > prev_body and
            close_price > prev_open and
            open_price < prev_close):
            return "Bullish Engulfing", "BUY"
        
        # Bearish Engulfing
        if (close_price < open_price and
            prev_close > prev_open and
            body > prev_body and
            open_price > prev_close and
            close_price < prev_open):
            return "Bearish Engulfing", "SELL"
        
        # Hammer (bullish reversal)
        if body > 0 and lower_shadow >= 2 * body and upper_shadow <= 0.3 * body:
            if close_price > open_price:
                return "Hammer", "BUY"
        
        # Shooting Star (bearish reversal)
        if body > 0 and upper_shadow >= 2 * body and lower_shadow <= 0.3 * body:
            if close_price < open_price:
                return "Shooting Star", "SELL"
        
        # Morning Star (3 candle bullish reversal)
        if prev_prev is not None:
            prev_prev_close = float(prev_prev["Close"])
            prev_prev_open = float(prev_prev["Open"])
            prev_prev_body = abs(prev_prev_close - prev_prev_open)
            
            if (prev_prev_close < prev_prev_open and  # First candle bearish
                prev_close < prev_open and  # Second candle bearish or small
                close_price > open_price and  # Third candle bullish
                close_price > (prev_prev_open + prev_prev_close) / 2):
                return "Morning Star", "BUY"
        
        # Evening Star (3 candle bearish reversal)
        if prev_prev is not None:
            prev_prev_close = float(prev_prev["Close"])
            prev_prev_open = float(prev_prev["Open"])
            
            if (prev_prev_close > prev_prev_open and  # First candle bullish
                prev_close > prev_open and  # Second candle bullish or small
                close_price < open_price and  # Third candle bearish
                close_price < (prev_prev_open + prev_prev_close) / 2):
                return "Evening Star", "SELL"
        
        return "No Pattern", None
    except:
        return "No Pattern", None


def calculate_sector_strength(symbol, sector_map):
    """Check if stock's sector is strong (above 20 EMA)."""
    stock_clean = symbol.upper().replace(".NS", "").replace(".BO", "")
    
    # Default sector mapping for major stocks
    default_sector_map = {
        'COALINDIA': '^CNXENERGY', 'HDFCBANK': '^NSEBANK', 'RELIANCE': '^CNXENERGY',
        'TCS': '^CNXIT', 'ICICIBANK': '^NSEBANK', 'INFY': '^CNXIT',
        'TATASTEEL': '^CNXMETAL', 'SUNPHARMA': '^CNXPHARMA', 'WIPRO': '^CNXIT',
        'AXISBANK': '^NSEBANK', 'MARUTI': '^CNXAUTOMOB', 'BAJAJFINSV': '^CNXINFRA',
        'LT': '^CNXINFRA', 'ASIANPAINT': '^CNXINFRA', 'BRITANNIA': '^CNXINFRA'
    }
    
    sector_index = default_sector_map.get(stock_clean, None)
    if not sector_index:
        return None, None
    
    try:
        sector_data = yf.download(sector_index, period="100d", progress=False)
        if sector_data is None or len(sector_data) < 20:
            return None, None
        
        if isinstance(sector_data.columns, pd.MultiIndex):
            sector_data.columns = sector_data.columns.droplevel(1)
        
        close = sector_data["Close"].squeeze()
        ma20 = close.rolling(20).mean()
        latest_close = float(close.iloc[-1])
        latest_ma20 = float(ma20.iloc[-1])
        
        is_strong = latest_close > latest_ma20
        return "Strong" if is_strong else "Weak", {
            "sector_index": sector_index,
            "latest_close": latest_close,
            "ma20": latest_ma20
        }
    except:
        return None, None


def get_market_conditions():
    """Check overall market conditions: Nifty trend, VIX, FII activity."""
    conditions = {
        "nifty_trend": None,
        "nifty_above_ma50": False,
        "vix_level": None,
        "vix_category": None,
        "fii_bullish": None,
        "market_score": 0,
        "warnings": []
    }
    
    try:
        # Nifty 50 trend
        nifty_data = yf.download("^NSEI", period="100d", progress=False)
        if nifty_data is not None and len(nifty_data) > 50:
            if isinstance(nifty_data.columns, pd.MultiIndex):
                nifty_data.columns = nifty_data.columns.droplevel(1)
            
            close = nifty_data["Close"].squeeze()
            ma50 = close.rolling(50).mean()
            latest_close = float(close.iloc[-1])
            latest_ma50 = float(ma50.iloc[-1])
            
            conditions["nifty_above_ma50"] = latest_close > latest_ma50
            conditions["nifty_trend"] = "Bullish" if latest_close > latest_ma50 else "Bearish"
            
            if conditions["nifty_above_ma50"]:
                conditions["market_score"] += 1
            else:
                conditions["market_score"] -= 1
    except:
        pass
    
    try:
        # India VIX
        vix_data = yf.download("^INDIAVIX", period="5d", progress=False)
        if vix_data is not None and len(vix_data) > 0:
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix_data.columns = vix_data.columns.droplevel(1)
            
            vix_close = float(vix_data["Close"].iloc[-1])
            conditions["vix_level"] = vix_close
            
            if vix_close < 15:
                conditions["vix_category"] = "Low Fear (Good)"
                conditions["market_score"] += 1
            elif vix_close < 20:
                conditions["vix_category"] = "Normal"
            elif vix_close < 25:
                conditions["vix_category"] = "High Fear"
                conditions["market_score"] -= 1
                conditions["warnings"].append("VIX elevated")
            else:
                conditions["vix_category"] = "Extreme Fear"
                conditions["market_score"] -= 2
                conditions["warnings"].append("VIX critical - avoid new trades")
    except:
        pass
    
    return conditions


def calculate_improved_confidence_score(data, symbol, ad_ratio, fii_net, trend, trade_type, 
                                       price, ma20, ma50, rsi_val, volume_ratio, atr, 
                                       late_entry_risk, market_score):
    """Calculate improved confidence score with tech + penalty system."""
    
    advanced_ind = calculate_advanced_indicators(data)
    if advanced_ind is None:
        advanced_ind = {
            "ema9": ma20, "ema21": ma50, "ema_signal": "Neutral", "ema_cross": False,
            "adx": 0, "stoch_rsi": 50, "rsi": rsi_val, "breakout_distance": 0
        }
    
    # STEP 1: Calculate Base Tech Score (0-10)
    tech_score = 0
    
    # Price vs MA20
    if price > ma20:
        tech_score += 1
    # Price vs MA50
    if price > ma50:
        tech_score += 1
    # Price vs 9 EMA
    if price > advanced_ind["ema9"]:
        tech_score += 1
    # 9 EMA > 21 EMA (Bullish crossover)
    if advanced_ind["ema_signal"] == "Bullish":
        tech_score += 1
    # RSI zones
    if 50 <= rsi_val <= 70:  # Buy zone
        tech_score += 1
    elif 30 <= rsi_val <= 50:  # Sell zone
        tech_score += 1
    # MACD (detected in main function)
    if trend == "Bullish":
        tech_score += 1
    # Volume confirmation
    if volume_ratio >= 1.5:
        tech_score += 1
    # Weekly Trend
    if trend == "Bullish":
        tech_score += 1
    # ADX > 25 (Strong Trend)
    if advanced_ind["adx"] > 25:
        tech_score += 1
    
    # STEP 2: Calculate Penalty Score (0-5)
    penalty_score = 0
    penalty_reasons = []
    
    # A/D Ratio penalties
    if ad_ratio is not None:
        if ad_ratio < 0.5:
            penalty_score += 2
            penalty_reasons.append(f"A/D Ratio critical ({ad_ratio:.2f})")
        elif ad_ratio < 0.8:
            penalty_score += 1
            penalty_reasons.append(f"A/D Ratio weak ({ad_ratio:.2f})")
    
    # FII OI penalties
    if fii_net is not None:
        fii_in_cr = fii_net / 10000000  # Convert to crores
        if fii_in_cr < 1:
            penalty_score += 2
            penalty_reasons.append(f"FII OI very low ({fii_in_cr:.2f} Cr)")
        elif fii_in_cr < 5:
            penalty_score += 1
            penalty_reasons.append(f"FII OI low ({fii_in_cr:.2f} Cr)")
    
    # Late entry risk
    if late_entry_risk:
        penalty_score += 1
        penalty_reasons.append("Late entry (price already moved)")
    
    # Context conflicts
    if trend == "Bullish" and (ad_ratio is not None and ad_ratio < 0.5):
        penalty_score += 1
        penalty_reasons.append("Bullish setup but weak breadth")
    
    # RSI overbought/oversold extreme
    if rsi_val > 78:
        penalty_score += 2
        penalty_reasons.append("RSI extremely overbought (>78)")
    elif rsi_val < 22:
        penalty_score += 2
        penalty_reasons.append("RSI extremely oversold (<22)")
    
    # Breakout distance penalty
    if advanced_ind["breakout_distance"] > 3:
        penalty_score += 1
        penalty_reasons.append(f"Late entry - {advanced_ind['breakout_distance']:.1f}% moved")
    
    # ADX weakness
    if advanced_ind["adx"] < 20:
        penalty_score += 2
        penalty_reasons.append("ADX weak - no strong trend")
    
    # Sector strength (get sector info)
    sector_strength, sector_info = calculate_sector_strength(symbol, {})
    if sector_strength == "Weak" and trend == "Bullish":
        penalty_score += 2
        penalty_reasons.append("Sector trend is bearish")
    
    # STEP 3: Calculate Final Score
    final_score = tech_score - penalty_score
    
    # STEP 4: Signal Classification
    if final_score >= 8:
        signal_class = "STRONG BUY"
    elif 6 <= final_score <= 7:
        signal_class = "BUY"
    elif 4 <= final_score <= 5:
        signal_class = "WEAK BUY"
    elif 2 <= final_score <= 3:
        signal_class = "NEUTRAL"
    else:
        signal_class = "NO TRADE"
    
    # STEP 5: Confidence Level
    if penalty_score == 0 and final_score >= 8:
        confidence = "HIGH"
    elif penalty_score <= 1 and final_score >= 6:
        confidence = "MEDIUM"
    elif penalty_score <= 2 and final_score >= 4:
        confidence = "LOW"
    else:
        confidence = "VERY LOW"
    
    # Add market conditions to score
    if market_score >= 2:
        final_score += 1
    elif market_score <= -2:
        final_score -= 1
    
    return {
        "tech_score": tech_score,
        "penalty_score": penalty_score,
        "final_score": max(0, final_score),
        "signal_class": signal_class,
        "confidence": confidence,
        "penalty_reasons": penalty_reasons,
        "advanced_ind": advanced_ind,
        "sector_strength": sector_strength,
        "position_size_pct": 1.0 if confidence == "HIGH" else (0.75 if confidence == "MEDIUM" else (0.50 if confidence == "LOW" else 0.0))
    }


def get_suggested_strikes(current_price, direction, symbol_clean=None, expiry_date=None):
    """
    Suggest options strikes for swing trading based on current price and signal direction.
    Returns strikes for CALL (direction=1/BUY) or PUT (direction=-1/SELL).
    NSE options typically have ₹5 strike intervals.
    
    Args:
        current_price: Current market price of the stock
        direction: 1 for BUY (CALL), -1 for SELL (PUT), 0 for NEUTRAL
    
    Returns:
        dict with suggested strikes (ATM, OTM_1, OTM_2) and option type
    """
    if direction == 0:
        return None

    # Default fallback using fixed strike interval.
    atm = round(current_price / 5) * 5
    if direction == 1:
        fallback = {
            "option_type": "CALL (CE)",
            "atm": atm,
            "atm_note": "Most liquid, balanced premium/risk",
            "otm_1": atm + 5,
            "otm_1_note": "Lower premium, needs move up",
            "otm_2": atm + 10,
            "otm_2_note": "Cheap, wider miss range",
            "recommended": "ATM" if current_price % 5 < 2.5 else "OTM_1",
            "source": "Static",
            "chain_timestamp": None,
        }
    else:
        fallback = {
            "option_type": "PUT (PE)",
            "atm": atm,
            "atm_note": "Most liquid, balanced premium/risk",
            "otm_1": atm - 5,
            "otm_1_note": "Lower premium, protection if prices drop",
            "otm_2": atm - 10,
            "otm_2_note": "Cheap, needs bigger drop",
            "recommended": "ATM" if current_price % 5 < 2.5 else "OTM_1",
            "source": "Static",
            "chain_timestamp": None,
        }

    if not symbol_clean or not expiry_date:
        return fallback

    chain = get_nse_equity_option_chain_data(symbol_clean, expiry_date=expiry_date)
    if not chain or not chain.get("data"):
        return fallback

    available_strikes = []
    for row in chain["data"]:
        try:
            strike_price = float(row.get("strikePrice", 0) or 0)
            ce = row.get("CE")
            pe = row.get("PE")
            if direction == 1 and ce:
                available_strikes.append(strike_price)
            elif direction == -1 and pe:
                available_strikes.append(strike_price)
        except Exception:
            continue

    if not available_strikes:
        return fallback

    available_strikes = sorted(set(available_strikes))
    atm_chain = min(available_strikes, key=lambda x: abs(x - current_price))

    if direction == 1:
        upper = [s for s in available_strikes if s > atm_chain]
        otm_1 = upper[0] if len(upper) >= 1 else atm_chain
        otm_2 = upper[1] if len(upper) >= 2 else otm_1
        return {
            "option_type": "CALL (CE)",
            "atm": round(atm_chain, 2),
            "atm_note": "From NSE option chain (nearest ATM)",
            "otm_1": round(otm_1, 2),
            "otm_1_note": "1st OTM strike from live chain",
            "otm_2": round(otm_2, 2),
            "otm_2_note": "2nd OTM strike from live chain",
            "recommended": "ATM",
            "source": "NSE Option Chain",
            "chain_timestamp": chain.get("timestamp"),
        }

    lower = [s for s in available_strikes if s < atm_chain]
    lower_desc = sorted(lower, reverse=True)
    otm_1 = lower_desc[0] if len(lower_desc) >= 1 else atm_chain
    otm_2 = lower_desc[1] if len(lower_desc) >= 2 else otm_1
    return {
        "option_type": "PUT (PE)",
        "atm": round(atm_chain, 2),
        "atm_note": "From NSE option chain (nearest ATM)",
        "otm_1": round(otm_1, 2),
        "otm_1_note": "1st OTM strike from live chain",
        "otm_2": round(otm_2, 2),
        "otm_2_note": "2nd OTM strike from live chain",
        "recommended": "ATM",
        "source": "NSE Option Chain",
        "chain_timestamp": chain.get("timestamp"),
    }


def detect_swing_signals(symbol, period="6mo"):
    """
    Detect swing trading signals for a stock.
    Uses NSE India direct API (stock-nse-india equivalent) for fresh data,
    with a yfinance fallback for non-NSE symbols.
    Returns a dict with signal type, strength, details, and live price.
    """
    try:
        symbol_clean = symbol.upper().replace(".NS", "").replace(".BO", "")

        # Period → minimum required bars for indicators (~200 bars is enough for all periods)
        period_days = {
            "1wk": 200, "1mo": 200, "2mo": 200,
            "3mo": 200, "6mo": 200, "1y": 400,
        }
        hist_days = period_days.get(period, 200)

        # --- Step 1: fetch history by exchange path ---
        is_bse_symbol = symbol.upper().endswith(".BO")
        if is_bse_symbol:
            # SENSEX/BSE path: avoid NSE historical endpoint.
            fetch_period = get_fetch_period(period)
            data = safe_download(symbol.upper(), fetch_period)
            data_source = "SENSEX"
            if data is None or data.empty:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
        else:
            data = nse_get_equity_history(symbol_clean, days=hist_days)
            data_source = "NSE"

            if data is None or data.empty:
                # Fallback to yfinance only for NSE path fallback.
                fetch_period = get_fetch_period(period)
                data = yf.download(symbol.upper(), period=fetch_period, progress=False)
                data_source = "yfinance"

                if data is None or data.empty:
                    return None

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)

        # --- Step 2: fetch the freshest available quote ---
        live_quote = get_live_market_quote(symbol_clean, symbol)
        live_price = None
        live_change_pct = None
        live_source = None
        live_quote_time = None
        if live_quote and live_quote.get("lastPrice", 0) > 0:
            live_price = live_quote["lastPrice"]
            live_change_pct = live_quote.get("pChange")
        elif live_quote and live_quote.get("price", 0) > 0:
            live_price = live_quote["price"]
            live_change_pct = live_quote.get("change_pct")
            live_source = live_quote.get("source")
            live_quote_time = live_quote.get("quote_time")

        # Technical indicators
        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()

        rsi_indicator = ta.momentum.RSIIndicator(close=data["Close"].squeeze(), window=14)
        data["RSI"] = rsi_indicator.rsi()

        macd_indicator = ta.trend.MACD(close=data["Close"].squeeze())
        data["MACD"] = macd_indicator.macd()
        data["MACD_Signal"] = macd_indicator.macd_signal()
        data["MACD_Hist"] = macd_indicator.macd_diff()

        # --- Step 3: compute technical indicators ---
        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()

        rsi_indicator = ta.momentum.RSIIndicator(close=data["Close"].squeeze(), window=14)
        data["RSI"] = rsi_indicator.rsi()

        macd_indicator = ta.trend.MACD(close=data["Close"].squeeze())
        data["MACD"] = macd_indicator.macd()
        data["MACD_Signal"] = macd_indicator.macd_signal()
        data["MACD_Hist"] = macd_indicator.macd_diff()

        bb = ta.volatility.BollingerBands(close=data["Close"].squeeze(), window=20, window_dev=2)
        data["BB_Upper"] = bb.bollinger_hband()
        data["BB_Lower"] = bb.bollinger_lband()
        data["BB_Mid"] = bb.bollinger_mavg()

        data["Volume_Avg20"] = data["Volume"].rolling(20).mean()
        data["ATR"] = ta.volatility.AverageTrueRange(
            high=data["High"].squeeze(),
            low=data["Low"].squeeze(),
            close=data["Close"].squeeze(),
            window=14
        ).average_true_range()

        data = data.dropna()
        if len(data) < 5:
            return None

        latest = data.iloc[-1]
        prev = data.iloc[-2]
        data_date = data.index[-1].strftime("%d %b %Y")

        # Historical close (last candle from OHLCV data)
        hist_close = float(latest["Close"].item() if hasattr(latest["Close"], "item") else latest["Close"])

        # Use live NSE price if available, otherwise use historical close
        close = live_price if live_price and live_price > 0 else hist_close

        rsi_val = latest["RSI"].item()
        ma20 = latest["MA20"].item()
        ma50 = latest["MA50"].item()
        macd_val = latest["MACD"].item()
        macd_signal = latest["MACD_Signal"].item()
        macd_hist = latest["MACD_Hist"].item()
        prev_macd_hist = prev["MACD_Hist"].item()
        bb_upper = latest["BB_Upper"].item()
        bb_lower = latest["BB_Lower"].item()
        volume = latest["Volume"].item()
        vol_avg = latest["Volume_Avg20"].item()
        atr = latest["ATR"].item()

        prev_ma20 = prev["MA20"].item()
        prev_ma50 = prev["MA50"].item()

        signals = []
        nse_data = {}
        score = 0
        tech_score = 0
        market_score = 0
        risk_notes = []

        prev_close = float(prev["Close"].item() if hasattr(prev["Close"], "item") else prev["Close"])
        volume_ratio = float(volume / vol_avg) if vol_avg > 0 else 0.0

        # Step 1: primary trend filter.
        if close > ma20 > ma50:
            trend = "Bullish"
            score += 2
        elif close < ma20 < ma50:
            trend = "Bearish"
            score -= 2
        else:
            trend = "Sideways"
        signals.append(f"Trend: {trend} (Price {close:.2f}, MA20 {ma20:.2f}, MA50 {ma50:.2f})")

        # Step 2: momentum by RSI.
        if rsi_val > 60:
            momentum = "Bullish"
            score += 2
        elif rsi_val < 40:
            momentum = "Bearish"
            score -= 2
        else:
            momentum = "Neutral"
        signals.append(f"RSI: {rsi_val:.1f} ({momentum})")

        if prev_macd_hist <= 0 and macd_hist > 0:
            score += 1
            signals.append("MACD: Bullish crossover")
        elif prev_macd_hist >= 0 and macd_hist < 0:
            score -= 1
            signals.append("MACD: Bearish crossover")

        oversold = rsi_val < 30
        overbought = rsi_val > 70
        if oversold:
            risk_notes.append("Oversold bounce risk")
        if overbought:
            risk_notes.append("Overbought pullback risk")

        # Step 3: volume confirmation.
        volume_condition = (volume_ratio >= 1.5)
        if volume_condition:
            score += 2
            signals.append(f"Volume: {volume_ratio:.2f}x (validated)")
        else:
            signals.append(f"Volume: {volume_ratio:.2f}x (weak)")

        # Step 4: mandatory price action trigger.
        lookback = min(20, len(data) - 1)
        recent_high = float(pd.to_numeric(data["High"].tail(lookback + 1).iloc[:-1], errors="coerce").max())
        recent_low = float(pd.to_numeric(data["Low"].tail(lookback + 1).iloc[:-1], errors="coerce").min())

        breakout = close > recent_high and prev_close <= recent_high
        bounce = (
            (prev_close <= prev_ma20 and close > ma20 and close > prev_close)
            or (prev_close <= prev_ma50 and close > ma50 and close > prev_close)
        )
        breakdown = close < recent_low and prev_close >= recent_low
        pullback_rejection = (
            (prev_close >= prev_ma20 and close < ma20 and close < prev_close)
            or (prev_close >= prev_ma50 and close < ma50 and close < prev_close)
        )

        trade_type = "None"
        direction = 0  # +1 buy, -1 sell

        trigger_check_enabled = True
        if trigger_check_enabled and trend == "Bullish" and (breakout or bounce):
            direction = 1
            trade_type = "Breakout" if breakout else "Bounce"
            score += 2
        elif trigger_check_enabled and trend == "Bearish" and (breakdown or pullback_rejection):
            direction = -1
            trade_type = "Breakdown" if breakdown else "Pullback"
            score -= 2
        elif not trigger_check_enabled:
            if trend == "Bullish":
                direction = 1
                trade_type = "Trend Follow"
            elif trend == "Bearish":
                direction = -1
                trade_type = "Trend Follow"

        # Reject trades against trend or without mandatory trigger.
        if trade_type == "None":
            signals.append("No valid price-action trigger; setup rejected")
        else:
            signals.append(f"Trigger: {trade_type}")

        # Step 5: ATR late-entry check.
        late_entry_risk = False
        if direction == 1:
            move_from_trigger = close - recent_low
            if atr > 0 and move_from_trigger > 2 * atr:
                late_entry_risk = True
        elif direction == -1:
            move_from_trigger = recent_high - close
            if atr > 0 and move_from_trigger > 2 * atr:
                late_entry_risk = True
        if late_entry_risk:
            risk_notes.append("Late entry risk")

        # Step 6: market breadth context.
        breadth = get_market_breadth()
        ad_ratio = None
        if breadth:
            nse_data["breadth"] = breadth
            ad_ratio = float(breadth.get("ad_ratio", 0))
            if direction == 1:
                if ad_ratio > 1.2:
                    market_score += 2
                elif ad_ratio < 0.5:
                    market_score -= 2
            elif direction == -1:
                if ad_ratio < 0.5:
                    market_score -= 2
                elif ad_ratio > 1.2:
                    market_score += 2

        # Step 7: institutional activity context.
        oi_data = get_participant_oi()
        fii_net = None
        if oi_data:
            nse_data["participant_oi"] = oi_data
            fii = oi_data.get("FII")
            if fii:
                fii_net = float(fii.get("net", 0))
                if direction == 1:
                    if fii_net > 0:
                        market_score += 2
                    elif fii_net < 0:
                        market_score -= 2
                elif direction == -1:
                    if fii_net < 0:
                        market_score -= 2
                    elif fii_net > 0:
                        market_score += 2

        tech_score = score
        score = score + market_score

        # Entry/SL/targets only when trigger exists.
        entry_price = round(close, 2) if direction != 0 else None
        stop_loss_long = round(max(recent_low * 0.995, ma20 * 0.995), 2) if direction == 1 else round(close + atr, 2)
        stop_loss_short = round(min(recent_high * 1.005, ma20 * 1.005), 2) if direction == -1 else round(close + atr, 2)

        if direction == 1 and entry_price is not None:
            risk_per_share = max(entry_price - stop_loss_long, 0.01)
            swing_target_up = round(entry_price + 2 * risk_per_share, 2)
            resistance_window = pd.to_numeric(data["High"].tail(60), errors="coerce").dropna()
            next_res = float(resistance_window.max()) if len(resistance_window) > 0 else swing_target_up
            swing_target_up_2 = round(max(next_res, swing_target_up + risk_per_share), 2)
            swing_target_down = round(entry_price - 2 * atr, 2)
        elif direction == -1 and entry_price is not None:
            risk_per_share = max(stop_loss_short - entry_price, 0.01)
            swing_target_down = round(entry_price - 2 * risk_per_share, 2)
            support_window = pd.to_numeric(data["Low"].tail(60), errors="coerce").dropna()
            next_sup = float(support_window.min()) if len(support_window) > 0 else swing_target_down
            swing_target_up_2 = round(min(next_sup, swing_target_down - risk_per_share), 2)
            swing_target_up = round(entry_price + 2 * atr, 2)
        else:
            risk_per_share = 0.0
            swing_target_up = round(close + (2 * atr), 2)
            swing_target_down = round(close - (2 * atr), 2)
            swing_target_up_2 = swing_target_up

        # Step 8: final classification by score bands.
        if trade_type == "None" or trend == "Sideways":
            swing_type = "NEUTRAL"
        elif score >= 6:
            swing_type = "STRONG BUY"
        elif 3 <= score <= 5:
            swing_type = "BUY"
        elif -2 <= score <= 2:
            swing_type = "NEUTRAL"
        elif -5 <= score <= -3:
            swing_type = "SELL"
        else:
            swing_type = "STRONG SELL"

        # Guardrail: no strong sell on RSI<30 without fresh breakdown.
        if swing_type == "STRONG SELL" and oversold and not breakdown:
            swing_type = "SELL"
            risk_notes.append("Oversold without fresh breakdown")

        # Rule: never suggest trade without trigger.
        if trade_type == "None":
            swing_type = "NEUTRAL"

        color = "green" if "BUY" in swing_type else "red" if "SELL" in swing_type else "gray"

        # Step 9: Advanced confidence grading with new scoring system (v2.0)
        
        # Detect candlestick pattern
        candle_pattern, candle_signal = detect_candlestick_pattern_improved(data)
        if candle_pattern != "No Pattern":
            signals.append(f"Candlestick Pattern: {candle_pattern}")
        
        # Calculate improved confidence score
        score_result = calculate_improved_confidence_score(
            data, symbol_clean, ad_ratio, fii_net, trend, trade_type,
            close, ma20, ma50, rsi_val, volume_ratio, atr,
            late_entry_risk, market_score
        )
        
        # Hard stop conditions (Override all signals if triggered)
        hard_stop = False
        hard_stop_reasons = []
        
        if rsi_val > 78:
            hard_stop = True
            hard_stop_reasons.append("RSI extremely overbought (>78)")
        if rsi_val < 25:
            hard_stop = True
            hard_stop_reasons.append("RSI extremely oversold (<25)")
        if score_result["advanced_ind"]["adx"] < 15:
            hard_stop = True
            hard_stop_reasons.append("ADX < 15 (No trend)")
        if volume_ratio < 0.8:
            hard_stop = True
            hard_stop_reasons.append("Volume < 0.8x (Dead volume)")
        if score_result["final_score"] < 4:
            hard_stop = True
            hard_stop_reasons.append("Final score < 4 (Too weak)")
        if score_result["penalty_score"] > tech_score:
            hard_stop = True
            hard_stop_reasons.append("Penalties exceed tech score")
        if direction == 1 and score_result["advanced_ind"]["breakout_distance"] > 5:
            hard_stop = True
            hard_stop_reasons.append("Breakout distance > 5% (Too late)")
        
        # Apply hard stops
        if hard_stop:
            confidence = "VERY LOW"
            confidence_reason = "Hard stop conditions triggered: " + "; ".join(hard_stop_reasons)
            swing_type = "NEUTRAL"
            risk_notes.append("HARD STOP: NO TRADE")
        else:
            # Use improved confidence from new system
            confidence = score_result["confidence"]
            
            if score_result["signal_class"] == "NO TRADE" or trade_type == "None":
                confidence_reason = "Insufficient setup quality for entry"
            elif score_result["signal_class"] == "STRONG BUY" or score_result["signal_class"] == "BUY":
                if score_result["penalty_score"] == 0:
                    confidence_reason = f"Clean setup - {len(score_result['penalty_reasons'])} conflicts"
                else:
                    confidence_reason = f"Setup valid despite {len(score_result['penalty_reasons'])} concerns: " + "; ".join(score_result['penalty_reasons'][:2])
            else:
                if score_result["penalty_reasons"]:
                    confidence_reason = "Concerns: " + "; ".join(score_result['penalty_reasons'][:2])
                else:
                    confidence_reason = score_result["signal_class"]
            
            # Update swing_type based on new scoring if hard stops don't trigger
            if score_result["final_score"] >= 8 and direction != 0:
                swing_type = "STRONG BUY" if direction == 1 else "STRONG SELL"
            elif score_result["final_score"] >= 6 and direction != 0:
                swing_type = "BUY" if direction == 1 else "SELL"
            elif score_result["final_score"] >= 4 and direction != 0:
                swing_type = "BUY" if direction == 1 else "SELL"
            else:
                swing_type = "NEUTRAL"

        # Existing UI compatibility fields.
        weekly_trend = trend
        intraday_trigger_text = trade_type if trade_type != "None" else "Not triggered"
        framework_entry_ok = trade_type != "None" and swing_type != "NEUTRAL"
        missing_conditions = [] if framework_entry_ok else ["Trigger"]
        avoid_reasons = []
        if trend == "Sideways":
            avoid_reasons.append("Trend is sideways")
        if trade_type == "None":
            avoid_reasons.append("Mandatory price action trigger missing")
        if not volume_condition:
            avoid_reasons.append("Volume below 1.5x average")

        # Risk sizing with improved position sizing based on confidence
        confidence_position_size = score_result["position_size_pct"]
        risk_capital = 1500 * confidence_position_size  # Adjust based on confidence
        shares_to_buy = int(risk_capital // risk_per_share) if risk_per_share > 0 else 0
        risk_amount = round(shares_to_buy * risk_per_share, 2)
        position_note = {
            "HIGH": "Full Position (100%)",
            "MEDIUM": "Reduced Position (75%)",
            "LOW": "Half Position (50%)",
            "VERY LOW": "NO TRADE"
        }.get(confidence, "Hold")

        if direction == 1:
            hold_days = "5-12"
        elif direction == -1:
            hold_days = "4-10"
        else:
            hold_days = "N/A"

        if risk_notes:
            signals.append("Risk Note: " + "; ".join(risk_notes))

        # Compatibility aliases for downstream UI/return schema.
        macd_condition = bool(prev_macd_hist <= 0 and macd_hist > 0)
        rsi_condition = bool(rsi_val > 60)
        all_entry_conditions = (trade_type != "None" and swing_type != "NEUTRAL" and not hard_stop)
        capped_market = market_score

        macd_status = "Bullish crossover" if macd_condition else ("Bullish momentum" if macd_hist > 0 else "No bullish crossover")
        rsi_status = "Healthy momentum zone" if rsi_condition else ("Overbought" if rsi_val > 65 else "Below momentum zone")

        nse_expiry = get_nse_nearest_option_expiry(symbol_clean)
        expiry_date = nse_expiry if nse_expiry else get_swing_expiry_date(period)
        expiry_source = "NSE Option Chain" if nse_expiry else "Estimated"
        
        # Calculate suggested option strikes
        suggested_strikes = get_suggested_strikes(close, direction, symbol_clean=symbol_clean, expiry_date=expiry_date)

        return {
            "stock": symbol.upper(),
            "close": round(close, 2),
            "hist_close": round(hist_close, 2),
            "live_price": round(live_price, 2) if live_price else None,
            "live_change_pct": round(live_change_pct, 2) if live_change_pct is not None else None,
            "data_source": live_source or data_source,
            "swing_type": swing_type,
            "color": color,
            "score": score_result["final_score"],
            "signals": signals,
            "rsi": round(rsi_val, 2),
            "rsi_status": rsi_status,
            "macd_status": macd_status,
            "macd_hist": round(macd_hist, 4),
            "atr": round(atr, 2),
            "volume_ratio": round(volume_ratio, 2),
            "volume_confirmed": volume_condition,
            "weekly_trend": weekly_trend,
            "one_hour_trigger": trade_type if trade_type != "None" else "Not triggered",
            "entry_price": entry_price,
            "swing_target_up": swing_target_up,
            "swing_target_up_2": swing_target_up_2,
            "swing_target_down": swing_target_down,
            "stop_loss_long": stop_loss_long,
            "stop_loss_short": stop_loss_short,
            "risk_per_trade": 1500,
            "risk_per_share": round(risk_per_share, 2),
            "shares_to_buy": shares_to_buy,
            "risk_amount": risk_amount,
            "hold_period_days": hold_days,
            "framework_entry_ok": all_entry_conditions,
            "missing_conditions": [] if all_entry_conditions else ["Entry criteria not met"],
            "avoid_reasons": hard_stop_reasons if hard_stop else [],
            "confidence": confidence,
            "confidence_reason": confidence_reason,
            "expiry_date": expiry_date,
            "expiry_source": expiry_source,
            "ma20": round(ma20, 2),
            "ma50": round(ma50, 2),
            "nse_data": nse_data,
            "tech_score": score_result["tech_score"],
            "penalty_score": score_result["penalty_score"],
            "market_score": capped_market,
            "data_date": data_date,
            "quote_time": live_quote_time,
            # NEW: Advanced indicators
            "ema9": round(score_result["advanced_ind"]["ema9"], 2),
            "ema21": round(score_result["advanced_ind"]["ema21"], 2),
            "ema_signal": score_result["advanced_ind"]["ema_signal"],
            "ema_cross": score_result["advanced_ind"]["ema_cross"],
            "adx": round(score_result["advanced_ind"]["adx"], 2),
            "stoch_rsi": round(score_result["advanced_ind"]["stoch_rsi"], 2),
            "breakout_distance": round(score_result["advanced_ind"]["breakout_distance"], 2),
            "candlestick_pattern": candle_pattern,
            "candlestick_signal": candle_signal,
            "sector_strength": score_result["sector_strength"],
            "position_sizing": position_note,
            "ad_ratio": round(ad_ratio, 2) if ad_ratio else None,
            "fii_net": round(fii_net / 10000000, 2) if fii_net else None,  # In crores
            "penalty_reasons": score_result["penalty_reasons"],
            "suggested_strikes": suggested_strikes
        }

    except Exception as e:
        return None


def get_moneycontrol_news(symbol):
    try:
        stock = symbol.replace(".NS", "").replace(".BO", "").lower()

        url = f"https://www.moneycontrol.com/news/tags/{stock}.html"

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        articles = soup.select("li.clearfix")

        news = []

        for item in articles[:5]:
            title_tag = item.find("h2")

            if title_tag:
                title = title_tag.text.strip()
                link = title_tag.find("a")["href"]

                news.append({
                    "title": title,
                    "publisher": "Moneycontrol",
                    "link": link
                })

        return news

    except Exception:
        return []


def safe_download(symbol, period="5y"):
    """
    Safe wrapper around yfinance download.
    Prevents crashes if Yahoo fails or returns empty data.
    """
    base_symbol = symbol.upper()

    symbols_to_try = [base_symbol]

    # Yahoo occasionally fails for BSE (.BO) symbols. Fall back to NSE ticker.
    if base_symbol.endswith(".BO"):
        mapped_nse = BSE_TO_NSE_ALIASES.get(base_symbol)
        if mapped_nse:
            symbols_to_try.append(mapped_nse)

    # Include known alias chain for each candidate.
    expanded_symbols = []
    for sym in symbols_to_try:
        expanded_symbols.append(sym)
        expanded_symbols.extend(SYMBOL_ALIASES.get(sym, []))

    # De-duplicate while preserving order.
    symbols_to_try = list(dict.fromkeys(expanded_symbols))

    # For training, fall back to shorter history if 5y is unavailable for newer listings.
    periods_to_try = [period]
    if period == "5y":
        periods_to_try.extend(["2y", "1y"])

    for sym in symbols_to_try:
        for use_period in periods_to_try:
            for attempt in range(3):
                try:
                    data = yf.download(
                        sym,
                        period=use_period,
                        progress=False,
                        threads=False,
                        auto_adjust=False,
                    )

                    if data is None or data.empty:
                        # Secondary fallback path for transient Yahoo responses.
                        data = yf.Ticker(sym).history(period=use_period, auto_adjust=False)

                    if data is None or data.empty:
                        time.sleep(0.6 * (attempt + 1))
                        continue

                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(1)

                    return data
                except Exception:
                    time.sleep(0.6 * (attempt + 1))

    return None
def analyze_stock_data(symbol, period="1mo"):
    """Analyze a single stock"""
    try:
        if model is None:
            return {"error": "Model not trained"}

        fetch_period = get_fetch_period(period)
        data = safe_download(symbol.upper(), fetch_period)

        if data is None or data.empty:
            return {"error": f"No data found for {symbol}"}

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Indicators
        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()

        rsi_ind = ta.momentum.RSIIndicator(close=data["Close"].squeeze(), window=14)
        data["RSI"] = rsi_ind.rsi()

        macd_ind = ta.trend.MACD(close=data["Close"].squeeze())
        data["MACD"] = macd_ind.macd()
        data["MACD_Signal"] = macd_ind.macd_signal()
        data["MACD_Hist"] = macd_ind.macd_diff()

        bb = ta.volatility.BollingerBands(close=data["Close"].squeeze(), window=20, window_dev=2)
        data["BB_Upper"] = bb.bollinger_hband()
        data["BB_Lower"] = bb.bollinger_lband()
        data["BB_Mid"] = bb.bollinger_mavg()

        atr_ind = ta.volatility.AverageTrueRange(
            high=data["High"].squeeze(), low=data["Low"].squeeze(),
            close=data["Close"].squeeze(), window=14
        )
        data["ATR"] = atr_ind.average_true_range()

        data["Volume_Avg20"] = data["Volume"].rolling(20).mean()
        data["Volume_Avg5"] = data["Volume"].rolling(5).mean()
        data["Return"] = data["Close"].pct_change()

        # OBV for trend confirmation
        data["OBV"] = ta.volume.OnBalanceVolumeIndicator(
            close=data["Close"].squeeze(), volume=data["Volume"].squeeze()
        ).on_balance_volume()
        data["OBV_MA10"] = data["OBV"].rolling(10).mean()

        data = data.dropna()
        data = trim_data_to_period(data, period)

        if data.empty:
            return {"error": "Not enough data"}

        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else latest
        hist_close = float(latest["Close"])
        
        # Try to get live NSE price for more accurate current analysis
        symbol_clean = symbol.upper().replace(".NS", "").replace(".BO", "")
        live_quote = get_live_market_quote(symbol_clean, symbol)
        live_price = None
        if live_quote and live_quote.get("price", 0) > 0:
            live_price = live_quote["price"]
        
        # Use live NSE price if available, otherwise use historical close
        current_price = float(live_price) if live_price and live_price > 0 else hist_close

        rsi_val   = float(latest["RSI"])
        macd_val  = float(latest["MACD"])
        macd_sig  = float(latest["MACD_Signal"])
        macd_hist = float(latest["MACD_Hist"])
        prev_macd_hist = float(prev["MACD_Hist"])
        bb_upper  = float(latest["BB_Upper"])
        bb_lower  = float(latest["BB_Lower"])
        bb_mid    = float(latest["BB_Mid"])
        atr_val   = float(latest["ATR"])
        vol_cur   = float(latest["Volume"])
        vol_avg20 = float(latest["Volume_Avg20"])
        vol_avg5  = float(latest["Volume_Avg5"])
        ma20_val  = float(latest["MA20"])
        ma50_val  = float(latest["MA50"])
        obv_rising = float(latest["OBV"]) > float(latest["OBV_MA10"])
        candle_patterns = detect_candlestick_patterns(data)
        price_action = detect_price_action_setups(data, ma20_val=ma20_val)

        # ── Pro composite scoring ──────────────────────────────────────
        pro_score = 0

        # Trend context
        ma20_dist_pct = ((current_price - ma20_val) / ma20_val) * 100 if ma20_val > 0 else 0
        ma50_dist_pct = ((current_price - ma50_val) / ma50_val) * 100 if ma50_val > 0 else 0
        in_downtrend = current_price < ma20_val and current_price < ma50_val
        in_uptrend = current_price > ma20_val and current_price > ma50_val

        # 0. Price-to-MA distance
        if ma20_dist_pct < -5 and ma50_dist_pct < -5:
            pro_score -= 3
        elif ma20_dist_pct < -3:
            pro_score -= 2
        elif ma20_dist_pct > 5 and ma50_dist_pct > 5:
            pro_score += 3
        elif ma20_dist_pct > 3:
            pro_score += 2

        # 1. ML model base signal
        bb_range_val = float(bb_upper - bb_lower)
        bb_pband_val = (current_price - float(bb_lower)) / bb_range_val if bb_range_val > 0 else 0.5
        atr_ratio_val = atr_val / current_price if current_price > 0 else 0.0
        vol_ratio_val = vol_cur / vol_avg20 if vol_avg20 > 0 else 1.0
        return_5d_val = float(data["Close"].pct_change(5).iloc[-1])
        obv_series = data["OBV"] if "OBV" in data.columns else ta.volume.OnBalanceVolumeIndicator(
            close=data["Close"].squeeze(), volume=data["Volume"].squeeze()
        ).on_balance_volume()
        obv_ma10_val = float(obv_series.rolling(10).mean().iloc[-1])
        obv_ratio_val = float(obv_series.iloc[-1]) / abs(obv_ma10_val) if abs(obv_ma10_val) > 0 else 1.0

        features_df = pd.DataFrame([{
            "MA20":         ma20_val,
            "MA50":         ma50_val,
            "RSI":          rsi_val,
            "MACD_Hist":    macd_hist,
            "BB_pband":     bb_pband_val,
            "ATR_ratio":    atr_ratio_val,
            "Volume_ratio": vol_ratio_val,
            "Return":       float(latest["Return"]),
            "Return_5d":    return_5d_val,
            "OBV_ratio":    obv_ratio_val
        }])
        prediction = model.predict(features_df.values)[0]
        prob = float(model.predict_proba(features_df.values)[0].max())
        raw_direction = "UP" if prediction == 1 else "DOWN"

        # Confidence-aware ML weighting improves signal quality by avoiding
        # overreacting to uncertain model outputs.
        if prob < 0.60:
            direction = "NEUTRAL"
            ml_score = 0
        elif prob < 0.72:
            direction = raw_direction
            ml_score = 1 if direction == "UP" else -1
        elif prob < 0.85:
            direction = raw_direction
            ml_score = 2 if direction == "UP" else -2
        else:
            direction = raw_direction
            ml_score = 3 if direction == "UP" else -3

        pro_score += ml_score

        # 2. MACD trend confirmation
        if prev_macd_hist <= 0 and macd_hist > 0:
            pro_score += 3   # bullish crossover
        elif prev_macd_hist >= 0 and macd_hist < 0:
            pro_score -= 3   # bearish crossover
        elif macd_hist > 0 and macd_hist > prev_macd_hist:
            pro_score += 1   # rising bullish momentum
        elif macd_hist < 0 and macd_hist < prev_macd_hist:
            pro_score -= 1   # deepening bearish momentum

        # 3. RSI zones (trend-aware)
        if rsi_val < 30:
            if in_downtrend:
                pro_score -= 1  # oversold in downtrend confirms bearish pressure
            else:
                pro_score += 2
        elif rsi_val > 70:
            if in_uptrend:
                pass  # overbought in uptrend is momentum, neutral
            else:
                pro_score -= 2
        elif 30 <= rsi_val <= 45:
            if not in_downtrend:
                pro_score += 1
        elif 55 <= rsi_val <= 70:
            if not in_uptrend:
                pro_score -= 1

        # 4. MA alignment
        if current_price > ma20_val > ma50_val:
            pro_score += 2
        elif current_price < ma20_val < ma50_val:
            pro_score -= 2
        elif in_downtrend:
            pro_score -= 1
        elif in_uptrend:
            pro_score += 1

        # 5. Bollinger Band position (trend-aware)
        if current_price <= bb_lower:
            if in_downtrend:
                pro_score -= 1  # band walking in downtrend
            else:
                pro_score += 2
        elif current_price >= bb_upper:
            if in_uptrend:
                pro_score += 1  # band walking in uptrend
            else:
                pro_score -= 2

        # 6. Volume momentum (5-day vs 20-day avg)
        if vol_avg20 > 0:
            vol_ratio = vol_avg5 / vol_avg20
            if vol_ratio > 1.5 and direction == "UP":
                pro_score += 2
            elif vol_ratio > 1.5 and direction == "DOWN":
                pro_score -= 2
            elif vol_ratio > 1.2 and direction == "UP":
                pro_score += 1
            elif vol_ratio > 1.2 and direction == "DOWN":
                pro_score -= 1

        # 7. OBV trend
        if obv_rising and direction == "UP":
            pro_score += 1
        elif not obv_rising and direction == "DOWN":
            pro_score -= 1

        # 8. Candlestick reversal patterns
        if "Bullish Engulfing" in candle_patterns:
            pro_score += 2
        if "Bullish Harami" in candle_patterns:
            pro_score += 1
        if "Bearish Engulfing" in candle_patterns:
            pro_score -= 2
        if "Bearish Harami" in candle_patterns:
            pro_score -= 1

        # 9. Price action strategy bias
        if price_action["bias"] == "Bullish":
            pro_score += 2
        elif price_action["bias"] == "Bearish":
            pro_score -= 2

        # Clamp to [-10, 10]
        pro_score = max(-10, min(10, pro_score))

        # ── Final recommendation using pro score ───────────────────────
        if pro_score >= 5:
            recommendation = "STRONG BUY"
        elif pro_score >= 2:
            recommendation = "BUY"
        elif pro_score <= -5:
            recommendation = "STRONG SELL"
        elif pro_score <= -2:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"

        # ── Entry price ───────────────────────────────────────────────
        if "BUY" in recommendation:
            # Enter near the nearest support: max(MA20, BB_lower) or 0.5×ATR below current
            support = max(ma20_val, bb_lower)
            entry_price = round(min(current_price, support + atr_val * 0.3), 2)
        elif "SELL" in recommendation:
            # Short entry near nearest resistance: min(MA20, BB_upper) or current
            resistance = min(ma20_val, bb_upper)
            entry_price = round(max(current_price, resistance - atr_val * 0.3), 2)
        else:
            entry_price = round(current_price, 2)

        # ── Future entry price (next actionable zone) ────────────────
        recent_highs = pd.to_numeric(data["High"].tail(21).iloc[:-1], errors="coerce").dropna()
        recent_lows = pd.to_numeric(data["Low"].tail(21).iloc[:-1], errors="coerce").dropna()
        recent_resistance = float(recent_highs.max()) if len(recent_highs) > 0 else current_price
        recent_support = float(recent_lows.min()) if len(recent_lows) > 0 else current_price

        if "BUY" in recommendation:
            if entry_price < current_price * 0.985:
                # Old buy zone is already missed; wait for fresh breakout continuation.
                future_entry_price = round(max(current_price + atr_val * 0.2, recent_resistance * 1.002), 2)
                entry_status = "Previous buy zone crossed; wait for fresh breakout entry"
            else:
                future_entry_price = entry_price
                entry_status = "Current buy zone is active"
        elif "SELL" in recommendation:
            if entry_price > current_price * 1.015:
                # Old short zone is too far; wait for a fresh breakdown trigger.
                future_entry_price = round(min(current_price - atr_val * 0.2, recent_support * 0.998), 2)
                entry_status = "Previous sell zone invalid; wait for fresh breakdown entry"
            else:
                future_entry_price = entry_price
                entry_status = "Current sell zone is active"
        else:
            future_entry_price = round(current_price, 2)
            entry_status = "No trade setup"

        # ── Future price prediction (15-day horizon) ──────────────────
        forecast_horizon_days = 15
        # Scale ATR multiplier and percentage for a longer horizon
        if direction == "UP":
            forecast_price = round(current_price + max(atr_val * 6, current_price * 0.09), 2)
        elif direction == "DOWN":
            forecast_price = round(current_price - max(atr_val * 6, current_price * 0.09), 2)
        else:
            forecast_price = round(current_price, 2)

        forecast_move_pct = round(((forecast_price - current_price) / current_price) * 100, 2) if current_price > 0 else 0

        # Chart data
        prices = data["Close"].astype(float).tolist()
        ma20 = data["MA20"].astype(float).tolist()
        ma50 = data["MA50"].astype(float).tolist()
        rsi_values = data["RSI"].astype(float).tolist()
        dates = data.index.tolist()

        # OHLC
        ohlc = []
        for index, row in data.iterrows():
            ohlc.append({
                "date": str(index.date()),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"])
            })

        # News
        news = get_moneycontrol_news(symbol)

        if not news:
            try:
                ticker = yf.Ticker(symbol.upper())
                news_data = ticker.news or []

                for item in news_data[:5]:
                    news.append({
                        "title": item.get("title", ""),
                        "publisher": item.get("publisher", ""),
                        "link": item.get("link", "")
                    })
            except:
                pass

        ai_message = get_ai_explanation(
            recommendation, pro_score, rsi_val,
            ma20_val, ma50_val, current_price,
            macd_hist, bb_lower, bb_upper
        )

        llm_context = {
            "trend": "Bullish" if current_price > ma20_val > ma50_val else "Bearish" if current_price < ma20_val < ma50_val else "Sideways",
            "trigger": recommendation,
            "confidence": "High" if prob >= 0.85 else "Medium" if prob >= 0.72 else "Low",
            "risk_note": "Use strict risk control and wait for price-action confirmation.",
            "forecast": f"{forecast_price} ({forecast_move_pct:+.2f}% over {forecast_horizon_days} days)",
            "entry": f"Future actionable entry: {future_entry_price} ({entry_status})",
        }
        stock_description = _get_llm_stock_description(symbol.upper(), llm_context)

        return {
            "stock": symbol.upper(),
            "current_price": round(current_price, 2),
            "entry_price": entry_price,
            "future_entry_price": future_entry_price,
            "entry_status": entry_status,
            "forecast_price": forecast_price,
            "forecast_move_pct": forecast_move_pct,
            "forecast_horizon_days": forecast_horizon_days,
            "prediction": direction,
            "confidence": round(prob, 2),
            "confidence_percent": calculate_confidence_gauge(prob),
            "recommendation": recommendation,
            "pro_score": pro_score,
            "macd": data["MACD"].astype(float).tolist(),
            "macd_signal": data["MACD_Signal"].astype(float).tolist(),
            "macd_hist_list": data["MACD_Hist"].astype(float).tolist(),
            "bb_upper": data["BB_Upper"].astype(float).tolist(),
            "bb_lower": data["BB_Lower"].astype(float).tolist(),
            "bb_mid": data["BB_Mid"].astype(float).tolist(),
            "atr": round(atr_val, 2),
            "prices": prices,
            "ma20": ma20,
            "ma50": ma50,
            "rsi": rsi_values,
            "dates": dates,
            "ohlc": ohlc,
            "news": news,
            "ai_explanation": ai_message,
            "stock_description": stock_description,
            "rsi_value": round(rsi_val, 2),
            "candle_patterns": candle_patterns,
            "price_action_bias": price_action["bias"],
            "price_action_signals": price_action["signals"],
            "recent_price_action_high": price_action["recent_high"],
            "recent_price_action_low": price_action["recent_low"],
        }

    except Exception as e:
        return {"error": str(e)}


def scan_alert_candidate(symbol, period, include_swing, alert_threshold):
    """Analyze one stock for alert generation, returning only qualifying results."""
    analysis_pick = None
    swing_pick = None

    analysis = analyze_stock_data(symbol, period)
    if isinstance(analysis, dict) and "error" not in analysis:
        if abs(analysis.get("pro_score", 0)) >= alert_threshold:
            analysis_pick = analysis

    if include_swing:
        swing = detect_swing_signals(symbol, period=period)
        if swing and abs(swing.get("score", 0)) >= alert_threshold:
            swing_pick = swing

    return {
        "analysis": analysis_pick,
        "swing": swing_pick,
    }

# ========================
# SIDEBAR MENU
# ========================
auto_train_model_if_missing()

with st.sidebar:
    st.header("⚙️ Menu")

    page = st.radio(
        "Select Page",
        ["Single Stock Analysis", "Swing Trading Alerts", "Market Summary", "Train Model"]
    )

    st.divider()

    st.subheader("Model Status")
    if model is not None:
        st.success("✅ Model Loaded & Ready")
        if hasattr(model, "n_features_in_"):
            st.caption(f"Features: {int(model.n_features_in_)}")
        if os.path.exists(MODEL_PATH):
            st.caption("File: stock_model.pkl")
    else:
        st.error("❌ Model Not Loaded")
        if os.path.exists(MODEL_PATH):
            try:
                temp_model = joblib.load(MODEL_PATH)
                detected = getattr(temp_model, "n_features_in_", "unknown")
                st.caption(f"Detected model features: {detected}")
            except Exception:
                st.caption("Model file exists but could not be read.")
        else:
            st.caption("No trained model file found. Train once from Train Model page.")
            st.caption("You can also train and load it directly from here.")

            if st.button("🧠 Train & Load Model", use_container_width=True):
                with st.spinner("Training model. Please wait..."):
                    try:
                        result = train_model_func()
                        if result is not None and result.get("model") is not None:
                            st.session_state["last_train_metrics"] = result.get("metrics", {})
                            st.session_state["train_success"] = True
                            load_model.clear()
                            st.rerun()
                        else:
                            st.error("❌ Training did not complete. Please check warnings and retry.")
                    except Exception as e:
                        st.error(f"Training failed: {e}")

    if st.button("🔄 Reload Model", use_container_width=True):
        load_model.clear()
        st.rerun()

    st.divider()

# ========================
# PAGE: SINGLE STOCK ANALYSIS
# ========================
if page == "Single Stock Analysis":
    st.header("📊 Single Stock Analysis")

    col0, col1, col2, col3 = st.columns([1.2, 2, 2, 1])

    with col0:
        exchange = st.selectbox(
            "Exchange",
            ["NSE", "SENSEX"],
            index=0
        )

    is_bse = exchange == "SENSEX"

    with col1:
        if is_bse:
            picked = st.selectbox(
                "Pick from list",
                ["(type custom)"] + list(BSE_STOCKS.keys()),
                index=0,
                key="pick_bse_stock"
            )
        else:
            picked = st.selectbox(
                "Pick from list",
                ["(type custom)"] + SCANNER_STOCKS,
                index=0,
                key="pick_nse_stock"
            )

    with col2:
        if is_bse:
            custom = st.text_input(
                "Or enter symbol manually",
                value="",
                placeholder="e.g., 532540.BO, 500325.BO (SENSEX/BSE)",
                key="custom_bse_symbol"
            )
        else:
            custom = st.text_input(
                "Or enter symbol manually",
                value="",
                placeholder="e.g., TCS.NS, RELIANCE.NS",
                key="custom_nse_symbol"
            )

    if custom.strip():
        symbol = normalize_symbol_input(custom, is_bse=is_bse)
    elif is_bse and picked != "(type custom)":
        symbol = BSE_STOCKS[picked]
    elif not is_bse and picked != "(type custom)":
        symbol = picked
    else:
        symbol = "532540.BO" if is_bse else "TCS.NS"

    with col3:
        period = st.selectbox(
            "Time Period",
            ["1wk", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=0
        )

    if st.button("Analyze Stock", key="analyze_btn"):
        with st.spinner(f"Analyzing {symbol}..."):
            result = analyze_stock_data(symbol, period)
        
        if "error" not in result:
            # Store result in session state so it persists across reruns
            st.session_state["analysis_result"] = result
            st.session_state["analysis_period"] = period
        else:
            st.error(f"Error: {result['error']}")

    # Display stored analysis results (persists even when clicking other buttons)
    if "analysis_result" in st.session_state:
        result = st.session_state["analysis_result"]
        
        # ── Top metric row ─────────────────────────────────────────
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Current Rate",   f"₹{result['current_price']}")
        c2.metric("Future Entry",   f"₹{result.get('future_entry_price', result['entry_price'])}")
        c3.metric("ATR",            f"₹{result['atr']}")
        c4.metric("Model Bias",     result["prediction"])
        c5.metric("Confidence",     f"{result['confidence_percent']}%")
        c6.metric("Pro Score",      f"{result['pro_score']:+d} / 10")

        rec = result["recommendation"]
        if "STRONG BUY" in rec:
            c7.success(f"Final Trade Bias: {rec}")
        elif "BUY" in rec:
            c7.success(f"Final Trade Bias: {rec}")
        elif "STRONG SELL" in rec:
            c7.error(f"Final Trade Bias: {rec}")
        elif "SELL" in rec:
            c7.error(f"Final Trade Bias: {rec}")
        else:
            c7.warning(f"Final Trade Bias: {rec}")

        st.caption("Model Bias is model-only. NEUTRAL means low ML confidence. Final Trade Bias uses combined technical + ML Pro Score.")
        if result.get("entry_status"):
            st.caption(f"Entry Note: {result['entry_status']}")

        st.subheader("🕯️ Candlestick Patterns")
        patterns = result.get("candle_patterns", [])
        if patterns:
            for p in patterns:
                if "Bullish" in p:
                    st.success(f"{p} detected")
                elif "Bearish" in p:
                    st.error(f"{p} detected")
                else:
                    st.info(p)
        else:
            st.caption("No strong bullish/bearish candlestick reversal pattern on the latest candle.")

        st.subheader("📌 Price Action Strategies")
        pa_bias = result.get("price_action_bias", "Neutral")
        pa_signals = result.get("price_action_signals", [])
        pa_high = result.get("recent_price_action_high")
        pa_low = result.get("recent_price_action_low")

        if pa_bias == "Bullish":
            st.success(f"Price Action Bias: {pa_bias}")
        elif pa_bias == "Bearish":
            st.error(f"Price Action Bias: {pa_bias}")
        else:
            st.warning(f"Price Action Bias: {pa_bias}")

        if pa_high is not None and pa_low is not None:
            st.caption(f"Recent 20-candle range: High ₹{pa_high} | Low ₹{pa_low}")

        if pa_signals:
            for sig in pa_signals:
                st.write(f"• {sig}")
        else:
            st.caption("No breakout/breakdown/rejection setup triggered right now.")

        st.divider()
        st.info(f"🤖 {result['ai_explanation']}")
        sd = result.get('stock_description')
        if sd:
            st.info(f"🧠 {sd}")

        # Technical-only Buy/Not Buy Suggestion
        ai_verdict_key = f"ai_verdict_{result['stock']}"
        if st.button(f"🔧 Get Technical Verdict", key=ai_verdict_key):
            st.session_state[f"{ai_verdict_key}_clicked"] = True

        # Display technical verdict if generated (persists across reruns)
        if st.session_state.get(f"{ai_verdict_key}_clicked"):
            with st.spinner("🔄 Generating technical verdict..."):
                ai_verdict = get_ai_swing_trading_verdict(result['stock'], {
                    "close": result['current_price'],
                    "rsi": float(result['rsi'][0]) if result.get('rsi') else 50,
                    "ma20": result['ma20'][0] if result.get('ma20') else result['current_price'],
                    "ma50": result['ma50'][0] if result.get('ma50') else result['current_price'],
                    "signals": [result['recommendation']],
                    "score": result['pro_score'],
                    "atr": result['atr'],
                    "volume_ratio": 1.0
                })
                if ai_verdict:
                    if "BUY" in ai_verdict.upper():
                        st.success(f"**✅ Technical Verdict**\n\n{ai_verdict}")
                    elif "NOT BUY" in ai_verdict.upper() or "SELL" in ai_verdict.upper():
                        st.error(f"**❌ Technical Verdict**\n\n{ai_verdict}")
                    else:
                        st.info(f"**ℹ️ Technical Verdict**\n\n{ai_verdict}")
                else:
                    st.warning("Unable to generate AI verdict at this time.")

        future_move = result.get("forecast_move_pct", 0)
        future_price = result.get("forecast_price", result["current_price"])
        st.success(
            f"🔮 Future Prediction ({result.get('forecast_horizon_days', 5)}D): ₹{future_price} "
            f"({future_move:+.2f}% from current)"
        )

        st.subheader("📺 TradingView Chart")
        tradingview_mini_chart(result["stock"], height=320)

        # ── Price + BB + MA chart ──────────────────────────────────
        st.subheader("📈 Price Chart with Bollinger Bands & Moving Averages")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result['dates'], y=result['bb_upper'], name="BB Upper",
            line=dict(color="rgba(200,200,200,0.5)", width=1, dash="dot"), showlegend=True))
        fig.add_trace(go.Scatter(x=result['dates'], y=result['bb_lower'], name="BB Lower",
            fill="tonexty", fillcolor="rgba(200,200,200,0.15)",
            line=dict(color="rgba(200,200,200,0.5)", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=result['dates'], y=result['prices'], name="Price",
            line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=result['dates'], y=result['ma20'],   name="MA20",
            line=dict(color="green", width=1.5)))
        fig.add_trace(go.Scatter(x=result['dates'], y=result['ma50'],   name="MA50",
            line=dict(color="red",   width=1.5)))
        fig.update_layout(hovermode="x unified", height=420, xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # ── MACD chart ─────────────────────────────────────────────
        st.subheader("📊 MACD (Trend Confirmation)")
        fig_macd = go.Figure()
        colors = ["green" if v >= 0 else "red" for v in result["macd_hist_list"]]
        fig_macd.add_trace(go.Bar(x=result['dates'], y=result['macd_hist_list'],
            name="Histogram", marker_color=colors, opacity=0.6))
        fig_macd.add_trace(go.Scatter(x=result['dates'], y=result['macd'],
            name="MACD", line=dict(color="blue", width=1.5)))
        fig_macd.add_trace(go.Scatter(x=result['dates'], y=result['macd_signal'],
            name="Signal", line=dict(color="orange", width=1.5)))
        fig_macd.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_macd.update_layout(hovermode="x unified", height=300, xaxis_title="Date", yaxis_title="MACD")
        st.plotly_chart(fig_macd, use_container_width=True)

        # ── RSI chart ──────────────────────────────────────────────
        st.subheader("📊 RSI Indicator")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=result['dates'], y=result['rsi'], name="RSI",
            line=dict(color="purple", width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.update_layout(hovermode="x unified", height=280, yaxis=dict(range=[0, 100]),
            xaxis_title="Date", yaxis_title="RSI")
        st.plotly_chart(fig_rsi, use_container_width=True)

# ========================
# PAGE: SWING TRADING ALERTS
# ========================
elif page == "Swing Trading Alerts":
    st.header("🔔 Swing Trading Alerts")
    st.write("Scans stocks using MA crossovers, RSI, MACD, Bollinger Bands, volume spikes + **NSE delivery data, India VIX, market breadth & FII/DII OI**.")

    st.divider()

    col_ex, col_a, col_b = st.columns([1, 2, 1])
    with col_ex:
        swing_exchange = st.selectbox("Exchange", ["NSE", "SENSEX"], index=0, key="swing_exchange")
    with col_a:
        filter_type = st.multiselect(
            "Filter Signals",
            ["STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"],
            default=["STRONG BUY", "BUY", "SELL", "STRONG SELL"]
        )
    with col_b:
        scan_period = st.selectbox("Scan Period", ["1wk", "1mo", "2mo", "3mo", "6mo", "1y"], index=1)

    use_full_nse_universe = False
    if swing_exchange == "NSE":
        use_full_nse_universe = st.checkbox("Use Full NSE Universe", value=False)

    swing_scan_symbols = (
        get_nse_stock_universe() if swing_exchange == "NSE" and use_full_nse_universe else
        (SCANNER_STOCKS if swing_exchange == "NSE" else list(BSE_STOCKS.values()))
    )
    st.caption(f"Scanning universe: {swing_exchange} ({len(swing_scan_symbols)} tracked stocks)")

    # Reverse map for BSE symbols so the UI can show names instead of numeric tickers.
    bse_ticker_to_name = {v: k for k, v in BSE_STOCKS.items()}

    swing_results = []

    scan_swing_clicked = st.button("🔍 Scan for Swing Trades", type="primary")

    if scan_swing_clicked:
        swing_results = []
        all_results = []
        progress = st.progress(0)
        status = st.empty()

        # Show current market status before scan
        mkt_status = nse_get_market_status()
        if mkt_status:
            mkt_label = "🟢 Market Open" if mkt_status["isOpen"] else "🔴 Market Closed"
            st.info(f"{mkt_label} — Trade Date: {mkt_status.get('tradeDate', 'N/A')}  |  Data: NSE India API (live)")
        else:
            st.info("📡 Fetching data from NSE India API…")

        total = len(swing_scan_symbols)
        max_workers = min(12, max(4, (os.cpu_count() or 4) * 2))
        status.text(f"Scanning {total} {swing_exchange} stocks with {max_workers} workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(detect_swing_signals, symbol, scan_period): symbol
                for symbol in swing_scan_symbols
            }

            completed = 0
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                    if result and result["swing_type"] in filter_type:
                        swing_results.append(result)
                except Exception:
                    pass

                completed += 1
                status.text(f"Analyzed {completed}/{total}: {bse_ticker_to_name.get(symbol, symbol)}")
                progress.progress(completed / total)

        status.text("Scan complete. Sorting results...")

        if not swing_results and swing_exchange == "SENSEX" and all_results:
            swing_results = sorted(all_results, key=lambda x: abs(x["score"]), reverse=True)[:5]
            st.info("No strong SENSEX signals matched the selected filters. Showing the closest setups instead.")

    def get_display_stock_name(stock_symbol):
        return bse_ticker_to_name.get(stock_symbol, stock_symbol)

    # Fire toast notifications for actionable signals
    buy_alerts = [r for r in swing_results if "BUY" in r["swing_type"]]
    sell_alerts = [r for r in swing_results if "SELL" in r["swing_type"]]

    if buy_alerts:
        st.toast(f"🟢 {len(buy_alerts)} BUY signal(s) found!", icon="📈")
    if sell_alerts:
        st.toast(f"🔴 {len(sell_alerts)} SELL signal(s) found!", icon="📉")
    if not swing_results:
        st.toast("No swing signals found right now.", icon="😐")

    # Sort by absolute score (strongest signals first)
    swing_results.sort(key=lambda x: abs(x["score"]), reverse=True)

    st.divider()

    if not swing_results:
        st.info("No swing trading signals detected for the selected filters.")
    else:
        st.success(f"Found **{len(swing_results)}** swing trading signal(s) in {swing_exchange}")

        for res in swing_results:
            # Signal card
            with st.container(border=True):
                header_col, badge_col = st.columns([3, 1])

                with header_col:
                    stock_symbol = res['stock']
                    stock_name = get_display_stock_name(stock_symbol)
                    display_name = stock_name if swing_exchange == "SENSEX" else stock_symbol
                    st.subheader(display_name)
                    # Build live price label
                    live_p = res.get("live_price")
                    live_chg = res.get("live_change_pct")
                    src = res.get("data_source", "NSE")
                    hist_close = res.get("hist_close", res["close"])
                    if live_p:
                        chg_str = f" ({live_chg:+.2f}%)" if live_chg is not None else ""
                        price_label = f"🟢 Current Market Price: ₹{live_p}{chg_str}  |  Hist Close: ₹{hist_close}"
                    else:
                        price_label = f"Hist Close: ₹{hist_close}  |  Live quote unavailable"
                    st.caption(
                        f"{price_label}  |  ATR: {res['atr']}  |  Volume: {res['volume_ratio']}x avg"
                        f"  |  📅 Candle: {res.get('data_date', 'N/A')} [{src}]"
                        + (f"  |  ⏱ Quote: {res.get('quote_time')}" if res.get('quote_time') else "")
                    )

                with badge_col:
                    if "BUY" in res["swing_type"]:
                        st.success(f"**{res['swing_type']}**")
                    elif "SELL" in res["swing_type"]:
                        st.error(f"**{res['swing_type']}**")
                    else:
                        st.warning(f"**{res['swing_type']}**")

                # TradingView Mini Chart
                tradingview_mini_chart(res["stock"])

                # Metrics row
                m1, m2, m3, m4, m5 = st.columns(5)
                live_display = f"₹{res['live_price']}" if res.get("live_price") else f"₹{res['close']}"
                live_delta = f"{res['live_change_pct']:+.2f}%" if res.get("live_change_pct") is not None else None
                m1.metric("Current Market Price" if res.get("live_price") else "Historical Close", live_display, live_delta)
                m2.metric("MA20", f"₹{res['ma20']}")
                m3.metric("RSI", res["rsi"])
                m4.metric("Tech Score", f"{res.get('tech_score', res['score']):+d}")
                m5.metric("Total Score", f"{res['score']:+d}", f"Mkt: {res.get('market_score', 0):+d}")

                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Weekly Trend", res.get("weekly_trend", "N/A"))
                f2.metric("Volume Confirm", "YES" if res.get("volume_confirmed") else "NO")
                f3.metric("Trigger", "YES" if res.get("one_hour_trigger") not in (None, "", "Not triggered") else "NO")
                f4.metric("Confidence", res.get("confidence", "N/A"))
                st.caption(f"1H Signal: {res.get('one_hour_trigger', 'N/A')} | RSI: {res.get('rsi_status', 'N/A')} | MACD: {res.get('macd_status', 'N/A')}")

                confidence = res.get("confidence", "N/A")
                confidence_reason = res.get("confidence_reason")
                missing_conditions = res.get("missing_conditions", [])
                if confidence == "Low":
                    detail = confidence_reason or "Mandatory filters failed."
                    if missing_conditions:
                        detail += " Missing: " + ", ".join(missing_conditions)
                    st.warning(f"⚠️ Why confidence is low: {detail}")
                elif confidence_reason:
                    st.info(f"Confidence reason: {confidence_reason}")

                # Advanced analysis UI removed per user request
                
                # Legacy Technical Verdict
                if st.button(f"🔧 Get Technical Verdict", key=f"ai_verdict_{res['stock']}_{res['score']}"):
                    with st.spinner("🔄 Generating technical verdict..."):
                        ai_verdict = get_ai_swing_trading_verdict(res['stock'], {
                            "close": res['close'],
                            "rsi": res['rsi'],
                            "ma20": res['ma20'],
                            "ma50": res['ma50'],
                            "signals": res['signals'],
                            "score": res['score'],
                            "atr": res['atr'],
                            "volume_ratio": res['volume_ratio']
                        })
                        if ai_verdict:
                            if "BUY" in ai_verdict.upper():
                                st.success(f"**✅ Technical Verdict**\n\n{ai_verdict}")
                            elif "NOT BUY" in ai_verdict.upper() or "SELL" in ai_verdict.upper():
                                st.error(f"**❌ Technical Verdict**\n\n{ai_verdict}")
                            else:
                                st.info(f"**ℹ️ Technical Verdict**\n\n{ai_verdict}")
                        else:
                            st.warning("Unable to generate verdict at this time.")

                # Signals list
                st.markdown("**Detected Signals:**")
                for sig_idx, sig in enumerate(res["signals"]):
                    st.write(f"  {sig}")
                    # Advanced analysis UI removed per user request

                # NSE Data row
                nse = res.get("nse_data", {})
                if nse:
                    st.markdown("**📡 NSE India Data:**")
                    nc1, nc2, nc3, nc4 = st.columns(4)
                    dlv = nse.get("delivery")
                    if dlv:
                        nc1.metric("Delivery %", f"{dlv['latest_delivery_pct']}%", f"Avg: {dlv['avg_delivery_pct']}%")
                    vix = nse.get("vix")
                    if vix:
                        nc2.metric("India VIX", vix["vix_close"], f"{vix['vix_change_pct']:+.1f}%")
                    br = nse.get("breadth")
                    if br:
                        nc3.metric("A/D Ratio", br["ad_ratio"], f"{br['advances']}A / {br['declines']}D")
                    oi = nse.get("participant_oi", {})
                    fii_oi = oi.get("FII")
                    if fii_oi:
                        nc4.metric("FII Net OI", format_crore(fii_oi["net"]), "Long" if fii_oi["net"] > 0 else "Short")

                # Trade plan
                # Trade plan — entry uses live price when available
                entry_price = res.get("live_price") or res["close"]
                st.markdown("**📋 Swing Trade Plan:**")
                tp1, tp2, tp3, tp4 = st.columns(4)
                if "BUY" in res["swing_type"]:
                    tp1.metric("Entry Price", f"₹{res.get('entry_price', entry_price)}")
                    tp2.metric("Target 1 (1:2)", f"₹{res['swing_target_up']}")
                    tp3.metric("Stop Loss", f"₹{res['stop_loss_long']}")
                    tp4.metric("Target 2", f"₹{res.get('swing_target_up_2', res['swing_target_up'])}")
                elif "SELL" in res["swing_type"]:
                    tp1.metric("Entry Price", f"₹{entry_price}")
                    tp2.metric("Target (Down)", f"₹{res['swing_target_down']}")
                    tp3.metric("Stop Loss", f"₹{res['stop_loss_short']}")
                    tp4.metric("Expiry Date", res.get("expiry_date", "N/A"))
                else:
                    tp1.metric("Target (Up)", f"₹{res['swing_target_up']}")
                    tp2.metric("Target (Down)", f"₹{res['swing_target_down']}")
                    tp3.metric("Expiry Date", res.get("expiry_date", "N/A"))
                    tp4.metric("Signal", res["swing_type"])

                if "BUY" in res["swing_type"]:
                    rs1, rs2, rs3, rs4 = st.columns(4)
                    rs1.metric("Risk/Trade", f"₹{res.get('risk_per_trade', 1500)}")
                    rs2.metric("Risk/Share", f"₹{res.get('risk_per_share', 0)}")
                    rs3.metric("Shares to Buy", int(res.get("shares_to_buy", 0)))
                    rs4.metric("Hold Period", f"{res.get('hold_period_days', 'N/A')} days")

                # Suggested Options Strikes
                suggested = res.get("suggested_strikes")
                if suggested:
                    st.divider()
                    st.markdown(f"**📊 Suggested Options Strikes ({suggested['option_type']}):**")
                    
                    ostrike1, ostrike2, ostrike3 = st.columns(3)
                    
                    with ostrike1:
                        st.markdown(f"**ATM: ₹{suggested['atm']}**")
                        st.caption(suggested['atm_note'])
                        if suggested['recommended'] == "ATM":
                            st.success("✅ Recommended")
                    
                    with ostrike2:
                        st.markdown(f"**OTM 1: ₹{suggested['otm_1']}**")
                        st.caption(suggested['otm_1_note'])
                        if suggested['recommended'] == "OTM_1":
                            st.success("✅ Recommended")
                    
                    with ostrike3:
                        st.markdown(f"**OTM 2: ₹{suggested['otm_2']}**")
                        st.caption(suggested['otm_2_note'])

                    if suggested.get("source") or suggested.get("chain_timestamp"):
                        st.caption(
                            f"Source: {suggested.get('source', 'N/A')}"
                            + (f" | Chain Time: {suggested.get('chain_timestamp')}" if suggested.get("chain_timestamp") else "")
                        )
                    
                    st.caption(f"💡 **Trading Tip:** For swing trades, ATM or slightly OTM strikes offer best liquidity. Check bid-ask spread before entering. Use nearest expiry.")

                st.caption("Exit Plan: Book 50% at T1 and trail SL to breakeven. Full exit if RSI > 75 or daily close falls below 20 EMA.")
                if res.get("avoid_reasons"):
                    st.warning("Avoid Setup: " + ", ".join(res["avoid_reasons"]))

                st.caption(f"Expiry Source: {res.get('expiry_source', 'Estimated')}  |  Price Source: {'NSE Live' if res.get('live_price') else res.get('data_source', 'NSE')}")

# ========================
# PAGE: MARKET SUMMARY
# Simplified view: show only today's NIFTY, BANKNIFTY, SENSEX, India VIX, Gold, Silver and top movers
# ========================
elif page == "Market Summary":
    st.header("📈 Market Summary")

    movers = get_nse_top_movers() or {"gainers": [], "losers": []}
    market_watch = nse_get_all_indices()
    if not market_watch:
        try:
            market_watch = capital_market.market_watch_all_indices()
        except Exception:
            market_watch = None

    def _find_market_row(rows, keywords):
        if isinstance(rows, list):
            for row in rows:
                try:
                    normalized = format_nse_index_row(row)
                except Exception:
                    continue
                name = str(normalized.get("Index", "")).upper()
                if any(keyword in name for keyword in keywords):
                    return {
                        "label": str(normalized.get("Index") or "N/A"),
                        "last": float(normalized.get("Last", 0) or 0),
                        "change_pct": float(normalized.get("Change %", 0) or 0),
                    }
            return None
        if rows is None or getattr(rows, "empty", True):
            return None
        for _, row in rows.iterrows():
            name = str(row.get("indexSymbol") or row.get("index") or row.get("key") or "").upper()
            if any(keyword in name for keyword in keywords):
                return {
                    "label": str(row.get("indexSymbol") or row.get("index") or row.get("key") or "N/A"),
                    "last": float(row.get("last", 0) or 0),
                    "change_pct": float(row.get("percentChange", 0) or 0),
                }
        return None

    def _parse_float(value):
        cleaned = (
            str(value or "")
            .replace(",", "")
            .replace("%", "")
            .replace("+", "")
            .strip()
        )
        return float(cleaned) if cleaned else None

    def _get_live_sensex_summary():
        bse_url = "https://m.bseindia.com/IndicesView_New.aspx/Sensex.aspx?Scripflag=SPBSMSIP"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.bseindia.com/",
        }

        try:
            resp = requests.get(bse_url, headers=headers, timeout=12)
            if resp.ok:
                soup = BeautifulSoup(resp.text, "html.parser")
                last_node = soup.find(id="UcHeaderMenu1_sensexLtp")
                change_node = soup.find(id="UcHeaderMenu1_sensexChange")
                pct_node = soup.find(id="UcHeaderMenu1_sensexPerChange")

                last_value = _parse_float(last_node.get_text(strip=True) if last_node else None)
                change_value = _parse_float(change_node.get_text(strip=True) if change_node else None)
                pct_value = _parse_float(pct_node.get_text(strip=True) if pct_node else None)

                if last_value is not None and pct_value is not None:
                    return {
                        "current_price": round(last_value, 2),
                        "change_pct": round(pct_value, 2),
                        "points_change": round(change_value or 0, 2),
                        "source": "BSE Live",
                    }
        except Exception:
            pass

        return get_live_index_data("^BSESN") or {}

    nifty = _find_market_row(market_watch, ["NIFTY 50", "NIFTY"])
    banknifty = _find_market_row(market_watch, ["NIFTY BANK", "BANKNIFTY"])
    vix = _find_market_row(market_watch, ["INDIA VIX"])
    sensex = _get_live_sensex_summary()
    gold = get_live_index_data("GC=F") or {}
    silver = get_live_index_data("SI=F") or {}
    fii_dii = get_fii_dii_today_activity()

    def _fmt(val):
        return f"{val:.2f}" if isinstance(val, (int, float)) else "N/A"

    def _fmt_pct(val):
        try:
            return f"{float(val):+.2f}%"
        except Exception:
            return "N/A"

    def _fmt_cr(val):
        return f"{float(val):,.2f} Cr"

    def _fmt_activity_date(date_str):
        try:
            return pd.to_datetime(date_str).strftime("%d %b %Y")
        except Exception:
            return str(date_str or "N/A")

    def _render_nifty_market_chart(height=320):
        html = f"""
        <div class="tradingview-widget-container" style="width:100%;height:{height}px;">
          <div class="tradingview-widget-container__widget" style="width:100%;height:100%;"></div>
          <script type="text/javascript"
                  src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js"
                  async>
          {{
            "symbol": "NSE:NIFTY",
            "width": "100%",
            "height": "{height}",
            "locale": "en",
            "dateRange": "1D",
            "colorTheme": "dark",
            "isTransparent": true,
            "autosize": true,
            "largeChartUrl": "",
            "chartOnly": false,
            "noTimeScale": false
          }}
          </script>
        </div>
        """
        components.html(html, height=height + 10)

    top_left, top_right = st.columns([1.45, 1.0], gap="large")

    with top_left:
        c1, c2, c3 = st.columns(3)
        with c1:
            if nifty:
                st.metric("NIFTY 50", _fmt(nifty.get('last')), _fmt_pct(nifty.get('change_pct', 0)))
            else:
                st.info("NIFTY data unavailable")
        with c2:
            if banknifty:
                st.metric("NIFTY BANK", _fmt(banknifty.get('last')), _fmt_pct(banknifty.get('change_pct', 0)))
            else:
                st.info("BANKNIFTY data unavailable")
        with c3:
            if sensex and sensex.get('current_price') is not None:
                st.metric("SENSEX", _fmt(sensex.get('current_price')), _fmt_pct(sensex.get('change_pct', 0)))
            else:
                st.info("SENSEX data unavailable")

        d1, d2, d3 = st.columns(3)
        with d1:
            if vix:
                st.metric("India VIX", _fmt(vix.get('last')), _fmt_pct(vix.get('change_pct', 0)))
            else:
                st.info("VIX data unavailable")
        with d2:
            if gold and gold.get('current_price') is not None:
                st.metric("Gold ($)", f"${_fmt(gold.get('current_price'))}", _fmt_pct(gold.get('change_pct', 0)))
            else:
                st.info("Gold data unavailable")
        with d3:
            if silver and silver.get('current_price') is not None:
                st.metric("Silver ($)", f"${_fmt(silver.get('current_price'))}", _fmt_pct(silver.get('change_pct', 0)))
            else:
                st.info("Silver data unavailable")

    with top_right:
        st.markdown("### NIFTY 50 Chart")
        _render_nifty_market_chart()

    st.subheader("FII / DII Today")
    if fii_dii:
        st.caption(
            f"Latest available cash activity: {_fmt_activity_date(fii_dii['date'])}  |  Source: {fii_dii['source']}"
        )
        fcol, dcol = st.columns(2, gap="large")
        with fcol:
            st.markdown("**FII**")
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.metric("Buy", _fmt_cr(fii_dii["fii"]["buy"]))
            with fc2:
                st.metric("Sell", _fmt_cr(fii_dii["fii"]["sell"]))
            with fc3:
                st.metric("Net", _fmt_cr(fii_dii["fii"]["net"]))
        with dcol:
            st.markdown("**DII**")
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.metric("Buy", _fmt_cr(fii_dii["dii"]["buy"]))
            with dc2:
                st.metric("Sell", _fmt_cr(fii_dii["dii"]["sell"]))
            with dc3:
                st.metric("Net", _fmt_cr(fii_dii["dii"]["net"]))
    else:
        st.info("FII / DII buy-sell data unavailable right now")

    st.divider()

    # Top movers (NSE)
    st.subheader("Top Movers (NSE Live)")
    left, right = st.columns(2)
    with left:
        st.write("### Top Gainers")
        if movers.get('gainers'):
            for g in movers['gainers'][:10]:
                st.write(f"• **{g.get('symbol')}** — {g.get('ltp')}  ({g.get('change_pct'):+.2f}%)")
        else:
            st.info("Top gainers unavailable")
    with right:
        st.write("### Top Losers")
        if movers.get('losers'):
            for l in movers['losers'][:10]:
                st.write(f"• **{l.get('symbol')}** — {l.get('ltp')}  ({l.get('change_pct'):+.2f}%)")
        else:
            st.info("Top losers unavailable")


# ========================
# PAGE: TRAIN MODEL
# ========================
elif page == "Train Model":
    st.header("🧠 Train ML Model")

    if st.session_state.get("train_success"):
        st.success("✅ Model trained and loaded successfully!")
        m = st.session_state.get("last_train_metrics")
        if m:
            st.caption(
                f"Validation Metrics → Accuracy: {m['accuracy']:.2%} | Precision: {m['precision']:.2%} | Recall: {m['recall']:.2%} | F1: {m['f1']:.2%}"
            )
        st.session_state["train_success"] = False

    st.info(
        "This will download 5 years of historical data for all stocks "
        "and train a RandomForest model. This may take several minutes."
    )

    if st.button("Start Training", type="primary"):
        with st.spinner("Training model..."):
            try:
                result = train_model_func()
                if result is not None and result.get("model") is not None:
                    st.session_state["last_train_metrics"] = result.get("metrics", {})
                    st.session_state["train_success"] = True
                    load_model.clear()
                    st.rerun()
                else:
                    st.error("❌ Training did not complete. Please check the warnings above and try again.")
            except Exception as e:
                st.error(f"Error: {e}")

