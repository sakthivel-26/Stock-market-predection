import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib
import requests
from bs4 import BeautifulSoup
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from nselib import capital_market, derivatives
import base64
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import time


from dotenv import load_dotenv
import os

def send_gmail_oauth(recipient_email, subject, html_body):
    """
    Send email using Gmail OAuth with auto refresh token
    """

    import base64
    from email.mime.text import MIMEText
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = Credentials(
        None,
        refresh_token=GMAIL_REFRESH_TOKEN,
        client_id=GMAIL_CLIENT_ID,
        client_secret=GMAIL_CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token",
    )

    # auto refresh token
    if not creds.valid:
        creds.refresh(Request())

    service = build("gmail", "v1", credentials=creds)

    message = MIMEText(html_body, "html")
    message["to"] = recipient_email
    message["from"] = GMAIL_SENDER
    message["subject"] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    service.users().messages().send(
        userId="me",
        body={"raw": raw}
    ).execute()

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

GMAIL_CLIENT_ID = os.getenv("GMAIL_CLIENT_ID")
GMAIL_CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET")
GMAIL_REFRESH_TOKEN = os.getenv("GMAIL_REFRESH_TOKEN")
GMAIL_SENDER = os.getenv("GMAIL_SENDER")

def get_telegram_chat_id(bot_token):

    try:
        url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get("ok") and data.get("result"):
            return data["result"][-1]["message"]["chat"]["id"]

    except:
        pass

    return None
TELEGRAM_CHAT_ID = get_telegram_chat_id(TELEGRAM_BOT_TOKEN)
# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="AI Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
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

# ========================
# LOAD MODEL
# ========================
@st.cache_resource
def load_model():
    """Load pre-trained model"""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        m = joblib.load(MODEL_PATH)
        # Verify model expects 6 features
        if hasattr(m, 'n_features_in_') and m.n_features_in_ != 6:
            os.remove(MODEL_PATH)
            return None
        return m
    except Exception:
        return None

model = load_model()
if model is None and os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
    load_model.clear()
    model = None

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


def tradingview_mini_chart(symbol_nse, height=220):
    """Render a TradingView mini chart widget for an NSE stock."""
    # Convert "RELIANCE.NS" -> "NSE:RELIANCE"
    tv_symbol = "NSE:" + symbol_nse.upper().replace(".NS", "")
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
            data["MA20"] = data["Close"].rolling(20).mean()
            data["MA50"] = data["Close"].rolling(50).mean()

            rsi = ta.momentum.RSIIndicator(
                close=data["Close"].squeeze(),
                window=14
            )

            data["RSI"] = rsi.rsi()

            data["Volume_Avg20"] = data["Volume"].rolling(20).mean()

            data["Return"] = data["Close"].pct_change()

            # Target variable
            data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

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
    features_df = dataset[
        ["MA20", "MA50", "RSI", "Volume", "Volume_Avg20", "Return"]
    ]

    features = features_df.values
    target = dataset["Target"].values

    status_text.text("Training RandomForest model...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(features, target)

    joblib.dump(model, MODEL_PATH)

    status_text.text("Model trained successfully! ✅")
    progress_bar.progress(1.0)

    return model

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
def get_nse_top_movers():
    """Fetch real-time top gainers and losers from NSE."""
    try:
        gainers = capital_market.top_gainers_or_losers("gainers")
        losers = capital_market.top_gainers_or_losers("losers")
        result = {"gainers": [], "losers": []}
        if gainers is not None and not gainers.empty:
            for _, row in gainers.head(10).iterrows():
                result["gainers"].append({
                    "symbol": row.get("symbol", ""),
                    "ltp": float(row.get("ltp", 0)),
                    "change_pct": round(float(row.get("perChange", row.get("net_price", 0))), 2),
                })
        if losers is not None and not losers.empty:
            for _, row in losers.head(10).iterrows():
                result["losers"].append({
                    "symbol": row.get("symbol", ""),
                    "ltp": float(row.get("ltp", 0)),
                    "change_pct": round(float(row.get("perChange", row.get("net_price", 0))), 2),
                })
        return result
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
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
    }
    days = day_map.get(period, 90)
    expiry_dt = datetime.now() + timedelta(days=days)
    return expiry_dt.strftime("%d-%b-%Y")


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


# ========================
# SWING TRADING DETECTION
# ========================
def detect_swing_signals(symbol, period="6mo"):
    """
    Detect swing trading signals for a stock.
    Returns a dict with signal type, strength, and details.
    """
    try:
        fetch_period = get_fetch_period(period)
        data = yf.download(symbol.upper(), period=fetch_period, progress=False)

        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Technical indicators
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
        close = latest["Close"].item()
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
        score = 0  # Bullish positive, Bearish negative

        # Trend context: determine if price is clearly above or below key MAs
        ma20_dist_pct = ((close - ma20) / ma20) * 100 if ma20 > 0 else 0
        ma50_dist_pct = ((close - ma50) / ma50) * 100 if ma50 > 0 else 0
        in_downtrend = close < ma20 and close < ma50
        in_uptrend = close > ma20 and close > ma50

        # Recent price momentum (5-bar return)
        if len(data) >= 6:
            recent_close_5 = data.iloc[-6]["Close"].item()
            recent_return_pct = ((close - recent_close_5) / recent_close_5) * 100
        else:
            recent_return_pct = 0
        price_falling = recent_return_pct < -2

        # 0. Price-to-MA distance — catch stocks far from moving averages
        if ma20_dist_pct < -5 and ma50_dist_pct < -5:
            signals.append(f"🔴 PRICE BREAKDOWN: {ma20_dist_pct:.1f}% below MA20, {ma50_dist_pct:.1f}% below MA50 — Strong bearish")
            score -= 3
        elif ma20_dist_pct < -3:
            signals.append(f"🔴 PRICE WEAK: {ma20_dist_pct:.1f}% below MA20 — Bearish pressure")
            score -= 2
        elif ma20_dist_pct > 5 and ma50_dist_pct > 5:
            signals.append(f"🟢 PRICE BREAKOUT: +{ma20_dist_pct:.1f}% above MA20, +{ma50_dist_pct:.1f}% above MA50 — Strong bullish")
            score += 3
        elif ma20_dist_pct > 3:
            signals.append(f"🟢 PRICE STRONG: +{ma20_dist_pct:.1f}% above MA20 — Bullish momentum")
            score += 2

        # 1. MA Crossover (Golden Cross / Death Cross)
        if prev_ma20 <= prev_ma50 and ma20 > ma50:
            signals.append("🟢 GOLDEN CROSS: MA20 crossed above MA50 (Strong Bullish)")
            score += 3
        elif prev_ma20 >= prev_ma50 and ma20 < ma50:
            signals.append("🔴 DEATH CROSS: MA20 crossed below MA50 (Strong Bearish)")
            score -= 3

        # 2. RSI Reversal Zones (trend-aware)
        if rsi_val < 30:
            if in_downtrend:
                signals.append(f"🔴 RSI OVERSOLD IN DOWNTREND: RSI at {rsi_val:.1f} — Confirms bearish momentum")
                score -= 1
            else:
                signals.append(f"🟢 RSI OVERSOLD: RSI at {rsi_val:.1f} — Potential bounce incoming")
                score += 2
        elif rsi_val > 70:
            if in_uptrend:
                signals.append(f"🟡 RSI OVERBOUGHT IN UPTREND: RSI at {rsi_val:.1f} — Strong momentum, watch for pullback")
            else:
                signals.append(f"🔴 RSI OVERBOUGHT: RSI at {rsi_val:.1f} — Potential pullback")
                score -= 2
        elif 30 <= rsi_val <= 40:
            if in_downtrend:
                signals.append(f"🟡 RSI NEAR OVERSOLD: RSI at {rsi_val:.1f} — Weak momentum, not yet a reversal")
            else:
                signals.append(f"🟡 RSI NEAR OVERSOLD: RSI at {rsi_val:.1f} — Watch for reversal")
                score += 1
        elif 60 <= rsi_val <= 70:
            if in_uptrend:
                signals.append(f"🟡 RSI NEAR OVERBOUGHT: RSI at {rsi_val:.1f} — Strong momentum, watch for pullback")
            else:
                signals.append(f"🟡 RSI NEAR OVERBOUGHT: RSI at {rsi_val:.1f} — Watch for pullback")
                score -= 1

        # 3. MACD Crossover
        if prev_macd_hist <= 0 and macd_hist > 0:
            signals.append("🟢 MACD BULLISH CROSSOVER: MACD crossed above signal line")
            score += 2
        elif prev_macd_hist >= 0 and macd_hist < 0:
            signals.append("🔴 MACD BEARISH CROSSOVER: MACD crossed below signal line")
            score -= 2

        # 4. Bollinger Band Breakout (trend-aware)
        if close <= bb_lower:
            if in_downtrend:
                signals.append("🔴 BB LOWER BREAK: Price below lower Bollinger Band in downtrend — Band walking")
                score -= 1
            else:
                signals.append("🟢 BB LOWER TOUCH: Price at lower Bollinger Band — Potential reversal up")
                score += 2
        elif close >= bb_upper:
            if in_uptrend:
                signals.append("🟢 BB UPPER BREAK: Price above upper Bollinger Band in uptrend — Band walking")
                score += 1
            else:
                signals.append("🔴 BB UPPER TOUCH: Price at upper Bollinger Band — Potential reversal down")
                score -= 2

        # 5. Volume Spike
        if vol_avg > 0 and volume > vol_avg * 1.5:
            vol_ratio = volume / vol_avg
            signals.append(f"⚡ VOLUME SPIKE: {vol_ratio:.1f}x average volume — Strong momentum")
            if score > 0:
                score += 1
            elif score < 0:
                score -= 1

        # 6. Price above/below key MAs
        if close > ma20 > ma50:
            signals.append("🟢 TREND: Price > MA20 > MA50 — Bullish alignment")
            score += 1
        elif close < ma20 < ma50:
            signals.append("🔴 TREND: Price < MA20 < MA50 — Bearish alignment")
            score -= 1
        elif in_downtrend:
            signals.append("🔴 TREND: Price below both MA20 & MA50 — Bearish")
            score -= 1
        elif in_uptrend:
            signals.append("🟢 TREND: Price above both MA20 & MA50 — Bullish")
            score += 1

        # ---- NSE INDIA ENHANCED SIGNALS ----
        nse_data = {}  # Store NSE data for UI display
        market_score = 0  # Separate market-wide score (capped to +/-1)

        # 7. Delivery Volume Analysis (stock-specific, uses price direction)
        symbol_clean = symbol.upper().replace(".NS", "")
        delivery = get_delivery_data(symbol_clean)
        if delivery:
            nse_data["delivery"] = delivery
            dlv = delivery["latest_delivery_pct"]
            avg_dlv = delivery["avg_delivery_pct"]
            if dlv > 60:
                if price_falling:
                    signals.append(f"🔴 HIGH DELIVERY SELLOFF: {dlv}% delivery (avg {avg_dlv}%) — Strong selling conviction")
                    score -= 2
                else:
                    signals.append(f"🟢 HIGH DELIVERY: {dlv}% delivery (avg {avg_dlv}%) — Strong buying conviction")
                    score += 2
            elif dlv > avg_dlv * 1.2:
                if price_falling:
                    signals.append(f"🔴 DELIVERY SURGE: {dlv}% vs avg {avg_dlv}% — Above-normal selling conviction")
                    score -= 1
                else:
                    signals.append(f"🟢 DELIVERY SURGE: {dlv}% vs avg {avg_dlv}% — Above-normal conviction")
                    score += 1
            elif dlv < 30:
                signals.append(f"🟡 LOW DELIVERY: {dlv}% — Speculative activity, weak conviction")

        # 8. India VIX (Fear gauge) — market-wide
        vix_data = get_india_vix()
        if vix_data:
            nse_data["vix"] = vix_data
            vix_val = vix_data["vix_close"]
            vix_chg = vix_data["vix_change_pct"]
            if vix_val > 25:
                signals.append(f"🔴 HIGH VIX: {vix_val} — Market fear elevated, caution on longs")
                market_score -= 1
            elif vix_val < 13:
                signals.append(f"🟢 LOW VIX: {vix_val} — Market complacent, favorable for longs")
                market_score += 1
            if vix_chg > 15:
                signals.append(f"🔴 VIX SPIKE: +{vix_chg}% — Sudden fear, expect volatility")
            elif vix_chg < -15:
                signals.append(f"🟢 VIX CRASH: {vix_chg}% — Fear receding")

        # 9. Market Breadth (Advance/Decline) — market-wide
        breadth = get_market_breadth()
        if breadth:
            nse_data["breadth"] = breadth
            ad_ratio = breadth["ad_ratio"]
            if ad_ratio > 2:
                signals.append(f"🟢 STRONG BREADTH: A/D ratio {ad_ratio} ({breadth['advances']}A/{breadth['declines']}D) — Broad rally")
                market_score += 1
            elif ad_ratio < 0.5:
                signals.append(f"🔴 WEAK BREADTH: A/D ratio {ad_ratio} ({breadth['advances']}A/{breadth['declines']}D) — Broad selloff")
                market_score -= 1

        # 10. FII/DII Participant OI Positioning — market-wide
        oi_data = get_participant_oi()
        if oi_data:
            nse_data["participant_oi"] = oi_data
            fii = oi_data.get("FII")
            if fii:
                fii_net = fii["net"]
                if fii_net > 0:
                    signals.append(f"🟢 FII NET LONG: {fii_net:,} contracts — Institutional bullish")
                    market_score += 1
                elif fii_net < 0:
                    signals.append(f"🔴 FII NET SHORT: {fii_net:,} contracts — Institutional bearish")
                    market_score -= 1

        # Cap market-wide contribution to +/-1 so it doesn't overpower stock technicals
        capped_market = max(-1, min(1, market_score))
        tech_score = score  # Save raw technical score
        score = score + capped_market

        # Determine overall swing signal
        if score >= 4:
            swing_type = "STRONG BUY"
            color = "green"
        elif score >= 2:
            swing_type = "BUY"
            color = "lightgreen"
        elif score <= -4:
            swing_type = "STRONG SELL"
            color = "red"
        elif score <= -2:
            swing_type = "SELL"
            color = "salmon"
        else:
            swing_type = "NEUTRAL"
            color = "gray"

        # Calculate swing targets using ATR
        swing_target_up = round(close + (atr * 2), 2)
        swing_target_down = round(close - (atr * 2), 2)
        stop_loss_long = round(close - atr, 2)
        stop_loss_short = round(close + atr, 2)
        nse_expiry = get_nse_nearest_option_expiry(symbol_clean)
        expiry_date = nse_expiry if nse_expiry else get_swing_expiry_date(period)
        expiry_source = "NSE Option Chain" if nse_expiry else "Estimated"

        return {
            "stock": symbol.upper(),
            "close": round(close, 2),
            "swing_type": swing_type,
            "color": color,
            "score": score,
            "signals": signals,
            "rsi": round(rsi_val, 2),
            "macd_hist": round(macd_hist, 4),
            "atr": round(atr, 2),
            "volume_ratio": round(volume / vol_avg, 2) if vol_avg > 0 else 0,
            "swing_target_up": swing_target_up,
            "swing_target_down": swing_target_down,
            "stop_loss_long": stop_loss_long,
            "stop_loss_short": stop_loss_short,
            "expiry_date": expiry_date,
            "expiry_source": expiry_source,
            "ma20": round(ma20, 2),
            "ma50": round(ma50, 2),
            "nse_data": nse_data,
            "tech_score": tech_score,
            "market_score": capped_market,
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
    try:
        data = yf.download(symbol, period=period, progress=False, threads=False)

        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        return data

    except Exception:
        return None
def analyze_stock_data(symbol, period="3mo"):
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
        current_price = float(latest["Close"])

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
        features_df = pd.DataFrame([{
            "MA20":         ma20_val,
            "MA50":         ma50_val,
            "RSI":          rsi_val,
            "Volume":       vol_cur,
            "Volume_Avg20": vol_avg20,
            "Return":       float(latest["Return"])
        }])
        prediction = model.predict(features_df.values)[0]
        prob = float(model.predict_proba(features_df.values)[0].max())
        direction = "UP" if prediction == 1 else "DOWN"
        pro_score += 2 if direction == "UP" else -2
        if prob >= 0.7: pro_score += 1 if direction == "UP" else -1

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
            elif vol_ratio > 1.2:
                pro_score += 1 if direction == "UP" else -1

        # 7. OBV trend
        if obv_rising and direction == "UP":
            pro_score += 1
        elif not obv_rising and direction == "DOWN":
            pro_score -= 1

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

        return {
            "stock": symbol.upper(),
            "current_price": round(current_price, 2),
            "entry_price": entry_price,
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
            "rsi_value": round(rsi_val, 2)
        }

    except Exception as e:
        return {"error": str(e)}
# ========================
# SIDEBAR MENU
# ========================
with st.sidebar:
    st.header("⚙️ Menu")

    page = st.radio(
        "Select Page",
        ["Single Stock Analysis", "Swing Trading Alerts", "Market Scanner", "Market Summary", "Email Alerts", "Train Model"]
    )

    st.divider()

    st.subheader("Model Status")
    if model is not None:
        st.success("✅ Model Loaded & Ready")
    else:
        st.error("❌ Model Not Loaded")

# ========================
# PAGE: SINGLE STOCK ANALYSIS
# ========================
if page == "Single Stock Analysis":
    st.header("📊 Single Stock Analysis")

    col0, col1, col2, col3 = st.columns([1.2, 2, 2, 1])

    with col0:
        exchange = st.selectbox(
            "Exchange",
            ["NSE", "BSE"],
            index=0
        )

    is_bse = exchange == "BSE"

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
                placeholder="e.g., 532540.BO, 500325.BO",
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
        symbol = custom.strip().upper()
    elif is_bse and picked != "(type custom)":
        symbol = BSE_STOCKS[picked]
    elif not is_bse and picked != "(type custom)":
        symbol = picked
    else:
        symbol = "532540.BO" if is_bse else "TCS.NS"

    with col3:
        period = st.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=1
        )

    if st.button("Analyze Stock", key="analyze_btn"):
        with st.spinner(f"Analyzing {symbol}..."):
            result = analyze_stock_data(symbol, period)

        if "error" not in result:
            # ── Top metric row ─────────────────────────────────────────
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            c1.metric("Current Rate",   f"₹{result['current_price']}")
            c2.metric("Entry Price",    f"₹{result['entry_price']}")
            c3.metric("ATR",            f"₹{result['atr']}")
            c4.metric("Prediction",     result["prediction"])
            c5.metric("Confidence",     f"{result['confidence_percent']}%")
            c6.metric("Pro Score",      f"{result['pro_score']:+d} / 10")

            rec = result["recommendation"]
            if "STRONG BUY" in rec:
                c7.success(f"🟢 {rec}")
            elif "BUY" in rec:
                c7.success(f"🟩 {rec}")
            elif "STRONG SELL" in rec:
                c7.error(f"🔴 {rec}")
            elif "SELL" in rec:
                c7.error(f"🟥 {rec}")
            else:
                c7.warning(f"🟡 {rec}")

            st.divider()
            st.info(f"🤖 {result['ai_explanation']}")

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

        else:
            st.error(f"Error: {result['error']}")

# ========================
# PAGE: SWING TRADING ALERTS
# ========================
elif page == "Swing Trading Alerts":
    st.header("🔔 Swing Trading Alerts")
    st.write("Scans stocks using MA crossovers, RSI, MACD, Bollinger Bands, volume spikes + **NSE delivery data, India VIX, market breadth & FII/DII OI**.")

    st.divider()

    col_a, col_b = st.columns([2, 1])
    with col_a:
        filter_type = st.multiselect(
            "Filter Signals",
            ["STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"],
            default=["STRONG BUY", "BUY", "SELL", "STRONG SELL"]
        )
    with col_b:
        scan_period = st.selectbox("Scan Period", ["1wk", "1mo", "2mo", "3mo", "6mo", "1y"], index=3)

    if st.button("🔍 Scan for Swing Trades", type="primary"):
        swing_results = []
        progress = st.progress(0)

        for idx, symbol in enumerate(SCANNER_STOCKS):
            result = detect_swing_signals(symbol, period=scan_period)
            if result and result["swing_type"] in filter_type:
                swing_results.append(result)
            progress.progress((idx + 1) / len(SCANNER_STOCKS))

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
            st.success(f"Found **{len(swing_results)}** swing trading signal(s)")

            for res in swing_results:
                # Signal card
                with st.container(border=True):
                    header_col, badge_col = st.columns([3, 1])

                    with header_col:
                        st.subheader(f"{res['stock']}")
                        st.caption(f"Close: ₹{res['close']}  |  ATR: {res['atr']}  |  Volume: {res['volume_ratio']}x avg")

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
                    m1.metric("RSI", res["rsi"])
                    m2.metric("MA20", f"₹{res['ma20']}")
                    m3.metric("MA50", f"₹{res['ma50']}")
                    m4.metric("Tech Score", f"{res.get('tech_score', res['score']):+d}")
                    m5.metric("Total Score", f"{res['score']:+d}", f"Mkt: {res.get('market_score', 0):+d}")

                    # Signals list
                    st.markdown("**Detected Signals:**")
                    for sig in res["signals"]:
                        st.write(f"  {sig}")

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
                            nc4.metric("FII Net OI", f"{fii_oi['net']:,}", "Long" if fii_oi["net"] > 0 else "Short")

                    # Trade plan
                    st.markdown("**📋 Swing Trade Plan:**")
                    tp1, tp2, tp3, tp4 = st.columns(4)
                    if "BUY" in res["swing_type"]:
                        tp1.metric("Entry Price", f"₹{res['close']}")
                        tp2.metric("Target (Up)", f"₹{res['swing_target_up']}")
                        tp3.metric("Stop Loss", f"₹{res['stop_loss_long']}")
                        tp4.metric("Expiry Date", res.get("expiry_date", "N/A"))
                    elif "SELL" in res["swing_type"]:
                        tp1.metric("Entry Price", f"₹{res['close']}")
                        tp2.metric("Target (Down)", f"₹{res['swing_target_down']}")
                        tp3.metric("Stop Loss", f"₹{res['stop_loss_short']}")
                        tp4.metric("Expiry Date", res.get("expiry_date", "N/A"))
                    else:
                        tp1.metric("Target (Up)", f"₹{res['swing_target_up']}")
                        tp2.metric("Target (Down)", f"₹{res['swing_target_down']}")
                        tp3.metric("Expiry Date", res.get("expiry_date", "N/A"))
                        tp4.metric("Signal", res["swing_type"])

                    st.caption(f"Expiry Source: {res.get('expiry_source', 'Estimated')}")

# ========================
# PAGE: MARKET SCANNER
# ========================
elif page == "Market Scanner":
    st.header("🔍 Market Scanner")
    st.write("Scanning all configured stocks for trading opportunities...")

    if st.button("Scan Market Now"):
        with st.spinner("Scanning market..."):
            results = []
            progress_bar = st.progress(0)

            for idx, symbol in enumerate(SCANNER_STOCKS):
                analysis = analyze_stock_data(symbol)
                if "error" not in analysis:
                    results.append(analysis)
                progress_bar.progress((idx + 1) / len(SCANNER_STOCKS))

            st.success(f"✅ Scanned {len(results)} stocks")
            st.divider()

            recommendation_filter = st.multiselect(
                "Filter by Recommendation:",
                ["BUY", "SELL", "HOLD"],
                default=["BUY", "SELL", "HOLD"]
            )

            filtered_results = [r for r in results if r["recommendation"] in recommendation_filter]

            st.subheader("📊 Scanner Results")

            for stock in filtered_results:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(stock["stock"], stock["prediction"])
                with col2:
                    st.metric("Confidence", f"{stock['confidence_percent']}%")
                with col3:
                    rec_color = "🟢" if stock["recommendation"] == "BUY" else "🔴" if stock["recommendation"] == "SELL" else "🟡"
                    st.metric("Recommendation", f"{rec_color} {stock['recommendation']}")
                with col4:
                    st.metric("RSI", stock["rsi_value"])

                st.divider()

# ========================
# PAGE: MARKET SUMMARY
# ========================
elif page == "Market Summary":
    st.header("📈 Market Summary")

    # Live NSE Dashboard
    st.subheader("📡 Live NSE Data")
    dash1, dash2, dash3 = st.columns(3)

    breadth_data = get_market_breadth()
    if breadth_data:
        with dash1:
            st.metric("NIFTY 50", f"₹{breadth_data['nifty_last']:,.0f}", f"{breadth_data['nifty_change_pct']:+.2f}%")
            st.metric("Advance/Decline", breadth_data["ad_ratio"], f"{breadth_data['advances']}A / {breadth_data['declines']}D / {breadth_data['unchanged']}U")

    vix_data = get_india_vix()
    if vix_data:
        with dash2:
            vix_delta_color = "inverse"  # Higher VIX = bad
            st.metric("India VIX", vix_data["vix_close"], f"{vix_data['vix_change_pct']:+.1f}%", delta_color=vix_delta_color)
            st.metric("VIX Range", f"{vix_data['vix_low']} - {vix_data['vix_high']}")

    oi_data = get_participant_oi()
    if oi_data:
        with dash3:
            fii_oi = oi_data.get("FII")
            if fii_oi:
                st.metric("FII Net OI", f"{fii_oi['net']:,}", "Long" if fii_oi["net"] > 0 else "Short")
            dii_oi = oi_data.get("DII")
            if dii_oi:
                st.metric("DII Net OI", f"{dii_oi['net']:,}", "Long" if dii_oi["net"] > 0 else "Short")

    # Participant OI Breakdown
    if oi_data:
        st.divider()
        st.subheader("🏦 Participant-wise Open Interest")
        oi_rows = []
        for ct in ["FII", "DII", "Client", "Pro"]:
            d = oi_data.get(ct)
            if d:
                oi_rows.append({"Participant": ct, "Long": f"{d['long']:,}", "Short": f"{d['short']:,}", "Net": f"{d['net']:,}"})
        if oi_rows:
            st.table(pd.DataFrame(oi_rows))

    st.divider()

    # Real-time top movers from NSE
    movers = get_nse_top_movers()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 Top Gainers (NSE Live)")
        if movers and movers["gainers"]:
            for g in movers["gainers"]:
                st.write(f"• **{g['symbol']}** — ₹{g['ltp']}  ({g['change_pct']:+.2f}%)")
        else:
            for symbol in GAINERS:
                st.write(f"• {symbol}")

    with col2:
        st.subheader("🔴 Top Losers (NSE Live)")
        if movers and movers["losers"]:
            for l in movers["losers"]:
                st.write(f"• **{l['symbol']}** — ₹{l['ltp']}  ({l['change_pct']:+.2f}%)")
        else:
            for symbol in LOSERS:
                st.write(f"• {symbol}")

# ========================
# PAGE: TRAIN MODEL
# ========================
elif page == "Train Model":
    st.header("🧠 Train ML Model")

    st.info(
        "This will download 5 years of historical data for all stocks "
        "and train a RandomForest model. This may take several minutes."
    )

    if st.button("Start Training", type="primary"):
        with st.spinner("Training model..."):
            try:
                new_model = train_model_func()
                load_model.clear()
                st.success("✅ Model trained and saved successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# ========================
# PAGE: STOCK ALERTS (Telegram + Email)
# ========================


elif page == "Email Alerts":
    st.header("🔔 Stock Alerts")
    st.write("Scan all stocks and push **high-conviction signals** via **Telegram** or **Email**.")

    st.divider()

    notify_method = st.radio("Notification Method", ["📱 Telegram (Recommended)", "📧 Gmail"], horizontal=True)

    if "Telegram" in notify_method:
        st.subheader("📱 Telegram Setup")
        st.caption(
            "**One-time setup:**\n"
            "1. Open Telegram → search **@BotFather** → send `/newbot` → copy the **Bot Token** into `.env`\n"
            "2. Open your new bot and send it any message (e.g. `hello`)\n"
            "3. The system will auto-detect your Chat ID"
        )
        if TELEGRAM_BOT_TOKEN:
            st.success("✅ Bot Token loaded from .env")
        else:
            st.error("❌ TELEGRAM_BOT_TOKEN not found in .env")
    else:
        st.subheader("📧 Gmail OAuth Setup")
        st.caption(
            "Uses **OAuth 2.0** with refresh token. Configure `GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`, "
            "`GMAIL_REFRESH_TOKEN`, and `GMAIL_SENDER` in your `.env` file."
        )
        recipient_email = st.text_input("Send alerts to (email)", value="", placeholder="recipient@gmail.com", key="recipient_email")
        if GMAIL_REFRESH_TOKEN and GMAIL_REFRESH_TOKEN != "YOUR_REFRESH_TOKEN":
            st.success("✅ Gmail OAuth credentials loaded from .env")
        else:
            st.warning("⚠️ Set GMAIL_REFRESH_TOKEN in .env (run refresh.py to get it)")

    st.divider()

    ecol3, ecol4 = st.columns(2)
    with ecol3:
        alert_threshold = st.slider("Score threshold (absolute)", min_value=4, max_value=10, value=6,
                                     help="Alert when |score| ≥ this value")
    with ecol4:
        alert_period = st.selectbox("Scan Period", ["1wk", "1mo", "3mo", "6mo"], index=2, key="alert_period")

    include_swing = st.checkbox("Include Swing Trading signals", value=True)

    if st.button("🚀 Scan & Send Alert", type="primary"):
        # Validate inputs
        can_send = True
        if "Telegram" in notify_method:
            if not TELEGRAM_BOT_TOKEN:
                st.error("❌ TELEGRAM_BOT_TOKEN not found in .env file.")
                can_send = False
        else:
            if not GMAIL_REFRESH_TOKEN or GMAIL_REFRESH_TOKEN == "YOUR_REFRESH_TOKEN":
                st.error("❌ Gmail OAuth not configured. Set GMAIL_REFRESH_TOKEN in .env.")
                can_send = False
            elif not recipient_email:
                st.error("Please enter a recipient email address.")
                can_send = False

        if model is None:
            st.error("ML model not trained. Train the model first.")
            can_send = False

        if can_send:
            progress = st.progress(0)
            status = st.empty()

            strong_stocks = []
            swing_picks = []

            total = len(SCANNER_STOCKS)
            for idx, sym in enumerate(SCANNER_STOCKS):
                status.text(f"Analyzing {sym} ({idx+1}/{total})...")

                res = analyze_stock_data(sym, alert_period)
                if "error" not in res and abs(res.get("pro_score", 0)) >= alert_threshold:
                    strong_stocks.append(res)

                if include_swing:
                    sw = detect_swing_signals(sym, period=alert_period)
                    if sw and abs(sw.get("score", 0)) >= alert_threshold:
                        swing_picks.append(sw)

                progress.progress((idx + 1) / total)

            status.text("Scan complete. Building alert...")

            if not strong_stocks and not swing_picks:
                st.warning(f"No stocks found with |score| ≥ {alert_threshold}. Try a lower threshold.")
            else:
                now_str = datetime.now().strftime("%d-%b-%Y %I:%M %p")

                # ── Build Telegram message ──
                if "Telegram" in notify_method:
                    lines = []
                    lines.append(f"📈 *Stock Alert — {now_str}*")
                    lines.append(f"Threshold: |score| ≥ {alert_threshold}  |  Period: {alert_period}\n")

                    if strong_stocks:
                        strong_stocks.sort(key=lambda x: x.get("pro_score", 0), reverse=True)
                        lines.append("🎯 *High\\-Conviction Picks:*")
                        for s in strong_stocks:
                            emoji = "🟢" if s["pro_score"] > 0 else "🔴"
                            lines.append(
                                f"{emoji} *{s['stock'].replace('.', '\\.')}*  ₹{s['current_price']}  "
                                f"Score: *{s['pro_score']:+d}/10*  {s['recommendation']}  "
                                f"RSI: {s['rsi_value']}  Entry: ₹{s['entry_price']}"
                            )
                        lines.append("")

                    if swing_picks:
                        swing_picks.sort(key=lambda x: abs(x.get("score", 0)), reverse=True)
                        lines.append("🔔 *Swing Trading Setups:*")
                        for sw in swing_picks:
                            emoji = "🟢" if sw["score"] > 0 else "🔴"
                            sl = sw["stop_loss_long"] if "BUY" in sw["swing_type"] else sw["stop_loss_short"]
                            target = sw["swing_target_up"] if "BUY" in sw["swing_type"] else sw["swing_target_down"]
                            lines.append(
                                f"{emoji} *{sw['stock'].replace('.', '\\.')}*  ₹{sw['close']}  "
                                f"Score: *{sw['score']:+d}*  {sw['swing_type']}  "
                                f"RSI: {sw['rsi']}  Target: ₹{target}  SL: ₹{sl}"
                            )
                        lines.append("")

                    lines.append("_Sent by AI Stock Analyzer ❤️_")
                    tg_message = "\n".join(lines)

                    # Send via Telegram Bot API (auto chat ID)
                    chat_id = get_telegram_chat_id(TELEGRAM_BOT_TOKEN)

                    if not chat_id:
                        st.error("❌ No chat ID found. Send a message to your Telegram bot first (e.g. 'hello').")
                    else:
                        try:
                            tg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                            tg_payload = {
                                "chat_id": chat_id,
                                "text": tg_message,
                                "parse_mode": "Markdown",
                            }
                            tg_resp = requests.post(tg_url, json=tg_payload, timeout=15)
                            tg_result = tg_resp.json()

                            if tg_result.get("ok"):
                                st.success(f"✅ Telegram alert sent! ({len(strong_stocks)} picks + {len(swing_picks)} swing)")
                                st.toast("📱 Telegram notification sent!", icon="✅")
                            else:
                                err_desc = tg_result.get("description", "Unknown error")
                                st.error(f"❌ Telegram API error: {err_desc}")
                        except requests.exceptions.ConnectionError:
                            st.error("❌ Could not connect to Telegram. Check your internet connection.")
                        except Exception as e:
                            st.error(f"❌ Telegram send failed: {e}")

                    # Show preview
                    with st.expander("📬 Message Preview", expanded=True):
                        preview_text = tg_message.replace("\\.", ".").replace("\\-", "-").replace("*", "**")
                        st.markdown(preview_text)

                # ── Build & send Gmail ──
                else:
                    html_parts = []
                    html_parts.append(f"<h2>📈 Stock Alert — {now_str}</h2>")
                    html_parts.append(f"<p>Threshold: |score| ≥ {alert_threshold} &nbsp;|&nbsp; Period: {alert_period}</p>")

                    if strong_stocks:
                        strong_stocks.sort(key=lambda x: x.get("pro_score", 0), reverse=True)
                        html_parts.append("<h3>🎯 High-Conviction Analysis Picks</h3>")
                        html_parts.append('<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-family:Arial;">')
                        html_parts.append('<tr style="background:#222;color:#fff;"><th>Stock</th><th>Price</th><th>Pro Score</th><th>Recommendation</th><th>RSI</th><th>Entry</th><th>Confidence</th></tr>')
                        for s in strong_stocks:
                            color = "#27ae60" if s["pro_score"] > 0 else "#e74c3c"
                            html_parts.append(
                                f'<tr><td><b>{s["stock"]}</b></td>'
                                f'<td>₹{s["current_price"]}</td>'
                                f'<td style="color:{color};font-weight:bold;">{s["pro_score"]:+d}/10</td>'
                                f'<td>{s["recommendation"]}</td>'
                                f'<td>{s["rsi_value"]}</td>'
                                f'<td>₹{s["entry_price"]}</td>'
                                f'<td>{s["confidence_percent"]}%</td></tr>'
                            )
                        html_parts.append('</table>')

                    if swing_picks:
                        swing_picks.sort(key=lambda x: abs(x.get("score", 0)), reverse=True)
                        html_parts.append("<h3>🔔 Swing Trading Setups</h3>")
                        html_parts.append('<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-family:Arial;">')
                        html_parts.append('<tr style="background:#222;color:#fff;"><th>Stock</th><th>Price</th><th>Score</th><th>Signal</th><th>RSI</th><th>Target Up</th><th>Target Down</th><th>Stop Loss</th><th>Expiry</th></tr>')
                        for sw in swing_picks:
                            color = "#27ae60" if sw["score"] > 0 else "#e74c3c"
                            sl = sw["stop_loss_long"] if "BUY" in sw["swing_type"] else sw["stop_loss_short"]
                            html_parts.append(
                                f'<tr><td><b>{sw["stock"]}</b></td>'
                                f'<td>₹{sw["close"]}</td>'
                                f'<td style="color:{color};font-weight:bold;">{sw["score"]:+d}</td>'
                                f'<td>{sw["swing_type"]}</td>'
                                f'<td>{sw["rsi"]}</td>'
                                f'<td>₹{sw["swing_target_up"]}</td>'
                                f'<td>₹{sw["swing_target_down"]}</td>'
                                f'<td>₹{sl}</td>'
                                f'<td>{sw.get("expiry_date", "N/A")}</td></tr>'
                            )
                        html_parts.append('</table>')

                    html_parts.append("<br><p style='color:#888;font-size:12px;'>Sent by AI Stock Analyzer | Powered by S H A K T H I ❤️</p>")
                    html_body = "\n".join(html_parts)

                    try:
                        send_gmail_oauth(
                            recipient_email,
                            f"📈 Stock Alert: {len(strong_stocks)} analysis + {len(swing_picks)} swing picks ({now_str})",
                            html_body
                        )
                        st.success(f"✅ Email sent to **{recipient_email}**!")
                        st.toast(f"📧 Alert sent to {recipient_email}", icon="✅")
                        with st.expander("📬 Email Preview", expanded=True):
                            st.html(html_body)
                    except Exception as e:
                        st.error(f"❌ Failed to send email: {e}")


def get_gmail_service():
    creds = Credentials(
        None,
        refresh_token=GMAIL_REFRESH_TOKEN,
        client_id=GMAIL_CLIENT_ID,
        client_secret=GMAIL_CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token"
    )
    if not creds.valid:
        creds.refresh(Request())
    return build("gmail", "v1", credentials=creds)


def send_gmail_oauth(recipient_email, subject, html_body):

    creds = Credentials(
        None,
        refresh_token=GMAIL_REFRESH_TOKEN,
        client_id=GMAIL_CLIENT_ID,
        client_secret=GMAIL_CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token",
    )

    if not creds.valid:
        creds.refresh(Request())

    service = build("gmail", "v1", credentials=creds)

    message = MIMEText(html_body, "html")
    message["to"] = recipient_email
    message["from"] = GMAIL_SENDER
    message["subject"] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    service.users().messages().send(
        userId="me",
        body={"raw": raw}
    ).execute()
