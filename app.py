# nifty_options_bot_pro_ready.py
"""
Institutional-Grade NIFTY / BANKNIFTY Options Bot
- Paper & LIVE trading
- Auto ATM/OTM CE/PE selection
- Trailing SL / ATR target
- Candlestick patterns + SMA/RSI/MACD signals
- Dashboard always shows valid LTP & option preview
"""

import os
import json
import math
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, timedelta, time
import pytz
from streamlit_autorefresh import st_autorefresh

try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None

try:
    from scipy.stats import norm
except ImportError:
    norm = None
# ------------------- Token Save/Load -------------------
TOKEN_FILE = "token.json"

def save_token_file(token_data):
    try:
        with open(TOKEN_FILE, "w") as f:
            json.dump(token_data, f)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not save token: {e}")

def load_token_file():
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}
# Optional autorefresh (kept exactly as you had)
try:
    from streamlit_autorefresh import st_autorefresh
    count = st_autorefresh(interval=600_000, limit=None, key="news_refresh")  # refresh every 10 min
except Exception:
    pass

# App config
st.set_page_config(page_title="NSE Intraday Scanner — Merged Bot (Paper + Live)", layout="wide")

# Market session detection (India)
market_open = time(9, 15)
market_close = time(15, 30)
now = datetime.now(pytz.timezone("Asia/Kolkata")).time()

# Detect mode & market status
is_live = 'kite' in st.session_state and st.session_state.kite
is_market_open = market_open <= now <= market_close

if is_live:
    if is_market_open:
        st.success("✅ LIVE MODE — Market Open — Trading in real time")
    else:
        st.warning("✅ LIVE MODE — Market Closed — Orders will be AMO")
else:
    if is_market_open:
        st.info("📝 PAPER MODE — Market Open — Simulated orders")
    else:
        st.info("📝 PAPER MODE — Market Closed — Simulated AMO")


# ------------------- News & Global Trend -------------------
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Nifty Options Bot PRO", layout="wide")

# ------------------- Constants -------------------
UNDERLYINGS = {
    "NIFTY": {"ticker": "^NSEI", "nfo_prefix": "NIFTY", "lot": 75},
    "BANKNIFTY": {"ticker": "^NSEBANK", "nfo_prefix": "BANKNIFTY", "lot": 25}
}
ROUND_TO = {"NIFTY": 50, "BANKNIFTY": 100}

# ------------------- Indicators -------------------
def sma(series, window): return series.rolling(window).mean()
def rsi(series, period=14):
    delta=series.diff()
    up=delta.clip(lower=0)
    down=-1*delta.clip(upper=0)
    ma_up=up.ewm(com=period-1,adjust=False).mean()
    ma_down=down.ewm(com=period-1,adjust=False).mean()
    rs=ma_up/ma_down
    return 100-(100/(1+rs))
def macd(series,n_fast=12,n_slow=26,n_signal=9):
    ema_fast=series.ewm(span=n_fast,adjust=False).mean()
    ema_slow=series.ewm(span=n_slow,adjust=False).mean()
    macd_line=ema_fast-ema_slow
    signal_line=macd_line.ewm(span=n_signal,adjust=False).mean()
    hist=macd_line-signal_line
    return macd_line,signal_line,hist
def atr(df,period=14):
    df['H-L']=df['High']-df['Low']
    df['H-C']=abs(df['High']-df['Close'].shift())
    df['L-C']=abs(df['Low']-df['Close'].shift())
    tr=df[['H-L','H-C','L-C']].max(axis=1)
    return tr.rolling(period).mean().iloc[-1]
def supertrend(df, period=10, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    atr_val = atr(df, period)
    upperband = hl2 + multiplier*atr_val
    lowerband = hl2 - multiplier*atr_val
    trend = []
    for i in range(len(df)):
        if i==0: trend.append(True)
        else:
            if df['Close'].iloc[i] > upperband.iloc[i-1]: trend.append(True)
            elif df['Close'].iloc[i] < lowerband.iloc[i-1]: trend.append(False)
            else: trend.append(trend[-1])
    return trend


def atr_series(df, period=14):
    """Return ATR series instead of single value."""
    h_l = df['High'] - df['Low']
    h_pc = (df['High'] - df['Close'].shift()).abs()
    l_pc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    atr_s = tr.rolling(period).mean()
    return atr_s


def compute_supertrend_df(df, period=7, multiplier=2):
    """Compute Supertrend bands and direction on a DataFrame and return augmented DataFrame.

    Uses per-row ATR (series) so bands follow 5m volatility more tightly.
    """
    if df.empty or len(df) < period + 2:
        return df

    df = df.copy()
    hl2 = (df['High'] + df['Low']) / 2
    atr_s = atr_series(df, period)
    df['ATR'] = atr_s
    df['basic_ub'] = hl2 + multiplier * df['ATR']
    df['basic_lb'] = hl2 - multiplier * df['ATR']

    final_ub = [np.nan] * len(df)
    final_lb = [np.nan] * len(df)
    supertrend = [True] * len(df)

    for i in range(len(df)):
        if i == 0:
            final_ub[i] = df['basic_ub'].iloc[i]
            final_lb[i] = df['basic_lb'].iloc[i]
            supertrend[i] = True
            continue

        # Final upper band
        if df['basic_ub'].iloc[i] < final_ub[i-1] or df['Close'].iloc[i-1] > final_ub[i-1]:
            final_ub[i] = df['basic_ub'].iloc[i]
        else:
            final_ub[i] = final_ub[i-1]

        # Final lower band
        if df['basic_lb'].iloc[i] > final_lb[i-1] or df['Close'].iloc[i-1] < final_lb[i-1]:
            final_lb[i] = df['basic_lb'].iloc[i]
        else:
            final_lb[i] = final_lb[i-1]

        # Supertrend direction
        if df['Close'].iloc[i] > final_ub[i-1]:
            supertrend[i] = True
        elif df['Close'].iloc[i] < final_lb[i-1]:
            supertrend[i] = False
        else:
            supertrend[i] = supertrend[i-1]

    df['ST_upper'] = final_ub
    df['ST_lower'] = final_lb
    df['ST_dir'] = supertrend
    df['ST'] = np.where(df['ST_dir'], df['ST_lower'], df['ST_upper'])
    return df

def calculate_supertrend_levels(df, period=10, multiplier=3):
    """Calculate supertrend with exact levels for SL and target"""
    hl2 = (df['High'] + df['Low']) / 2
    atr_val = atr(df, period)
    upperband = hl2 + multiplier*atr_val
    lowerband = hl2 - multiplier*atr_val
    
    current_price = df['Close'].iloc[-1]
    current_high = df['High'].iloc[-1]
    current_low = df['Low'].iloc[-1]
    
    return {
        "current_price": current_price,
        "upper_band": upperband.iloc[-1],
        "lower_band": lowerband.iloc[-1],
        "atr_value": atr_val,
    }

def estimate_option_delta(spot_price, strike, days_to_expiry, is_call=True, volatility=0.25):
    """
    Estimate option delta using Black-Scholes approximation
    Delta represents how much option premium changes per 1 point move in spot
    """
    try:
        if norm is None:
            # Fallback if scipy not available
            moneyness = spot_price / strike
            if is_call:
                if moneyness > 1.05:
                    return 0.7
                elif moneyness < 0.95:
                    return 0.3
                else:
                    return 0.5
            else:
                if moneyness < 0.95:
                    return 0.7
                elif moneyness > 1.05:
                    return 0.3
                else:
                    return 0.5
        
        # Risk-free rate (approximate)
        r = 0.06
        
        # Calculate d1 from Black-Scholes
        sigma = volatility
        T = days_to_expiry / 365.0
        
        if T <= 0:
            return 1.0 if is_call else -1.0
        
        d1 = (np.log(spot_price / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        if is_call:
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1  # Put delta is negative
            delta = abs(delta)  # We'll use absolute value for moves
        
        return max(0.1, min(0.95, delta))  # Clamp between 0.1 and 0.95
    except:
        # Fallback to simple delta estimate based on moneyness
        moneyness = spot_price / strike
        if is_call:
            if moneyness > 1.05:
                return 0.7
            elif moneyness < 0.95:
                return 0.3
            else:
                return 0.5
        else:
            if moneyness < 0.95:
                return 0.7
            elif moneyness > 1.05:
                return 0.3
            else:
                return 0.5

def calculate_sl_target_from_indicators(df, spot_price, strike, entry_premium, days_to_expiry, is_call=True, chart_interval="5m"):
    """
    Calculate SL and Target for options accounting for delta and IV
    - Shows spot reference levels
    - Converts to option premium levels using delta
    """
    if df.empty or len(df) < 14:
        return None
    
    try:
        # For tighter and more responsive SL/Target use 5m indicators
        try:
            df5m = fetch_ohlc_safe(UNDERLYINGS[underlying_choice]["ticker"], period="7d", interval=chart_interval)
        except Exception:
            df5m = df.copy()

        # If fetch failed, fallback to provided df
        if df5m is None or df5m.empty:
            df5m = df

        st_df = compute_supertrend_df(df5m, period=7, multiplier=2)

        # Robust ATR: pick last non-null ATR from supertrend frame, else fallback
        atr_val = None
        if 'ATR' in st_df.columns:
            non_null_atr = st_df['ATR'].dropna()
            if len(non_null_atr) > 0:
                atr_val = float(non_null_atr.iloc[-1])

        if atr_val is None or pd.isna(atr_val):
            try:
                atr_val = float(atr(df5m, period=14))
            except Exception:
                atr_val = float(atr(df, period=14))

        # Calculate RSI on 5m where possible
        try:
            rsi14_series = rsi(df5m['Close'], 14).dropna()
            rsi14 = float(rsi14_series.iloc[-1]) if len(rsi14_series) > 0 else float(rsi(df['Close'], 14).iloc[-1])
        except Exception:
            rsi14 = float(rsi(df['Close'], 14).iloc[-1])

        # Recent 5m swings (safe)
        try:
            recent_high = float(df5m['High'].iloc[-30:].max())
        except Exception:
            recent_high = float(df['High'].iloc[-20:].max())
        try:
            recent_low = float(df5m['Low'].iloc[-30:].min())
        except Exception:
            recent_low = float(df['Low'].iloc[-20:].min())

        # Calculate option delta (safe fallback)
        delta = estimate_option_delta(spot_price, strike, days_to_expiry, is_call=is_call)

        # Use tighter ATR multiplier for target to avoid overly large targets
        atr_multiplier_target = 1.2
        atr_multiplier_sl_buffer = 0.5

        # Supertrend value fallback
        st_val = None
        if 'ST' in st_df.columns and not pd.isna(st_df['ST'].iloc[-1]):
            st_val = float(st_df['ST'].iloc[-1])

        if st_val is None:
            if is_call:
                st_val = max(recent_low, spot_price - atr_val) if not pd.isna(atr_val) else recent_low
            else:
                st_val = min(recent_high, spot_price + atr_val) if not pd.isna(atr_val) else recent_high

        # Compute spot SL/target with safe numeric ops
        if is_call:
            spot_stop_loss = float(st_val) - (atr_multiplier_sl_buffer * float(atr_val))
            spot_target = float(spot_price) + (atr_multiplier_target * float(atr_val))
        else:
            spot_stop_loss = float(st_val) + (atr_multiplier_sl_buffer * float(atr_val))
            spot_target = float(spot_price) - (atr_multiplier_target * float(atr_val))

        # Ensure target is reasonable vs recent swing
        if is_call:
            spot_target = min(spot_target, recent_high)
        else:
            spot_target = max(spot_target, recent_low)

        spot_move_to_sl = abs(float(spot_price) - float(spot_stop_loss))
        spot_move_to_target = abs(float(spot_target) - float(spot_price))

        premium_move_to_sl = spot_move_to_sl * float(delta)
        premium_move_to_target = spot_move_to_target * float(delta)

        # If premium moves end up NaN or zero, fallback to a small buffer
        if pd.isna(premium_move_to_sl) or premium_move_to_sl == 0:
            premium_move_to_sl = max(0.5, 0.5 * float(delta))
        if pd.isna(premium_move_to_target) or premium_move_to_target == 0:
            premium_move_to_target = max(0.5, 0.8 * float(delta))

        option_stop_loss = round(float(entry_premium) - float(premium_move_to_sl), 2)
        option_target = round(float(entry_premium) + float(premium_move_to_target), 2)

        option_stop_loss = max(0.05, option_stop_loss)

        return {
            "spot_price": round(spot_price, 2),
            "spot_stop_loss": round(spot_stop_loss, 2),
            "spot_target": round(spot_target, 2),
            "spot_move_to_sl": round(spot_move_to_sl, 2),
            "spot_move_to_target": round(spot_move_to_target, 2),
            "option_entry": round(entry_premium, 2),
            "option_stop_loss": option_stop_loss,
            "option_target": option_target,
            "premium_move_to_sl": round(premium_move_to_sl, 2),
            "premium_move_to_target": round(premium_move_to_target, 2),
            "delta": round(delta, 3),
            "atr_value": round(float(atr_val), 2) if not pd.isna(atr_val) else None,
            "rsi": round(rsi14, 2),
            "supertrend_upper": round(st_df['ST_upper'].iloc[-1], 2) if 'ST_upper' in st_df.columns else None,
            "supertrend_lower": round(st_df['ST_lower'].iloc[-1], 2) if 'ST_lower' in st_df.columns else None,
        }
    except Exception as e:
        st.warning(f"Could not calculate indicators: {e}")
        return None

def adx(df, period=14):
    df['TR'] = df['High'] - df['Low']
    df['+DM'] = df['High'].diff()
    df['-DM'] = df['Low'].diff().abs()
    return df['TR'].rolling(period).mean().iloc[-1]

def vwap(df):
    return (df['Close']*df['Volume']).cumsum()/df['Volume'].cumsum()

# ------------------- OHLC Fetch -------------------
@st.cache_data(ttl=60)
def fetch_ohlc(ticker, period="30d", interval="15m"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        return df
    except:
        return pd.DataFrame()

def fetch_ohlc_safe(ticker, period="30d", interval="15m"):
    df = fetch_ohlc(ticker, period, interval)
    if df.empty:
        df = fetch_ohlc(ticker, period="60d", interval="1h")
    return df

def get_option_premium(kite, nfo_symbol, fallback=None):
    """
    Get the current premium (LTP) for an option symbol
    In live mode: get from Zerodha API
    In paper mode: estimate from spot price and delta
    """
    try:
        if kite:
            inst = f"NFO:{nfo_symbol}"
            ltp_data = kite.ltp(inst)
            return float(ltp_data[inst]["last_price"])
    except:
        pass
    
    # Fallback to provided value (from entry or session state)
    if fallback is not None:
        return fallback
    
    return 0.0

# ------------------- LTP -------------------
def get_underlying_ltp(kite, symbol, fallback_df=None):
    try:
        if kite:
            inst = f"NSE:{symbol}"
            ltp_data = kite.ltp(inst)
            return float(ltp_data[inst]["last_price"])
    except:
        pass
    if fallback_df is not None and not fallback_df.empty:
        return float(fallback_df["Close"].iloc[-1])
    return None

# ------------------- Option Helpers -------------------
def nearest_strike(price, step):
    return int(round(price / step) * step)

def build_nfo_symbol(underlying, expiry_date, strike, opt_type):
    typ = "CE" if opt_type.upper().startswith("C") else "PE"
    y = expiry_date.strftime("%d%b%Y").upper()
    return f"{underlying}{y}{strike}{typ}"

# ------------------- Global Trend & News -------------------
def get_global_trend():
    indices = {"S&P500":"^GSPC", "Dow":"^DJI", "Nasdaq":"^IXIC", "DAX":"^GDAXI", "Nikkei":"^N225"}
    bullish = bearish = 0
    for name, ticker in indices.items():
        try:
            df = yf.download(ticker, period="2d", interval="1d", progress=False)
            if len(df) < 2:
                continue
            if df["Close"].iloc[-1] > df["Close"].iloc[-2]:
                bullish += 1
            else:
                bearish += 1
        except:
            continue
    return "BULL" if bullish >= bearish else "BEAR"

def fetch_news_sentiment(query="SENSEX", max_news=5):
    try:
        api_key = st.secrets.get("NEWSAPI_KEY", "")
    except:
        api_key = os.environ.get("NEWSAPI_KEY", "")
    if not api_key:
        return 0, []
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&pageSize={max_news}&apiKey={api_key}"
    try:
        response = requests.get(url).json()
        articles = response.get("articles", [])
        if not articles:
            return 0, []
        scores = []
        out_articles = []
        for art in articles:
            title = art.get("title", "")
            score = analyzer.polarity_scores(title)["compound"]
            scores.append(score)
            out_articles.append({
                "title": title,
                "description": art.get("description", ""),
                "url": art.get("url", ""),
                "source": art.get("source", {}).get("name", ""),
                "publishedAt": art.get("publishedAt", ""),
                "score": score,
            })
        avg = sum(scores) / len(scores) if scores else 0
        return avg, out_articles
    except:
        return 0, []

# ------------------- Candlestick -------------------
def detect_candlestick_patterns(df):
    patterns = {}
    if df.empty or len(df) < 2:
        return patterns
    last, prev = df.iloc[-1], df.iloc[-2]
    if prev["Close"] < prev["Open"] and last["Close"] > last["Open"] and last["Close"] > prev["Open"] and last["Open"] < prev["Close"]:
        patterns["Bullish Engulfing"] = True
    if prev["Close"] > prev["Open"] and last["Close"] < last["Open"] and last["Open"] > prev["Close"] and last["Close"] < prev["Open"]:
        patterns["Bearish Engulfing"] = True
    return patterns

# ------------------- Signal -------------------
def underlying_signal(df):
    if df.empty or len(df) < 10:
        return "NEUTRAL", {}

    s20, s50 = sma(df["Close"], 20), sma(df["Close"], 50)
    macd_line, macd_sig, macd_hist = macd(df["Close"])
    rsi14 = rsi(df["Close"], 14)

    bull_tech = (s20.iloc[-2] <= s50.iloc[-2] and s20.iloc[-1] > s50.iloc[-1] and macd_hist.iloc[-1] > 0)
    bear_tech = (s20.iloc[-2] >= s50.iloc[-2] and s20.iloc[-1] < s50.iloc[-1] and macd_hist.iloc[-1] < 0)

    global_trend = get_global_trend()
    bull_global = global_trend == "BULL"
    bear_global = global_trend == "BEAR"

    news_score, news_articles = fetch_news_sentiment()
    bull_news = news_score > 0.1
    bear_news = news_score < -0.1

    bull_count = sum([bull_tech, bull_global, bull_news])
    bear_count = sum([bear_tech, bear_global, bear_news])

    if bull_count > bear_count:
        signal = "BULL"
    elif bear_count > bull_count:
        signal = "BEAR"
    else:
        signal = "NEUTRAL"

    meta = {
        "rsi": rsi14.iloc[-1],
        "macd_hist": macd_hist.iloc[-1],
        "news_score": news_score,
        "news_articles": news_articles,
        "global_trend": global_trend,
    }
    return signal, meta

# ------------------- Position Sizing -------------------
def size_qty_by_capital(capital, price, allocation_pct, underlying_choice):
    if price <= 0:
        return 0
    alloc = capital * allocation_pct
    qty = int(math.floor(alloc / price / UNDERLYINGS[underlying_choice]["lot"]))
    return max(qty, 1)

# ------------------- Orders -------------------
def place_real_order(kite, tradingsymbol, tx_type, qty, product="MIS"):
    try:
        return kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=tradingsymbol,
            transaction_type=tx_type,
            quantity=int(qty * UNDERLYINGS[underlying_choice]["lot"]),
            order_type=kite.ORDER_TYPE_MARKET,
            product=getattr(kite, f"PRODUCT_{product}", "MIS"),
        )
    except:
        return None

def place_sell_order(kite, tradingsymbol, qty, price_info=""):
    """Place a SELL order to close an open position"""
    try:
        if kite:
            order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NFO,
                tradingsymbol=tradingsymbol,
                transaction_type=kite.TRANSACTION_TYPE_SELL,
                quantity=int(qty * UNDERLYINGS.get(UNDERLYINGS.get(tradingsymbol.split()[0], {}).get("nfo_prefix", "NIFTY"), {"lot": 75}).get("lot", 75)),
                order_type=kite.ORDER_TYPE_MARKET,
                product="MIS",
            )
            st.success(f"✅ LIVE SELL order placed for {tradingsymbol} {price_info}")
            return order_id
        return None
    except Exception as e:
        st.error(f"❌ Error placing SELL order: {e}")
        return None

# ------------------- Trailing SL -------------------
def monitor_trailing_sl(kite):
    if "open_positions" not in st.session_state:
        return
    for sym, trade in list(st.session_state["open_positions"].items()):
        ltp = get_underlying_ltp(kite, sym)
        if ltp and (ltp <= trade.get("trail_sl", 0) or ltp >= trade.get("target", 0)):
            st.info(f"Closing {sym} at LTP={ltp:.2f}")
            st.session_state["open_positions"].pop(sym)


# ------------------- Streamlit UI -------------------
st.title("⚡ Nifty/BANKNIFTY Options J-Bot PRO")
st.caption("Paper-mode by default. Test carefully before enabling LIVE trading.")
with st.sidebar:
    st.header("⚙️ Trade Settings")
    sl_pct = st.slider("Stop Loss %", 2, 20, 10) / 100      # default 10%
    tgt_pct = st.slider("Target %", 2, 50, 20) / 100        # default 20%
    trail_pct = st.slider("Trailing SL %", 1, 10, 1) / 100  # default 1%

with st.sidebar:
    st.header("Settings")
    underlying_choice = st.selectbox("Underlying", list(UNDERLYINGS.keys()))
    option_type = st.selectbox("Option Type", ["CE (Call)", "PE (Put)"])
    allocation_pct = st.slider("Capital per trade (%)", 1, 100, 50) / 100.0
    starting_cap = st.number_input("Starting Capital (₹)", 20000, 2000000, 20000, 1000)
    st.session_state['starting_capital'] = starting_cap
    chart_interval = st.selectbox("Chart Interval", ["1m", "5m", "15m", "30m"], index=1, key="chart_interval")
    live_trading = st.checkbox("Enable LIVE trading", False)

st.markdown("---")
st.subheader("Kite API")

# --- Load API credentials: try secrets first, fallback to UI input ---
try:
    API_KEY = st.secrets.get("API_KEY", "")
    API_SECRET = st.secrets.get("API_SECRET", "")
except:
    API_KEY = ""
    API_SECRET = ""

# Allow UI input as fallback
with st.expander("🔑 API Credentials (J if not in secrets.toml)"):
    api_key_input = st.text_input("API Key", value=API_KEY, type="password", key="api_key_input")
    api_secret_input = st.text_input("API Secret", value=API_SECRET, type="password", key="api_secret_input")
    if api_key_input:
        API_KEY = api_key_input
    if api_secret_input:
        API_SECRET = api_secret_input

# Try to load saved token from file (if any)
token_data = load_token_file()
kite = None
access_token_to_use = token_data.get("access_token") if token_data.get("date") == date.today().isoformat() else None

# Auto-connect if valid token exists
if access_token_to_use:
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(access_token_to_use)
        st.session_state.kite = kite
        st.success("✅ Auto-connected using saved Access Token!")
    except Exception as e:
        st.warning(f"Still Saved token invalid/expired: {e}")
        access_token_to_use = None

# Request Token input (only if auto-login fails)
request_token = st.text_input("Request Token (only if needed)", type="password")

# Manual refresh if needed
if st.button("Create Access Token") or not access_token_to_use:
    if request_token:
        try:
            kite = KiteConnect(api_key=API_KEY)
            data = kite.generate_session(request_token, api_secret=API_SECRET)
            kite.set_access_token(data["access_token"])
            st.session_state.kite = kite

            # Save token to file
            save_token_file({
                "access_token": data["access_token"],
                "date": date.today().isoformat(),
            })

            st.success("✅ Zerodha connected! New token generated and saved.")
            st.info("⚠️ Token saved locally and auto-used until midnight.")
        except Exception as e:
            st.error(f"Login failed: {e}")
    else:
        st.warning("Please enter the Request Token to generate Access Token.")

# Status
if "kite" in st.session_state:
    st.success("Connected to Zerodha ✅")
else:
    st.warning("Not connected yet.")




# ------------------- Tabs (mobile safe) -------------------
try:
    tab1, tab2, tab3 = st.tabs(["📊 Signals", "📑 Order Preview", "📂 Positions"])

    # ------------------- Tab 1: Signals -------------------
    with tab1:
        # Use global sidebar chart interval selection
        interval = st.session_state.get("chart_interval", "5m")
        lookback = st.selectbox("History", ["7d", "14d", "30d"], index=1)

        with st.spinner("Fetching data..."):
            try:
                df = fetch_ohlc_safe(
                    UNDERLYINGS[underlying_choice]["ticker"],
                    period=lookback,
                    interval=interval,
                )
            except Exception as e:
                st.error(f"⚠️ Error fetching OHLC: {e}")
                df = pd.DataFrame()

        st.metric("Data rows", len(df))

        if not df.empty:
            with st.expander("Show last 8 rows"):
                st.dataframe(df.tail(8))

            try:
                signal, meta = underlying_signal(df)
                st.subheader(f"Signal: {signal}")
                # show meta except the full articles list (we'll render articles separately)
                meta_shallow = {k: v for k, v in meta.items() if k != 'news_articles'}
                st.json(meta_shallow)
                st.markdown(f"**Global Trend:** {meta.get('global_trend', 'N/A')}")

                # News Sentiment panel with collapsed articles (score shown in title)
                with st.expander(f"News Sentiment Score: {meta.get('news_score', 0):+.2f}"):
                    st.metric("Score", f"{meta.get('news_score', 0):.2f}")
                    articles = meta.get('news_articles', []) or []
                    if articles:
                        for art in articles:
                            title_short = (art.get('title') or '')[:120]
                            with st.expander(f"{art.get('score', 0):+.2f} — {title_short}"):
                                if art.get('description'):
                                    st.write(art.get('description'))
                                if art.get('source'):
                                    st.markdown(f"**Source:** {art.get('source')}")
                                if art.get('publishedAt'):
                                    st.markdown(f"**Published:** {art.get('publishedAt')}")
                                if art.get('url'):
                                    st.markdown(f"[Read more]({art.get('url')})")
                    else:
                        st.write("No news articles available or API key missing.")

                patterns = detect_candlestick_patterns(df)
                if patterns:
                    with st.expander("Candlestick Patterns"):
                        st.write(patterns)
            except Exception as e:
                st.warning(f"⚠️ Could not calculate signals: {e}")
        else:
            st.info("No data available for signals")

    # ------------------- Tab 2: Order Preview -------------------
    with tab2:
      st.subheader("Order Preview & Risk Management")
    try:
        ltp = get_underlying_ltp(
            kite,
            UNDERLYINGS[underlying_choice]["ticker"],
            fallback_df=df if not df.empty else None,
        )
    except Exception:
        ltp = None

    if ltp is None and not df.empty:
        ltp = df["Close"].iloc[-1]

    st.metric("Underlying LTP (NIFTY/BANKNIFTY)", f"{ltp:.2f}" if ltp else "N/A")

    if ltp:
        try:
            expiry_input = date.today() + timedelta(days=7)
            days_to_expiry = (expiry_input - date.today()).days
            
            strike = nearest_strike(ltp, ROUND_TO[underlying_choice])
            opt_side = "CE" if option_type.startswith("CE") else "PE"
            nfo_symbol = build_nfo_symbol(
                UNDERLYINGS[underlying_choice]["nfo_prefix"],
                expiry_input,
                strike,
                opt_side,
            )
            approx_premium = max(1.0, ltp * 0.01)
            qty_preview = size_qty_by_capital(
                starting_cap, approx_premium, allocation_pct, underlying_choice
            )

            # --- SUPERTREND-BASED SL/TARGET WITH DELTA ADJUSTMENT ---
            is_call = opt_side == "CE"
            indicator_levels = calculate_sl_target_from_indicators(
                df, 
                spot_price=ltp, 
                strike=strike,
                entry_premium=approx_premium,
                days_to_expiry=days_to_expiry,
                is_call=is_call,
                chart_interval=interval,
            )
            
            st.write("**Candidate Option:**", nfo_symbol)
            st.write(f"**Spot Price (NIFTY):** ₹{ltp:.2f} | **Strike:** {strike}")
            st.write(f"**Approx Option Premium:** ₹{approx_premium:.2f} | **Qty:** {qty_preview}")
            
            st.markdown("---")
            st.subheader("📊 Technical Analysis (Supertrend & Greeks)")
            
            if indicator_levels:
                col_greek1, col_greek2, col_greek3, col_greek4 = st.columns(4)
                with col_greek1:
                    st.metric("Delta", f"{indicator_levels['delta']:.3f}")
                    st.caption(f"Premium moves ₹{indicator_levels['delta']:.2f} per spot point")
                with col_greek2:
                    st.metric("ATR Value", f"₹{indicator_levels['atr_value']:.2f}")
                with col_greek3:
                    st.metric("RSI (14)", f"{indicator_levels['rsi']:.2f}")
                with col_greek4:
                    st.metric("Days to Expiry", days_to_expiry)
            
            st.markdown("---")
            
            # Combined reference panel: NIFTY spot levels and Option levels side-by-side
            st.markdown("---")
            st.subheader("📍 Reference Levels — Spot & Option")
            ref_col_spot, ref_col_opt = st.columns([1, 1])
            with ref_col_spot:
                st.info("**NIFTY Spot Levels**")
                st.write(f"Current Spot: ₹{indicator_levels['spot_price']:.2f}")
                st.write(f"Spot SL: ₹{indicator_levels['spot_stop_loss']:.2f} ({indicator_levels['spot_move_to_sl']:.2f} pt)")
                st.write(f"Spot Target: ₹{indicator_levels['spot_target']:.2f} (+{indicator_levels['spot_move_to_target']:.2f} pt)")
                if indicator_levels.get('supertrend_upper') is not None:
                    st.caption(f"Supertrend UB: ₹{indicator_levels['supertrend_upper']:.2f} | LB: ₹{indicator_levels['supertrend_lower']:.2f}")
            with ref_col_opt:
                st.info("**Option Levels**")
                st.write(f"Entry Premium: ₹{indicator_levels['option_entry']:.2f}")
                st.write(f"Option SL: ₹{indicator_levels['option_stop_loss']:.2f} (≈₹{indicator_levels['premium_move_to_sl']:.2f})")
                st.write(f"Option Target: ₹{indicator_levels['option_target']:.2f} (≈₹{indicator_levels['premium_move_to_target']:.2f})")
                st.caption(f"Delta: {indicator_levels['delta']:.3f} | ATR: ₹{indicator_levels['atr_value']:.2f} | RSI: {indicator_levels['rsi']:.2f}")

            st.markdown("---")
            st.subheader("🎯 OPTION Premium - SL & Target Configuration")
            st.write("**These are the prices that will trigger auto-close when option premium hits them**")
            
            # --- Create expandable section for OPTION SL/Target ---
            col_opt_entry, col_opt_sl, col_opt_tgt = st.columns(3)
            
            with col_opt_entry:
                st.write("**Entry Premium**")
                st.info(f"₹{indicator_levels['option_entry']:.2f}")
                st.caption("(Auto-calculated from strike/spot)")
            
            with col_opt_sl:
                st.write("**Stop Loss Premium**")
                if indicator_levels:
                    manual_opt_sl = st.number_input(
                        "Option SL (₹)", 
                        value=float(indicator_levels['option_stop_loss']),
                        min_value=0.05,
                        step=0.1,
                        key=f"manual_opt_sl_{interval}"
                    )
                else:
                    manual_opt_sl = st.number_input(
                        "Option SL (₹)", 
                        value=0.5,
                        min_value=0.05,
                        step=0.1,
                        key=f"manual_opt_sl_{interval}"
                    )
                loss_from_entry = indicator_levels['option_entry'] - manual_opt_sl
                st.caption(f"↓ Loss: ₹{loss_from_entry:.2f} ({(loss_from_entry/indicator_levels['option_entry']*100) if indicator_levels['option_entry'] > 0 else 0:.1f}%)")
            
            with col_opt_tgt:
                st.write("**Target Premium**")
                if indicator_levels:
                    manual_opt_target = st.number_input(
                        "Option Target (₹)", 
                        value=float(indicator_levels['option_target']),
                        min_value=0.01,
                        step=0.1,
                        key=f"manual_opt_target_{interval}"
                    )
                else:
                    manual_opt_target = st.number_input(
                        "Option Target (₹)", 
                        value=approx_premium * 1.5,
                        min_value=0.01,
                        step=0.1,
                        key=f"manual_opt_target_{interval}"
                    )
                profit_from_entry = manual_opt_target - indicator_levels['option_entry']
                st.caption(f"↑ Profit: ₹{profit_from_entry:.2f} ({(profit_from_entry/indicator_levels['option_entry']*100) if indicator_levels['option_entry'] > 0 else 0:.1f}%)")
            
            # Use manually entered values
            option_entry_price = indicator_levels['option_entry']
            option_stop_loss_price = manual_opt_sl
            option_target_price = manual_opt_target
            
            # Show summary
            st.markdown("---")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.info(f"**Entry Premium:** ₹{option_entry_price:.2f}")
            with summary_col2:
                st.warning(f"**Option SL:** ₹{option_stop_loss_price:.2f}")
            with summary_col3:
                st.success(f"**Option Target:** ₹{option_target_price:.2f}")
            
            st.markdown("---")
            st.write("**How it works:**")
            st.write(f"✓ When you BUY {nfo_symbol} at ₹{option_entry_price:.2f}, we'll monitor the option premium")
            st.write(f"✓ If premium falls to ₹{option_stop_loss_price:.2f} or lower → Automatic SELL order (LOSS)")
            st.write(f"✓ If premium rises to ₹{option_target_price:.2f} or higher → Automatic SELL order (PROFIT)")

        except Exception as e:
            st.warning(f"⚠️ Could not build order preview: {e}")

    st.markdown("---")
    col_exec1, col_exec2 = st.columns(2)

    with col_exec1:
        if st.button("Scan & Place (paper/live)"):
            try:
                if signal == "BULL":
                    opt_type, tx_type = "CE", "BUY"
                elif signal == "BEAR":
                    opt_type, tx_type = "PE", "BUY"
                else:
                    st.info("Signal NEUTRAL")
                    tx_type = None

                if tx_type and ltp:
                    strike = nearest_strike(ltp, ROUND_TO[underlying_choice])
                    nfo_symbol = build_nfo_symbol(
                        UNDERLYINGS[underlying_choice]["nfo_prefix"],
                        expiry_input,
                        strike,
                        opt_type,
                    )
                    qty = size_qty_by_capital(
                        starting_cap, option_entry_price, allocation_pct, underlying_choice
                    )

                    if live_trading and kite:
                        order_id = place_real_order(kite, nfo_symbol, tx_type, qty)
                        if order_id:
                            st.success(f"LIVE order placed. ID:{order_id}")
                            # Store order details for monitoring
                            st.session_state.setdefault("open_positions", {})[nfo_symbol] = {
                                "id": order_id,
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "symbol": nfo_symbol,
                                "underlying": underlying_choice,
                                "side": "LONG",
                                "entry_premium": option_entry_price,
                                "entry_spot": ltp,
                                "strike": strike,
                                "is_call": is_call,
                                "stop_loss_premium": option_stop_loss_price,
                                "target_premium": option_target_price,
                                "qty": int(qty),
                                "mode": "LIVE",
                                "status": "ACTIVE",
                                "days_to_expiry": days_to_expiry,
                            }
                    else:
                        trade_id = f"PAPER-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        st.session_state.setdefault("open_positions", {})[nfo_symbol] = {
                            "id": trade_id,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "symbol": nfo_symbol,
                            "underlying": underlying_choice,
                            "side": "LONG",
                            "entry_premium": option_entry_price,
                            "entry_spot": ltp,
                            "strike": strike,
                            "is_call": is_call,
                            "stop_loss_premium": option_stop_loss_price,
                            "target_premium": option_target_price,
                            "qty": int(qty),
                            "mode": "PAPER",
                            "status": "ACTIVE",
                            "days_to_expiry": days_to_expiry,
                        }
                        st.success(f"PAPER trade recorded: {nfo_symbol} qty={qty} "
                                   f"Premium Entry=₹{option_entry_price:.2f} "
                                   f"SL=₹{option_stop_loss_price:.2f} "
                                   f"TGT=₹{option_target_price:.2f}")
            except Exception as e:
                st.error(f"⚠️ Could not place trade: {e}")

        with col_exec2:
            if st.button("Close All Paper Positions"):
                if "open_positions" in st.session_state and st.session_state["open_positions"]:
                    for sym in list(st.session_state["open_positions"].keys()):
                        st.session_state["open_positions"].pop(sym)
                        st.info(f"Closed paper pos {sym}")
                else:
                    st.info("No open positions")

    # ------------------- Tab 3: Positions -------------------
    with tab3:
        st.subheader("🎯 Active Positions - Real-Time Monitoring (Option Premium)")
        
        open_pos = st.session_state.get("open_positions", {})
        
        if open_pos:
            # Create a placeholder for live updates
            pos_container = st.container()
            
            for sym, pos in list(open_pos.items()):
                with pos_container:
                    st.markdown("---")
                    st.write(f"**Option:** {sym} | **Status:** {pos.get('status', 'UNKNOWN')}")
                    
                    # Get current option premium (from option symbol)
                    # Note: In production, fetch from Zerodha option chain
                    current_option_premium = get_option_premium(kite, sym, fallback=pos.get("entry_premium"))
                    
                    entry_premium = pos.get("entry_premium", 0)
                    entry_spot = pos.get("entry_spot", 0)
                    strike = pos.get("strike", 0)
                    sl_premium = pos.get("stop_loss_premium", 0)
                    target_premium = pos.get("target_premium", 0)
                    
                    # Calculate P&L (on premium)
                    pnl_premium = current_option_premium - entry_premium
                    pnl_pct = (pnl_premium / entry_premium * 100) if entry_premium > 0 else 0
                    
                    # Determine status based on OPTION PREMIUM
                    hit_sl = current_option_premium <= sl_premium
                    hit_target = current_option_premium >= target_premium
                    
                    # Display position info in columns
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        st.write("**Entry Premium:** ₹{:.2f}".format(entry_premium))
                        st.write("**Entry Spot:** ₹{:.2f}".format(entry_spot))
                        st.write("**Strike:** {}".format(strike))
                    
                    with col_b:
                        st.write("**Current Premium:** ₹{:.2f}".format(current_option_premium))
                        if pnl_pct >= 0:
                            st.success(f"**P&L:** +₹{pnl_premium:.2f} (+{pnl_pct:.2f}%)")
                        else:
                            st.error(f"**P&L:** ₹{pnl_premium:.2f} ({pnl_pct:.2f}%)")
                    
                    with col_c:
                        st.write(f"**SL Premium:** ₹{sl_premium:.2f}")
                        st.write(f"**Target Premium:** ₹{target_premium:.2f}")
                    
                    with col_d:
                        st.write(f"**Qty:** {pos.get('qty', 0)}")
                        st.write(f"**Mode:** {pos.get('mode', 'PAPER')}")
                        st.write(f"**Days to Expiry:** {pos.get('days_to_expiry', 'N/A')}")
                    
                    # Show status
                    st.markdown("---")
                    
                    if hit_target:
                        st.balloons()
                        st.success(f"✅ TARGET HIT! Premium ₹{current_option_premium:.2f} >= Target ₹{target_premium:.2f}")
                        
                        # Auto close position
                        if pos.get("mode") == "LIVE" and kite:
                            place_sell_order(kite, sym, pos.get("qty", 1), f"@ ₹{current_option_premium:.2f} premium (TARGET)")
                        
                        st.session_state["open_positions"].pop(sym)
                        st.info(f"Position closed - PROFIT: ₹{pnl_premium:.2f}")
                        
                    elif hit_sl:
                        st.error(f"❌ STOP LOSS HIT! Premium ₹{current_option_premium:.2f} <= SL ₹{sl_premium:.2f}")
                        
                        # Auto close position
                        if pos.get("mode") == "LIVE" and kite:
                            place_sell_order(kite, sym, pos.get("qty", 1), f"@ ₹{current_option_premium:.2f} premium (STOPLOSS)")
                        
                        st.session_state["open_positions"].pop(sym)
                        st.warning(f"Position closed - LOSS: ₹{pnl_premium:.2f}")
                    
                    else:
                        # Show progress
                        dist_to_sl = current_option_premium - sl_premium
                        dist_to_target = target_premium - current_option_premium
                        
                        st.info(f"🔄 ACTIVE | Premium to SL: ₹{dist_to_sl:.2f} | Premium to Target: ₹{dist_to_target:.2f}")
                        
                        # Manual close button
                        if st.button(f"Close {sym} Manually", key=f"close_{sym}"):
                            if pos.get("mode") == "LIVE" and kite:
                                place_sell_order(kite, sym, pos.get("qty", 1), f"@ ₹{current_option_premium:.2f} premium (MANUAL)")
                            st.session_state["open_positions"].pop(sym)
                            st.info(f"Manual close: {sym}")
            
            # Summary stats
            st.subheader("📊 Summary")
            total_pnl = 0
            total_positions = len(open_pos)
            
            for sym, pos in open_pos.items():
                current_option_premium = get_option_premium(kite, sym, fallback=pos.get("entry_premium"))
                pnl_premium = current_option_premium - pos.get("entry_premium", 0)
                total_pnl += pnl_premium
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Active Positions", total_positions)
            with col_s2:
                if total_pnl >= 0:
                    st.metric("Total P&L", f"₹{total_pnl:.2f}", delta=f"+{total_pnl:.2f}", delta_color="off")
                else:
                    st.metric("Total P&L", f"₹{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="off")
            with col_s3:
                st.write("Auto-refresh: Every 30s")
        
        else:
            st.info("📭 No open positions. Place an order from the Order Preview tab.")

except Exception as e:
    st.error(f"⚠️ Tabs could not load. Please refresh. ({e})")

# ------------------- Auto-refresh & Trailing SL -------------------
st_autorefresh(interval=30000, key="auto_refresh")

def monitor_option_premiums(kite=None):
    """Monitor active positions based on OPTION PREMIUM moves (not spot moves)"""
    open_pos = st.session_state.get("open_positions", {})
    if not open_pos:
        return

    closed_positions = []
    
    for sym, pos in list(open_pos.items()):
        # Get current option premium
        current_option_premium = get_option_premium(kite, sym, fallback=pos.get("entry_premium"))
        
        if current_option_premium <= 0:
            continue

        entry_premium = pos.get("entry_premium", 0)
        sl_premium = pos.get("stop_loss_premium", 0)
        target_premium = pos.get("target_premium", 0)
        
        # Check if SL or Target is hit (based on OPTION PREMIUM)
        hit_sl = current_option_premium <= sl_premium
        hit_target = current_option_premium >= target_premium
        
        if hit_target or hit_sl:
            # Close position
            if pos.get("mode") == "LIVE" and kite:
                trigger_msg = "TARGET" if hit_target else "STOPLOSS"
                place_sell_order(kite, sym, pos.get("qty", 1), f"@ ₹{current_option_premium:.2f} premium ({trigger_msg})")
            
            closed_positions.append(sym)
            st.session_state["open_positions"].pop(sym)

try:
    monitor_option_premiums(kite if "kite" in st.session_state else None)
except Exception as e:
    st.warning(f"⚠️ Monitoring error: {e}")


