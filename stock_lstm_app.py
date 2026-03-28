"""
╔══════════════════════════════════════════════════════════════════╗
║   STOCK FUND INVESTMENT ANALYZER — LSTM + Streamlit             ║
║   Theme: Professional RED & BLACK                               ║
║   Enhanced with advanced UI + new features                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys
import warnings
import math
import datetime

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

try:
    import yfinance as yf
    _yf_ver = tuple(int(x) for x in yf.__version__.split(".")[:3])
    if _yf_ver < (0, 2, 40):
        st.warning(
            f"⚠️ You have yfinance **{yf.__version__}** — too old.\n\n"
            "Run:\n```\npip uninstall yfinance multitasking -y\n"
            'pip install "yfinance>=0.2.40"\n```'
        )
except TypeError:
    st.error("yfinance import failed.\n\nFix:\n```\npip uninstall yfinance multitasking -y\n"
             'pip install "yfinance>=0.2.40"\n```')
    st.stop()
except Exception as e:
    st.error(f"Cannot import yfinance: {e}")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LSTM Stock Analyzer Pro",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
# RED & BLACK PROFESSIONAL THEME
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;700;900&display=swap');

  /* ── Global Background ── */
  [data-testid="stAppViewContainer"] {
    background: #000000;
    background-image:
      radial-gradient(ellipse at 20% 20%, #1a000088 0%, transparent 50%),
      radial-gradient(ellipse at 80% 80%, #0d000066 0%, transparent 50%),
      repeating-linear-gradient(
        0deg,
        transparent,
        transparent 40px,
        #1a000015 40px,
        #1a000015 41px
      ),
      repeating-linear-gradient(
        90deg,
        transparent,
        transparent 40px,
        #1a000015 40px,
        #1a000015 41px
      );
    font-family: 'Exo 2', sans-serif;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0a0000 !important;
    border-right: 1px solid #cc000055 !important;
    box-shadow: 4px 0 30px #cc000020;
  }
  [data-testid="stSidebar"] * {
    font-family: 'Exo 2', sans-serif !important;
  }

  /* ── Headings ── */
  h1, h2, h3, h4 {
    font-family: 'Rajdhani', sans-serif !important;
    color: #ff2222 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
  }
  h1 { font-size: 2.4rem !important; font-weight: 700 !important; }

  /* ── Paragraph text ── */
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li {
    color: #ccaaaa !important;
    font-family: 'Exo 2', sans-serif !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #1a0000, #2d0000) !important;
    border: 1px solid #cc0000 !important;
    color: #ff3333 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    border-radius: 2px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 0 15px #cc000030 !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #2d0000, #4d0000) !important;
    box-shadow: 0 0 30px #cc000066, inset 0 0 10px #ff000010 !important;
    border-color: #ff3333 !important;
    transform: translateY(-1px) !important;
  }

  /* ── Metrics ── */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d0000, #1a0505) !important;
    border: 1px solid #cc000040 !important;
    border-left: 3px solid #cc0000 !important;
    padding: 1rem !important;
    border-radius: 2px !important;
    box-shadow: 0 4px 20px #00000080 !important;
  }
  div[data-testid="metric-container"] label {
    color: #884444 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ff4444 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
  }

  /* ── Progress bar ── */
  .stProgress > div > div {
    background: linear-gradient(90deg, #660000, #cc0000, #ff4444) !important;
    box-shadow: 0 0 10px #cc000066 !important;
  }
  .stProgress > div {
    background: #1a0000 !important;
  }

  /* ── Code blocks ── */
  code {
    color: #ff6666 !important;
    background: #0d0000 !important;
    border: 1px solid #cc000030 !important;
    font-family: 'Share Tech Mono', monospace !important;
  }
  pre {
    background: #0a0000 !important;
    border: 1px solid #cc000040 !important;
    border-left: 3px solid #cc0000 !important;
  }

  /* ── Selectbox / Sliders ── */
  .stSelectbox > div > div {
    background: #0d0000 !important;
    border: 1px solid #cc000055 !important;
    color: #ffaaaa !important;
    font-family: 'Exo 2', sans-serif !important;
  }
  .stSlider > div > div > div {
    background: #cc0000 !important;
  }

  /* ── Expander ── */
  [data-testid="stExpander"] {
    background: #0a0000 !important;
    border: 1px solid #cc000040 !important;
    border-radius: 2px !important;
  }
  [data-testid="stExpander"] summary {
    color: #ff4444 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
  }

  /* ── Info / Warning / Error boxes ── */
  .stInfo, [data-testid="stInfo"] {
    background: #0d0010 !important;
    border: 1px solid #cc000066 !important;
    color: #ffaaaa !important;
  }
  .stWarning, [data-testid="stWarning"] {
    background: #1a0d00 !important;
    border-left: 3px solid #ff6600 !important;
  }

  /* ── Divider ── */
  hr {
    border-color: #cc000033 !important;
    margin: 1.5rem 0 !important;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {
    border: 1px solid #cc000040 !important;
  }

  /* ── Caption ── */
  [data-testid="stCaptionContainer"] p {
    color: #664444 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0a0000; }
  ::-webkit-scrollbar-thumb { background: #660000; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #cc0000; }

  /* ── Sidebar label text ── */
  .stSelectbox label, .stSlider label, .stRadio label {
    color: #aa4444 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    font-size: 0.85rem !important;
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TICKER CATALOGUE
# ══════════════════════════════════════════════════════════════════
TICKERS = {
    "SPY  — S&P 500 ETF":                    "SPY",
    "QQQ  — NASDAQ 100 ETF":                 "QQQ",
    "DIA  — Dow Jones ETF":                  "DIA",
    "IWM  — Russell 2000 Small-Cap ETF":     "IWM",
    "VTI  — Total US Stock Market ETF":      "VTI",
    "VOO  — Vanguard S&P 500 ETF":           "VOO",
    "XLK  — Technology Sector ETF":          "XLK",
    "XLF  — Financial Sector ETF":           "XLF",
    "XLE  — Energy Sector ETF":              "XLE",
    "XLV  — Healthcare Sector ETF":          "XLV",
    "XLY  — Consumer Discretionary ETF":     "XLY",
    "AAPL — Apple Inc.":                     "AAPL",
    "MSFT — Microsoft Corporation":          "MSFT",
    "GOOGL— Alphabet (Google)":              "GOOGL",
    "AMZN — Amazon.com Inc.":                "AMZN",
    "NVDA — NVIDIA Corporation":             "NVDA",
    "META — Meta Platforms":                 "META",
    "TSLA — Tesla Inc.":                     "TSLA",
    "GLD  — Gold ETF":                       "GLD",
    "TLT  — US Treasury Bond ETF":           "TLT",
    "BTC-USD — Bitcoin":                     "BTC-USD",
}

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-family: Rajdhani, sans-serif; font-size:1.6rem;
                    font-weight:700; color:#ff2222; letter-spacing:0.15em;
                    text-transform:uppercase;'>
            ◈ LSTM ANALYZER
        </div>
        <div style='font-family: Share Tech Mono, monospace; font-size:0.7rem;
                    color:#663333; letter-spacing:0.2em; margin-top:0.2rem;'>
            PROFESSIONAL EDITION
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    ticker_label = st.selectbox("Select Asset", options=list(TICKERS.keys()), index=0)
    ticker = TICKERS[ticker_label]
    st.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#884444;margin-top:-0.5rem;margin-bottom:0.5rem;'>SYMBOL: <span style='color:#ff4444;'>{ticker}</span></div>", unsafe_allow_html=True)

    period_label = st.selectbox(
        "Historical Period",
        ["1 Year", "2 Years", "5 Years", "10 Years", "Max Available"],
        index=2
    )
    period_map = {"1 Year": 1, "2 Years": 2, "5 Years": 5, "10 Years": 10, "Max Available": 20}
    period_years = period_map[period_label]

    st.markdown("---")
    st.markdown("<div style='font-family:Rajdhani,sans-serif;font-weight:700;color:#cc3333;font-size:0.85rem;letter-spacing:0.1em;text-transform:uppercase;'>MODEL PARAMETERS</div>", unsafe_allow_html=True)

    window_size    = st.slider("Lookback Window (days)", 20, 120, 60, 5)
    forecast_days  = st.slider("Forecast Horizon (days)", 5, 30, 10, 5)
    return_threshold = st.slider("BUY Threshold (%)", 0.5, 5.0, 1.5, 0.5) / 100.0

    st.markdown("---")
    st.markdown("<div style='font-family:Rajdhani,sans-serif;font-weight:700;color:#cc3333;font-size:0.85rem;letter-spacing:0.1em;text-transform:uppercase;'>ARCHITECTURE</div>", unsafe_allow_html=True)

    lstm_units_1 = st.select_slider("LSTM Layer 1", [32, 64, 128, 256], value=128)
    lstm_units_2 = st.select_slider("LSTM Layer 2", [16, 32, 64, 128], value=64)
    dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.3, 0.05)
    epochs       = st.slider("Max Epochs", 20, 150, 80, 10)

    st.markdown("---")
    run_btn = st.button("⬛  EXECUTE ANALYSIS", use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PLOT THEME CONSTANTS
# ══════════════════════════════════════════════════════════════════
BG       = "#000000"
BG2      = "#0a0000"
RED      = "#cc0000"
RED_BR   = "#ff3333"
RED_DIM  = "#660000"
GREEN    = "#00cc44"
GRID     = "#1a0000"
TEXT     = "#ccaaaa"
MONO     = "Share Tech Mono"

# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_data(ticker: str, years: int) -> pd.DataFrame:
    end_dt   = datetime.date.today()
    start_dt = end_dt - datetime.timedelta(days=365 * years)
    tk = yf.Ticker(ticker)
    df = tk.history(
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        actions=False
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[needed].dropna()

# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["body"]         = df["Close"] - df["Open"]
    f["body_ratio"]   = f["body"].abs() / (df["High"] - df["Low"] + 1e-8)
    f["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    f["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    f["direction"]    = np.sign(f["body"])
    f["return_1d"]    = df["Close"].pct_change(1)
    f["return_5d"]    = df["Close"].pct_change(5)
    f["return_10d"]   = df["Close"].pct_change(10)
    sma20 = df["Close"].rolling(20).mean()
    sma50 = df["Close"].rolling(50).mean()
    f["price_vs_sma20"] = df["Close"] / (sma20 + 1e-8)
    f["price_vs_sma50"] = df["Close"] / (sma50 + 1e-8)
    f["sma_cross"]      = sma20 / (sma50 + 1e-8)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    f["macd"]           = (ema12 - ema26) / (df["Close"] + 1e-8)
    f["volatility_20d"] = f["return_1d"].rolling(20).std()
    f["atr_ratio"]      = (df["High"] - df["Low"]).rolling(14).mean() / (df["Close"] + 1e-8)
    vol_avg20           = df["Volume"].rolling(20).mean()
    f["volume_ratio"]   = df["Volume"] / (vol_avg20 + 1)
    obv                 = (np.sign(f["return_1d"]) * df["Volume"]).cumsum()
    f["obv_ratio"]      = obv / (obv.abs().rolling(20).mean() + 1)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"]            = (100 - 100 / (1 + gain / (loss + 1e-8))) / 100.0
    # ── NEW: Bollinger Band position ──
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    f["bb_position"] = (df["Close"] - bb_mid) / (2 * bb_std + 1e-8)
    # ── NEW: Momentum ──
    f["momentum_5"]  = df["Close"] / (df["Close"].shift(5) + 1e-8) - 1
    f["momentum_20"] = df["Close"] / (df["Close"].shift(20) + 1e-8) - 1
    return f.dropna()

def create_labels(df_orig, feat_index, horizon, threshold):
    aligned = df_orig.loc[feat_index, "Close"]
    ret     = aligned.shift(-horizon) / aligned - 1
    return (ret > threshold).astype(int)

def make_sequences(features_arr, labels_arr, window):
    X, y = [], []
    for i in range(window, len(features_arr)):
        X.append(features_arr[i - window: i])
        y.append(labels_arr[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def build_lstm_model(window, n_features, units1, units2, drop):
    mdl = models.Sequential([
        layers.LSTM(units1, return_sequences=True,
                    input_shape=(window, n_features), name="lstm_1"),
        layers.Dropout(drop, name="drop_1"),
        layers.LSTM(units2, return_sequences=False, name="lstm_2"),
        layers.Dropout(drop, name="drop_2"),
        layers.Dense(64, activation="relu", name="dense_1"),
        layers.Dropout(drop / 2, name="drop_3"),
        layers.Dense(32, activation="relu", name="dense_2"),
        layers.Dense(1, activation="sigmoid", name="output")
    ], name="LSTM_RedBlack_Pro")
    mdl.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return mdl

def decode_signal(prob):
    if prob >= 0.65:   return "STRONG BUY",  "buy",   "▲"
    elif prob >= 0.52: return "BUY",          "buy",   "▲"
    elif prob >= 0.35: return "HOLD",         "hold",  "◆"
    elif prob >= 0.20: return "SELL",         "sell",  "▼"
    else:              return "STRONG SELL",  "sell",  "▼"

# ══════════════════════════════════════════════════════════════════
# CHART HELPERS — RED & BLACK THEME
# ══════════════════════════════════════════════════════════════════
def chart_layout(height=480):
    return dict(
        plot_bgcolor=BG, paper_bgcolor=BG2,
        font=dict(color=TEXT, family=MONO, size=11),
        height=height,
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID,
                   showline=True, linecolor=RED_DIM),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID,
                   showline=True, linecolor=RED_DIM),
        legend=dict(bgcolor=BG2, bordercolor=RED_DIM, borderwidth=1,
                    font=dict(family=MONO, size=10)),
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_rangeslider_visible=False,
    )

def candlestick_fig(df, ticker):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.02,
        subplot_titles=["", "VOLUME", "RSI"]
    )
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=GREEN, decreasing_line_color=RED_BR,
        increasing_fillcolor="rgba(0,204,68,0.2)", decreasing_fillcolor="rgba(255,51,51,0.2)",
        name=ticker), row=1, col=1)
    # SMAs
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(20).mean(),
        line=dict(color=RED, width=1.5, dash="solid"), name="SMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(50).mean(),
        line=dict(color="#ff6600", width=1.5, dash="dot"), name="SMA 50"), row=1, col=1)
    # Bollinger Bands
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    fig.add_trace(go.Scatter(x=df.index, y=bb_mid + 2*bb_std,
        line=dict(color="#660000", width=0.8, dash="dash"), name="BB Upper",
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_mid - 2*bb_std,
        line=dict(color="#660000", width=0.8, dash="dash"), name="BB Lower",
        fill="tonexty", fillcolor="rgba(204,0,0,0.04)", showlegend=False), row=1, col=1)
    # Volume
    colors = [GREEN if c >= o else RED_BR
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
        marker_color=colors, opacity=0.6, name="Volume"), row=2, col=1)
    # RSI
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - 100 / (1 + gain / (loss + 1e-8))
    fig.add_trace(go.Scatter(x=df.index, y=rsi,
        line=dict(color=RED, width=1.5), name="RSI",
        fill="tozeroy", fillcolor="rgba(204,0,0,0.06)"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#ff6600", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=GREEN, opacity=0.5, row=3, col=1)

    layout = chart_layout(580)
    layout["yaxis3"] = dict(range=[0, 100], gridcolor=GRID, zerolinecolor=GRID)
    fig.update_layout(**layout)
    for ann in fig.layout.annotations:
        ann.font.color = RED_DIM
        ann.font.family = MONO
        ann.font.size = 10
    return fig

def loss_fig(history):
    ep  = list(range(1, len(history.history["loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=history.history["loss"],
        name="Train Loss", line=dict(color=RED_BR, width=2),
        fill="tozeroy", fillcolor="rgba(204,0,0,0.1)"))
    fig.add_trace(go.Scatter(x=ep, y=history.history["val_loss"],
        name="Val Loss", line=dict(color="#ff6600", width=2, dash="dash")))
    layout = chart_layout(300)
    layout["xaxis"]["title"] = "EPOCH"
    layout["yaxis"]["title"] = "LOSS"
    fig.update_layout(**layout)
    return fig

def signal_history_fig(dates, probs):
    fig = go.Figure()
    fig.add_hrect(y0=0.65, y1=1.0, fillcolor=GREEN,    opacity=0.06,
                  annotation_text="BUY ZONE",   annotation_font_color=GREEN)
    fig.add_hrect(y0=0.35, y1=0.65, fillcolor="#ffb800", opacity=0.04,
                  annotation_text="HOLD ZONE",  annotation_font_color="#ffb800")
    fig.add_hrect(y0=0.0,  y1=0.35, fillcolor=RED_BR,  opacity=0.06,
                  annotation_text="SELL ZONE",  annotation_font_color=RED_BR)
    fig.add_trace(go.Scatter(x=dates, y=probs, mode="lines",
        name="BUY Probability", line=dict(color=RED, width=2),
        fill="tozeroy", fillcolor="rgba(204,0,0,0.13)"))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#ffffff", opacity=0.2)
    layout = chart_layout(280)
    layout["yaxis"]["range"] = [0, 1]
    layout["yaxis"]["title"] = "BUY PROBABILITY"
    fig.update_layout(**layout)
    return fig

def signals_on_price_fig(test_close, y_pred, y_prob):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_close.index, y=test_close.values,
        mode="lines", name="CLOSE",
        line=dict(color=TEXT, width=1.5)))
    buy_idx  = test_close.index[y_pred[:len(test_close)] == 1]
    sell_idx = test_close.index[y_pred[:len(test_close)] == 0]
    fig.add_trace(go.Scatter(x=buy_idx, y=test_close.loc[buy_idx].values,
        mode="markers", name="BUY",
        marker=dict(color=GREEN, size=8, symbol="triangle-up",
                    line=dict(color="#003300", width=1))))
    fig.add_trace(go.Scatter(x=sell_idx, y=test_close.loc[sell_idx].values,
        mode="markers", name="SELL",
        marker=dict(color=RED_BR, size=8, symbol="triangle-down",
                    line=dict(color="#330000", width=1))))
    layout = chart_layout(380)
    fig.update_layout(**layout)
    return fig

def rsi_gauge(rsi_val):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_val,
        title={"text": "RSI (14)", "font": {"color": TEXT, "family": MONO, "size": 13}},
        number={"font": {"color": RED_BR, "family": MONO}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT,
                     "tickfont": {"family": MONO, "size": 9}},
            "bar":  {"color": RED},
            "bgcolor": BG2,
            "bordercolor": RED_DIM,
            "steps": [
                {"range": [0, 30],   "color": "rgba(0,204,68,0.13)"},
                {"range": [30, 70],  "color": "#111100"},
                {"range": [70, 100], "color": "rgba(204,0,0,0.13)"},
            ],
            "threshold": {
                "line": {"color": RED_BR, "width": 2},
                "thickness": 0.75,
                "value": rsi_val
            }
        }
    ))
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG2,
        font=dict(color=TEXT, family=MONO),
        height=220, margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig

# ══════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style='border-bottom: 1px solid #cc000044; padding-bottom: 1rem; margin-bottom: 1rem;'>
  <div style='font-family: Rajdhani, sans-serif; font-size: 2.5rem; font-weight: 900;
              color: #cc0000; letter-spacing: 0.1em; text-transform: uppercase;
              text-shadow: 0 0 30px #cc000066;'>
    ◈ LSTM STOCK FUND ANALYZER
  </div>
  <div style='font-family: Share Tech Mono, monospace; font-size: 0.78rem;
              color: #663333; letter-spacing: 0.25em; margin-top: 0.3rem;'>
    PROFESSIONAL EDITION  ·  POWERED BY TENSORFLOW  ·  RED & BLACK THEME
  </div>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### ◈ HOW IT WORKS")
        st.markdown("""
1. Downloads real OHLCV data via Yahoo Finance
2. Engineers **20 candlestick + technical features**
3. Trains a 2-layer stacked LSTM classifier
4. Outputs BUY / HOLD / SELL with confidence %
5. Shows RSI gauge + Bollinger Bands overlay
        """)
    with c2:
        st.markdown("#### ◈ LSTM ARCHITECTURE")
        st.code("""LSTM(128, return_sequences=True)
Dropout(0.3)
LSTM(64,  return_sequences=False)
Dropout(0.3)
Dense(64, activation='relu')
Dense(32, activation='relu')
Dense(1,  activation='sigmoid')""", language="python")
    with c3:
        st.markdown("#### ◈ NEW FEATURES")
        st.markdown("""
- 🔴 **Bollinger Bands** on price chart
- 🔴 **RSI Panel** below candlestick
- 🔴 **RSI Gauge** indicator
- 🔴 **Momentum features** (5d & 20d)
- 🔴 **Deeper Dense layers** for accuracy
- 🔴 **Red & Black Pro Theme**
        """)
    st.markdown("""
    <div style='background:#0a0000;border:1px solid #cc000044;border-left:3px solid #cc0000;
                padding:1rem 1.5rem;border-radius:2px;margin-top:1rem;
                font-family:Share Tech Mono,monospace;color:#884444;font-size:0.85rem;'>
        ◈ SELECT AN ASSET FROM THE SIDEBAR AND CLICK  
        <span style='color:#ff3333;font-weight:700;'> ⬛ EXECUTE ANALYSIS</span>
        TO BEGIN
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════
bar = st.progress(0, text="")

# STEP 1: Download
bar.progress(5, text=f"▶ DOWNLOADING {ticker} — {period_label}...")
try:
    df = load_data(ticker, period_years)
except Exception as e:
    st.error(f"Download failed for **{ticker}**: {e}")
    st.stop()

if df is None or df.empty:
    st.error(f"**`{ticker}` returned empty data.**")
    st.stop()

if len(df) < 200:
    st.warning(f"Only **{len(df)} rows** downloaded. Try a longer period.")

# STEP 2: Price chart
bar.progress(15, text="▶ RENDERING CHART...")
st.markdown(f"### ◈ {ticker}  —  {period_label.upper()} PRICE HISTORY")
st.plotly_chart(candlestick_fig(df, ticker), use_container_width=True)

last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2])
chg        = (last_close - prev_close) / prev_close * 100
hi52       = float(df["Close"].tail(252).max())
lo52       = float(df["Close"].tail(252).min())
vol_avg    = float(df["Volume"].tail(20).mean())

# ── RSI current value ──
delta_r = df["Close"].diff()
gain_r  = delta_r.clip(lower=0).rolling(14).mean()
loss_r  = (-delta_r.clip(upper=0)).rolling(14).mean()
rsi_now = float((100 - 100 / (1 + gain_r / (loss_r + 1e-8))).iloc[-1])

co1, co2, co3, co4, co5 = st.columns(5)
co1.metric("LAST CLOSE",    f"${last_close:.2f}", f"{chg:+.2f}%")
co2.metric("52W HIGH",      f"${hi52:.2f}")
co3.metric("52W LOW",       f"${lo52:.2f}")
co4.metric("TRADING DAYS",  f"{len(df):,}")
co5.metric("AVG VOLUME 20D",f"{vol_avg/1e6:.1f}M")
st.markdown("---")

# STEP 3: Features
bar.progress(25, text="▶ ENGINEERING FEATURES...")
features_df  = engineer_features(df)
feature_cols = features_df.columns.tolist()
n_features   = len(feature_cols)

with st.expander(f"◈ ENGINEERED FEATURES ({n_features} columns) — LAST 10 ROWS"):
    st.dataframe(features_df.tail(10).style.format("{:.4f}"), use_container_width=True)

# STEP 4: Labels
bar.progress(30, text="▶ CREATING LABELS...")
labels           = create_labels(df, features_df.index, forecast_days, return_threshold)
labels           = labels.iloc[:-forecast_days]
features_trimmed = features_df.iloc[:-forecast_days]

l1, l2, l3 = st.columns(3)
l1.metric("BUY LABELS",  f"{labels.sum():,} ({labels.mean()*100:.1f}%)")
l2.metric("SELL LABELS", f"{(labels==0).sum():,} ({(1-labels.mean())*100:.1f}%)")
l3.metric("FEATURE COLS", str(n_features))

# STEP 5: Scale
bar.progress(35, text="▶ SCALING FEATURES...")
split_train     = int(len(features_trimmed) * 0.80)
scaler          = MinMaxScaler()
scaler.fit(features_trimmed.values[:split_train])
features_scaled = scaler.transform(features_trimmed.values)

# STEP 6: Sequences
bar.progress(40, text="▶ BUILDING SEQUENCES...")
X, y   = make_sequences(features_scaled, labels.values, window_size)
trn_n  = int(len(X) * 0.80)
val_n  = int(len(X) * 0.10)

X_train, y_train = X[:trn_n],             y[:trn_n]
X_val,   y_val   = X[trn_n:trn_n+val_n],  y[trn_n:trn_n+val_n]
X_test,  y_test  = X[trn_n+val_n:],       y[trn_n+val_n:]
test_dates        = features_trimmed.index[window_size + trn_n + val_n:]

st.markdown("**◈ CHRONOLOGICAL DATA SPLIT:**")
s1, s2, s3 = st.columns(3)
s1.metric("TRAIN", f"{len(X_train):,}")
s2.metric("VAL",   f"{len(X_val):,}")
s3.metric("TEST",  f"{len(X_test):,}")

# STEP 7: Build
bar.progress(50, text="▶ BUILDING LSTM MODEL...")
model = build_lstm_model(window_size, n_features, lstm_units_1, lstm_units_2, dropout_rate)

with st.expander("◈ MODEL ARCHITECTURE"):
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    st.code("\n".join(lines), language="text")

# STEP 8: Train
bar.progress(55, text="▶ TRAINING LSTM — PLEASE WAIT...")
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                   patience=6, min_lr=1e-6, verbose=0)
    ],
    verbose=0
)
actual_epochs = len(history.history["loss"])

# STEP 9: Evaluate
bar.progress(85, text="▶ EVALUATING MODEL...")
y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob > 0.5).astype(int)
acc    = accuracy_score(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, y_prob)
except Exception:
    auc = float("nan")

# STEP 10: Signal
bar.progress(95, text="▶ GENERATING SIGNAL...")
latest_prob                     = float(model.predict(
    features_scaled[-window_size:].reshape(1, window_size, n_features), verbose=0)[0][0])
signal_label, signal_type, signal_icon = decode_signal(latest_prob)
bar.progress(100, text="✅ ANALYSIS COMPLETE")
bar.empty()

# ══════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### ◈ INVESTMENT DECISION")

sig_color = {"buy": "#00cc44", "hold": "#ffb800", "sell": "#cc0000"}[signal_type]
sig_bg    = {"buy": "#001a00", "hold": "#1a1000", "sell": "#1a0000"}[signal_type]

st.markdown(f"""
<div style="background:{sig_bg};border:2px solid {sig_color};
            padding:2.5rem 2rem;text-align:center;border-radius:2px;
            margin-bottom:1.5rem;box-shadow:0 0 40px {sig_color}22;
            position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:2px;
              background:linear-gradient(90deg,transparent,{sig_color},transparent);"></div>
  <div style="font-family:Rajdhani,sans-serif;font-size:3.8rem;font-weight:900;
              color:{sig_color};letter-spacing:0.15em;text-shadow:0 0 20px {sig_color}88;">
    {signal_icon} {signal_label}
  </div>
  <div style="font-family:Share Tech Mono,monospace;color:#664444;font-size:0.8rem;
              letter-spacing:0.18em;margin-top:0.8rem;">
    BUY PROBABILITY: <span style="color:{sig_color};font-size:1.1rem;">{latest_prob*100:.1f}%</span>
    &nbsp;|&nbsp; SYMBOL: <span style="color:{sig_color};">{ticker}</span>
    &nbsp;|&nbsp; HORIZON: <span style="color:{sig_color};">{forecast_days}D</span>
    &nbsp;|&nbsp; THRESHOLD: <span style="color:{sig_color};">{return_threshold*100:.1f}%</span>
  </div>
  <div style="position:absolute;bottom:0;left:0;right:0;height:2px;
              background:linear-gradient(90deg,transparent,{sig_color},transparent);"></div>
</div>
""", unsafe_allow_html=True)

# ── RSI Gauge + Advice ──
adv_col, gauge_col = st.columns([2, 1])

with adv_col:
    advice = {
        "buy": f"""
**◈ WHY BUY?** The LSTM detected a **bullish pattern** over the last {window_size} trading days of **{ticker}**.
Model gives **{latest_prob*100:.1f}% probability** of price rising more than {return_threshold*100:.1f}%
in the next {forecast_days} trading days.

**Suggested action:** Consider entering a position. Place a stop-loss ~2% below entry.
Watch for confirmation via increasing volume on up-candles.
""",
        "hold": f"""
**◈ WHY HOLD?** The LSTM sees a **mixed, sideways pattern** in **{ticker}**.
BUY probability is {latest_prob*100:.1f}% — not strong enough to commit capital.

**Suggested action:** Stay on the sidelines. Wait for a cleaner breakout.
Re-run after major earnings or economic events.
""",
        "sell": f"""
**◈ WHY SELL / AVOID?** The LSTM detected a **bearish pattern** in **{ticker}**.
BUY probability is only {latest_prob*100:.1f}%, meaning **{(1-latest_prob)*100:.1f}% bearish signal**
for the next {forecast_days} trading days.

**Suggested action:** Avoid new positions. Tighten stop-loss if already holding.
Wait for confirmed bullish reversal before re-entering.
"""
    }[signal_type]
    st.markdown(advice)

with gauge_col:
    st.plotly_chart(rsi_gauge(rsi_now), use_container_width=True)

st.warning("⚠️ DISCLAIMER: For educational purposes only. Never make real investment decisions based solely on ML signals. Always consult a certified financial advisor.")

# ── Performance Metrics ──
st.markdown("---")
st.markdown("### ◈ MODEL PERFORMANCE — HELD-OUT TEST DATA")

p1, p2, p3, p4 = st.columns(4)
p1.metric("TEST ACCURACY",  f"{acc*100:.1f}%")
p2.metric("ROC-AUC",        f"{auc:.3f}" if not math.isnan(auc) else "N/A")
p3.metric("EPOCHS TRAINED", str(actual_epochs))
p4.metric("TEST SAMPLES",   f"{len(X_test):,}")

with st.expander("◈ FULL CLASSIFICATION REPORT"):
    rpt = classification_report(y_test, y_pred,
                                target_names=["SELL/HOLD", "BUY"],
                                output_dict=True)
    st.dataframe(pd.DataFrame(rpt).T.style.format("{:.3f}"), use_container_width=True)

st.markdown("---")
col_l, col_r = st.columns(2)
with col_l:
    st.markdown("### ◈ TRAINING LOSS CURVE")
    st.plotly_chart(loss_fig(history), use_container_width=True)
    st.caption("Val loss rising while train loss falls = overfitting → increase Dropout.")
with col_r:
    st.markdown("### ◈ BUY PROBABILITY — TEST PERIOD")
    st.plotly_chart(signal_history_fig(test_dates[:len(y_prob)], y_prob), use_container_width=True)
    st.caption("Green >0.65 = BUY · Yellow 0.35–0.65 = HOLD · Red <0.35 = SELL")

# ── Signals on Price ──
st.markdown("---")
st.markdown("### ◈ BUY / SELL SIGNALS ON PRICE CHART")
test_close = df["Close"].reindex(test_dates[:len(y_prob)]).dropna()
if not test_close.empty:
    st.plotly_chart(signals_on_price_fig(test_close, y_pred, y_prob), use_container_width=True)
    st.caption("▲ Green triangle = BUY signal  ·  ▼ Red triangle = SELL/HOLD signal")

# ── Tuning Guide ──
st.markdown("---")
st.markdown("### ◈ TUNING GUIDE")
st.markdown(f"""
| PARAMETER | CURRENT VALUE | EFFECT IF INCREASED |
|---|---|---|
| Lookback Window | {window_size} days | More historical context; slower training |
| LSTM Layer 1 | {lstm_units_1} units | More pattern capacity; risk of overfitting |
| LSTM Layer 2 | {lstm_units_2} units | Same as above |
| Dropout Rate | {dropout_rate} | More regularization; use 0.3–0.4 if overfitting |
| Forecast Horizon | {forecast_days} days | Longer = harder to predict |
| BUY Threshold | {return_threshold*100:.1f}% | Fewer but higher-conviction BUY signals |
""")

st.markdown("---")
st.markdown("""
<div style='font-family:Share Tech Mono,monospace;font-size:0.72rem;color:#442222;
            text-align:center;letter-spacing:0.1em;padding:0.5rem;'>
    BUILT WITH  tf.keras.layers.LSTM  ·  yfinance  ·  Streamlit  ·  Plotly  ·  RED & BLACK PRO
</div>
""", unsafe_allow_html=True)
