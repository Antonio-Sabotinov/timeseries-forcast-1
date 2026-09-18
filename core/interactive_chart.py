"""Interactive TradingView-style candlestick chart with Plotly for notebooks.

Usage in a notebook:

    from interactive_chart import build_chart

    fig = build_chart("META", interval="1d", start="2024-01-01",
                       indicators=["sma", "rsi", "macd"])
    fig.show()

Requirements:
    uv pip install yfinance plotly pandas
"""

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

VALID_INDICATORS = {"sma", "ema", "bbands", "rsi", "macd"}


def load_data(ticker: str, interval: str, start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(ticker, interval=interval, start=start, end=end,
                      auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {ticker} from {start} to {end} (interval {interval}).")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else "Datetime"
    df = df.rename(columns={date_col: "ds"})
    return df


def add_sma(df, windows=(20, 50)):
    cols = []
    for w in windows:
        col = f"SMA_{w}"
        df[col] = df["Close"].rolling(w).mean()
        cols.append(col)
    return cols


def add_ema(df, windows=(12, 26)):
    cols = []
    for w in windows:
        col = f"EMA_{w}"
        df[col] = df["Close"].ewm(span=w, adjust=False).mean()
        cols.append(col)
    return cols


def add_bbands(df, window=20, n_std=2.0):
    mid = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    df["BB_MID"] = mid
    df["BB_UPPER"] = mid + n_std * std
    df["BB_LOWER"] = mid - n_std * std
    return ["BB_UPPER", "BB_MID", "BB_LOWER"]


def add_rsi(df, window=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))


def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]


def get_data(
    ticker: str,
    interval: str = "1d",
    start: str = "2023-01-01",
    end: str | None = None,
    indicators: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return raw and indicator data as a DataFrame without building a chart.
    This is the data interface for algorithms, backtesting, and ML.

    Spalten: ds, Open, High, Low, Close, Volume, + je nach indicators:
             SMA_20, SMA_50, EMA_12, EMA_26, BB_UPPER/MID/LOWER, RSI, MACD, MACD_SIGNAL, MACD_HIST

    Example:
        df = get_data("META", start="2024-01-01", indicators=["sma", "rsi", "macd"])
        df.to_csv("meta_data.csv", index=False)   # optional als Datei speichern
    """
    indicators = [i.lower() for i in (indicators or [])]
    unknown = set(indicators) - VALID_INDICATORS
    if unknown:
        raise ValueError(f"Unbekannte Indikatoren: {unknown}. Erlaubt: {VALID_INDICATORS}")

    df = load_data(ticker, interval, start, end)

    if "sma" in indicators:
        add_sma(df)
    if "ema" in indicators:
        add_ema(df)
    if "bbands" in indicators:
        add_bbands(df)
    if "rsi" in indicators:
        add_rsi(df)
    if "macd" in indicators:
        add_macd(df)

    return df


def build_chart(
    ticker: str,
    interval: str = "1d",
    start: str = "2023-01-01",
    end: str | None = None,
    indicators: list[str] | None = None,
) -> go.Figure:
    """
    Build and return an interactive candlestick chart.
    In a notebook: ``fig = build_chart(...); fig.show()``

    Use ``get_data()`` when raw data is needed as a DataFrame.

    ticker:     e.g. "META"
    interval:   "1d", "1h", "1wk", ...
    start/end:  "YYYY-MM-DD"
    indicators: list of "sma", "ema", "bbands", "rsi", "macd"
    """
    df = get_data(ticker, interval=interval, start=start, end=end, indicators=indicators)
    indicators = [i.lower() for i in (indicators or [])]

    overlay_cols = []
    if "sma" in indicators:
        overlay_cols += [c for c in df.columns if c.startswith("SMA_")]
    if "ema" in indicators:
        overlay_cols += [c for c in df.columns if c.startswith("EMA_")]
    if "bbands" in indicators:
        overlay_cols += ["BB_UPPER", "BB_MID", "BB_LOWER"]

    row_specs = [("price", 0.55), ("volume", 0.15)]
    if "rsi" in indicators:
        row_specs.append(("rsi", 0.15))
    if "macd" in indicators:
        row_specs.append(("macd", 0.15))

    n_rows = len(row_specs)
    row_heights = [h for _, h in row_specs]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=[f"{ticker} ({interval})"] + [""] * (n_rows - 1),
    )
    row_map = {name: i + 1 for i, (name, _) in enumerate(row_specs)}

    fig.add_trace(
        go.Candlestick(
            x=df["ds"], open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name=ticker, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ),
        row=row_map["price"], col=1,
    )

    palette = ["#f5a623", "#7b61ff", "#00bcd4", "#ff6b6b", "#4caf50"]
    for i, col in enumerate(overlay_cols):
        fig.add_trace(
            go.Scatter(x=df["ds"], y=df[col], name=col, mode="lines",
                       line=dict(width=1.3, color=palette[i % len(palette)])),
            row=row_map["price"], col=1,
        )

    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(
        go.Bar(x=df["ds"], y=df["Volume"], name="Volume", marker_color=colors, showlegend=False),
        row=row_map["volume"], col=1,
    )

    if "rsi" in indicators:
        fig.add_trace(
            go.Scatter(x=df["ds"], y=df["RSI"], name="RSI(14)", line=dict(color="#7b61ff")),
            row=row_map["rsi"], col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="grey", row=row_map["rsi"], col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="grey", row=row_map["rsi"], col=1)
        fig.update_yaxes(range=[0, 100], row=row_map["rsi"], col=1, title_text="RSI")

    if "macd" in indicators:
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_HIST"]]
        fig.add_trace(
            go.Bar(x=df["ds"], y=df["MACD_HIST"], name="MACD Hist",
                   marker_color=hist_colors, showlegend=False),
            row=row_map["macd"], col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["ds"], y=df["MACD"], name="MACD", line=dict(color="#f5a623", width=1.3)),
            row=row_map["macd"], col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["ds"], y=df["MACD_SIGNAL"], name="Signal", line=dict(color="#00bcd4", width=1.3)),
            row=row_map["macd"], col=1,
        )
        fig.update_yaxes(title_text="MACD", row=row_map["macd"], col=1)

    fig.update_yaxes(title_text="Price", row=row_map["price"], col=1)
    fig.update_yaxes(title_text="Volume", row=row_map["volume"], col=1)

    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        ),
        row=1, col=1,
    )
    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.04, row=n_rows, col=1)

    fig.update_layout(
        height=750 + 130 * (n_rows - 2),
        template="plotly_dark",
        title=f"{ticker} — {interval} — {start} bis {end or 'heute'}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=40),
        hovermode="x unified",
    )

    return fig