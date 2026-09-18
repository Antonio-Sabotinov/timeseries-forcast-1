"""Send time-series data from ``get_data()`` to a local Ollama model.

Usage in a notebook:

    from interactive_chart import get_data
    from llm_analysis import analyze_dataframe

    df = get_data("META", start="2024-01-01", indicators=["sma", "rsi", "macd"])
    answer = analyze_dataframe(df, "What is the current trend? Are there warning signs?")
    print(answer)

Requirements:
    uv pip install openai
    Ollama must be running (ollama serve) with a model available (ollama pull llama3.2)
"""

import pandas as pd
from openai import OpenAI

from core.fundamentals.context import build_fundamental_context

CLIENT = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def summarize_dataframe(df: pd.DataFrame, n_recent: int = 15) -> str:
    """
    Convert a DataFrame into a compact text summary for an LLM prompt.

    Raw DataFrames with hundreds or thousands of rows should not be sent directly:
    - They can quickly exceed the context window.
    - LLMs are less reliable at precise arithmetic over raw tables.
    - A summary with relevant metrics produces better analyses.
    """
    numeric_cols = df.select_dtypes(include="number").columns

    stats = df[numeric_cols].describe().round(2)
    recent = df.tail(n_recent).copy()

    if "ds" in recent.columns:
        recent["ds"] = pd.to_datetime(recent["ds"]).dt.strftime("%Y-%m-%d")

    summary = (
        f"Period: {df['ds'].min()} to {df['ds'].max()}\n"
        f"Number of data points: {len(df)}\n\n"
        f"=== Statistical metrics (full period) ===\n"
        f"{stats.to_string()}\n\n"
        f"=== Latest {n_recent} data points ===\n"
        f"{recent.to_string(index=False)}"
    )
    return summary


def analyze_dataframe(
    df: pd.DataFrame,
    question: str,
    model: str = "llama3.2",
    n_recent: int = 15,
    system_prompt: str | None = None,
) -> str:
    """
    Send a time-series summary and a question to Ollama.

    df:        DataFrame from get_data() with OHLCV and indicator columns
    question:  Your specific question, such as "What is the trend?"
    model:     Ollama model name; it must be pulled beforehand
    n_recent:  Number of recent rows to include in detail
    """
    data_summary = summarize_dataframe(df, n_recent=n_recent)

    default_prompt = (
        "You are an analyst of financial time series. You receive statistical "
        "metrics and the latest data points for a stock, including technical "
        "indicators. Answer precisely and support your assessment with concrete "
        "numbers. Do not provide investment advice; give a factual, scientific "
        "interpretation of the data."
    )

    response = CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt or default_prompt},
            {"role": "user", "content": f"{data_summary}\n\nFrage: {question}"},
        ],
    )
    return response.choices[0].message.content


def analyze_fundamental_report(
    report: dict,
    question: str,
    model: str = "llama3.2",
    system_prompt: str | None = None,
) -> str:
    """Send a bounded SEC fundamentals report to Ollama."""
    context = build_fundamental_context(report)
    default_prompt = (
        "You are an analyst of company fundamentals. Use only the provided SEC "
        "data and sources. Separate observed facts from interpretations, mention "
        "data-quality warnings, and do not provide investment advice. Do not "
        "invent missing values."
    )
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt or default_prompt},
            {"role": "user", "content": f"SEC-Fundamentalkontext:\n{context}\n\nFrage: {question}"},
        ],
    )
    return response.choices[0].message.content