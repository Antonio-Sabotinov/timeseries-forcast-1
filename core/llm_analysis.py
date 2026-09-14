"""
Uebergibt Zeitreihen-Daten (aus get_data()) an ein lokales Ollama-Modell zur Analyse.

Nutzung im Notebook:

    from interactive_chart import get_data
    from llm_analysis import analyze_dataframe

    df = get_data("META", start="2024-01-01", indicators=["sma", "rsi", "macd"])
    antwort = analyze_dataframe(df, "Wie ist der aktuelle Trend? Gibt es Warnsignale?")
    print(antwort)

Voraussetzungen:
    uv pip install openai
    ollama muss laufen (ollama serve) und ein Modell geladen haben (ollama pull llama3.2)
"""

import pandas as pd
from openai import OpenAI

from core.fundamentals.context import build_fundamental_context

CLIENT = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def summarize_dataframe(df: pd.DataFrame, n_recent: int = 15) -> str:
    """
    Wandelt einen DataFrame in eine kompakte Text-Zusammenfassung um,
    die gut in einen LLM-Prompt passt: Kennzahlen + die letzten n_recent Zeilen.

    Rohe DataFrames (hunderte/tausende Zeilen) sollte man NICHT direkt reingeben:
    - Sprengt schnell den Kontext
    - LLMs sind bei Rohzahlen-Tabellen schlecht in praeziser Arithmetik
    - Eine Zusammenfassung + relevante Kennzahlen fuehrt zu besseren Analysen
    """
    numeric_cols = df.select_dtypes(include="number").columns

    stats = df[numeric_cols].describe().round(2)
    recent = df.tail(n_recent).copy()

    if "ds" in recent.columns:
        recent["ds"] = pd.to_datetime(recent["ds"]).dt.strftime("%Y-%m-%d")

    summary = (
        f"Zeitraum: {df['ds'].min()} bis {df['ds'].max()}\n"
        f"Anzahl Datenpunkte: {len(df)}\n\n"
        f"=== Statistische Kennzahlen (gesamter Zeitraum) ===\n"
        f"{stats.to_string()}\n\n"
        f"=== Letzte {n_recent} Datenpunkte ===\n"
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
    Schickt eine Zusammenfassung der Zeitreihen-Daten + eine Frage an Ollama.

    df:        DataFrame aus get_data() (mit OHLCV + Indikator-Spalten)
    question:  Deine konkrete Frage, z.B. "Wie ist der Trend?" oder
               "Gibt es Anzeichen fuer eine Trendumkehr?"
    model:     Ollama-Modellname (muss vorher per `ollama pull` geladen sein)
    n_recent:  Wie viele der juengsten Zeilen im Detail mitgegeben werden
    """
    data_summary = summarize_dataframe(df, n_recent=n_recent)

    default_prompt = (
        "Du bist ein Analyst fuer Finanz-Zeitreihen. Du bekommst statistische "
        "Kennzahlen und die juengsten Datenpunkte einer Aktie inkl. technischer "
        "Indikatoren. Antworte praezise und begruende deine Einschaetzung anhand "
        "der konkreten Zahlen. Gib keine Anlageempfehlung, sondern eine "
        "sachliche/wissenschaftliche Einordnung der Daten."
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
        "Du bist ein Analyst fuer Unternehmensfundamentaldaten. Nutze nur die "
        "bereitgestellten SEC-Daten und Quellen. Trenne beobachtete Fakten von "
        "Interpretationen, nenne Datenqualitaetswarnungen und gib keine "
        "Anlageempfehlung. Erfinde keine fehlenden Werte."
    )
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt or default_prompt},
            {"role": "user", "content": f"SEC-Fundamentalkontext:\n{context}\n\nFrage: {question}"},
        ],
    )
    return response.choices[0].message.content