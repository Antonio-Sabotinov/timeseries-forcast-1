"""Run an end-to-end IREN market and fundamentals analysis with Ollama."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

from core.edgar_poller import get_filing_history, get_saved_documents, save_filing_reports
from core.fundamentals import evaluate_ticker
from core.interactive_chart import VALID_INDICATORS, get_data

DEFAULT_TICKER = "IREN"
DEFAULT_START = "2025-01-01"
DEFAULT_MODEL = "llama3.2"
DEFAULT_OUTPUT = Path("data/processed/iren_analysis.json")


def _filter_rows_from_start(rows: list[dict[str, Any]], start: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("period_end", "") >= start]


def _market_context(frame: pd.DataFrame, recent_rows: int = 20) -> dict[str, Any]:
    numeric = frame.select_dtypes(include="number")
    stats = numeric.describe().round(4).to_dict()
    recent = frame.tail(recent_rows).copy()
    recent["ds"] = pd.to_datetime(recent["ds"]).dt.strftime("%Y-%m-%d")
    recent = recent.astype(object).where(pd.notna(recent), None)
    return {
        "period": {
            "start": str(frame["ds"].min().date()),
            "end": str(frame["ds"].max().date()),
        },
        "rows": len(frame),
        "indicators": sorted(VALID_INDICATORS),
        "statistics": stats,
        "recent_rows": recent.to_dict(orient="records"),
    }


def _chunk_text(text: str, chunk_size: int = 1800, overlap: int = 250) -> list[str]:
    words = text.split()
    step = max(1, chunk_size - overlap)
    return [" ".join(words[index : index + chunk_size]) for index in range(0, len(words), step)]


def retrieve_filing_context(
    ticker: str,
    query: str,
    start: str,
    top_k: int = 6,
) -> list[dict[str, str]]:
    """Retrieve relevant chunks from locally saved SEC filing text."""
    history_by_accession = {
        item["accession_number"]: item for item in get_filing_history(ticker)
    }
    documents = [
        document
        for document in get_saved_documents(ticker)
        if (
            history_by_accession.get(document.get("accession_number", ""), {}).get(
                "report_date", ""
            )
            >= start
            or history_by_accession.get(document.get("accession_number", ""), {}).get(
                "filing_date", ""
            )
            >= start
        )
    ]
    chunks: list[dict[str, str]] = []
    for document in documents:
        text_path = Path(document["text_path"])
        if not text_path.exists():
            continue
        for index, text in enumerate(_chunk_text(text_path.read_text(encoding="utf-8"))):
            chunks.append(
                {
                    "text": text,
                    "source": document.get("source_url", ""),
                    "accession_number": document.get("accession_number", ""),
                    "form": document.get("form", ""),
                    "chunk": str(index),
                }
            )

    if not chunks:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", max_features=25000)
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
    ranked = sorted(zip(scores, chunks), key=lambda item: item[0], reverse=True)
    return [
        {**chunk, "relevance": f"{score:.4f}"}
        for score, chunk in ranked[:top_k]
        # if score > 0
    ]


def _build_prompt(
    ticker: str,
    start: str,
    end: str,
    market: dict[str, Any],
    fundamentals: dict[str, Any],
    filing_context: list[dict[str, str]],
) -> str:
    return f"""Analyze {ticker} from {start} through {end} using only the supplied evidence.

Market data and technical indicators:
{json.dumps(market, indent=2, default=str)}

SEC fundamentals:
{json.dumps(fundamentals, indent=2, default=str)}

Retrieved SEC filing excerpts:
{json.dumps(filing_context, indent=2, default=str)}

Write a concise but complete analysis with these sections:
1. Executive summary
2. Price and technical-indicator trends
3. Fundamental trends
4. Evidence from SEC filings, citing accession numbers when present
5. Conflicts, missing data, and limitations
6. Conclusion without investment advice

Separate measured observations from interpretations. Do not invent values or facts."""


def analyze_iren(
    *,
    ticker: str = DEFAULT_TICKER,
    start: str = DEFAULT_START,
    model: str = DEFAULT_MODEL,
    output_path: Path | str = DEFAULT_OUTPUT,
    refresh_sec: bool = False,
) -> dict[str, Any]:
    """Run the complete market, SEC, RAG, and LLM workflow."""
    today = date.today()
    end = today.isoformat()
    market_end = (today + timedelta(days=1)).isoformat()

    market_frame = get_data(
        ticker,
        interval="1d",
        start=start,
        end=market_end,
        indicators=sorted(VALID_INDICATORS),
    )
    fundamental_report = evaluate_ticker(ticker, as_of=end, refresh=refresh_sec)
    fundamental_report["annual_metrics"] = _filter_rows_from_start(
        fundamental_report["annual_metrics"], start
    )
    fundamental_report["quarterly_metrics"] = _filter_rows_from_start(
        fundamental_report["quarterly_metrics"], start
    )

    save_filing_reports(ticker, form_types=("10-K", "10-Q"), limit=50)
    query = (
        f"{ticker} revenue profitability cash flow debt bitcoin mining operations "
        "guidance risks outlook capacity expansion"
    )
    filing_context = retrieve_filing_context(ticker, query, start)
    market_context = _market_context(market_frame)
    prompt = _build_prompt(
        ticker,
        start,
        end,
        market_context,
        fundamental_report,
        filing_context,
    )

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful financial-data analyst. Use only the supplied "
                    "market, SEC, and retrieved filing evidence. Do not provide investment advice."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    summary = response.choices[0].message.content or ""

    result = {
        "ticker": ticker.upper(),
        "start": start,
        "end": end,
        "model": model,
        "market": market_context,
        "fundamentals": fundamental_report,
        "retrieved_filings": filing_context,
        "summary": summary,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


if __name__ == "__main__":
    analysis = analyze_iren()
    print(analysis["summary"])
