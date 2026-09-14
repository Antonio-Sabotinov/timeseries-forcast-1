"""Normalize SEC XBRL facts into compact, provenance-aware reports."""

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import pandas as pd

from .sec_client import get_company_facts

FACT_TAGS = {
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "debt": ("LongTermDebtAndFinanceLeaseObligationsCurrent", "LongTermDebtNoncurrent"),
}


@dataclass(frozen=True)
class FundamentalReport:
    ticker: str
    cik: str
    company_name: str
    as_of_date: str
    annual_metrics: list[dict[str, Any]]
    quarterly_metrics: list[dict[str, Any]]
    quality_flags: list[str]
    sources: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _facts_for_tag(facts: dict, tag: str) -> list[dict]:
    for taxonomy in ("us-gaap", "dei"):
        units = facts.get("facts", {}).get(taxonomy, {}).get(tag, {}).get("units", {})
        for unit_values in units.values():
            return unit_values
    return []


def _period_type(item: dict) -> str | None:
    if not item.get("end"):
        return None
    if not item.get("start"):
        return "instant"
    days = (pd.Timestamp(item["end"]) - pd.Timestamp(item["start"])).days
    if 300 <= days <= 400:
        return "annual"
    if 70 <= days <= 110:
        return "quarterly"
    return None


def _select_metric_rows(facts: dict, as_of: str) -> list[dict]:
    rows = []
    for metric, tags in FACT_TAGS.items():
        for tag in tags:
            candidates = [
                {**item, "metric": metric, "tag": tag, "period_type": _period_type(item)}
                for item in _facts_for_tag(facts, tag)
                if item.get("form") in {"10-K", "10-Q", "10-K/A", "10-Q/A"}
                and item.get("filed", "9999-12-31") <= as_of
                and _period_type(item) in {"annual", "quarterly", "instant"}
            ]
            if candidates:
                rows.extend(candidates)
                break
    return rows


def normalize_companyfacts(facts: dict, ticker: str, *, as_of: str | None = None) -> FundamentalReport:
    """Create annual and quarterly metric rows from a raw companyfacts payload."""
    as_of = as_of or date.today().isoformat()
    rows = _select_metric_rows(facts, as_of)
    grouped: dict[tuple[str, str], dict] = {}
    sources: dict[str, dict[str, str]] = {}

    for item in rows:
        period_type = item["period_type"]
        key = (item["end"], period_type)
        current = grouped.setdefault(key, {"period_end": item["end"], "period_type": period_type})
        if item["metric"] not in current or item.get("filed", "") > current[item["metric"]].get("filed", ""):
            current[item["metric"]] = item
        accession = item.get("accn")
        if accession:
            sources[accession] = {
                "accession_number": accession,
                "form": item.get("form", ""),
                "filed": item.get("filed", ""),
            }

    def make_rows(period_type: str) -> list[dict]:
        output = []
        for row in sorted((v for (end, kind), v in grouped.items() if kind == period_type), key=lambda v: v["period_end"]):
            result = {"period_end": row["period_end"], "period_type": period_type}
            for metric in FACT_TAGS:
                item = row.get(metric)
                result[metric] = item["val"] if item else None
                if item:
                    result[f"{metric}_source"] = item.get("accn")
            output.append(result)
        frame = pd.DataFrame(output)
        if frame.empty:
            return []
        for metric in FACT_TAGS:
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        frame["revenue_growth"] = frame["revenue"].pct_change()
        frame["gross_margin"] = frame["gross_profit"] / frame["revenue"]
        frame["operating_margin"] = frame["operating_income"] / frame["revenue"]
        frame["cash_flow_margin"] = frame["operating_cash_flow"] / frame["revenue"]
        frame["free_cash_flow"] = frame["operating_cash_flow"] - frame["capex"].abs()
        return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")

    annual = make_rows("annual")
    quarterly = make_rows("quarterly")
    flags = []
    if not annual and not quarterly:
        flags.append("No comparable 10-K or 10-Q duration facts found.")
    for metric in FACT_TAGS:
        if not any(row.get(metric) is not None for row in annual + quarterly):
            flags.append(f"Missing metric: {metric}.")
    return FundamentalReport(
        ticker=ticker.upper(), cik=str(facts.get("cik", "")).zfill(10),
        company_name=facts.get("entityName", ""), as_of_date=as_of,
        annual_metrics=annual, quarterly_metrics=quarterly,
        quality_flags=flags, sources=list(sources.values()),
    )


def evaluate_ticker(ticker: str, *, as_of: str | None = None, refresh: bool = False) -> dict:
    """Fetch and normalize one ticker's SEC fundamentals."""
    return normalize_companyfacts(get_company_facts(ticker, refresh=refresh), ticker, as_of=as_of).to_dict()


def evaluate_portfolio(tickers: list[str], *, as_of: str | None = None, refresh: bool = False) -> dict:
    """Evaluate each ticker independently and return a portfolio report."""
    reports = [evaluate_ticker(ticker, as_of=as_of, refresh=refresh) for ticker in dict.fromkeys(tickers)]
    return {"as_of_date": as_of or date.today().isoformat(), "companies": reports}