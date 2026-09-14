"""Small SEC company-facts client with a local raw-response cache."""

import json
from pathlib import Path
from time import sleep

import requests

from core.edgar_poller import get_cik

SEC_HEADERS = {"User-Agent": "Antonio Timeseries Research antonio.sabotinov.AS@gmail.com"}
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
FACTS_CACHE_PATH = Path.home() / ".cache" / "timeseries-project" / "companyfacts"


def get_company_facts(ticker: str, *, refresh: bool = False) -> dict:
    """Return the SEC companyfacts payload for ``ticker``."""
    cik = get_cik(ticker)
    cache_path = FACTS_CACHE_PATH / f"CIK{cik}.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    FACTS_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        SEC_FACTS_URL.format(cik=cik), headers=SEC_HEADERS, timeout=30
    )
    response.raise_for_status()
    sleep(0.1)
    payload = response.json()
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload