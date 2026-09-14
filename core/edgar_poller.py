"""
EDGAR-Poller mit persistiertem Zustand: Ticker -> CIK -> Filings (10-K/10-Q/8-K)
Notebook-importierbar, kein CLI. SQLite-basierte Historie fuer Diff-Erkennung
und spaetere Fundamentaldaten-Zeitreihenbetrachtung.
"""
import requests
import json
import sqlite3
import hashlib
from html.parser import HTMLParser
from pathlib import Path
from datetime import datetime, timezone

HEADERS = {"User-Agent": "Antonio Research antonio@example.com"}  # anpassen!
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
CACHE_PATH = Path.home() / ".cache" / "edgar_ticker_map.json"
DB_PATH = Path.home() / "projects" / "timeseries-project" / "data" / "edgar_filings.db"
DOCUMENTS_PATH = DB_PATH.parent / "filing_documents"


class _HTMLTextExtractor(HTMLParser):
    """Converts SEC HTML/XML documents into plain text for RAG ingestion."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip_content = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_content = True

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_content = False
        elif tag.lower() in {"p", "div", "br", "tr", "li", "section"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip_content:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts).strip()


def _document_to_text(content: bytes, encoding: str | None = None) -> str:
    """Extract plain text from an HTML, XML, or text filing."""
    decoded = content.decode(encoding or "utf-8", errors="replace")
    parser = _HTMLTextExtractor()
    parser.feed(decoded)
    return parser.text()


def _load_ticker_map(force_refresh: bool = False) -> dict:
    """Lädt Ticker->CIK Mapping, cached lokal (Datei ändert sich selten)."""
    if CACHE_PATH.exists() and not force_refresh:
        return json.loads(CACHE_PATH.read_text())

    resp = requests.get(TICKER_MAP_URL, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    raw = resp.json()  # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}

    ticker_to_cik = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in raw.values()}

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(ticker_to_cik))
    return ticker_to_cik


def get_cik(ticker: str) -> str:
    """Ticker (z.B. 'AAPL') -> zero-padded CIK-String."""
    ticker_map = _load_ticker_map()
    cik = ticker_map.get(ticker.upper())
    if cik is None:
        # evtl. neu gelisteter Ticker -> Cache erzwungen refreshen
        ticker_map = _load_ticker_map(force_refresh=True)
        cik = ticker_map.get(ticker.upper())
    if cik is None:
        raise ValueError(f"Ticker '{ticker}' nicht in SEC-Mapping gefunden.")
    return cik


def get_recent_filings(ticker: str, form_types: tuple = ("10-K", "10-Q", "8-K"), limit: int = 10) -> list[dict]:
    """
    Holt die letzten `limit` Filings eines Tickers, gefiltert nach form_types.
    Rückgabe: Liste von dicts mit form, filingDate, reportDate, accessionNumber, primaryDocument, url
    """
    cik = get_cik(ticker)
    resp = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    n = len(recent["form"])

    results = []
    for i in range(n):
        form = recent["form"][i]
        if form not in form_types:
            continue

        accession_nodash = recent["accessionNumber"][i].replace("-", "")
        primary_doc = recent["primaryDocument"][i]
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{accession_nodash}/{primary_doc}"
        )

        results.append({
            "ticker": ticker.upper(),
            "form": form,
            "filingDate": recent["filingDate"][i],
            "reportDate": recent["reportDate"][i],
            "accessionNumber": recent["accessionNumber"][i],
            "primaryDocument": primary_doc,
            "url": filing_url,
        })

        if len(results) >= limit:
            break

    return results


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS filings (
            accession_number TEXT PRIMARY KEY,
            ticker           TEXT NOT NULL,
            cik              TEXT NOT NULL,
            form             TEXT NOT NULL,
            filing_date      TEXT NOT NULL,
            report_date      TEXT NOT NULL,
            primary_document TEXT NOT NULL,
            url              TEXT NOT NULL,
            first_seen_at    TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON filings(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_report_date ON filings(ticker, report_date)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS filing_documents (
            accession_number TEXT PRIMARY KEY,
            ticker           TEXT NOT NULL,
            form             TEXT NOT NULL,
            source_url       TEXT NOT NULL,
            raw_path         TEXT NOT NULL,
            text_path        TEXT NOT NULL,
            sha256            TEXT NOT NULL,
            downloaded_at    TEXT NOT NULL
        )
    """)
    return conn


def save_filing_report(filing: dict, cik: str | None = None) -> dict | None:
    """Download one SEC filing and save its original and plain-text versions."""
    accession_number = filing["accessionNumber"]
    conn = _get_conn()

    existing = conn.execute(
        "SELECT 1 FROM filing_documents WHERE accession_number = ?",
        (accession_number,),
    ).fetchone()
    if existing is not None:
        conn.close()
        return None

    try:
        response = requests.get(filing["url"], headers=HEADERS, timeout=30)
        response.raise_for_status()
        content = response.content
        if not content:
            raise ValueError(f"SEC returned an empty document: {filing['url']}")

        accession_safe = accession_number.replace("-", "")
        primary_document = Path(filing["primaryDocument"]).name
        suffix = Path(primary_document).suffix.lower() or ".html"
        document_dir = DOCUMENTS_PATH / filing["ticker"] / accession_safe
        document_dir.mkdir(parents=True, exist_ok=True)

        raw_path = document_dir / f"{accession_safe}{suffix}"
        text_path = document_dir / f"{accession_safe}.txt"
        raw_tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
        text_tmp = text_path.with_suffix(".txt.tmp")

        raw_tmp.write_bytes(content)
        raw_tmp.replace(raw_path)
        text_tmp.write_text(
            _document_to_text(content, response.encoding),
            encoding="utf-8",
        )
        text_tmp.replace(text_path)

        downloaded_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT OR IGNORE INTO filings
            (accession_number, ticker, cik, form, filing_date, report_date,
             primary_document, url, first_seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filing["accessionNumber"], filing["ticker"], cik or "",
                filing["form"], filing["filingDate"], filing["reportDate"],
                filing["primaryDocument"], filing["url"], downloaded_at,
            ),
        )
        conn.execute(
            """
            INSERT INTO filing_documents
            (accession_number, ticker, form, source_url, raw_path, text_path,
             sha256, downloaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                accession_number, filing["ticker"], filing["form"],
                filing["url"], str(raw_path), str(text_path),
                hashlib.sha256(content).hexdigest(), downloaded_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        **filing,
        "raw_path": str(raw_path),
        "text_path": str(text_path),
        "downloaded_at": downloaded_at,
    }


def save_filing_reports(
    ticker: str,
    form_types: tuple = ("10-K", "10-Q", "8-K"),
    limit: int = 25,
) -> list[dict]:
    """Save recent filings not already stored; safe to call repeatedly."""
    cik = get_cik(ticker)
    candidates = get_recent_filings(ticker, form_types=form_types, limit=limit)
    saved = []

    for filing in candidates:
        document = save_filing_report(filing, cik=cik)
        if document is not None:
            saved.append(document)

    return saved


def get_saved_documents(ticker: str) -> list[dict]:
    """Return saved document metadata for a ticker."""
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM filing_documents WHERE ticker = ? ORDER BY downloaded_at ASC",
        (ticker.upper(),),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_new_filings(ticker: str, form_types: tuple = ("10-K", "10-Q", "8-K"), limit: int = 25) -> list[dict]:
    """
    Diff-Scan: holt aktuelle Filings von EDGAR, vergleicht gegen die lokale
    Historie und persistiert neue Eintraege. Gibt NUR die neuen Filings zurueck.
    Bereits bekannte Filings bleiben unveraendert in der DB (Historie bleibt erhalten).
    """
    cik = get_cik(ticker)
    candidates = get_recent_filings(ticker, form_types=form_types, limit=limit)

    conn = _get_conn()
    new_filings = []
    now = datetime.now(timezone.utc).isoformat()

    for f in candidates:
        existing = conn.execute(
            "SELECT 1 FROM filings WHERE accession_number = ?",
            (f["accessionNumber"],)
        ).fetchone()

        if existing is None:
            conn.execute(
                """INSERT INTO filings
                   (accession_number, ticker, cik, form, filing_date, report_date,
                    primary_document, url, first_seen_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (f["accessionNumber"], f["ticker"], cik, f["form"], f["filingDate"],
                 f["reportDate"], f["primaryDocument"], f["url"], now)
            )
            new_filings.append(f)

    conn.commit()
    conn.close()
    return new_filings


def get_filing_history(ticker: str, form_types: tuple | None = None) -> list[dict]:
    """
    Liest die persistierte Historie eines Tickers aus der lokalen DB
    (chronologisch nach report_date) - Basis fuer spaetere
    Fundamentaldaten-Zeitreihenbetrachtung.
    """
    conn = _get_conn()
    query = "SELECT * FROM filings WHERE ticker = ?"
    params = [ticker.upper()]

    if form_types:
        placeholders = ",".join("?" * len(form_types))
        query += f" AND form IN ({placeholders})"
        params.extend(form_types)

    query += " ORDER BY report_date ASC"

    conn.row_factory = sqlite3.Row
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    # Quick manual test - nicht als CLI gedacht, nur zur lokalen Verifikation
    print("--- Erster Lauf (alles ist neu) ---")
    for f in get_new_filings("AAPL"):
        print(f["form"], f["filingDate"], f["reportDate"], f["url"])

    print("--- Zweiter Lauf direkt danach (sollte leer sein) ---")
    for f in get_new_filings("AAPL"):
        print(f["form"], f["filingDate"], f["reportDate"], f["url"])

    print("--- Persistierte Historie ---")
    for f in get_filing_history("AAPL"):
        print(f["form"], f["report_date"], f["accession_number"])