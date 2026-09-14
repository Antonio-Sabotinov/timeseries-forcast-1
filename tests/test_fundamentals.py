from core.fundamentals.reports import normalize_companyfacts


def _fact(value, start, end, accn="0000000000-26-000001"):
    return {"val": value, "start": start, "end": end, "filed": "2026-05-01", "form": "10-K", "accn": accn}


def test_normalizes_annual_metrics_and_preserves_provenance():
    payload = {
        "cik": 320193,
        "entityName": "Example Inc.",
        "facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [_fact(100, "2025-01-01", "2025-12-31"), _fact(120, "2026-01-01", "2026-12-31", "0000000000-27-000001")]}},
            "OperatingIncomeLoss": {"units": {"USD": [_fact(20, "2025-01-01", "2025-12-31"), _fact(30, "2026-01-01", "2026-12-31", "0000000000-27-000001")]}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [_fact(25, "2025-01-01", "2025-12-31"), _fact(35, "2026-01-01", "2026-12-31", "0000000000-27-000001")]}},
            "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [_fact(5, "2025-01-01", "2025-12-31"), _fact(7, "2026-01-01", "2026-12-31", "0000000000-27-000001")]}},
        }},
    }
    report = normalize_companyfacts(payload, "EXM", as_of="2026-12-31").to_dict()
    row = report["annual_metrics"][1]
    assert row["revenue"] == 120
    assert row["operating_margin"] == 0.25
    assert row["free_cash_flow"] == 28
    assert row["revenue_source"] == "0000000000-27-000001"
    assert report["cik"] == "0000320193"


def test_missing_capex_is_reported_as_none():
    payload = {
        "cik": 1,
        "entityName": "Example Inc.",
        "facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [_fact(100, "2025-01-01", "2025-12-31")]}},
            "NetCashProvidedByUsedInOperatingActivities": {
                "units": {"USD": [_fact(20, "2025-01-01", "2025-12-31")]},
            },
        }},
    }
    row = normalize_companyfacts(payload, "EXM").to_dict()["annual_metrics"][0]
    assert row["capex"] is None
    assert row["free_cash_flow"] is None