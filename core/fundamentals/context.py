"""Build bounded, evidence-linked context for an LLM or agent."""

import json
from typing import Any


def build_fundamental_context(report: dict[str, Any]) -> str:
    """Serialize a report without sending raw SEC payloads to the model."""
    if "companies" in report:
        body = {
            "as_of_date": report.get("as_of_date"),
            "companies": [
                {
                    "ticker": company["ticker"],
                    "company_name": company["company_name"],
                    "annual_metrics": company["annual_metrics"][-5:],
                    "quarterly_metrics": company["quarterly_metrics"][-8:],
                    "quality_flags": company["quality_flags"],
                    "sources": company["sources"],
                }
                for company in report["companies"]
            ],
        }
    else:
        body = report
    return json.dumps(body, indent=2, default=str)