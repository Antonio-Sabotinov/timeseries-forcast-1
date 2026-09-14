# META Stock Forecast & LLM Analysis Playground

🇩🇪 [Deutsch](#-deutsch) | 🇬🇧 [English](#-english)

---

## 🇩🇪 Deutsch

### Über das Projekt

Dies ist eine **Testumgebung** zum Vergleich verschiedener Open-Source-Modelle (via [Ollama](https://ollama.com)) für die Analyse und Vorhersage von Kursentwicklungen bei Zeitreihen. Die Kursdaten (DataFrames) werden über die Bibliothek [`yfinance`](https://pypi.org/project/yfinance/) extrahiert.

Das Projekt kombiniert:
- **Statistisches Forecasting** – Walk-Forward-Vergleich verschiedener Modelle (Naive, AutoARIMA, LightGBM)
- **Interaktive Visualisierung** – TradingView-ähnliche Candlestick-Charts mit Indikatoren (SMA, EMA, Bollinger Bands, RSI, MACD)
- **LLM-gestützte Analyse** – lokale Auswertung der Kursdaten durch Open-Source-LLMs über eine lokal laufende Ollama-Instanz

> Alle verwendeten Ressourcen, Bibliotheken und Modelle sind öffentlich zugänglich (Stand: 02.08.2026).

### Projektstruktur

```
├── utils/
│   ├── meta_forecast_compare.py   # Walk-Forward Modellvergleich (Naive, AutoARIMA, LightGBM)
│   ├── interactive_chart.py       # Interaktive Candlestick-Charts (Plotly)
│   └── llm_analysis.py            # DataFrame-Zusammenfassung + LLM-Analyse via Ollama
├── notebooks/                     # Jupyter Notebooks zur Nutzung der Module
└── pyproject.toml                 # Editable Package Install
```

### Tech Stack

| Bereich | Tools |
|---|---|
| Daten | `yfinance` |
| Forecasting | `statsforecast`, `lightgbm` |
| Visualisierung | `Plotly` |
| LLM-Integration | `Ollama` (OpenAI-kompatible API) |
| Environment | Python, `uv`, Jupyter |

### Getestete LLM-Modelle

- `phi4-mini:latest`
- `deepseek-r1:8b`
- `llama3.2:latest`

Alle Modelle laufen **lokal via Ollama** (CPU-Inferenz), keine Cloud-API-Kosten.

Voraussetzung: [Ollama](https://ollama.com) lokal installiert, gewünschte Modelle via `ollama pull <modell>` geladen.


### ⚠️ Hinweis / Disclaimer

Dieses Projekt dient **ausschließlich Test- und Lernzwecken**. Die verwendeten Kurs- und Fundamentaldaten werden aus öffentlich verfügbaren Online-Quellen bezogen und sind möglicherweise nicht vollständig, aktuell oder dauerhaft verfügbar. Das Projekt stellt **keine Finanzberatung** dar und sollte nicht als Grundlage für reale Anlageentscheidungen verwendet werden. Alle Vorhersagen basieren auf experimentellen Modellen ohne Garantie auf Richtigkeit.

---

## 🇬🇧 English

### About the Project

This is a **testing environment** for comparing different open-source models (via [Ollama](https://ollama.com)) for analyzing and forecasting price movements in time series data. Price data (DataFrames) is extracted using the [`yfinance`](https://pypi.org/project/yfinance/) library.

The project combines:
- **Statistical forecasting** – walk-forward comparison of multiple models (Naive, AutoARIMA, LightGBM)
- **Interactive visualization** – TradingView-style candlestick charts with configurable indicators (SMA, EMA, Bollinger Bands, RSI, MACD)
- **LLM-assisted analysis** – local evaluation of price data using open-source LLMs via a locally running Ollama instance

> All resources, libraries, and models used are publicly available (as of 2026-08-02).

### Project Structure

```
├── utils/
│   ├── meta_forecast_compare.py   # Walk-forward model comparison (Naive, AutoARIMA, LightGBM)
│   ├── interactive_chart.py       # Interactive candlestick charts (Plotly)
│   └── llm_analysis.py            # DataFrame summarization + LLM analysis via Ollama
├── notebooks/                     # Jupyter notebooks for using the modules
└── pyproject.toml                 # Editable package install
```

### Tech Stack

| Area | Tools |
|---|---|
| Data | `yfinance` |
| Forecasting | `statsforecast`, `lightgbm` |
| Visualization | `Plotly` |
| LLM integration | `Ollama` (OpenAI-compatible API) |
| Environment | Python, `uv`, Jupyter |

### Tested LLM Models

- `phi4-mini:latest`
- `deepseek-r1:8b`
- `llama3.2:latest`

All models run **locally via Ollama** (CPU inference), no cloud API costs involved.

Requirement: [Ollama](https://ollama.com) installed locally, desired models pulled via `ollama pull <model>`.

### SEC fundamentals

The first fundamentals slice is available through importable APIs. It fetches
SEC `companyfacts`, caches raw responses locally, normalizes a small core metric
set, preserves filing accession numbers, and builds bounded context for Ollama.

```python
from core.fundamentals import evaluate_portfolio, evaluate_ticker
from core.llm_analysis import analyze_fundamental_report

company = evaluate_ticker("AAPL")
portfolio = evaluate_portfolio(["AAPL", "MSFT", "NVDA"])
answer = analyze_fundamental_report(
	company,
	"Which fundamental trends and data-quality warnings are visible?",
)
```

SEC requests require a descriptive User-Agent with a reachable contact address.
The implementation uses comparable 10-K and 10-Q duration facts and reports
unsupported or missing metrics as quality flags instead of silently estimating.

### ⚠️ Disclaimer

This project is intended **for testing and learning purposes only**. The market and fundamental data used are obtained from publicly available online sources and may be incomplete, outdated, or unavailable at any time. This project does **not constitute financial advice** and should not be used as a basis for real investment decisions. All predictions are based on experimental models with no guarantee of accuracy.

---

**License:** MIT (or your preferred license)