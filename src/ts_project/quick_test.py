# src/ts_project/quick_test.py
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

dates = pd.date_range("2023-01-01", periods=100, freq="D")
df = pd.DataFrame({
    "unique_id": "series_1",
    "ds": dates,
    "y": np.sin(np.arange(100) / 5) * 10 + 50 + np.random.normal(0, 1, 100)
})

sf = StatsForecast(models=[AutoARIMA()], freq="D")
sf.fit(df)
forecast = sf.predict(h=14)
print(forecast)