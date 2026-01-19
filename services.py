import requests
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# FX / ML
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Prophet (vais descomentar no requirements.txt)
from prophet import Prophet


class PortfolioService:
    def __init__(self, timeout: int = 30, data_dir: Optional[str] = None):
        self.timeout = timeout
        self.data_dir = Path(data_dir) if data_dir else None

    # =========================================================
    # WEATHER (Open-Meteo)
    # =========================================================
    def geocode_city(self, name: str, count: int = 1, lang: str = "pt") -> dict:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": name, "count": count, "language": lang, "format": "json"}
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        js = r.json()
        res = js.get("results") or []
        if not res:
            return {}
        top = res[0]
        return {
            "name": top.get("name"),
            "country": top.get("country"),
            "latitude": float(top["latitude"]),
            "longitude": float(top["longitude"]),
            "timezone": top.get("timezone"),
        }

    def fetch_weather_forecast(
        self,
        lat: float,
        lon: float,
        days: int = 7,
        temp_unit: str = "celsius",
        wind_unit: str = "kmh",
        lang: str = "pt",
    ) -> dict:
        url = "https://api.open-meteo.com/v1/forecast"
        daily = [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ]
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(daily),
            "forecast_days": max(1, min(30, int(days))),
            "timezone": "auto",
            "temperature_unit": temp_unit,
            "wind_speed_unit": wind_unit,
            "past_days": 0,
            "language": lang,
        }
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # =========================================================
    # AMES
    # =========================================================
    @lru_cache(maxsize=1)
    def load_ames_data(self) -> pd.DataFrame:
        if not self.data_dir:
            raise RuntimeError("data_dir não configurado.")

        csv_path = self.data_dir / "ames.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} não encontrado.")

        df = pd.read_csv(csv_path)

        for c in df.columns:
            if c in {"faixa_preco", "bairro", "Neighborhood"}:
                continue
            if df[c].dtype == object:
                converted = pd.to_numeric(df[c], errors="coerce")
                if converted.notna().sum() > 0:
                    df[c] = converted

        return df

    def calcular_estatisticas_1d(self, serie: pd.Series) -> Dict[str, Any]:
        s = serie.dropna()

        if len(s) == 0:
            return {k: None for k in [
                "n", "media", "mediana", "moda", "minimo", "maximo",
                "variancia", "desvio_padrao", "q1", "q3", "iqr",
                "assimetria", "curtose", "stat_shapiro", "p_valor_shapiro"
            ]}

        sample = s if len(s) <= 5000 else s.sample(5000, random_state=42)

        try:
            stat_sh, p_valor = stats.shapiro(sample)
        except Exception:
            stat_sh, p_valor = np.nan, np.nan

        return {
            "n": int(len(s)),
            "media": float(s.mean()),
            "mediana": float(s.median()),
            "moda": float(s.mode().iloc[0]) if not s.mode().empty else None,
            "minimo": float(s.min()),
            "maximo": float(s.max()),
            "variancia": float(s.var()),
            "desvio_padrao": float(s.std()),
            "q1": float(s.quantile(0.25)),
            "q3": float(s.quantile(0.75)),
            "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
            "assimetria": float(s.skew()),
            "curtose": float(s.kurtosis()),
            "stat_shapiro": float(stat_sh) if not np.isnan(stat_sh) else None,
            "p_valor_shapiro": float(p_valor) if not np.isnan(p_valor) else None,
        }

    # =========================================================
    # FX / MARKETS
    # =========================================================
    FX_PAIRS = {
        "USD/BRL": "BRL=X",
        "EUR/USD": "EURUSD=X",
        "USD/JPY": "JPY=X",
        "GBP/USD": "GBPUSD=X",
    }

    FX_WINDOW_SIZE = 30
    FX_RANDOM_STATE = 42

    def fx_download_history(self, ticker: str, period: str = "3y") -> pd.Series:
        data = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
        if data is None or data.empty:
            raise ValueError("Sem dados históricos.")
        s = data["Close"].dropna().copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s.name = "rate"
        return s

    def _fx_metrics(self, y_true, y_pred) -> Dict[str, float]:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        denom = np.clip(np.abs(y_true), 1e-8, None)
        mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
        return {"mae": mae, "rmse": rmse, "mape": mape}

    def fx_rf(self, series: pd.Series, n_days: int) -> Tuple[Dict, pd.DataFrame]:
        values = series.values.astype(float)
        X, y = [], []
        for i in range(self.FX_WINDOW_SIZE, len(values)):
            X.append(values[i - self.FX_WINDOW_SIZE:i])
            y.append(values[i])
        X, y = np.array(X), np.array(y)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=self.FX_RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        metrics = self._fx_metrics(y_test, model.predict(X_test))

        history = values[-self.FX_WINDOW_SIZE:].tolist()
        future, dates = [], []
        last_date = series.index[-1]

        for i in range(n_days):
            X_in = scaler.transform([history])
            pred = float(model.predict(X_in)[0])
            future.append(pred)
            dates.append(last_date + pd.Timedelta(days=i + 1))
            history = history[1:] + [pred]

        forecast = pd.DataFrame(
            {"forecast_rate": future},
            index=pd.DatetimeIndex(dates, name="date")
        )
        return metrics, forecast

    def fx_arima(self, series: pd.Series, n_days: int) -> Tuple[Dict, pd.DataFrame]:
        split = int(len(series) * 0.8)
        train, test = series.iloc[:split], series.iloc[split:]

        model = ARIMA(train, order=(1, 1, 1)).fit()
        metrics = self._fx_metrics(test.values, model.forecast(len(test)).values)

        full = ARIMA(series, order=(1, 1, 1)).fit()
        future = full.forecast(n_days)

        dates = [series.index[-1] + pd.Timedelta(days=i + 1) for i in range(n_days)]
        forecast = pd.DataFrame(
            {"forecast_rate": future.values},
            index=pd.DatetimeIndex(dates, name="date")
        )
        return metrics, forecast

    def fx_prophet(self, series: pd.Series, n_days: int) -> Tuple[Dict, pd.DataFrame]:
        df = series.reset_index()
        df.columns = ["ds", "y"]

        split = int(len(df) * 0.8)
        train, test = df.iloc[:split], df.iloc[split:]

        m = Prophet()
        m.fit(train)

        future_test = m.make_future_dataframe(periods=len(test), freq="D")
        forecast_test = m.predict(future_test).iloc[-len(test):]

        metrics = self._fx_metrics(
            test["y"].values,
            forecast_test["yhat"].values
        )

        m_full = Prophet()
        m_full.fit(df)
        future_full = m_full.make_future_dataframe(periods=n_days, freq="D")
        forecast = m_full.predict(future_full).iloc[-n_days:]

        forecast_df = forecast[["ds", "yhat"]].set_index("ds")
        forecast_df.columns = ["forecast_rate"]
        forecast_df.index.name = "date"

        return metrics, forecast_df
