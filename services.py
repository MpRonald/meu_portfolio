import requests
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

# ===== FX deps =====
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


class PortfolioService:
    def __init__(self, timeout: int = 30, data_dir: Optional[str] = None):
        self.timeout = timeout
        self.data_dir = Path(data_dir) if data_dir else None

    # =========================
    # Weather (Open-Meteo)
    # =========================
    def geocode_city(self, name: str, count: int = 1, lang: str = "pt") -> dict:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": name, "count": count, "language": lang, "format": "json"}
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        js = r.json()
        res = (js.get("results") or [])
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

    # =========================
    # Ames (dataset + stats)
    # =========================
    @lru_cache(maxsize=1)
    def load_ames_data(self) -> pd.DataFrame:
        if not self.data_dir:
            raise RuntimeError("data_dir não configurado no PortfolioService.")

        csv_path = self.data_dir / "ames.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Ficheiro {csv_path} não encontrado.")

        df = pd.read_csv(csv_path)

        # tenta garantir tipos numéricos em colunas que façam sentido
        for c in df.columns:
            if c in {"faixa_preco", "bairro", "Neighborhood"}:
                continue
            if df[c].dtype == object:
                # converter o que der; se não der, mantém
                converted = pd.to_numeric(df[c], errors="coerce")
                # se converteu alguma coisa, usa; senão mantém original
                if converted.notna().sum() > 0:
                    df[c] = converted

        return df

    def calcular_estatisticas_1d(self, serie: pd.Series) -> Dict[str, Any]:
        s = serie.dropna()

        if len(s) == 0:
            return {
                "n": 0,
                "media": None,
                "mediana": None,
                "moda": None,
                "minimo": None,
                "maximo": None,
                "variancia": None,
                "desvio_padrao": None,
                "q1": None,
                "q3": None,
                "iqr": None,
                "assimetria": None,
                "curtose": None,
                "stat_shapiro": None,
                "p_valor_shapiro": None,
            }

        media = float(s.mean())
        mediana = float(s.median())
        moda_vals = s.mode()
        moda = float(moda_vals.iloc[0]) if not moda_vals.empty else None
        minimo = float(s.min())
        maximo = float(s.max())
        variancia = float(s.var())
        desvio_padrao = float(s.std())
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1

        assimetria = float(s.skew())
        curtose = float(s.kurtosis())

        sample = s
        if len(s) > 5000:
            sample = s.sample(5000, random_state=42)

        try:
            stat_sh, p_valor = stats.shapiro(sample)
        except Exception:
            stat_sh, p_valor = np.nan, np.nan

        return {
            "n": int(len(s)),
            "media": media,
            "mediana": mediana,
            "moda": moda,
            "minimo": minimo,
            "maximo": maximo,
            "variancia": variancia,
            "desvio_padrao": desvio_padrao,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "assimetria": assimetria,
            "curtose": curtose,
            "stat_shapiro": float(stat_sh) if not np.isnan(stat_sh) else None,
            "p_valor_shapiro": float(p_valor) if not np.isnan(p_valor) else None,
        }

    def calcular_testes_adicionais(
        self,
        serie: pd.Series,
        df_filtrado: pd.DataFrame,
        var: str,
        df_completo: Optional[pd.DataFrame],
    ) -> Dict[str, Optional[float]]:
        resultados: Dict[str, Optional[float]] = {}
        s = serie.dropna()

        # Jarque–Bera
        try:
            jb_stat, jb_p = stats.jarque_bera(s)
            resultados["jb_stat"] = float(jb_stat)
            resultados["jb_p"] = float(jb_p)
        except Exception:
            resultados["jb_stat"] = None
            resultados["jb_p"] = None

        # Pearson com preco e preco_m2
        for alvo in ["preco", "preco_m2"]:
            r_key = f"corr_{alvo}_r"
            p_key = f"corr_{alvo}_p"
            if alvo in df_filtrado.columns and var in df_filtrado.columns and var != alvo:
                subset = df_filtrado[[var, alvo]].dropna()
                if len(subset) >= 3:
                    try:
                        r, p = stats.pearsonr(subset[var], subset[alvo])
                        resultados[r_key] = float(r)
                        resultados[p_key] = float(p)
                    except Exception:
                        resultados[r_key] = None
                        resultados[p_key] = None
                else:
                    resultados[r_key] = None
                    resultados[p_key] = None
            else:
                resultados[r_key] = None
                resultados[p_key] = None

        # Spearman com preco
        if "preco" in df_filtrado.columns and var in df_filtrado.columns and var != "preco":
            subset = df_filtrado[[var, "preco"]].dropna()
            if len(subset) >= 3:
                try:
                    rho, p_s = stats.spearmanr(subset[var], subset["preco"])
                    resultados["corr_spearman_r"] = float(rho)
                    resultados["corr_spearman_p"] = float(p_s)
                except Exception:
                    resultados["corr_spearman_r"] = None
                    resultados["corr_spearman_p"] = None
            else:
                resultados["corr_spearman_r"] = None
                resultados["corr_spearman_p"] = None
        else:
            resultados["corr_spearman_r"] = None
            resultados["corr_spearman_p"] = None

        # Kruskal–Wallis por faixa_preco (só quando sem filtro: df_completo não None)
        if df_completo is not None and "faixa_preco" in df_completo.columns and var in df_completo.columns:
            grupos = []
            for faixa in sorted(df_completo["faixa_preco"].dropna().unique().tolist()):
                vals = df_completo.loc[df_completo["faixa_preco"] == faixa, var].dropna()
                if len(vals) >= 3:
                    grupos.append(vals.values)

            if len(grupos) >= 2:
                try:
                    H, p_kw = stats.kruskal(*grupos)
                    resultados["kruskal_H"] = float(H)
                    resultados["kruskal_p"] = float(p_kw)
                except Exception:
                    resultados["kruskal_H"] = None
                    resultados["kruskal_p"] = None
            else:
                resultados["kruskal_H"] = None
                resultados["kruskal_p"] = None
        else:
            resultados["kruskal_H"] = None
            resultados["kruskal_p"] = None

        # Regressão linear simples: preco ~ var
        if "preco" in df_filtrado.columns and var in df_filtrado.columns and var != "preco":
            subset = df_filtrado[[var, "preco"]].dropna()
            if len(subset) >= 3:
                try:
                    res = stats.linregress(subset[var], subset["preco"])
                    resultados["reg_beta0"] = float(res.intercept)
                    resultados["reg_beta1"] = float(res.slope)
                    resultados["reg_r2"] = float(res.rvalue ** 2)
                    resultados["reg_p_beta1"] = float(res.pvalue)
                except Exception:
                    resultados["reg_beta0"] = None
                    resultados["reg_beta1"] = None
                    resultados["reg_r2"] = None
                    resultados["reg_p_beta1"] = None
            else:
                resultados["reg_beta0"] = None
                resultados["reg_beta1"] = None
                resultados["reg_r2"] = None
                resultados["reg_p_beta1"] = None
        else:
            resultados["reg_beta0"] = None
            resultados["reg_beta1"] = None
            resultados["reg_r2"] = None
            resultados["reg_p_beta1"] = None

        return resultados

    # ============================================================
    # ==== FX (RF / ARIMA / Prophet) =============================
    # ============================================================

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
            raise ValueError(f"Sem dados históricos para o ticker {ticker}.")
        s = data["Close"].dropna().copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = s.sort_index()
        s.name = "rate"
        return s

    def stock_download_history(self, ticker: str, period: str = "3y") -> pd.Series:
        data = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
        if data is None or data.empty:
            raise ValueError(f"Sem dados históricos para o ticker {ticker}.")
        s = data["Close"].dropna().copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = s.sort_index()
        s.name = "price"
        return s

    def fx_compute_metrics(self, y_true, y_pred) -> dict:
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))

        denom = np.clip(np.abs(y_true), 1e-8, None)
        mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

        return {"mae": float(mae), "rmse": rmse, "mape": mape}

    def fx_create_supervised(self, series: pd.Series, window_size: int = FX_WINDOW_SIZE):
        values = np.asarray(series.values, dtype="float64").reshape(-1)
        X, y = [], []
        for i in range(window_size, len(values)):
            X.append(values[i - window_size:i])
            y.append(values[i])
        X = np.array(X)
        y = np.array(y)
        return X, y

    def fx_forecast_n_days_rf(
        self,
        model,
        scaler,
        history_series: pd.Series,
        n_days: int,
        window_size: int = FX_WINDOW_SIZE
    ) -> pd.DataFrame:
        if n_days <= 0:
            raise ValueError("n_days deve ser > 0")
        if n_days > 180:
            raise ValueError("Não é permitido prever mais de 180 dias.")

        history = history_series.copy().sort_index()
        history_values = np.asarray(history.values, dtype="float64").reshape(-1)
        if len(history_values) < window_size:
            raise ValueError("Histórico insuficiente para formar a janela inicial.")

        last_values = history_values[-window_size:].tolist()
        last_date = history.index[-1]

        future_dates = []
        future_values = []

        for step in range(1, n_days + 1):
            X_input = np.array(last_values, dtype="float64").reshape(1, -1)
            X_input_scaled = scaler.transform(X_input)
            next_value = float(model.predict(X_input_scaled)[0])
            next_date = last_date + pd.Timedelta(days=step)

            future_dates.append(next_date)
            future_values.append(next_value)

            last_values = last_values[1:] + [next_value]

        forecast_df = pd.DataFrame(
            {"forecast_rate": future_values},
            index=pd.DatetimeIndex(future_dates, name="date")
        )
        return forecast_df

    def fx_train_and_forecast_rf(self, series: pd.Series, n_days: int):
        X, y = self.fx_create_supervised(series, self.FX_WINDOW_SIZE)
        if len(X) < 50:
            raise ValueError("Dados insuficientes para treinar Random Forest (menos de 50 amostras).")

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=self.FX_RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        y_pred_test = model.predict(X_test_scaled)

        metrics = self.fx_compute_metrics(y_test, y_pred_test)
        forecast_df = self.fx_forecast_n_days_rf(model, scaler, series, n_days, self.FX_WINDOW_SIZE)
        return metrics, forecast_df

    def fx_train_and_forecast_arima(self, series: pd.Series, n_days: int):
        if len(series) < 80:
            raise ValueError("Dados insuficientes para treinar ARIMA (menos de 80 observações).")

        split = int(len(series) * 0.8)
        train = series.iloc[:split]
        test = series.iloc[split:]

        model = ARIMA(train, order=(1, 1, 1))
        model_fit = model.fit()

        forecast_test = model_fit.forecast(steps=len(test))
        metrics = self.fx_compute_metrics(test.values, forecast_test.values)

        full_model = ARIMA(series, order=(1, 1, 1)).fit()
        forecast_future = full_model.forecast(steps=n_days)

        last_date = series.index[-1]
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, n_days + 1)]
        forecast_df = pd.DataFrame(
            {"forecast_rate": forecast_future.values},
            index=pd.DatetimeIndex(future_dates, name="date")
        )
        return metrics, forecast_df

    def fx_train_and_forecast_prophet(self, series: pd.Series, n_days: int):
        from prophet import Prophet
        if len(series) < 80:
            raise ValueError("Dados insuficientes para treinar Prophet (menos de 80 observações).")

        df = series.reset_index()
        df.columns = ["ds", "y"]

        split = int(len(df) * 0.8)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]

        m_metrics = Prophet()
        m_metrics.fit(train_df)

        future_test = m_metrics.make_future_dataframe(periods=len(test_df), freq="D")
        forecast_test = m_metrics.predict(future_test)
        forecast_test_tail = forecast_test.iloc[-len(test_df):]

        y_true = test_df.set_index("ds")["y"]
        y_pred = forecast_test_tail.set_index("ds")["yhat"]
        common_idx = y_true.index.intersection(y_pred.index)
        metrics = self.fx_compute_metrics(y_true.loc[common_idx], y_pred.loc[common_idx])

        m_full = Prophet()
        m_full.fit(df)
        future_full = m_full.make_future_dataframe(periods=n_days, freq="D")
        forecast_full = m_full.predict(future_full)
        forecast_future = forecast_full.iloc[-n_days:][["ds", "yhat"]]

        forecast_df = forecast_future.set_index("ds").rename(columns={"yhat": "forecast_rate"})
        forecast_df.index.name = "date"
        return metrics, forecast_df
