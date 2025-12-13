import os
import re
import json
import pickle
import logging
from io import StringIO
from datetime import datetime
from pathlib import Path
from functools import lru_cache
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from flask import (
    Flask, render_template, request, jsonify,
    make_response, redirect, url_for
)

from werkzeug.middleware.proxy_fix import ProxyFix
from jinja2 import TemplateNotFound

# ------------------ ML / Stats ------------------
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import load as joblib_load
from scipy import stats

# ------------------ Time series ------------------
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# ------------------ Plotly ------------------
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import plot


# ============================================================
# ==== Paths seguros (cross-platform) ========================
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))).resolve()
ARTIF_DIR = Path(os.getenv("ARTIF_DIR", str(BASE_DIR / "artifacts"))).resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIF_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ==== Logging ===============================================
# ============================================================

def _configure_logging(app: Flask) -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
    else:
        root.setLevel(level)

    app.logger.setLevel(level)
    app.logger.info(
        "App a iniciar | ENV=%s | DEBUG=%s | DATA_DIR=%s | ARTIF_DIR=%s",
        app.config.get("ENV_NAME"),
        app.config.get("DEBUG"),
        app.config.get("DATA_DIR"),
        app.config.get("ARTIF_DIR")
    )


# ============================================================
# ==== App factory (deploy-safe) =============================
# ============================================================

def create_app() -> Flask:
    app = Flask(__name__)

    # ProxyFix: necessário atrás do Railway/Render
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

    # Config
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")
    env = (os.getenv("FLASK_ENV") or os.getenv("ENV") or "production").lower().strip()
    app.config["ENV_NAME"] = env
    debug_flag = (os.getenv("FLASK_DEBUG") or "").strip() == "1"
    app.config["DEBUG"] = True if env == "development" or debug_flag else False

    # Paths (fonte única de verdade)
    app.config["BASE_DIR"] = str(BASE_DIR)
    app.config["DATA_DIR"] = str(DATA_DIR)
    app.config["ARTIF_DIR"] = str(ARTIF_DIR)

    _configure_logging(app)

    @app.after_request
    def add_default_headers(resp):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return resp

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "not_found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        env_name = (app.config.get("ENV_NAME") or "production").lower()
        is_prod = env_name == "production" or os.getenv("RAILWAY_ENVIRONMENT") is not None
        if is_prod:
            return jsonify({"error": "internal_server_error"}), 500
        return jsonify({"error": "internal_server_error", "detail": str(e)}), 500

    @app.route("/healthz")
    def healthz():
        return jsonify({
            "status": "ok",
            "time": datetime.utcnow().isoformat() + "Z",
            "env": app.config.get("ENV_NAME"),
        })

    return app


# ============================================================
# ==== Instância global (gunicorn procura "app") =============
# ============================================================

app = create_app()


# ============================================================
# ==== Render seguro (enquanto templates não existem) =========
# ============================================================

def safe_render(template_name: str, **ctx):
    try:
        return render_template(template_name, **ctx)
    except TemplateNotFound:
        return jsonify({
            "template_missing": template_name,
            "hint": "Ainda não criaste o template. Isto é esperado nesta fase.",
            "context_keys": sorted(list(ctx.keys()))
        }), 200


# ============================================================
# ==== Helpers: carregar CSV do /data =========================
# ============================================================

def load_dataset(filename: str) -> pd.DataFrame:
    path = Path(app.config["DATA_DIR"]) / filename
    return pd.read_csv(path)


# ============================================================
# ==== Presets de Ações (quotes) =============================
# ============================================================

PRESETS = [
    ("Apple (EUA)", "AAPL"),
    ("Microsoft (EUA)", "MSFT"),
    ("Tesla (EUA)", "TSLA"),
    ("Petrobras PN (Brasil)", "PETR4.SA"),
    ("Vale ON (Brasil)", "VALE3.SA"),
    ("Itaú Unibanco PN (Brasil)", "ITUB4.SA"),
    ("GALP (Portugal)", "GALP.LS"),
    ("EDP (Portugal)", "EDP.LS"),
    ("BCP (Portugal)", "BCP.LS"),
]


def _coerce_period_month(period: str) -> str:
    return "1mo" if period == "1m" else period


def normalize_interval_and_period(interval: str, period: str):
    original_interval, original_period = interval, period
    notice = None
    period = _coerce_period_month(period)

    if interval == "1m":
        if period not in ["1d", "5d", "7d"]:
            period = "5d"
    elif interval in ["2m", "5m", "15m", "30m"]:
        if period not in ["1d", "5d", "7d", "1mo", "2mo"]:
            period = "1mo"
    elif interval in ["60m", "90m", "1h"]:
        if period not in ["1mo", "3mo", "6mo", "1y", "2y"]:
            period = "1y"

    if (interval, period) != (original_interval, original_period):
        notice = f"Compatibilizei interval={original_interval} / period={original_period} para interval={interval} / period={period}."
    return interval, period, notice


def fetch_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise ValueError("Ticker inválido.")

    interval, period, _ = normalize_interval_and_period(interval, period)
    data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if data is None or data.empty:
        raise ValueError("Sem dados para o ticker/período informado.")

    data = data.reset_index()
    date_col = "Date" if "Date" in data.columns else "Datetime"
    data[date_col] = pd.to_datetime(data[date_col]).dt.tz_localize(None)
    data = data[[date_col, "Open", "High", "Low", "Close", "Volume"]].rename(columns={date_col: "Date"})
    return data


# ============================================================
# ==== Weather (Open-Meteo) ==================================
# ============================================================

WEATHER_DEFAULT_CITY = "Lisboa"


def geocode_city(name: str, count: int = 1, lang: str = "pt") -> dict:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": name, "count": count, "language": lang, "format": "json"}
    r = requests.get(url, params=params, timeout=30)
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


def fetch_weather_forecast(lat: float, lon: float, days: int = 7,
                           temp_unit: str = "celsius", wind_unit: str = "kmh", lang: str = "pt") -> dict:
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
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


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


def fx_download_history(ticker: str, period: str = "3y") -> pd.Series:
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
    if data is None or data.empty:
        raise ValueError(f"Sem dados históricos para o ticker {ticker}.")
    s = data["Close"].dropna().copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s.sort_index()
    s.name = "rate"
    return s


def stock_download_history(ticker: str, period: str = "3y") -> pd.Series:
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
    if data is None or data.empty:
        raise ValueError(f"Sem dados históricos para o ticker {ticker}.")
    s = data["Close"].dropna().copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s.sort_index()
    s.name = "price"
    return s


def fx_compute_metrics(y_true, y_pred) -> dict:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    return {"mae": float(mae), "rmse": rmse, "mape": mape}


def fx_create_supervised(series: pd.Series, window_size: int = FX_WINDOW_SIZE):
    values = np.asarray(series.values, dtype="float64").reshape(-1)
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i - window_size:i])
        y.append(values[i])
    X = np.array(X)
    y = np.array(y)
    return X, y


def fx_forecast_n_days_rf(model, scaler, history_series: pd.Series,
                          n_days: int, window_size: int = FX_WINDOW_SIZE) -> pd.DataFrame:
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


def fx_train_and_forecast_rf(series: pd.Series, n_days: int):
    X, y = fx_create_supervised(series, FX_WINDOW_SIZE)
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
        random_state=FX_RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    y_pred_test = model.predict(X_test_scaled)

    metrics = fx_compute_metrics(y_test, y_pred_test)
    forecast_df = fx_forecast_n_days_rf(model, scaler, series, n_days, FX_WINDOW_SIZE)
    return metrics, forecast_df


def fx_train_and_forecast_arima(series: pd.Series, n_days: int):
    if len(series) < 80:
        raise ValueError("Dados insuficientes para treinar ARIMA (menos de 80 observações).")

    split = int(len(series) * 0.8)
    train = series.iloc[:split]
    test = series.iloc[split:]

    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit()

    forecast_test = model_fit.forecast(steps=len(test))
    metrics = fx_compute_metrics(test.values, forecast_test.values)

    full_model = ARIMA(series, order=(1, 1, 1)).fit()
    forecast_future = full_model.forecast(steps=n_days)

    last_date = series.index[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, n_days + 1)]
    forecast_df = pd.DataFrame(
        {"forecast_rate": forecast_future.values},
        index=pd.DatetimeIndex(future_dates, name="date")
    )
    return metrics, forecast_df


def fx_train_and_forecast_prophet(series: pd.Series, n_days: int):
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
    metrics = fx_compute_metrics(y_true.loc[common_idx], y_pred.loc[common_idx])

    m_full = Prophet()
    m_full.fit(df)
    future_full = m_full.make_future_dataframe(periods=n_days, freq="D")
    forecast_full = m_full.predict(future_full)
    forecast_future = forecast_full.iloc[-n_days:][["ds", "yhat"]]

    forecast_df = forecast_future.set_index("ds").rename(columns={"yhat": "forecast_rate"})
    forecast_df.index.name = "date"
    return metrics, forecast_df


# ============================================================
# ==== E-commerce dataset (carrega no boot) ===================
# ============================================================

ECOM_PATH = Path(app.config["DATA_DIR"]) / "e_commerce.csv"
ECOM_DF = None
ECOM_LOAD_ERROR = None

def _load_ecom_data():
    global ECOM_DF, ECOM_LOAD_ERROR
    try:
        df = pd.read_csv(ECOM_PATH, encoding="latin1")
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["InvoiceDate"])
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

        if "CustomerID" in df.columns:
            df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")

        ECOM_DF = df
        ECOM_LOAD_ERROR = None
    except Exception as e:
        ECOM_DF = None
        ECOM_LOAD_ERROR = f"Erro ao carregar dataset de e-commerce: {e}"

_load_ecom_data()


# ============================================================
# ==== Churn artifacts (carrega no boot) ======================
# ============================================================

_CHURN_MODEL = None
_CHURN_SCALER = None
_CHURN_FEATURES = None
_CHURN_ERROR = None

def _load_churn_artifacts():
    global _CHURN_MODEL, _CHURN_SCALER, _CHURN_FEATURES, _CHURN_ERROR
    try:
        art_dir = Path(app.config["ARTIF_DIR"])
        _CHURN_MODEL = joblib_load(art_dir / "modelo_churn.joblib")
        _CHURN_SCALER = joblib_load(art_dir / "modelo_churn_scaler.joblib")
        _CHURN_FEATURES = joblib_load(art_dir / "modelo_churn_feature_names.joblib")
        _CHURN_ERROR = None
    except Exception as e:
        _CHURN_MODEL = None
        _CHURN_SCALER = None
        _CHURN_FEATURES = None
        _CHURN_ERROR = f"Falha ao carregar artifacts de churn: {e}"

_load_churn_artifacts()


def preprocess_telco(df_raw: pd.DataFrame,
                     scaler=None,
                     feature_names=None) -> pd.DataFrame:
    df = df_raw.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    cat_cols_to_dummies = [
        'gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ]
    existing_cat_cols = [c for c in cat_cols_to_dummies if c in df.columns]
    if existing_cat_cols:
        df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)

    if "tenure" in df.columns and "TotalCharges" in df.columns:
        df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=["0-12m", "12-24m", "24-36m", "36-48m", "48-60m", "60-72m"]
        )
        df["tenure_group"] = LabelEncoder().fit_transform(df["tenure_group"].astype(str))

    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]
    numeric_existing = [c for c in numerical_columns if c in df.columns]

    if scaler is not None and numeric_existing:
        df[numeric_existing] = scaler.transform(df[numeric_existing])

    if feature_names is not None:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

    return df


def predict_churn_from_raw(df_raw: pd.DataFrame):
    if _CHURN_MODEL is None or _CHURN_SCALER is None or _CHURN_FEATURES is None:
        raise RuntimeError(_CHURN_ERROR or "Modelo de churn indisponível.")
    df_proc = preprocess_telco(df_raw, scaler=_CHURN_SCALER, feature_names=_CHURN_FEATURES)
    proba = _CHURN_MODEL.predict_proba(df_proc)[0, 1]
    pred = _CHURN_MODEL.predict(df_proc)[0]
    return int(pred), float(proba)


# ============================================================
# ==== NLP PT (modelo salvo em ARTIF_DIR) =====================
# ============================================================

NLP_MODEL_PATH = os.getenv(
    "NLP_MODEL_PATH",
    str(Path(app.config["ARTIF_DIR"]) / "modelo_sentimento.joblib")
)

_NLP_MODEL = None
_NLP_ERROR = None
_NLP_STATS = None


def _clean_html(s: str) -> str:
    s = (s or "")
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_list_or_none(x):
    if x is None:
        return None
    try:
        return np.array(x).ravel().tolist()
    except Exception:
        return list(x) if isinstance(x, (list, tuple)) else [x]


def _load_pt_model():
    global _NLP_MODEL, _NLP_ERROR, _NLP_STATS
    try:
        p = Path(NLP_MODEL_PATH)
        if not p.exists():
            _NLP_MODEL = None
            _NLP_STATS = None
            _NLP_ERROR = (
                f"Modelo de NLP não encontrado em: {p}. "
                "Coloque o ficheiro em /artifacts ou defina a variável NLP_MODEL_PATH."
            )
            return

        try:
            bundle = joblib_load(str(p))
        except Exception as e1:
            alt = p.with_suffix(".pkl")
            if alt.exists():
                try:
                    with open(alt, "rb") as f:
                        bundle = pickle.load(f)
                except Exception as e2:
                    _NLP_MODEL = None
                    _NLP_STATS = None
                    _NLP_ERROR = f"Falha ao carregar modelo NLP (.joblib/.pkl): {e1} | {e2}"
                    return
            else:
                _NLP_MODEL = None
                _NLP_STATS = None
                _NLP_ERROR = f"Falha ao carregar modelo NLP (.joblib): {e1}"
                return

        if not isinstance(bundle, dict) or "tfidf" not in bundle or "clf" not in bundle:
            _NLP_MODEL = None
            _NLP_STATS = None
            _NLP_ERROR = "Arquivo de modelo inválido (esperado dict com 'tfidf' e 'clf')."
            return

        classes = bundle.get("classes_", None)
        if classes is None:
            classes = getattr(bundle["clf"], "classes_", None)

        classes_list = _to_list_or_none(classes)
        _NLP_MODEL = {"tfidf": bundle["tfidf"], "clf": bundle["clf"], "classes_": classes_list}
        _NLP_STATS = {"classes": [str(c) for c in classes_list] if classes_list is not None else None}
        _NLP_ERROR = None

    except Exception as e:
        _NLP_MODEL = None
        _NLP_STATS = None
        _NLP_ERROR = f"Falha inesperada ao iniciar NLP: {e}"


def _positive_index(classes, proba) -> int:
    if classes is None:
        return int(np.argmax(proba))
    cls = [str(c).strip().lower() for c in classes]
    aliases_pos = {"positivo", "positive", "pos", "1", "true", "sim", "bom"}
    aliases_neg = {"negativo", "negative", "neg", "0", "false", "nao", "não", "ruim"}

    for i, c in enumerate(cls):
        if c in aliases_pos:
            return i
    if len(cls) == 2:
        for i, c in enumerate(cls):
            if c in aliases_neg:
                return 1 - i
    return int(np.argmax(proba))


def _pt_label_from_pred(pred) -> str:
    p = str(pred).strip().lower()
    if p in {"positivo", "positive", "pos", "1", "true", "sim", "bom"}:
        return "bom"
    if p in {"negativo", "negative", "neg", "0", "false", "nao", "não", "ruim"}:
        return "ruim"
    if p in {"neutro", "neutral", "neu"}:
        return "neutro"
    return "bom" if p in {"1"} else "ruim"


_load_pt_model()


def predict_sentiment_label(text: str) -> Tuple[str, float, dict]:
    if _NLP_MODEL is None:
        raise RuntimeError(_NLP_ERROR or "Modelo de NLP não carregado.")

    tfidf = _NLP_MODEL["tfidf"]
    clf = _NLP_MODEL["clf"]
    classes = _NLP_MODEL.get("classes_")

    s = _clean_html(text)
    Xq = tfidf.transform([s])

    proba = None
    score_pos = 0.5
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Xq)[0]
        pos_idx = _positive_index(classes, proba)
        score_pos = float(proba[pos_idx])

    pred = clf.predict(Xq)[0]
    label_pt = _pt_label_from_pred(pred)

    if 0.45 <= score_pos <= 0.55:
        label_pt = "neutro"

    meta = {
        "classes": classes,
        "raw_pred": str(pred),
        "score_strategy": "predict_proba" if proba is not None else "heuristic_from_predict",
        "model_path": str(NLP_MODEL_PATH),
    }
    return label_pt, score_pos, meta


# ============================================================
# ==== Heart Disease (treina via CSV se não houver artifact) ==
# ============================================================

DEFAULT_HEART_FEATURES = [
    "sexo", "idade", "cigarros_por_dia", "uso_medicamento_pressao", "AVC",
    "hipertensao", "diabetes", "colesterol_total", "pressao_arterial_sistolica",
    "pressao_arterial_diastolica", "IMC", "freq_cardiaca", "glicemia", "fumante",
]
HEART_TARGET = "risco_DAC_decada"

HEART_RAW_URL = os.getenv(
    "HEART_RAW_URL",
    "https://raw.githubusercontent.com/MpRonald/datasets/main/doenca_cardiaca_final.csv"
)

_HEART_FEATURES = DEFAULT_HEART_FEATURES[:]
_HEART_MODEL = None
_HEART_STATS = None
_HEART_LOAD_ERROR = None
_HEART_SOURCE = None  # "artifact" ou "csv-train"


def _attempt_load_heart_artifact():
    art_dir = Path(app.config["ARTIF_DIR"])
    if not art_dir.exists():
        return None, None, None
    try:
        metas = sorted(art_dir.glob("meta_*.joblib"), reverse=True)
        for mp in metas:
            meta = joblib_load(mp)
            if meta.get("task") != "heart_disease":
                continue
            version = meta["version"]
            mpath = art_dir / f"model_{version}.joblib"
            if not mpath.exists():
                continue
            art = joblib_load(mpath)
            if isinstance(art, dict) and "model" in art:
                model = art["model"]
                features = art.get("feature_names") or meta.get("feature_names") or DEFAULT_HEART_FEATURES
            else:
                model = art
                features = meta.get("feature_names") or DEFAULT_HEART_FEATURES
            return model, list(features), meta
    except Exception:
        pass
    return None, None, None


def _load_heart_from_csv_and_train():
    r = requests.get(HEART_RAW_URL, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))

    missing = [c for c in DEFAULT_HEART_FEATURES + [HEART_TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando no CSV: {missing}")

    for c in DEFAULT_HEART_FEATURES + [HEART_TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42
        ))
    ])
    X = df[DEFAULT_HEART_FEATURES].copy()
    y = df[HEART_TARGET].astype(int).copy()
    model.fit(X, y)

    stats_dict = {
        c: {"min": float(np.nanmin(X[c])), "max": float(np.nanmax(X[c])), "median": float(np.nanmedian(X[c]))}
        for c in DEFAULT_HEART_FEATURES
    }
    return model, DEFAULT_HEART_FEATURES[:], stats_dict


def _init_heart_model():
    global _HEART_MODEL, _HEART_FEATURES, _HEART_STATS, _HEART_LOAD_ERROR, _HEART_SOURCE
    try:
        model, features, meta = _attempt_load_heart_artifact()
        if model is not None:
            _HEART_MODEL = model
            _HEART_FEATURES = features
            try:
                _, _, stats_dict = _load_heart_from_csv_and_train()
                _HEART_STATS = {c: stats_dict[c] for c in _HEART_FEATURES if c in stats_dict}
            except Exception:
                _HEART_STATS = None
            _HEART_LOAD_ERROR = None
            _HEART_SOURCE = "artifact"
            return

        model, features, stats_dict = _load_heart_from_csv_and_train()
        _HEART_MODEL, _HEART_FEATURES, _HEART_STATS = model, features, stats_dict
        _HEART_LOAD_ERROR = None
        _HEART_SOURCE = "csv-train"
    except Exception as e:
        _HEART_MODEL, _HEART_FEATURES, _HEART_STATS = None, DEFAULT_HEART_FEATURES[:], None
        _HEART_LOAD_ERROR = str(e)
        _HEART_SOURCE = None


_init_heart_model()


# ============================================================
# ==== Loan Default (pipeline joblib) =========================
# ============================================================

LOAN_MODEL_PATH = os.getenv("LOAN_MODEL_PATH", str(Path(app.config["ARTIF_DIR"]) / "molde_credit_risk.joblib"))

_LOAN_PIPE = None
_LOAN_META = None
_LOAN_ERROR = None


def _extract_pipeline(obj):
    if not isinstance(obj, dict):
        return obj, None
    for k in ("pipeline", "model", "pipe", "estimator"):
        if k in obj:
            return obj[k], obj
    keys = ", ".join(obj.keys())
    raise ValueError(
        "Objeto .joblib é um dict sem chave de estimador ('pipeline'/'model'/'pipe'/'estimator'). "
        f"Chaves: {keys}"
    )


try:
    _loaded = joblib_load(LOAN_MODEL_PATH)
    _LOAN_PIPE, _LOAN_META = _extract_pipeline(_loaded)
except Exception as e:
    _LOAN_PIPE = None
    _LOAN_META = None
    _LOAN_ERROR = f"Falha ao carregar pipeline de empréstimo em '{LOAN_MODEL_PATH}': {e}"


LOAN_CAMPO_NUM = [
    'idade', 'rendimento_anual', 'anos_emprego', 'valor_emprestimo',
    'taxa_juros', 'percent_rendimento', 'anos_historico_credito'
]
LOAN_CAMPO_CAT = [
    'tipo_habitacao', 'finalidade_emprestimo', 'grau_emprestimo', 'historico_inadimplencia'
]

LOAN_OPCOES = {
    'tipo_habitacao': ['Aluguel', 'Própria', 'Hipoteca', 'Outro'],
    'finalidade_emprestimo': [
        'Pessoal', 'Educação', 'Médico', 'Empresarial',
        'Reforma residencial', 'Consolidação de dívidas'
    ],
    'grau_emprestimo': [
        'Excelente', 'Bom', 'Regular', 'Abaixo da média',
        'Ruim', 'Muito ruim', 'Altamente arriscado'
    ],
    'historico_inadimplencia': [('Y', 'Sim (há histórico)'), ('N', 'Não (sem histórico)')]
}


def _to_float_or_none(x: str):
    if x is None:
        return None
    x = x.strip().replace(',', '.')
    if x == '':
        return None
    try:
        return float(x)
    except Exception:
        return None


# ============================================================
# ==== Ames (helpers que já tinhas) ===========================
# ============================================================

COLUNAS_PROJETO = [
    "preco", "quartos", "banheiros", "area_habitavel", "area_lote",
    "andares", "area_acima_solo", "area_porao", "ano_construcao",
    "latitude", "longitude", "area_habitavel_viz", "area_lote_viz",
    "faixa_preco", "idade_imovel", "area_total", "densidade_construcao", "preco_m2",
]

NOMES_AMIGAVEIS = {
    "preco": "Preço do Imóvel ($)",
    "quartos": "Número de Quartos",
    "banheiros": "Número de WC",
    "area_habitavel": "Área Habitável (m²)",
    "area_lote": "Área do Lote (m²)",
    "andares": "Número de Andares",
    "area_acima_solo": "Área acima do Solo (m²)",
    "area_porao": "Área da Cave/PORÃO (m²)",
    "ano_construcao": "Ano de Construção",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "area_habitavel_viz": "Área Habitável Média dos Vizinhos (m²)",
    "area_lote_viz": "Área do Lote Média dos Vizinhos (m²)",
    "faixa_preco": "Faixa de Preço",
    "idade_imovel": "Idade do Imóvel (anos)",
    "area_total": "Área Total (m²)",
    "densidade_construcao": "Densidade de Construção",
    "preco_m2": "Preço por m² ($)",
}

NOMES_FAIXA = {
    "baixo": "Preço Baixo",
    "medio": "Preço Médio",
    "alto": "Preço Alto",
    "muito_alto": "Preço Muito Alto"
}

VARIAVEIS_IMPORTANTES = list(NOMES_AMIGAVEIS.keys())


@lru_cache(maxsize=1)
def load_ames_data() -> pd.DataFrame:
    csv_path = Path(app.config["DATA_DIR"]) / "ames.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Ficheiro {csv_path} não encontrado.")
    df = pd.read_csv(csv_path)
    colunas_validas = [c for c in COLUNAS_PROJETO if c in df.columns]
    df = df[colunas_validas]
    return df


def calcular_estatisticas_1d(serie: pd.Series) -> dict:
    s = serie.dropna()

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
    serie: pd.Series,
    df_filtrado: pd.DataFrame,
    var: str,
    df_completo: Optional[pd.DataFrame],
) -> dict:
    resultados: dict[str, float | None] = {}
    s = serie.dropna()

    try:
        jb_stat, jb_p = stats.jarque_bera(s)
        resultados["jb_stat"] = float(jb_stat)
        resultados["jb_p"] = float(jb_p)
    except Exception:
        resultados["jb_stat"] = None
        resultados["jb_p"] = None

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
# ==== ROTAS (mantidas, com safe_render) ======================
# ============================================================

@app.route("/")
def index():
    return safe_render("index.html")


@app.route("/fx", methods=["GET", "POST"])
def fx():
    erro = None
    resultado = None

    if request.method == "POST":
        try:
            pair_name = request.form.get("pair")
            algoritmo = request.form.get("algoritmo")
            n_days_raw = request.form.get("n_days", "30")

            try:
                n_days = int(n_days_raw)
            except Exception:
                n_days = 30
            n_days = max(1, min(180, n_days))

            if pair_name not in FX_PAIRS:
                raise ValueError("Par de moedas inválido.")
            if algoritmo not in ["rf", "arima", "prophet"]:
                raise ValueError("Algoritmo inválido.")

            ticker = FX_PAIRS[pair_name]
            series = fx_download_history(ticker, period="3y")

            if algoritmo == "rf":
                metrics, forecast_df = fx_train_and_forecast_rf(series, n_days)
                alg_label = "Random Forest"
            elif algoritmo == "arima":
                metrics, forecast_df = fx_train_and_forecast_arima(series, n_days)
                alg_label = "ARIMA (1,1,1)"
            else:
                metrics, forecast_df = fx_train_and_forecast_prophet(series, n_days)
                alg_label = "Prophet"

            history_last_year = series.last("365D")

            x_hist = [d.date() for d in history_last_year.index]
            y_hist = [float(v) for v in history_last_year.values]
            x_fore = [d.date() for d in forecast_df.index]
            y_fore = [float(v) for v in forecast_df["forecast_rate"].values]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_hist, y=y_hist, mode="lines+markers", name="Histórico", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=x_fore, y=y_fore, mode="lines+markers", name="Previsão", line=dict(width=2, dash="dash")))

            fig.update_layout(
                title=f"Histórico e previsão — {pair_name}",
                xaxis_title="Data",
                yaxis_title=f"Taxa {pair_name}",
                margin=dict(l=40, r=20, t=40, b=40),
                template="plotly_dark",
                height=400,
                hovermode="x unified"
            )
            fig.update_yaxes(tickformat=".3f")

            fx_plot_div = plot(fig, include_plotlyjs="cdn", output_type="div")

            resultado = {
                "pair_name": pair_name,
                "algoritmo": algoritmo,
                "algoritmo_label": alg_label,
                "n_days": n_days,
                "metrics": metrics,
                "forecast": forecast_df,
                "history": history_last_year,
                "fx_plot_div": fx_plot_div,
            }

        except Exception as e:
            erro = str(e)

    return safe_render("fx.html", pairs=FX_PAIRS, erro=erro, resultado=resultado)


@app.route("/quotes", methods=["GET", "POST"])
def quotes():
    DEFAULT_TICKER = "AAPL"
    DEFAULT_ALGO = "rf"
    DEFAULT_N_DAYS = 30

    preset_ticker = request.values.get("preset_ticker")
    default_ticker = preset_ticker if preset_ticker else DEFAULT_TICKER

    raw_period = request.values.get("period", "1y")
    raw_interval = request.values.get("interval", "1d")
    period = _coerce_period_month(raw_period)
    interval, period, notice = normalize_interval_and_period(raw_interval, period)

    algoritmo = request.values.get("algoritmo", DEFAULT_ALGO)
    n_days_raw = request.values.get("n_days", str(DEFAULT_N_DAYS))
    try:
        n_days = int(n_days_raw)
    except Exception:
        n_days = DEFAULT_N_DAYS
    n_days = max(1, min(180, n_days))

    form = {
        "ticker": request.values.get("ticker", default_ticker).upper(),
        "period": period,
        "interval": interval,
        "algoritmo": algoritmo,
        "n_days": n_days,
    }

    chart_data = None
    error = None
    metrics = None
    forecast_df = None
    stock_plot_div = None
    alg_label = None

    try:
        df = fetch_history(form["ticker"], form["period"], form["interval"])
        intraday = form["interval"] in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
        date_fmt = "%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d"
        chart_data = {
            "labels": df["Date"].dt.strftime(date_fmt).tolist(),
            "close": df["Close"].round(4).tolist(),
            "ohlc": df[["Open", "High", "Low", "Close"]].round(4).values.tolist(),
        }

        series = stock_download_history(form["ticker"], period="3y")

        if algoritmo == "rf":
            metrics, forecast_df = fx_train_and_forecast_rf(series, n_days)
            alg_label = "Random Forest"
        elif algoritmo == "arima":
            metrics, forecast_df = fx_train_and_forecast_arima(series, n_days)
            alg_label = "ARIMA (1,1,1)"
        elif algoritmo == "prophet":
            metrics, forecast_df = fx_train_and_forecast_prophet(series, n_days)
            alg_label = "Prophet"
        else:
            raise ValueError("Algoritmo inválido.")

        history_last_year = series.last("365D")

        x_hist = [d.date() for d in history_last_year.index]
        y_hist = [float(v) for v in history_last_year.values]
        x_fore = [d.date() for d in forecast_df.index]
        y_fore = [float(v) for v in forecast_df["forecast_rate"].values]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_hist, y=y_hist, mode="lines+markers", name="Histórico (fecho)", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=x_fore, y=y_fore, mode="lines+markers", name="Previsão", line=dict(width=2, dash="dash")))

        fig.update_layout(
            title=f"Histórico e previsão — {form['ticker']}",
            xaxis_title="Data",
            yaxis_title=f"Preço {form['ticker']}",
            margin=dict(l=40, r=20, t=40, b=40),
            template="plotly_dark",
            height=400,
            hovermode="x unified",
        )
        fig.update_yaxes(tickformat=".3f")

        stock_plot_div = plot(fig, include_plotlyjs="cdn", output_type="div")

    except Exception as e:
        error = str(e)

    return safe_render(
        "quotes.html",
        form=form,
        chart_data=chart_data,
        error=error,
        presets=PRESETS,
        notice=notice,
        metrics=metrics,
        forecast=forecast_df,
        alg_label=alg_label,
        stock_plot_div=stock_plot_div,
    )


@app.route("/quotes.csv")
def quotes_csv():
    ticker = request.args.get("ticker", "AAPL").upper()
    period = _coerce_period_month(request.args.get("period", "1y"))
    interval = request.args.get("interval", "1d")
    interval, period, _ = normalize_interval_and_period(interval, period)

    df = fetch_history(ticker, period, interval)
    csv_buf = StringIO()
    df.to_csv(csv_buf, index=False)

    resp = make_response(csv_buf.getvalue())
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = f"attachment; filename={ticker}_{period}_{interval}.csv"
    return resp


@app.route("/weather")
def weather():
    city = (request.args.get("city") or WEATHER_DEFAULT_CITY).strip()
    try:
        days = int(request.args.get("days", 7))
    except Exception:
        days = 7
    days = max(1, min(30, days))

    units = request.args.get("units", "metric")
    if units == "imperial":
        temp_unit, wind_unit = "fahrenheit", "mph"
    else:
        units, temp_unit, wind_unit = "metric", "celsius", "kmh"

    error, meta, chart = None, {}, None
    try:
        loc = geocode_city(city or WEATHER_DEFAULT_CITY, count=1, lang="pt")
        if not loc:
            raise RuntimeError("Cidade não encontrada. Tenta outro nome (ex.: 'Porto', 'Coimbra').")

        meta = {
            "city": f"{loc['name']}, {loc.get('country','')}".strip().strip(","),
            "lat": loc["latitude"],
            "lon": loc["longitude"],
            "timezone": loc.get("timezone", "auto"),
            "days": days,
            "temp_unit": "°C" if temp_unit == "celsius" else "°F",
            "wind_unit": "km/h" if wind_unit == "kmh" else wind_unit,
        }

        js = fetch_weather_forecast(loc["latitude"], loc["longitude"], days, temp_unit, wind_unit, lang="pt")
        daily = js.get("daily") or {}
        dates = daily.get("time") or []
        tmax = daily.get("temperature_2m_max") or []
        tmin = daily.get("temperature_2m_min") or []
        rain = daily.get("precipitation_sum") or []
        wmax = daily.get("wind_speed_10m_max") or []

        if not dates:
            raise RuntimeError("Sem dados de previsão para este local.")

        chart = {
            "labels": dates,
            "tmax": [round(x, 2) if x is not None else None for x in tmax],
            "tmin": [round(x, 2) if x is not None else None for x in tmin],
            "rain": [round(x, 2) if x is not None else None for x in rain],
            "wmax": [round(x, 2) if x is not None else None for x in wmax],
        }
    except Exception as e:
        error = str(e)

    return safe_render("weather.html", city=city, meta=meta, chart=chart, units=units, error=error)


@app.route("/weather.csv")
def weather_csv():
    city = (request.args.get("city") or WEATHER_DEFAULT_CITY).strip()
    try:
        days = int(request.args.get("days", 7))
    except Exception:
        days = 7
    days = max(1, min(30, days))

    units = request.args.get("units", "metric")
    temp_unit = "celsius" if units != "imperial" else "fahrenheit"
    wind_unit = "kmh" if units != "imperial" else "mph"

    loc = geocode_city(city or WEATHER_DEFAULT_CITY, count=1, lang="pt")
    if not loc:
        return make_response("Cidade não encontrada", 400)

    js = fetch_weather_forecast(loc["latitude"], loc["longitude"], days, temp_unit, wind_unit, lang="pt")
    daily = js.get("daily") or {}
    df = pd.DataFrame({
        "date": daily.get("time") or [],
        "temp_max": daily.get("temperature_2m_max") or [],
        "temp_min": daily.get("temperature_2m_min") or [],
        "precipitation_mm": daily.get("precipitation_sum") or [],
        "wind_speed_max": daily.get("wind_speed_10m_max") or [],
    })
    csv_buf = StringIO()
    df.to_csv(csv_buf, index=False)
    resp = make_response(csv_buf.getvalue())
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    safe_city = (loc.get("name") or city).replace(" ", "_")
    resp.headers["Content-Disposition"] = f"attachment; filename=weather_{safe_city}_{days}d.csv"
    return resp


@app.route("/ml/heart", methods=["GET", "POST"])
def ml_heart():
    error = _HEART_LOAD_ERROR
    result = None
    prob = None

    def _dflt(col, fallback):
        try:
            if _HEART_STATS and col in _HEART_STATS:
                med = _HEART_STATS[col]["median"]
                return str(int(med)) if float(med).is_integer() else str(med)
        except Exception:
            pass
        return str(fallback)

    defaults = {
        "sexo": "1",
        "idade": _dflt("idade", 50),
        "cigarros_por_dia": _dflt("cigarros_por_dia", 0),
        "uso_medicamento_pressao": "0",
        "AVC": "0",
        "hipertensao": "0",
        "diabetes": "0",
        "colesterol_total": _dflt("colesterol_total", 220),
        "pressao_arterial_sistolica": _dflt("pressao_arterial_sistolica", 130),
        "pressao_arterial_diastolica": _dflt("pressao_arterial_diastolica", 80),
        "IMC": _dflt("IMC", 27.0),
        "freq_cardiaca": _dflt("freq_cardiaca", 80),
        "glicemia": _dflt("glicemia", 90),
        "fumante": "0",
    }
    defaults = {k: v for k, v in defaults.items() if k in _HEART_FEATURES}

    if request.method == "POST" and _HEART_MODEL is not None:
        try:
            vals = {}
            for k in _HEART_FEATURES:
                v = request.form.get(k, "").strip().replace(",", ".")
                vals[k] = float(v) if v != "" else 0.0
            Xq = pd.DataFrame([vals], columns=_HEART_FEATURES)
            y_hat = int(_HEART_MODEL.predict(Xq)[0])
            p_hat = None
            if hasattr(_HEART_MODEL, "predict_proba"):
                p_all = _HEART_MODEL.predict_proba(Xq)[0]
                p_hat = float(max(p_all))
            result = "Risco ALTO (provável presença)" if y_hat == 1 else "Risco BAIXO (provável ausência)"
            prob = p_hat
            for k in _HEART_FEATURES:
                defaults[k] = str(vals[k])
        except Exception as e:
            error = f"Entrada inválida: {e}"

    heart_info = {"source": _HEART_SOURCE or "indisponível"}
    try:
        art_dir = Path(app.config["ARTIF_DIR"])
        metas = sorted((art_dir.glob("meta_*.joblib")), reverse=True)
        for mp in metas:
            meta = joblib_load(mp)
            if meta.get("task") == "heart_disease":
                heart_info["version"] = meta.get("version")
                break
    except Exception:
        pass

    return safe_render(
        "ml_heart.html",
        error=error,
        result=result,
        prob=prob,
        defaults=defaults,
        stats=_HEART_STATS,
        features=_HEART_FEATURES,
        heart_info=heart_info
    )


@app.route("/ml/heart/api", methods=["GET", "POST"])
def ml_heart_api():
    if request.method == "GET":
        return jsonify({
            "ok": _HEART_MODEL is not None,
            "error": _HEART_LOAD_ERROR,
            "features": _HEART_FEATURES,
            "source": _HEART_SOURCE
        })
    if _HEART_MODEL is None:
        return jsonify({"error": f"Modelo indisponível: {_HEART_LOAD_ERROR or 'falha ao carregar'}"}), 503
    try:
        data = request.get_json(force=True)
        row = [float(str(data.get(k, "0")).replace(",", ".")) for k in _HEART_FEATURES]
        Xq = pd.DataFrame([row], columns=_HEART_FEATURES)
        y_hat = int(_HEART_MODEL.predict(Xq)[0])
        proba = _HEART_MODEL.predict_proba(Xq)[0].tolist() if hasattr(_HEART_MODEL, "predict_proba") else None
        return jsonify({"prediction": y_hat, "probabilities": proba, "features": _HEART_FEATURES})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/nlp/supervised", methods=["GET", "POST"])
def nlp_supervised():
    error = _NLP_ERROR
    result = None
    score = None
    meta = {}
    engine = f"Modelo salvo (PT) — {os.path.basename(NLP_MODEL_PATH)}"

    defaults = {"text": request.values.get("text", "")}

    if request.method == "POST":
        try:
            txt = request.form.get("text", "").strip()
            defaults["text"] = txt
            if not txt:
                raise ValueError("Digite um texto para analisar.")
            label, s, m = predict_sentiment_label(txt)
            result, score, meta = label, s, m
        except Exception as e:
            error = f"Erro: {e}"

    return safe_render(
        "nlp_supervised.html",
        error=error,
        result=result,
        score=score,
        meta=meta,
        engine=engine,
        stats=_NLP_STATS,
        defaults=defaults
    )


@app.route("/api/nlp/supervised", methods=["POST"])
def nlp_supervised_api():
    if _NLP_MODEL is None and _NLP_ERROR:
        return jsonify({"ok": False, "error": _NLP_ERROR}), 503
    try:
        js = request.get_json(force=True)
        txt = (js.get("text") or "").strip()
        if not txt:
            return jsonify({"ok": False, "error": "texto vazio"}), 400
        label, s, m = predict_sentiment_label(txt)
        return jsonify({"ok": True, "label": label, "score_pos": s, "meta": m})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/loan", methods=["GET"])
def loan_form():
    return safe_render("loan.html", opcoes=LOAN_OPCOES, erro=_LOAN_ERROR)


@app.route("/loan/prever", methods=["POST"])
def loan_prever():
    if _LOAN_PIPE is None:
        return safe_render("loan.html", opcoes=LOAN_OPCOES, erro=_LOAN_ERROR or "Pipeline indisponível.")

    payload = {}
    for c in LOAN_CAMPO_NUM:
        payload[c] = _to_float_or_none(request.form.get(c))
    for c in LOAN_CAMPO_CAT:
        payload[c] = request.form.get(c) or None

    cols = LOAN_CAMPO_NUM + LOAN_CAMPO_CAT
    X_in = pd.DataFrame([payload], columns=cols)

    try:
        prob_default = None
        pred = None

        if hasattr(_LOAN_PIPE, "predict_proba"):
            prob_default = float(_LOAN_PIPE.predict_proba(X_in)[:, 1][0])
            pred = int(prob_default >= 0.5)
        elif hasattr(_LOAN_PIPE, "decision_function"):
            import math
            score = float(_LOAN_PIPE.decision_function(X_in)[0])
            prob_default = 1.0 / (1.0 + math.exp(-score))
            pred = int(prob_default >= 0.5)
        else:
            pred = int(_LOAN_PIPE.predict(X_in)[0])
            prob_default = 0.5

    except Exception as e:
        return safe_render("loan.html", opcoes=LOAN_OPCOES, erro=f"Erro ao prever: {e}")

    risco_txt = "ALTO (tende a default)" if pred == 1 else "BAIXO (tende a pagar)"
    prob_fmt = f"{prob_default:.2%}"

    return safe_render(
        "loan.html",
        prob=prob_fmt,
        classe=pred,
        risco=risco_txt,
        entrada=payload,
        opcoes=LOAN_OPCOES,
        erro=None
    )


@app.route("/churn_xai", methods=["GET", "POST"])
def churn_xai_dashboard():
    erro = _CHURN_ERROR
    result = None
    prob_pct = None

    defaults = {
        "gender": request.form.get("gender", "Female"),
        "SeniorCitizen": request.form.get("SeniorCitizen", "0"),
        "Partner": request.form.get("Partner", "No"),
        "Dependents": request.form.get("Dependents", "No"),
        "tenure": request.form.get("tenure", "12"),
        "PhoneService": request.form.get("PhoneService", "Yes"),
        "MultipleLines": request.form.get("MultipleLines", "No"),
        "InternetService": request.form.get("InternetService", "Fiber optic"),
        "OnlineSecurity": request.form.get("OnlineSecurity", "No"),
        "OnlineBackup": request.form.get("OnlineBackup", "No"),
        "DeviceProtection": request.form.get("DeviceProtection", "No"),
        "TechSupport": request.form.get("TechSupport", "No"),
        "StreamingTV": request.form.get("StreamingTV", "Yes"),
        "StreamingMovies": request.form.get("StreamingMovies", "Yes"),
        "Contract": request.form.get("Contract", "Month-to-month"),
        "PaperlessBilling": request.form.get("PaperlessBilling", "Yes"),
        "PaymentMethod": request.form.get("PaymentMethod", "Electronic check"),
        "MonthlyCharges": request.form.get("MonthlyCharges", "70.0"),
        "TotalCharges": request.form.get("TotalCharges", "1400.0"),
    }

    if request.method == "POST":
        if _CHURN_MODEL is None or _CHURN_SCALER is None or _CHURN_FEATURES is None:
            erro = _CHURN_ERROR or "Modelo de churn não está totalmente carregado (modelo/scaler/features)."
        else:
            try:
                row = {
                    "gender": defaults["gender"],
                    "SeniorCitizen": int(defaults["SeniorCitizen"]),
                    "Partner": defaults["Partner"],
                    "Dependents": defaults["Dependents"],
                    "tenure": float(str(defaults["tenure"]).replace(",", ".")),
                    "PhoneService": defaults["PhoneService"],
                    "MultipleLines": defaults["MultipleLines"],
                    "InternetService": defaults["InternetService"],
                    "OnlineSecurity": defaults["OnlineSecurity"],
                    "OnlineBackup": defaults["OnlineBackup"],
                    "DeviceProtection": defaults["DeviceProtection"],
                    "TechSupport": defaults["TechSupport"],
                    "StreamingTV": defaults["StreamingTV"],
                    "StreamingMovies": defaults["StreamingMovies"],
                    "Contract": defaults["Contract"],
                    "PaperlessBilling": defaults["PaperlessBilling"],
                    "PaymentMethod": defaults["PaymentMethod"],
                    "MonthlyCharges": float(str(defaults["MonthlyCharges"]).replace(",", ".")),
                    "TotalCharges": float(str(defaults["TotalCharges"]).replace(",", ".")),
                }
                df_raw = pd.DataFrame([row])
                pred, prob = predict_churn_from_raw(df_raw)
                prob_pct = round(prob * 100, 1)
                result = "ALTO risco de churn" if pred == 1 else "BAIXO risco de churn"
            except Exception as e:
                erro = f"Erro ao processar os dados ou prever: {e}"

    return safe_render("churn_xai.html", erro=erro, result=result, prob_pct=prob_pct, defaults=defaults)


@app.route("/ames", methods=["GET", "POST"])
def ames_dashboard():
    df_completo = load_ames_data()
    df = df_completo.copy()

    numeric_cols = [c for c in COLUNAS_PROJETO if c in df.columns and c != "faixa_preco"]
    if not numeric_cols:
        return jsonify({"error": "Sem colunas numéricas disponíveis no dataset Ames."}), 500

    default_var = "preco" if "preco" in numeric_cols else numeric_cols[0]
    var = request.form.get("variavel", default_var)

    faixas_unicas = ["Todos"]
    if "faixa_preco" in df.columns:
        faixas_unicas += sorted(df["faixa_preco"].dropna().unique().tolist())

    faixa_selecionada = request.form.get("faixa_preco", "Todos")

    df_filtrado = df.copy()
    if faixa_selecionada != "Todos" and "faixa_preco" in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado["faixa_preco"] == faixa_selecionada]

    if var not in df_filtrado.columns:
        var = default_var

    serie = df_filtrado[var].dropna()
    stats_dict = calcular_estatisticas_1d(serie)

    testes_extra = calcular_testes_adicionais(
        serie=serie,
        df_filtrado=df_filtrado,
        var=var,
        df_completo=df_completo if faixa_selecionada == "Todos" else None,
    )

    label = NOMES_AMIGAVEIS.get(var, var)

    fig_hist = px.histogram(
        df_filtrado,
        x=var,
        nbins=40,
        marginal="box",
        title=f"Distribuição de {label}",
        labels={var: label},
    )

    fig_box = px.box(
        df_filtrado,
        y=var,
        points="outliers",
        title=f"Boxplot de {label}",
        labels={var: label},
    )

    graph_hist_json = json.dumps(fig_hist, cls=plotly.utils.PlotlyJSONEncoder)
    graph_box_json = json.dumps(fig_box, cls=plotly.utils.PlotlyJSONEncoder)

    interpretacao_normalidade = None
    if stats_dict["p_valor_shapiro"] is not None:
        alpha = 0.05
        if stats_dict["p_valor_shapiro"] < alpha:
            interpretacao_normalidade = (
                "p < 0,05 ⇒ rejeitamos a hipótese de normalidade "
                "(a distribuição não é aproximadamente normal)."
            )
        else:
            interpretacao_normalidade = (
                "p ≥ 0,05 ⇒ não rejeitamos a hipótese de normalidade "
                "(a distribuição pode ser considerada aproximadamente normal)."
            )

    return safe_render(
        "ames.html",
        variavel_selecionada=var,
        variaveis=numeric_cols,
        faixa_selecionada=faixa_selecionada,
        faixas=faixas_unicas,
        estatisticas=stats_dict,
        testes_extra=testes_extra,
        graph_hist_json=graph_hist_json,
        graph_box_json=graph_box_json,
        interpretacao_normalidade=interpretacao_normalidade,
        nomes_amigaveis=NOMES_AMIGAVEIS,
        nomes_faixa=NOMES_FAIXA,
    )


# ============================================================
# ==== Entrypoint local (produção Railway usa gunicorn) ======
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=app.config.get("DEBUG", False))
