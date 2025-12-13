import os
from datetime import datetime
from io import StringIO
from typing import Tuple, Optional
import re
import pickle
from pathlib import Path
from scipy import stats
from functools import lru_cache
import json
import statsmodels.api as sm


import numpy as np
import pandas as pd
import yfinance as yf
import requests

from flask import (
    Flask, render_template, request, jsonify, make_response, redirect, url_for
)

# ==== Sklearn imports ====
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from joblib import load as joblib_load

# ==== Extras para FX (câmbio) ====
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor  # regressão, diferente do Classifier lá de cima

from statsmodels.tsa.arima.model import ARIMA       # pip install statsmodels
from prophet import Prophet                         # pip install prophet
import plotly.graph_objs as go
from plotly.offline import plot
import plotly
import plotly.express as px

app = Flask(__name__)

# ======================================================
# Caminhos base (funcionam localmente e no Railway)
# ======================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def load_dataset(filename: str) -> pd.DataFrame:
    """Carrega um CSV da pasta data/ de forma segura."""
    path = DATA_DIR / filename
    return pd.read_csv(path)


# ======== Presets de Ações (legenda) ========
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

# ======== Helpers — Cotações ========


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


# ======== Helpers — Tempo (Open-Meteo) ========

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


# ======== FX — Previsão de Câmbio (RF / ARIMA / Prophet) ========

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

    return {"mae": mae, "rmse": rmse, "mape": mape}


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


# ======== ML — Heart Disease (novo dataset em PT) ========

DEFAULT_HEART_FEATURES = [
    "sexo",
    "idade",
    "cigarros_por_dia",
    "uso_medicamento_pressao",
    "AVC",
    "hipertensao",
    "diabetes",
    "colesterol_total",
    "pressao_arterial_sistolica",
    "pressao_arterial_diastolica",
    "IMC",
    "freq_cardiaca",
    "glicemia",
    "fumante",
]
HEART_TARGET = "risco_DAC_decada"

HEART_RAW_URL = os.getenv(
    "HEART_RAW_URL",
    "https://raw.githubusercontent.com/MpRonald/datasets/main/doenca_cardiaca_final.csv"
)

ARTIF_DIR = BASE_DIR / "artifacts"

_HEART_FEATURES = DEFAULT_HEART_FEATURES[:]
_HEART_MODEL = None
_HEART_STATS = None
_HEART_LOAD_ERROR = None
_HEART_SOURCE = None  # "artifact" ou "csv-train"


def _attempt_load_heart_artifact():
    if not ARTIF_DIR.exists():
        return None, None, None
    try:
        metas = sorted(ARTIF_DIR.glob("meta_*.joblib"), reverse=True)
        for mp in metas:
            meta = joblib_load(mp)
            if meta.get("task") != "heart_disease":
                continue
            version = meta["version"]
            mpath = ARTIF_DIR / f"model_{version}.joblib"
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

# ====================== NLP — Modelo PT carregado de arquivo ======================

# Se NLP_MODEL_PATH não vier via env, usa um ficheiro na pasta do projeto
NLP_MODEL_PATH = os.getenv("NLP_MODEL_PATH")
if NLP_MODEL_PATH is None:
    NLP_MODEL_PATH = str(BASE_DIR / "modelo_sentimento.joblib")

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
        bundle = joblib_load(NLP_MODEL_PATH)
    except Exception as e1:
        try:
            alt = os.path.splitext(NLP_MODEL_PATH)[0] + ".pkl"
            with open(alt, "rb") as f:
                bundle = pickle.load(f)
        except Exception as e2:
            _NLP_MODEL, _NLP_ERROR = None, f"Falha ao carregar modelo salvo: {e1} | {e2}"
            _NLP_STATS = None
            return

    if not isinstance(bundle, dict) or "tfidf" not in bundle or "clf" not in bundle:
        _NLP_MODEL, _NLP_ERROR = None, "Arquivo de modelo inválido (esperado dict com 'tfidf' e 'clf')."
        _NLP_STATS = None
        return

    classes = bundle.get("classes_", None)
    if classes is None:
        classes = getattr(bundle["clf"], "classes_", None)

    classes_list = _to_list_or_none(classes)
    _NLP_MODEL = {"tfidf": bundle["tfidf"], "clf": bundle["clf"], "classes_": classes_list}
    _NLP_STATS = {"classes": [str(c) for c in classes_list] if classes_list is not None else None}
    _NLP_ERROR = None


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
        "score_strategy": "predict_proba" if proba is not None else "heuristic_from_predict"
    }
    return label_pt, score_pos, meta


# ========== ML — Loan Default (estado_emprestimo) ==========

LOAN_MODEL_PATH = os.getenv("LOAN_MODEL_PATH", "molde_credit_risk.joblib")

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
    raise ValueError(f"Objeto .joblib é um dict sem chave de estimador ('pipeline'/'model'/'pipe'/'estimator'). Chaves: {keys}")


try:
    _loaded = joblib_load(LOAN_MODEL_PATH)
    _LOAN_PIPE, _LOAN_META = _extract_pipeline(_loaded)
except Exception as e:
    _LOAN_PIPE = None
    _LOAN_META = None
    _LOAN_ERROR = f"Falha ao carregar pipeline de empréstimo em '{LOAN_MODEL_PATH}': {e}"

LOAN_CAMPO_NUM = [
    'idade',
    'rendimento_anual',
    'anos_emprego',
    'valor_emprestimo',
    'taxa_juros',
    'percent_rendimento',
    'anos_historico_credito'
]
LOAN_CAMPO_CAT = [
    'tipo_habitacao',
    'finalidade_emprestimo',
    'grau_emprestimo',
    'historico_inadimplencia'
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


@app.route("/loan", methods=["GET"])
def loan_form():
    return render_template("loan.html", opcoes=LOAN_OPCOES, erro=_LOAN_ERROR)


@app.route("/loan/prever", methods=["POST"])
def loan_prever():
    if _LOAN_PIPE is None:
        return render_template("loan.html", opcoes=LOAN_OPCOES, erro=_LOAN_ERROR or "Pipeline indisponível.")

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
        return render_template("loan.html", opcoes=LOAN_OPCOES, erro=f"Erro ao prever: {e}")

    risco_txt = "ALTO (tende a default)" if pred == 1 else "BAIXO (tende a pagar)"
    prob_fmt = f"{prob_default:.2%}"

    return render_template(
        "loan.html",
        prob=prob_fmt,
        classe=pred,
        risco=risco_txt,
        entrada=payload,
        opcoes=LOAN_OPCOES,
        erro=None
    )


# ========== ML — Churn de Clientes (Telco) ==========

CHURN_MODEL_PATH = os.getenv(
    "CHURN_MODEL_PATH",
    str(BASE_DIR / "modelo_churn.joblib")
)
CHURN_SCALER_PATH = os.getenv(
    "CHURN_SCALER_PATH",
    str(BASE_DIR / "modelo_churn_scaler.joblib")
)
CHURN_FEATURES_PATH = os.getenv(
    "CHURN_FEATURES_PATH",
    str(BASE_DIR / "modelo_churn_feature_names.joblib")
)

CHURN_DATA_PATH = DATA_DIR / "telco_customer_churn.csv"

_CHURN_MODEL = None
_CHURN_SCALER = None
_CHURN_FEATURES = None
_CHURN_ERROR = None


def _load_churn_artifacts():
    global _CHURN_MODEL, _CHURN_SCALER, _CHURN_FEATURES, _CHURN_ERROR
    try:
        _CHURN_MODEL = joblib_load(CHURN_MODEL_PATH)
        _CHURN_SCALER = joblib_load(CHURN_SCALER_PATH)
        _CHURN_FEATURES = joblib_load(CHURN_FEATURES_PATH)
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
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    binary_columns = ['Partner', 'Dependents', 'PhoneService',
                      'PaperlessBilling', 'Churn']
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    cat_cols_to_dummies = [
        'gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract',
        'PaymentMethod'
    ]
    existing_cat_cols = [c for c in cat_cols_to_dummies if c in df.columns]
    if existing_cat_cols:
        df = pd.get_dummies(
            df,
            columns=existing_cat_cols,
            drop_first=True
        )

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

    df_proc = preprocess_telco(
        df_raw,
        scaler=_CHURN_SCALER,
        feature_names=_CHURN_FEATURES
    )

    proba = _CHURN_MODEL.predict_proba(df_proc)[0, 1]
    pred = _CHURN_MODEL.predict(df_proc)[0]

    return int(pred), float(proba)


# ============================================================
# 🔹 DICIONÁRIO DE TRADUÇÃO DAS VARIÁVEIS IMPORTANTES DO AMES
# ============================================================

COLUNAS_PROJETO = [
    "preco",
    "quartos",
    "banheiros",
    "area_habitavel",
    "area_lote",
    "andares",
    "area_acima_solo",
    "area_porao",
    "ano_construcao",
    "latitude",
    "longitude",
    "area_habitavel_viz",
    "area_lote_viz",
    "faixa_preco",
    "idade_imovel",
    "area_total",
    "densidade_construcao",
    "preco_m2",
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

CORR_VARS = [
    "preco",
    "preco_m2",
    "area_total",
    "area_habitavel",
    "area_lote",
    "quartos",
    "banheiros",
    "andares",
    "idade_imovel",
    "densidade_construcao",
]

VARIAVEIS_IMPORTANTES = list(NOMES_AMIGAVEIS.keys())


@lru_cache(maxsize=1)
def load_ames_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "ames.csv"
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
        stat, p_valor = stats.shapiro(sample)
    except Exception:
        stat, p_valor = np.nan, np.nan

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
        "stat_shapiro": float(stat) if not np.isnan(stat) else None,
        "p_valor_shapiro": float(p_valor) if not np.isnan(p_valor) else None,
    }


def calcular_testes_adicionais(
    serie: pd.Series,
    df_filtrado: pd.DataFrame,
    var: str,
    df_completo: pd.DataFrame | None,
) -> dict:
    resultados: dict[str, float | None] = {}

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

    # Kruskal–Wallis por faixa_preco (global)
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



# ======== Rotas ========

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/fx", methods=["GET", "POST"])
def fx():
    erro = None
    resultado = None

    if request.method == "POST":
        try:
            pair_name = request.form.get("pair")
            algoritmo = request.form.get("algoritmo")
            n_days_raw = request.form.get("n_days", "30")

            # Sanitizar nº de dias
            try:
                n_days = int(n_days_raw)
            except Exception:
                n_days = 30

            if n_days < 1:
                n_days = 1
            if n_days > 180:
                n_days = 180

            # Validar inputs
            if pair_name not in FX_PAIRS:
                raise ValueError("Par de moedas inválido.")
            if algoritmo not in ["rf", "arima", "prophet"]:
                raise ValueError("Algoritmo inválido.")

            # Série histórica (3 anos)
            ticker = FX_PAIRS[pair_name]
            series = fx_download_history(ticker, period="3y")

            # Treinar / prever conforme o algoritmo
            if algoritmo == "rf":
                metrics, forecast_df = fx_train_and_forecast_rf(series, n_days)
                alg_label = "Random Forest"
            elif algoritmo == "arima":
                metrics, forecast_df = fx_train_and_forecast_arima(series, n_days)
                alg_label = "ARIMA (1,1,1)"
            else:
                metrics, forecast_df = fx_train_and_forecast_prophet(series, n_days)
                alg_label = "Prophet"

            # Histórico do último ano (para deixar o gráfico mais limpo)
            history_last_year = series.last("365D")

            # Dados para o gráfico
            x_hist = [d.date() for d in history_last_year.index]
            y_hist = [float(v) for v in history_last_year.values]

            x_fore = [d.date() for d in forecast_df.index]
            y_fore = [float(v) for v in forecast_df["forecast_rate"].values]

            # Figura Plotly
            fig = go.Figure()

            # Série histórica – linha lisa, sem marcadores
            fig.add_trace(
                go.Scatter(
                    x=x_hist,
                    y=y_hist,
                    mode="lines",
                    name="Histórico",
                    line=dict(width=2)
                )
            )

            # Série de previsão – linha tracejada, sem marcadores
            fig.add_trace(
                go.Scatter(
                    x=x_fore,
                    y=y_fore,
                    mode="lines",
                    name="Previsão",
                    line=dict(width=2, dash="dash")
                )
            )

            fig.update_layout(
                title=f"Histórico e previsão — {pair_name}",
                xaxis_title="Data",
                yaxis_title=f"Taxa {pair_name}",
                margin=dict(l=40, r=20, t=40, b=40),
                template="plotly_white",      # tema claro, alinhado com o resto do site
                height=400,
                hovermode="x unified",
                paper_bgcolor="white",
                plot_bgcolor="white",
            )

            fig.update_yaxes(tickformat=".3f")

            fx_plot_div = plot(fig, include_plotlyjs="cdn", output_type="div")

            # Monta o resultado enviado ao template
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

    return render_template(
        "fx.html",
        pairs=FX_PAIRS,
        erro=erro,
        resultado=resultado
    )


# ---------- Cotações + Previsão ----------

@app.route("/quotes", methods=["GET", "POST"])
def quotes():
    # Defaults para formulário
    DEFAULT_TICKER = "AAPL"
    DEFAULT_ALGO = "rf"
    DEFAULT_N_DAYS = 30

    # --- Ler inputs do formulário/URL ---
    preset_ticker = request.values.get("preset_ticker")
    default_ticker = preset_ticker if preset_ticker else DEFAULT_TICKER

    raw_period   = request.values.get("period", "1y")
    raw_interval = request.values.get("interval", "1d")
    period = _coerce_period_month(raw_period)
    interval, period, notice = normalize_interval_and_period(raw_interval, period)

    algoritmo = request.values.get("algoritmo", DEFAULT_ALGO)
    n_days_raw = request.values.get("n_days", str(DEFAULT_N_DAYS))
    try:
        n_days = int(n_days_raw)
    except Exception:
        n_days = DEFAULT_N_DAYS
    if n_days < 1:
        n_days = 1
    if n_days > 180:
        n_days = 180

    # Form dict para o template
    form = {
        "ticker":   request.values.get("ticker", default_ticker).upper(),
        "period":   period,
        "interval": interval,
        "algoritmo": algoritmo,
        "n_days":   n_days,
    }

    error = None
    metrics = None
    forecast_df = None
    stock_plot_div = None
    alg_label = None

    try:
        # ---------- PREVISÃO (usando últimos 3 anos diários) ----------
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

        # Histórico recente (1 ano) para o gráfico de ML
        history_last_year = series.last("365D")

        x_hist = [d.date() for d in history_last_year.index]
        y_hist = [float(v) for v in history_last_year.values]

        x_fore = [d.date() for d in forecast_df.index]
        y_fore = [float(v) for v in forecast_df["forecast_rate"].values]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_hist,
                y=y_hist,
                mode="lines",   # sem marcadores
                name="Histórico (fecho)",
                line=dict(width=2)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_fore,
                y=y_fore,
                mode="lines",   # sem marcadores
                name="Previsão",
                line=dict(width=2, dash="dash")
            )
        )

        fig.update_layout(
            title=f"Histórico e previsão — {form['ticker']}",
            xaxis_title="Data",
            yaxis_title=f"Preço {form['ticker']}",
            margin=dict(l=40, r=20, t=40, b=40),
            template="plotly_white",      # tema claro
            height=400,
            hovermode="x unified",
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        fig.update_yaxes(tickformat=".3f")

        stock_plot_div = plot(fig, include_plotlyjs="cdn", output_type="div")

    except Exception as e:
        error = str(e)

    return render_template(
        "quotes.html",
        form=form,
        error=error,
        presets=PRESETS,
        notice=notice,
        metrics=metrics,
        forecast=forecast_df,
        alg_label=alg_label,
        stock_plot_div=stock_plot_div,
    )


# ========= E-COMMERCE DASHBOARD (VENDAS) =========

# Caminho do CSV de e-commerce (reutiliza DATA_DIR definido no topo)
ECOM_PATH = Path(DATA_DIR) / "e_commerce.csv"

ECOM_DF = None
ECOM_LOAD_ERROR = None


def _load_ecom_data():
    """
    Carrega e limpa o dataset de e-commerce.
    Espera um CSV com colunas: 
    InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
    """
    global ECOM_DF, ECOM_LOAD_ERROR

    try:
        df = pd.read_csv(ECOM_PATH, encoding="latin1")

        # Converter data (dataset é dd/mm/yyyy hh:mm)
        df["InvoiceDate"] = pd.to_datetime(
            df["InvoiceDate"],
            dayfirst=True,
            errors="coerce"
        )

        # Remover linhas sem data
        df = df.dropna(subset=["InvoiceDate"])

        # Filtrar vendas válidas (sem devoluções)
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

        # Valor total da linha
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

        # Garantir tipo numérico de CustomerID
        if "CustomerID" in df.columns:
            df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")

        ECOM_DF = df
        ECOM_LOAD_ERROR = None
    except Exception as e:
        ECOM_DF = None
        ECOM_LOAD_ERROR = f"Erro ao carregar dataset de e-commerce: {e}"


# carregar ao subir a app
_load_ecom_data()


# ---------- Meteo ----------
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
            "wind_unit": "km/h" if wind_unit == "kmh" else ("mph" if wind_unit == "mph" else wind_unit),
        }

        js = fetch_weather_forecast(
            loc["latitude"],
            loc["longitude"],
            days,
            temp_unit,
            wind_unit,
            lang="pt"
        )

        daily = js.get("daily") or {}
        dates = daily.get("time") or []
        tmax  = daily.get("temperature_2m_max") or []
        tmin  = daily.get("temperature_2m_min") or []
        rain  = daily.get("precipitation_sum") or []
        wmax  = daily.get("wind_speed_10m_max") or []
        prob  = daily.get("precipitation_probability_max") or []  # <- probabilidade de chuva (%)

        if not dates:
            raise RuntimeError("Sem dados de previsão para este local.")

        chart = {
            "labels": dates,
            "tmax": [round(x, 2) if x is not None else None for x in tmax],
            "tmin": [round(x, 2) if x is not None else None for x in tmin],
            "rain": [round(x, 2) if x is not None else None for x in rain],
            "wmax": [round(x, 2) if x is not None else None for x in wmax],
            "prob": [int(x) if x is not None else None for x in prob],
        }

    except Exception as e:
        error = str(e)

    return render_template(
        "weather.html",
        city=city,
        meta=meta,
        chart=chart,
        units=units,
        error=error,
    )


# ---------- ML: Heart Disease ----------
@app.route("/ml/heart", methods=["GET", "POST"])
def ml_heart():
    error = _HEART_LOAD_ERROR
    result = None
    prob = None
    prob_pct = None
    risk_level = None
    risk_insights = []

    def _dflt(col, fallback):
        try:
            if _HEART_STATS and col in _HEART_STATS:
                med = _HEART_STATS[col]["median"]
                return str(int(med)) if float(med).is_integer() else str(med)
        except Exception:
            pass
        return str(fallback)

    # defaults iniciais (usamos medianas quando possível)
    defaults = {
        "sexo":                          "1",
        "idade":                         _dflt("idade", 50),
        "cigarros_por_dia":              _dflt("cigarros_por_dia", 0),
        "uso_medicamento_pressao":       "0",
        "AVC":                           "0",
        "hipertensao":                   "0",
        "diabetes":                      "0",
        "colesterol_total":              _dflt("colesterol_total", 220),
        "pressao_arterial_sistolica":    _dflt("pressao_arterial_sistolica", 130),
        "pressao_arterial_diastolica":   _dflt("pressao_arterial_diastolica", 80),
        "IMC":                           _dflt("IMC", 27.0),
        "freq_cardiaca":                 _dflt("freq_cardiaca", 80),
        "glicemia":                      _dflt("glicemia", 90),
        "fumante":                       "0",
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
                p_hat = float(max(p_all))  # confiança na classe prevista

            result = (
                "Risco ALTO (provável presença de doença coronária em 10 anos)"
                if y_hat == 1
                else "Risco BAIXO (provável ausência de doença coronária em 10 anos)"
            )
            prob = p_hat
            if prob is not None:
                prob_pct = int(round(prob * 100))

            risk_level = "alto" if y_hat == 1 else "baixo"

            # Atualizar defaults com o que o utilizador preencheu
            for k in _HEART_FEATURES:
                defaults[k] = str(vals[k]).replace(".", ",")

            # ------- Análises simples dos fatores de risco (educacional) -------
            idade = vals.get("idade", 0)
            col = vals.get("colesterol_total", 0)
            pas = vals.get("pressao_arterial_sistolica", 0)
            pad = vals.get("pressao_arterial_diastolica", 0)
            imc = vals.get("IMC", 0)
            glic = vals.get("glicemia", 0)
            cigs = vals.get("cigarros_por_dia", 0)
            fumante = vals.get("fumante", 0)

            # Idade
            if idade >= 60:
                risk_insights.append("Idade ≥ 60 anos: risco cardiovascular naturalmente aumentado.")
            elif idade >= 45:
                risk_insights.append("Idade entre 45 e 59 anos: atenção a fatores de risco adicionais.")

            # Colesterol
            if col >= 240:
                risk_insights.append("Colesterol total muito elevado (≥ 240 mg/dL).")
            elif col >= 200:
                risk_insights.append("Colesterol total limítrofe/alto (≥ 200 mg/dL).")

            # Pressão arterial
            if pas >= 140 or pad >= 90:
                risk_insights.append("Pressão arterial em faixa de hipertensão (≥ 140/90 mmHg).")
            elif pas >= 130 or pad >= 85:
                risk_insights.append("Pressão arterial em faixa limítrofe (≥ 130/85 mmHg).")

            # IMC
            if imc >= 30:
                risk_insights.append("IMC em faixa de obesidade (≥ 30).")
            elif imc >= 25:
                risk_insights.append("IMC em faixa de excesso de peso (25–29,9).")

            # Glicemia
            if glic >= 126:
                risk_insights.append("Glicemia em faixa compatível com diabetes (≥ 126 mg/dL).")
            elif glic >= 100:
                risk_insights.append("Glicemia em faixa de risco/metabolismo alterado (100–125 mg/dL).")

            # Tabagismo
            if fumante >= 3 or cigs >= 20:
                risk_insights.append("Perfil de tabagismo intenso: recomendada cessação tabágica.")
            elif fumante >= 1 or cigs > 0:
                risk_insights.append("Consumo de tabaco presente: reduzir/cessar reduz significativamente o risco.")

        except Exception as e:
            error = f"Entrada inválida: {e}"

    heart_info = {"source": _HEART_SOURCE or "indisponível"}
    try:
        metas = sorted((ARTIF_DIR.glob("meta_*.joblib")), reverse=True)
        for mp in metas:
            meta = joblib_load(mp)
            if meta.get("task") == "heart_disease":
                heart_info["version"] = meta.get("version")
                break
    except Exception:
        pass

    return render_template(
        "ml_heart.html",
        error=error,
        result=result,
        prob=prob,
        prob_pct=prob_pct,
        risk_level=risk_level,
        risk_insights=risk_insights,
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


# ---------- NLP: Supervised (modelo PT carregado) ----------
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

    return render_template(
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


# ---------- Dashboard de E-commerce (Vendas) ----------
@app.route("/ecom", methods=["GET"])
def ecom_dashboard():
    """
    Dashboard de vendas da loja online.
    Mostra KPIs + filtros + gráficos de receita mensal, top produtos e top países.
    """
    erro = ECOM_LOAD_ERROR
    kpis = None
    revenue_plot_div = None
    products_plot_div = None
    countries_plot_div = None
    top_products_table = None
    top_countries_table = None

    # Filtros vindos da query string (?start_date=...&end_date=...)
    filtros = {
        "start_date": request.args.get("start_date", "").strip(),
        "end_date": request.args.get("end_date", "").strip(),
        "country": request.args.get("country", "").strip(),
        "product": request.args.get("product", "").strip(),  # agora virá do <select>
    }

    # Listas para os <select>
    countries_list = []
    products_list = []

    if ECOM_DF is not None:
        countries_list = sorted(
            ECOM_DF["Country"].dropna().unique().tolist()
        )
        products_list = sorted(
            ECOM_DF["Description"].dropna().unique().tolist()
        )

    if ECOM_DF is not None and erro is None:
        df = ECOM_DF.copy()

        # ----- Aplicar filtros -----
        # Filtro de data inicial
        if filtros["start_date"]:
            try:
                dt_ini = pd.to_datetime(filtros["start_date"])
                df = df[df["InvoiceDate"] >= dt_ini]
            except Exception:
                pass

        # Filtro de data final
        if filtros["end_date"]:
            try:
                dt_fim = pd.to_datetime(filtros["end_date"])
                df = df[df["InvoiceDate"] <= dt_fim]
            except Exception:
                pass

        # Filtro de país
        if filtros["country"]:
            df = df[df["Country"] == filtros["country"]]

        # Filtro de produto
        if filtros["product"]:
            df = df[df["Description"] == filtros["product"]]

        if df.empty:
            erro = "Nenhum dado encontrado para os filtros selecionados."
        else:
            # ----- KPIs -----
            total_revenue = float(df["TotalPrice"].sum())
            num_orders = int(df["InvoiceNo"].nunique())
            num_customers = int(df["CustomerID"].nunique())
            total_qty = float(df["Quantity"].sum())  # total de itens

            avg_ticket = float(total_revenue / num_orders) if num_orders > 0 else 0.0
            avg_items_per_order = float(total_qty / num_orders) if num_orders > 0 else 0.0
            avg_revenue_per_customer = float(total_revenue / num_customers) if num_customers > 0 else 0.0

            # Período coberto pelos dados filtrados
            first_date = pd.to_datetime(df["InvoiceDate"].min())
            last_date = pd.to_datetime(df["InvoiceDate"].max())
            period_days = max((last_date - first_date).days + 1, 1)
            avg_daily_revenue = float(total_revenue / period_days) if period_days > 0 else 0.0

            kpis = {
                "total_revenue": total_revenue,
                "num_orders": num_orders,
                "num_customers": num_customers,
                "avg_ticket": avg_ticket,
                "avg_items_per_order": avg_items_per_order,
                "avg_revenue_per_customer": avg_revenue_per_customer,
                "period_days": period_days,
                "avg_daily_revenue": avg_daily_revenue,
            }

            # ----- Receita mensal -----
            df_month = (
                df.set_index("InvoiceDate")
                  .resample("M")["TotalPrice"]
                  .sum()
                  .reset_index()
            )
            df_month["MonthStr"] = df_month["InvoiceDate"].dt.strftime("%Y-%m")

            fig_rev = go.Figure()
            fig_rev.add_trace(
                go.Scatter(
                    x=df_month["MonthStr"],
                    y=df_month["TotalPrice"],
                    mode="lines",
                    name="Receita mensal",
                    line=dict(width=2),
                )
            )
            fig_rev.update_layout(
                title="Receita mensal",
                xaxis_title="Mês",
                yaxis_title="Receita",
                margin=dict(l=40, r=20, t=40, b=40),
                template="plotly_white",
                height=350,
                hovermode="x unified",
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            fig_rev.update_yaxes(tickformat=".2f")
            revenue_plot_div = plot(fig_rev, include_plotlyjs="cdn", output_type="div")

            # ----- Top 10 produtos por receita -----
            top_products = (
                df.groupby("Description", as_index=False)["TotalPrice"]
                .sum()
                .sort_values("TotalPrice", ascending=False)
                .head(10)
            )

            # Para o GRÁFICO: inverter a ordem para que o maior apareça em cima
            plot_top_products = top_products.sort_values("TotalPrice", ascending=True)

            fig_prod = go.Figure()
            fig_prod.add_trace(
                go.Bar(
                    x=plot_top_products["TotalPrice"],
                    y=plot_top_products["Description"],
                    orientation="h",
                    name="Receita por produto",
                )
            )

            fig_prod.update_layout(
                title="Top 10 produtos por receita",
                xaxis_title="Receita",
                yaxis_title="Produto",
                margin=dict(l=120, r=20, t=40, b=40),
                template="plotly_white",
                height=450,
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            products_plot_div = plot(fig_prod, include_plotlyjs=False, output_type="div")

            top_products_table = top_products.to_dict(orient="records")

            # ----- Top 10 países por receita -----
            top_countries = (
                df.groupby("Country", as_index=False)["TotalPrice"]
                  .sum()
                  .sort_values("TotalPrice", ascending=False)
                  .head(10)
            )

            fig_ctry = go.Figure()
            fig_ctry.add_trace(
                go.Bar(
                    x=top_countries["Country"],
                    y=top_countries["TotalPrice"],
                    name="Receita por país",
                )
            )
            fig_ctry.update_layout(
                title="Top 10 países por receita",
                xaxis_title="País",
                yaxis_title="Receita",
                margin=dict(l=40, r=20, t=40, b=40),
                template="plotly_white",
                height=350,
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            countries_plot_div = plot(fig_ctry, include_plotlyjs=False, output_type="div")

            top_countries_table = top_countries.to_dict(orient="records")

    return render_template(
        "ecom.html",
        erro=erro,
        kpis=kpis,
        revenue_plot_div=revenue_plot_div,
        products_plot_div=products_plot_div,
        countries_plot_div=countries_plot_div,
        top_products_table=top_products_table,
        top_countries_table=top_countries_table,
        filtros=filtros,
        countries=countries_list,
        products=products_list,
    )


# ---------- Dashboard de E-commerce: RFM / Segmentação ----------
@app.route("/ecom/rfm", methods=["GET"])
def ecom_rfm():
    """
    Análise RFM (Recency, Frequency, Monetary) por cliente
    + segmentação em grupos (VIP, Leal, etc.).
    """
    erro = ECOM_LOAD_ERROR
    rfm_table = None
    seg_summary = None
    seg_plot_div = None

    filtros = {
        "country": request.args.get("country", "").strip(),
    }

    countries_list = []
    if ECOM_DF is not None:
        countries_list = sorted(
            ECOM_DF["Country"].dropna().unique().tolist()
        )

    if ECOM_DF is not None and erro is None:
        df = ECOM_DF.copy()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        if filtros["country"]:
            df = df[df["Country"] == filtros["country"]]

        df = df.dropna(subset=["CustomerID"])

        if df.empty:
            erro = "Nenhum dado encontrado para os filtros selecionados."
        else:
            ref_date = df["InvoiceDate"].max()

            rfm = (
                df.groupby("CustomerID")
                  .agg(
                      Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
                      Frequency=("InvoiceNo", "nunique"),
                      Monetary=("TotalPrice", "sum"),
                  )
                  .reset_index()
            )

            if rfm.empty:
                erro = "Não foi possível calcular RFM (sem clientes válidos)."
            else:
                try:
                    rfm["R_score"] = pd.qcut(
                        rfm["Recency"], 4, labels=[4, 3, 2, 1]
                    ).astype(int)
                except Exception:
                    rfm["R_score"] = 2

                try:
                    rfm["F_score"] = pd.qcut(
                        rfm["Frequency"], 4, labels=[1, 2, 3, 4]
                    ).astype(int)
                except Exception:
                    rfm["F_score"] = 2

                try:
                    rfm["M_score"] = pd.qcut(
                        rfm["Monetary"], 4, labels=[1, 2, 3, 4]
                    ).astype(int)
                except Exception:
                    rfm["M_score"] = 2

                rfm["RFM_score"] = (
                    rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
                )

                def _segment(row):
                    s = row["RFM_score"]
                    if s >= 10:
                        return "Clientes VIP"
                    elif s >= 8:
                        return "Clientes Leais"
                    elif s >= 5:
                        return "Em crescimento"
                    else:
                        return "Em risco"

                rfm["Segment"] = rfm.apply(_segment, axis=1)

                seg_summary_df = (
                    rfm.groupby("Segment", as_index=False)
                       .agg(
                           num_customers=("CustomerID", "nunique"),
                           avg_recency=("Recency", "mean"),
                           avg_frequency=("Frequency", "mean"),
                           total_monetary=("Monetary", "sum"),
                       )
                       .sort_values("total_monetary", ascending=False)
                )

                seg_summary = seg_summary_df.to_dict(orient="records")

                fig_seg = go.Figure()
                fig_seg.add_trace(
                    go.Bar(
                        x=seg_summary_df["Segment"],
                        y=seg_summary_df["num_customers"],
                        name="Nº de clientes",
                    )
                )
                fig_seg.update_layout(
                    title="Distribuição de clientes por segmento RFM",
                    xaxis_title="Segmento",
                    yaxis_title="Nº de clientes",
                    margin=dict(l=40, r=20, t=40, b=40),
                    template="plotly_white",
                    height=350,
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                )
                seg_plot_div = plot(fig_seg, include_plotlyjs="cdn", output_type="div")

                rfm_table = (
                    rfm.sort_values(["RFM_score", "Monetary"], ascending=[False, False])
                       .head(200)
                       .to_dict(orient="records")
                )

    return render_template(
        "ecom_rfm.html",
        erro=erro,
        filtros=filtros,
        countries=countries_list,
        seg_plot_div=seg_plot_div,
        seg_summary=seg_summary,
        rfm_table=rfm_table,
    )


# ---------- Dashboard de E-commerce: Previsão de Receita ----------
@app.route("/ecom/forecast", methods=["GET"])
def ecom_forecast():
    """
    Previsão de receita mensal usando Prophet.
    Agrupa as vendas por mês, treina o modelo e prevê os próximos N meses.
    """
    erro = ECOM_LOAD_ERROR
    kpis = None
    forecast_plot_div = None

    periods_raw = request.args.get("periods", "12")
    try:
        periods = int(periods_raw)
    except Exception:
        periods = 12
    periods = max(1, min(36, periods))

    if ECOM_DF is not None and erro is None:
        df = ECOM_DF.copy()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        df_month = (
            df.set_index("InvoiceDate")
              .resample("M")["TotalPrice"]
              .sum()
              .reset_index()
        )

        if df_month.empty or len(df_month) < 6:
            erro = "Poucos dados mensais para treinar o modelo de previsão."
        else:
            df_prophet = df_month.rename(columns={"InvoiceDate": "ds", "TotalPrice": "y"})

            m = Prophet()
            m.fit(df_prophet)

            future = m.make_future_dataframe(periods=periods, freq="M")
            forecast = m.predict(future)

            last_ds = df_prophet["ds"].max()
            hist_mask = forecast["ds"] <= last_ds
            fut_mask = forecast["ds"] > last_ds

            hist = forecast.loc[hist_mask, ["ds", "yhat"]].copy()
            fut = forecast.loc[fut_mask, ["ds", "yhat"]].copy()

            hist["MonthStr"] = hist["ds"].dt.strftime("%Y-%m")
            fut["MonthStr"] = fut["ds"].dt.strftime("%Y-%m")

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=df_prophet["ds"].dt.strftime("%Y-%m"),
                    y=df_prophet["y"],
                    name="Receita real (mensal)",
                )
            )

            if not fut.empty:
                fig.add_trace(
                    go.Scatter(
                        x=fut["MonthStr"],
                        y=fut["yhat"],
                        mode="lines",
                        name="Previsão (yhat)",
                        line=dict(width=2, dash="dash"),
                    )
                )

            fig.update_layout(
                title=f"Receita mensal e previsão ({periods} meses à frente)",
                xaxis_title="Mês",
                yaxis_title="Receita",
                margin=dict(l=40, r=20, t=40, b=40),
                template="plotly_white",
                height=400,
                hovermode="x unified",
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            fig.update_yaxes(tickformat=".2f")

            forecast_plot_div = plot(fig, include_plotlyjs="cdn", output_type="div")

            df_sorted = df_prophet.sort_values("ds").reset_index(drop=True)

            last12_hist = df_sorted.tail(12)
            last12_revenue = float(last12_hist["y"].sum())

            next12_fut = fut.sort_values("ds").head(12)
            next12_revenue = float(next12_fut["yhat"].sum()) if not next12_fut.empty else None

            cagr = None
            try:
                first_val = float(df_sorted["y"].iloc[0])
                last_val = float(df_sorted["y"].iloc[-1])
                years = (df_sorted["ds"].iloc[-1] - df_sorted["ds"].iloc[0]).days / 365.25
                if years > 0 and first_val > 0:
                    cagr = (last_val / first_val) ** (1 / years) - 1
            except Exception:
                cagr = None

            avg_monthly_hist = float(df_sorted["y"].mean())
            avg_monthly_forecast = None
            try:
                if not next12_fut.empty:
                    avg_monthly_forecast = float(next12_fut["yhat"].mean())
            except Exception:
                avg_monthly_forecast = None

            kpis = {
                "periods": periods,
                "last12_revenue": last12_revenue,
                "next12_revenue": next12_revenue,
                "cagr": cagr,
                "avg_monthly_hist": avg_monthly_hist,
                "avg_monthly_forecast": avg_monthly_forecast,
            }

    return render_template(
        "ecom_forecast.html",
        erro=erro,
        kpis=kpis,
        periods=periods,
        forecast_plot_div=forecast_plot_div,
    )


# ---------- Dashboard de E-commerce: Clusters (K-Means sobre RFM) ----------
@app.route("/ecom/clusters", methods=["GET"])
def ecom_clusters():
    """
    Clusterização de clientes com K-Means usando RFM (Recency, Frequency, Monetary).
    Mostra:
      - Distribuição de clientes por cluster
      - Scatter plot (Frequency x Monetary) colorido por cluster
      - Tabela-resumo por cluster
      - Tabela de clientes (top 300)
    """
    erro = ECOM_LOAD_ERROR
    cluster_scatter_div = None
    cluster_bar_div = None
    cluster_summary = None
    cluster_table = None

    filtros = {
        "country": request.args.get("country", "").strip(),
    }

    try:
        k = int(request.args.get("k", "4"))
    except Exception:
        k = 4
    if k < 2:
        k = 2
    if k > 8:
        k = 8

    countries_list = []
    if ECOM_DF is not None:
        countries_list = sorted(
            ECOM_DF["Country"].dropna().unique().tolist()
        )

    if ECOM_DF is not None and erro is None:
        df = ECOM_DF.copy()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        if filtros["country"]:
            df = df[df["Country"] == filtros["country"]]

        df = df.dropna(subset=["CustomerID"])

        if df.empty:
            erro = "Nenhum dado encontrado para os filtros selecionados."
        else:
            ref_date = df["InvoiceDate"].max()

            rfm = (
                df.groupby("CustomerID")
                  .agg(
                      Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
                      Frequency=("InvoiceNo", "nunique"),
                      Monetary=("TotalPrice", "sum"),
                  )
                  .reset_index()
            )

            rfm = rfm[(rfm["Monetary"] > 0) & (rfm["Frequency"] > 0)]

            if rfm.empty:
                erro = "Não foi possível calcular clusters (sem clientes válidos após filtragem)."
            else:
                X = rfm[["Recency", "Frequency", "Monetary"]].copy()

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                km = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=10,
                )
                rfm["Cluster"] = km.fit_predict(X_scaled)

                cluster_summary_df = (
                    rfm.groupby("Cluster", as_index=False)
                       .agg(
                           num_customers=("CustomerID", "nunique"),
                           avg_recency=("Recency", "mean"),
                           avg_frequency=("Frequency", "mean"),
                           avg_monetary=("Monetary", "mean"),
                           total_monetary=("Monetary", "sum"),
                       )
                       .sort_values("total_monetary", ascending=False)
                )
                cluster_summary = cluster_summary_df.to_dict(orient="records")

                fig_bar = go.Figure()
                fig_bar.add_trace(
                    go.Bar(
                        x=[f"Cluster {int(c)}" for c in cluster_summary_df["Cluster"]],
                        y=cluster_summary_df["num_customers"],
                        name="Nº de clientes",
                    )
                )
                fig_bar.update_layout(
                    title=f"Distribuição de clientes por cluster (k={k})",
                    xaxis_title="Cluster",
                    yaxis_title="Nº de clientes",
                    margin=dict(l=40, r=20, t=40, b=40),
                    template="plotly_white",
                    height=350,
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                )
                cluster_bar_div = plot(fig_bar, include_plotlyjs="cdn", output_type="div")

                fig_scatter = go.Figure()
                fig_scatter.add_trace(
                    go.Scatter(
                        x=rfm["Frequency"],
                        y=rfm["Monetary"],
                        mode="markers",
                        text=[f"Cliente {cid}" for cid in rfm["CustomerID"]],
                        marker=dict(
                            size=7,
                            color=rfm["Cluster"],
                            colorscale="Viridis",
                            showscale=True,
                        ),
                    )
                )
                fig_scatter.update_layout(
                    title=f"Clusters de clientes (Frequency x Monetary) — k={k}",
                    xaxis_title="Frequency (nº de encomendas)",
                    yaxis_title="Monetary (receita total)",
                    margin=dict(l=40, r=20, t=40, b=40),
                    template="plotly_white",
                    height=400,
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                )
                cluster_scatter_div = plot(fig_scatter, include_plotlyjs=False, output_type="div")

                cluster_table = (
                    rfm.sort_values(["Cluster", "Monetary"], ascending=[True, False])
                       .head(300)
                       .to_dict(orient="records")
                )

    return render_template(
        "ecom_clusters.html",
        erro=erro,
        filtros=filtros,
        countries=countries_list,
        k=k,
        cluster_bar_div=cluster_bar_div,
        cluster_scatter_div=cluster_scatter_div,
        cluster_summary=cluster_summary,
        cluster_table=cluster_table,
    )


@app.route("/churn_xai", methods=["GET", "POST"])
def churn_xai_dashboard():
    """
    Formulário interativo para churn:
    - utilizador preenche dados de um cliente (Telco)
    - modelo prevê probabilidade de churn
    """

    erro = _CHURN_ERROR
    result = None
    prob_pct = None

    # Defaults do formulário (GET ou POST)
    defaults = {
        "gender":           request.form.get("gender", "Female"),
        "SeniorCitizen":    request.form.get("SeniorCitizen", "0"),
        "Partner":          request.form.get("Partner", "No"),
        "Dependents":       request.form.get("Dependents", "No"),
        "tenure":           request.form.get("tenure", "12"),
        "PhoneService":     request.form.get("PhoneService", "Yes"),
        "MultipleLines":    request.form.get("MultipleLines", "No"),
        "InternetService":  request.form.get("InternetService", "Fiber optic"),
        "OnlineSecurity":   request.form.get("OnlineSecurity", "No"),
        "OnlineBackup":     request.form.get("OnlineBackup", "No"),
        "DeviceProtection": request.form.get("DeviceProtection", "No"),
        "TechSupport":      request.form.get("TechSupport", "No"),
        "StreamingTV":      request.form.get("StreamingTV", "Yes"),
        "StreamingMovies":  request.form.get("StreamingMovies", "Yes"),
        "Contract":         request.form.get("Contract", "Month-to-month"),
        "PaperlessBilling": request.form.get("PaperlessBilling", "Yes"),
        "PaymentMethod":    request.form.get("PaymentMethod", "Electronic check"),
        "MonthlyCharges":   request.form.get("MonthlyCharges", "70.0"),
        "TotalCharges":     request.form.get("TotalCharges", "1400.0"),
    }

    if request.method == "POST":
        if _CHURN_MODEL is None or _CHURN_SCALER is None or _CHURN_FEATURES is None:
            erro = _CHURN_ERROR or "Modelo de churn não está totalmente carregado (modelo/scaler/features)."
        else:
            try:
                # Lê valores do form (aceita vírgula ou ponto em numéricos)
                gender = request.form.get("gender", defaults["gender"])
                senior = request.form.get("SeniorCitizen", defaults["SeniorCitizen"])
                partner = request.form.get("Partner", defaults["Partner"])
                dependents = request.form.get("Dependents", defaults["Dependents"])

                tenure_str = (request.form.get("tenure", defaults["tenure"]) or "").strip().replace(",", ".")
                monthly_str = (request.form.get("MonthlyCharges", defaults["MonthlyCharges"]) or "").strip().replace(",", ".")
                total_str = (request.form.get("TotalCharges", defaults["TotalCharges"]) or "").strip().replace(",", ".")

                phone_service = request.form.get("PhoneService", defaults["PhoneService"])
                multiple_lines = request.form.get("MultipleLines", defaults["MultipleLines"])
                internet_service = request.form.get("InternetService", defaults["InternetService"])
                online_security = request.form.get("OnlineSecurity", defaults["OnlineSecurity"])
                online_backup = request.form.get("OnlineBackup", defaults["OnlineBackup"])
                device_protection = request.form.get("DeviceProtection", defaults["DeviceProtection"])
                tech_support = request.form.get("TechSupport", defaults["TechSupport"])
                streaming_tv = request.form.get("StreamingTV", defaults["StreamingTV"])
                streaming_movies = request.form.get("StreamingMovies", defaults["StreamingMovies"])
                contract = request.form.get("Contract", defaults["Contract"])
                paperless_billing = request.form.get("PaperlessBilling", defaults["PaperlessBilling"])
                payment_method = request.form.get("PaymentMethod", defaults["PaymentMethod"])

                # Converte numéricos
                tenure = float(tenure_str) if tenure_str else 0.0
                monthly_charges = float(monthly_str) if monthly_str else 0.0
                total_charges = float(total_str) if total_str else 0.0

                # Atualiza defaults para manter o formulário preenchido
                defaults.update({
                    "gender": gender,
                    "SeniorCitizen": senior,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": str(tenure).replace(".", ","),  # volta a mostrar com vírgula se quiseres
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": str(monthly_charges).replace(".", ","),
                    "TotalCharges": str(total_charges).replace(".", ","),
                })

                # Monta df_raw com as colunas do CSV original
                row = {
                    "gender": gender,
                    "SeniorCitizen": int(senior),
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }

                df_raw = pd.DataFrame([row])

                pred, prob = predict_churn_from_raw(df_raw)
                prob_pct = round(prob * 100, 1)
                result = "ALTO risco de churn" if pred == 1 else "BAIXO risco de churn"

            except Exception as e:
                erro = f"Erro ao processar os dados ou prever: {e}"

    return render_template(
        "churn_xai.html",
        erro=erro,
        result=result,
        prob_pct=prob_pct,
        defaults=defaults,
    )


@app.route("/ames", methods=["GET", "POST"])
def ames_dashboard():
    # df completo, sem filtro (para análises globais como Kruskal)
    df_completo = load_ames_data()
    df = df_completo.copy()

    # Variáveis numéricas do projeto (exceto faixa_preco, que é categórica)
    numeric_cols = [c for c in COLUNAS_PROJETO if c in df.columns and c != "faixa_preco"]

    # Variável selecionada
    default_var = "preco" if "preco" in numeric_cols else numeric_cols[0]
    var = request.form.get("variavel", default_var)

    # Filtro de faixa de preço
    faixas_unicas = ["Todos"]
    if "faixa_preco" in df.columns:
        faixas_unicas += sorted(df["faixa_preco"].dropna().unique().tolist())

    faixa_selecionada = request.form.get("faixa_preco", "Todos")

    # Aplica filtro se não for "Todos"
    df_filtrado = df.copy()
    if faixa_selecionada != "Todos" and "faixa_preco" in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado["faixa_preco"] == faixa_selecionada]

    # Garante que a variável existe no df filtrado
    if var not in df_filtrado.columns:
        var = default_var

    # Série da variável (já filtrada)
    serie = df_filtrado[var].dropna()

    # Estatísticas descritivas
    stats_dict = calcular_estatisticas_1d(serie)

    # Testes adicionais
    testes_extra = calcular_testes_adicionais(
        serie=serie,
        df_filtrado=df_filtrado,
        var=var,
        df_completo=df_completo if faixa_selecionada == "Todos" else None,
    )

    # Nome amigável da variável
    label = NOMES_AMIGAVEIS.get(var, var)

    # ----------------------
    # Gráfico 1: Histograma
    # ----------------------
    fig_hist = px.histogram(
        df_filtrado,
        x=var,
        nbins=40,
        marginal="box",
        title=f"Distribuição de {label}",
        labels={var: label},
    )

    # -------------------
    # Gráfico 2: Boxplot
    # -------------------
    fig_box = px.box(
        df_filtrado,
        y=var,
        points="outliers",
        title=f"Boxplot de {label}",
        labels={var: label},
    )

    # -----------------------------------------
    # Gráfico 3: Dispersão Preço vs variável
    # -----------------------------------------
    graph_scatter_json = None
    if "preco" in df_filtrado.columns and var in df_filtrado.columns and var != "preco":
        fig_scatter = px.scatter(
            df_filtrado,
            x=var,
            y="preco",
            color="faixa_preco" if "faixa_preco" in df_filtrado.columns else None,
            labels={
                var: label,
                "preco": "Preço do Imóvel (€)",
                "faixa_preco": "Faixa de Preço",
            },
            title=f"Preço do Imóvel vs {label}",
        )
        graph_scatter_json = json.dumps(
            fig_scatter, cls=plotly.utils.PlotlyJSONEncoder
        )

    # -----------------------------------------
    # Gráfico 3b: Preço vs Ano de Construção
    # -----------------------------------------
    graph_preco_ano_json = None
    if all(c in df_filtrado.columns for c in ["preco", "ano_construcao"]):
        df_ano = df_filtrado.dropna(subset=["preco", "ano_construcao"]).copy()
        if not df_ano.empty:
            fig_preco_ano = px.scatter(
                df_ano,
                x="ano_construcao",
                y="preco",
                color="faixa_preco" if "faixa_preco" in df_ano.columns else None,
                labels={
                    "ano_construcao": "Ano de Construção",
                    "preco": "Preço do Imóvel (€)",
                    "faixa_preco": "Faixa de Preço",
                },
                title="Preço do Imóvel ao longo do Ano de Construção",
            )
            fig_preco_ano.update_traces(marker=dict(size=6, opacity=0.7))
            fig_preco_ano.update_layout(
                xaxis=dict(dtick=5),  # marcação a cada 5 anos (ajusta se quiser)
            )

            graph_preco_ano_json = json.dumps(
                fig_preco_ano, cls=plotly.utils.PlotlyJSONEncoder
            )

    # -----------------------------------------
    # Gráfico 4: Boxplot variável por faixa de preço
    # -----------------------------------------
    graph_box_faixa_json = None
    if "faixa_preco" in df_filtrado.columns and var in df_filtrado.columns:
        fig_box_faixa = px.box(
            df_filtrado,
            x="faixa_preco",
            y=var,
            labels={
                "faixa_preco": "Faixa de Preço",
                var: label,
            },
            title=f"{label} por Faixa de Preço",
        )
        graph_box_faixa_json = json.dumps(
            fig_box_faixa, cls=plotly.utils.PlotlyJSONEncoder
        )

    # ------------------------------
    # Gráfico 5: Heatmap de correlação
    # ------------------------------
    graph_heatmap_json = None
    vars_corr = [c for c in CORR_VARS if c in df_filtrado.columns]

    if len(vars_corr) >= 2:
        df_corr = df_filtrado[vars_corr].corr(method="pearson")

        cols_amigaveis = [NOMES_AMIGAVEIS.get(c, c) for c in vars_corr]
        df_corr_named = df_corr.copy()
        df_corr_named.columns = cols_amigaveis
        df_corr_named.index = cols_amigaveis

        fig_heatmap = px.imshow(
            df_corr_named,
            text_auto=".4f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            title="Matriz de Correlação (Pearson) - Variáveis principais",
        )
        fig_heatmap.update_layout(
            xaxis_title="Variáveis",
            yaxis_title="Variáveis",
            width=1400,   # mais largo
            height=900,   # mais alto
            font=dict(size=14),
            margin=dict(
                l=180,   # mais espaço à esquerda para labels
                r=40,
                t=100,
                b=220    # mais espaço para labels do eixo X inclinadas
            ),
            xaxis=dict(tickangle=-40)  # garante rótulos inclinados, menos sobreposição
        )

        fig_heatmap.update_traces(
            textfont=dict(size=12, color="black")  # números bem visíveis
        )

        graph_heatmap_json = json.dumps(
            fig_heatmap, cls=plotly.utils.PlotlyJSONEncoder
        )

    # ------------------------------
    # Gráfico 6: Mapa (lat/long)
    # ------------------------------
    graph_map_json = None
    if all(col in df_filtrado.columns for col in ["latitude", "longitude"]):
        df_mapa = df_filtrado.dropna(subset=["latitude", "longitude"]).copy()
        if not df_mapa.empty:
            fig_map = px.scatter_mapbox(
                df_mapa,
                lat="latitude",
                lon="longitude",
                color="faixa_preco" if "faixa_preco" in df_mapa.columns else None,
                size="preco" if "preco" in df_mapa.columns else None,
                hover_data=[
                    c for c in ["preco", "preco_m2", "area_total", "quartos", "banheiros"]
                    if c in df_mapa.columns
                ],
                zoom=11,
                height=550,
                title="Mapa de Imóveis (localização geográfica)",
            )
            fig_map.update_layout(
                mapbox_style="open-street-map",
                margin=dict(l=0, r=0, t=60, b=0),
            )

            graph_map_json = json.dumps(
                fig_map, cls=plotly.utils.PlotlyJSONEncoder
            )

    # Converte os outros gráficos para JSON
    graph_hist_json = json.dumps(fig_hist, cls=plotly.utils.PlotlyJSONEncoder)
    graph_box_json = json.dumps(fig_box, cls=plotly.utils.PlotlyJSONEncoder)

    # Interpretação da normalidade (Shapiro)
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

    return render_template(
        "ames.html",
        variavel_selecionada=var,
        variaveis=numeric_cols,
        faixa_selecionada=faixa_selecionada,
        faixas=faixas_unicas,
        estatisticas=stats_dict,
        testes_extra=testes_extra,
        graph_hist_json=graph_hist_json,
        graph_box_json=graph_box_json,
        graph_scatter_json=graph_scatter_json,
        graph_box_faixa_json=graph_box_faixa_json,
        graph_heatmap_json=graph_heatmap_json,
        graph_map_json=graph_map_json,
        graph_preco_ano_json=graph_preco_ano_json,
        interpretacao_normalidade=interpretacao_normalidade,
        nomes_amigaveis=NOMES_AMIGAVEIS,
        nomes_faixa=NOMES_FAIXA,
    )


# Healthcheck
@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

import os
from datetime import datetime
from io import StringIO
from typing import Tuple, Optional
import re
import pickle
from pathlib import Path
from scipy import stats
from functools import lru_cache
import json

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from flask import (
    Flask, render_template, request, jsonify, make_response, redirect, url_for
)

# ==== Sklearn imports ====
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from joblib import load as joblib_load

# ==== Extras para FX (câmbio) ====
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor  # regressão, diferente do Classifier lá de cima

from statsmodels.tsa.arima.model import ARIMA       # pip install statsmodels
from prophet import Prophet                         # pip install prophet
import plotly.graph_objs as go
from plotly.offline import plot
import plotly
import plotly.express as px



app = Flask(__name__)

# --- (opcional) URL do Streamlit FX: define FX_URL para usar /fx ---
FX_URL = os.getenv("FX_URL", "http://127.0.0.1:8501")

# ======== Presets de Ações (legenda) ========
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

# ======== Helpers — Cotações ========

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

# ======== Helpers — Tempo (Open-Meteo) ========

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

# ======== FX — Previsão de Câmbio (RF / ARIMA / Prophet) ========

# Pares de moedas mais comuns
FX_PAIRS = {
    "USD/BRL": "BRL=X",      # quantos BRL por 1 USD
    "EUR/USD": "EURUSD=X",   # quantos USD por 1 EUR
    "USD/JPY": "JPY=X",      # quantos JPY por 1 USD
    "GBP/USD": "GBPUSD=X",   # quantos USD por 1 GBP
}

FX_WINDOW_SIZE = 30
FX_RANDOM_STATE = 42


def fx_download_history(ticker: str, period: str = "3y") -> pd.Series:
    """
    Baixa histórico diário via yfinance e devolve uma Series de 'Close'.
    Usado para os modelos de câmbio.
    """
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
    if data is None or data.empty:
        raise ValueError(f"Sem dados históricos para o ticker {ticker}.")
    s = data["Close"].dropna().copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)  # remover timezone
    s = s.sort_index()
    s.name = "rate"
    return s

def stock_download_history(ticker: str, period: str = "3y") -> pd.Series:
    """
    Faz download dos últimos 'period' de um ticker de ação
    e devolve APENAS a série de preços de FECHO (Close).
    Será usada pelos modelos de previsão (RF / ARIMA / Prophet).
    """
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
    if data is None or data.empty:
        raise ValueError(f"Sem dados históricos para o ticker {ticker}.")

    s = data["Close"].dropna().copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s.sort_index()
    s.name = "price"
    return s


def fx_compute_metrics(y_true, y_pred) -> dict:
    """
    Calcula métricas de regressão: MAE, RMSE, MAPE (%).
    Compatível com versões mais antigas do scikit-learn (sem argumento 'squared').
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)

    # Em versões antigas não há squared=False, então fazemos a raiz "na mão"
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    # evita divisão por zero no MAPE
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    return {"mae": mae, "rmse": rmse, "mape": mape}



# ---------- Random Forest (regressão) ----------

def fx_create_supervised(series: pd.Series, window_size: int = FX_WINDOW_SIZE):
    """
    Transforma série 1D em X, y com janelas deslizantes.
    X.shape = (n_amostras, window_size), y.shape = (n_amostras,)
    """
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
    """
    Previsão iterativa de n_days à frente usando Random Forest.
    """
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
    """
    Treina Random Forest (a cada chamada, para simplicidade didática),
    calcula métricas no teste e gera previsão n_days à frente.
    """
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


# ---------- ARIMA ----------

def fx_train_and_forecast_arima(series: pd.Series, n_days: int):
    """
    Treina ARIMA(1,1,1) para demo, calcula métricas no teste e gera previsão n_days.
    """
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


# ---------- Prophet ----------

def fx_train_and_forecast_prophet(series: pd.Series, n_days: int):
    """
    Treina Prophet, calcula métricas com backtest simples e gera previsão n_days.
    """
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


# ======== ML — Heart Disease (novo dataset em PT) ========

# Colunas do novo CSV
DEFAULT_HEART_FEATURES = [
    "sexo",
    "idade",
    "cigarros_por_dia",
    "uso_medicamento_pressao",
    "AVC",
    "hipertensao",
    "diabetes",
    "colesterol_total",
    "pressao_arterial_sistolica",
    "pressao_arterial_diastolica",
    "IMC",
    "freq_cardiaca",
    "glicemia",
    "fumante",
]
HEART_TARGET = "risco_DAC_decada"

# URL ATUALIZADA
HEART_RAW_URL = os.getenv(
    "HEART_RAW_URL",
    "https://raw.githubusercontent.com/MpRonald/datasets/main/doenca_cardiaca_final.csv"
)

ARTIF_DIR = Path(__file__).resolve().parent / "artifacts"

_HEART_FEATURES = DEFAULT_HEART_FEATURES[:]  # pode ser sobrescrito por artifact
_HEART_MODEL = None
_HEART_STATS = None
_HEART_LOAD_ERROR = None
_HEART_SOURCE = None  # "artifact" ou "csv-train"

def _attempt_load_heart_artifact():
    """
    Carrega artifact salvo (task=='heart_disease') se existir.
    Suporta dois formatos:
      - artifact dict {"model","feature_names",...}
      - model puro + meta com "feature_names"
    """
    if not ARTIF_DIR.exists():
        return None, None, None
    try:
        metas = sorted(ARTIF_DIR.glob("meta_*.joblib"), reverse=True)
        for mp in metas:
            meta = joblib_load(mp)
            if meta.get("task") != "heart_disease":
                continue
            version = meta["version"]
            mpath = ARTIF_DIR / f"model_{version}.joblib"
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
    """
    Baixa o CSV novo (colunas em PT) e treina um RandomForest com imputação.
    Retorna model, features, stats (min/max/mediana por coluna).
    """
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

    stats = {
        c: {"min": float(np.nanmin(X[c])), "max": float(np.nanmax(X[c])), "median": float(np.nanmedian(X[c]))}
        for c in DEFAULT_HEART_FEATURES
    }
    return model, DEFAULT_HEART_FEATURES[:], stats

def _init_heart_model():
    """
    Tenta carregar artifact; se não houver, treina a partir do CSV novo.
    """
    global _HEART_MODEL, _HEART_FEATURES, _HEART_STATS, _HEART_LOAD_ERROR, _HEART_SOURCE
    try:
        model, features, meta = _attempt_load_heart_artifact()
        if model is not None:
            _HEART_MODEL = model
            _HEART_FEATURES = features
            # Stats só para defaults do formulário (opcionais)
            try:
                _, _, stats = _load_heart_from_csv_and_train()
                _HEART_STATS = {c: stats[c] for c in _HEART_FEATURES if c in stats}
            except Exception:
                _HEART_STATS = None
            _HEART_LOAD_ERROR = None
            _HEART_SOURCE = "artifact"
            return
        # Fallback: treinar com CSV novo
        model, features, stats = _load_heart_from_csv_and_train()
        _HEART_MODEL, _HEART_FEATURES, _HEART_STATS = model, features, stats
        _HEART_LOAD_ERROR = None
        _HEART_SOURCE = "csv-train"
    except Exception as e:
        _HEART_MODEL, _HEART_FEATURES, _HEART_STATS = None, DEFAULT_HEART_FEATURES[:], None
        _HEART_LOAD_ERROR = str(e)
        _HEART_SOURCE = None

_init_heart_model()

# ====================== NLP — Modelo PT carregado de arquivo ======================

# Caminho padrão do modelo salvo (pode sobrescrever via env NLP_MODEL_PATH)
NLP_MODEL_PATH = os.getenv(
    "NLP_MODEL_PATH",
    r"C:\Users\datap\OneDrive\Documentos\GitHub\portfolio\portfolio_flask\modelo_sentimento.joblib"
)

_NLP_MODEL = None         # bundle: {"tfidf": ..., "clf": ..., "classes_": ...}
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
    """Carrega bundle salvo (tfidf+clf). Tenta .joblib e, se falhar, .pkl."""
    global _NLP_MODEL, _NLP_ERROR, _NLP_STATS
    try:
        bundle = joblib_load(NLP_MODEL_PATH)
    except Exception as e1:
        try:
            alt = os.path.splitext(NLP_MODEL_PATH)[0] + ".pkl"
            with open(alt, "rb") as f:
                bundle = pickle.load(f)
        except Exception as e2:
            _NLP_MODEL, _NLP_ERROR = None, f"Falha ao carregar modelo salvo: {e1} | {e2}"
            _NLP_STATS = None
            return

    if not isinstance(bundle, dict) or "tfidf" not in bundle or "clf" not in bundle:
        _NLP_MODEL, _NLP_ERROR = None, "Arquivo de modelo inválido (esperado dict com 'tfidf' e 'clf')."
        _NLP_STATS = None
        return

    classes = bundle.get("classes_", None)
    if classes is None:
        classes = getattr(bundle["clf"], "classes_", None)

    classes_list = _to_list_or_none(classes)
    _NLP_MODEL = {"tfidf": bundle["tfidf"], "clf": bundle["clf"], "classes_": classes_list}
    _NLP_STATS = {"classes": [str(c) for c in classes_list] if classes_list is not None else None}
    _NLP_ERROR = None

def _positive_index(classes, proba) -> int:
    """Determina o índice da classe 'positivo' (PT) de forma robusta."""
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
    """Normaliza rótulos do classificador para 'bom'/'ruim'/'neutro'."""
    p = str(pred).strip().lower()
    if p in {"positivo", "positive", "pos", "1", "true", "sim", "bom"}:
        return "bom"
    if p in {"negativo", "negative", "neg", "0", "false", "nao", "não", "ruim"}:
        return "ruim"
    if p in {"neutro", "neutral", "neu"}:
        return "neutro"
    return "bom" if p in {"1"} else "ruim"

# carrega o modelo ao subir a app
_load_pt_model()

def predict_sentiment_label(text: str) -> Tuple[str, float, dict]:
    """
    Usa o modelo em PORTUGUÊS salvo (tfidf + clf) para prever.
    Retorna (label_pt_bom/ruim/neutro, score_pos, meta)
    """
    if _NLP_MODEL is None:
        raise RuntimeError(_NLP_ERROR or "Modelo de NLP não carregado.")

    tfidf = _NLP_MODEL["tfidf"]
    clf = _NLP_MODEL["clf"]
    classes = _NLP_MODEL.get("classes_")

    s = _clean_html(text)
    Xq = tfidf.transform([s])

    # probabilidade da classe positiva (se disponível)
    proba = None
    score_pos = 0.5
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Xq)[0]
        pos_idx = _positive_index(classes, proba)
        score_pos = float(proba[pos_idx])

    pred = clf.predict(Xq)[0]
    label_pt = _pt_label_from_pred(pred)

    # regra de neutralidade
    if 0.45 <= score_pos <= 0.55:
        label_pt = "neutro"

    meta = {
        "classes": classes,
        "raw_pred": str(pred),
        "score_strategy": "predict_proba" if proba is not None else "heuristic_from_predict"
    }
    return label_pt, score_pos, meta


# ========== ML — Loan Default (estado_emprestimo) ==========
# Carregamento do pipeline salvo e rotas /loan e /loan/prever

# Caminho do pipeline salvo (podes sobrescrever via env)
LOAN_MODEL_PATH = os.getenv("LOAN_MODEL_PATH", "molde_credit_risk.joblib")

# Tenta carregar o pipeline completo (ColumnTransformer + SMOTE + modelo)
_LOAN_PIPE = None
_LOAN_ERROR = None

def _extract_pipeline(obj):
    """Aceita objetos salvos como dict e tenta extrair o pipeline/estimator."""
    if not isinstance(obj, dict):
        return obj, None  # já é o estimador/pipeline

    # tenta chaves comuns
    for k in ("pipeline", "model", "pipe", "estimator"):
        if k in obj:
            return obj[k], obj  # retorna também o dict para metadados

    # nenhum estimador dentro do dict
    keys = ", ".join(obj.keys())
    raise ValueError(f"Objeto .joblib é um dict sem chave de estimador ('pipeline'/'model'/'pipe'/'estimator'). Chaves: {keys}")

try:
    _loaded = joblib_load(LOAN_MODEL_PATH)
    _LOAN_PIPE, _LOAN_META = _extract_pipeline(_loaded)
except Exception as e:
    _LOAN_PIPE = None
    _LOAN_META = None
    _LOAN_ERROR = f"Falha ao carregar pipeline de empréstimo em '{LOAN_MODEL_PATH}': {e}"


# Campos esperados (mesmos usados no treino)
LOAN_CAMPO_NUM = [
    'idade',
    'rendimento_anual',
    'anos_emprego',
    'valor_emprestimo',
    'taxa_juros',
    'percent_rendimento',
    'anos_historico_credito'
]
LOAN_CAMPO_CAT = [
    'tipo_habitacao',
    'finalidade_emprestimo',
    'grau_emprestimo',
    'historico_inadimplencia'
]

# Opções para selects (labels/values devem bater com o treino)
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
    # Enviaremos Y/N por padrão; adapte se o treino foi com "Sim/Não"
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

@app.route("/loan", methods=["GET"])
def loan_form():
    # Renderiza o formulário (usa templates/loan.html)
    # Se preferires, podes reaproveitar um base.html teu e iterar o dicionário LOAN_OPCOES
    return render_template("loan.html", opcoes=LOAN_OPCOES, erro=_LOAN_ERROR)

@app.route("/loan/prever", methods=["POST"])
def loan_prever():
    if _LOAN_PIPE is None:
        return render_template("loan.html", opcoes=LOAN_OPCOES, erro=_LOAN_ERROR or "Pipeline indisponível.")

    # Lê payload do formulário
    payload = {}
    for c in LOAN_CAMPO_NUM:
        payload[c] = _to_float_or_none(request.form.get(c))
    for c in LOAN_CAMPO_CAT:
        payload[c] = request.form.get(c) or None

    # Se o treino foi com "Sim/Não", faça o mapeamento aqui (descomentar conforme necessário):
    # mapa_hist = {'Y': 'Sim', 'N': 'Não'}
    # payload['historico_inadimplencia'] = mapa_hist.get(payload['historico_inadimplencia'])

    # DataFrame de uma linha com a ordem/nomes esperados
    cols = LOAN_CAMPO_NUM + LOAN_CAMPO_CAT
    X_in = pd.DataFrame([payload], columns=cols)

    try:
        # Probabilidade se disponível
        prob_default = None
        pred = None

        if hasattr(_LOAN_PIPE, "predict_proba"):
            prob_default = float(_LOAN_PIPE.predict_proba(X_in)[:, 1][0])
            pred = int(prob_default >= 0.5)

        elif hasattr(_LOAN_PIPE, "decision_function"):
            # Converte score para "probabilidade" via sigmóide (aproximação binária)
            import math
            score = float(_LOAN_PIPE.decision_function(X_in)[0])
            prob_default = 1.0 / (1.0 + math.exp(-score))
            pred = int(prob_default >= 0.5)

        else:
            # Fallback: só há predict
            pred = int(_LOAN_PIPE.predict(X_in)[0])
            # Sem proba disponível; usa 0.5 neutro apenas para exibir
            prob_default = 0.5

    except Exception as e:
        return render_template("loan.html", opcoes=LOAN_OPCOES, erro=f"Erro ao prever: {e}")


    risco_txt = "ALTO (tende a default)" if pred == 1 else "BAIXO (tende a pagar)"
    prob_fmt = f"{prob_default:.2%}"

    # Renderiza resultado no mesmo template (loan.html), reaproveitando o formulário
    return render_template(
        "loan.html",
        prob=prob_fmt,
        classe=pred,
        risco=risco_txt,
        entrada=payload,
        opcoes=LOAN_OPCOES,  # <-- IMPORTANTE
        erro=None            # opcional, só pra limpar mensagem de erro
    )


# ========== ML — Churn de Clientes (Telco) ==========

CHURN_MODEL_PATH = os.getenv(
    "CHURN_MODEL_PATH",
    str(Path(__file__).resolve().parent / "modelo_churn.joblib")
)
CHURN_SCALER_PATH = os.getenv(
    "CHURN_SCALER_PATH",
    str(Path(__file__).resolve().parent / "modelo_churn_scaler.joblib")
)
CHURN_FEATURES_PATH = os.getenv(
    "CHURN_FEATURES_PATH",
    str(Path(__file__).resolve().parent / "modelo_churn_feature_names.joblib")
)

CHURN_DATA_PATH = Path(__file__).resolve().parent / "data" / "telco_customer_churn.csv"

_CHURN_MODEL = None
_CHURN_SCALER = None
_CHURN_FEATURES = None
_CHURN_ERROR = None


def _load_churn_artifacts():
    """Carrega modelo de churn + scaler + nomes de features, se existirem."""
    global _CHURN_MODEL, _CHURN_SCALER, _CHURN_FEATURES, _CHURN_ERROR
    try:
        _CHURN_MODEL = joblib_load(CHURN_MODEL_PATH)
        _CHURN_SCALER = joblib_load(CHURN_SCALER_PATH)
        _CHURN_FEATURES = joblib_load(CHURN_FEATURES_PATH)
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
    """
    Reproduz o pré-processamento usado no notebook de churn Telco.
    Espera colunas originais do CSV (gender, SeniorCitizen, Partner, etc.).
    """
    df = df_raw.copy()

    # --- 1. TotalCharges: tratar espaços > NaN > float > preencher mediana ---
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # --- 2. Remover ID ---
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # --- 3. Binárias Yes/No -> 1/0 ---
    binary_columns = ['Partner', 'Dependents', 'PhoneService',
                      'PaperlessBilling', 'Churn']
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # --- 4. One-Hot (igual ao notebook) ---
    cat_cols_to_dummies = [
        'gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract',
        'PaymentMethod'
    ]
    existing_cat_cols = [c for c in cat_cols_to_dummies if c in df.columns]
    if existing_cat_cols:
        df = pd.get_dummies(
            df,
            columns=existing_cat_cols,
            drop_first=True
        )

    # --- 5. Feature engineering ---
    if "tenure" in df.columns and "TotalCharges" in df.columns:
        df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=["0-12m", "12-24m", "24-36m", "36-48m", "48-60m", "60-72m"]
        )
        df["tenure_group"] = LabelEncoder().fit_transform(df["tenure_group"].astype(str))

    # --- 6. Booleanos True/False -> 0/1 ---
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # --- 7. Escalonamento das numéricas ---
    numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]
    numeric_existing = [c for c in numerical_columns if c in df.columns]

    if scaler is not None and numeric_existing:
        df[numeric_existing] = scaler.transform(df[numeric_existing])

    # --- 8. Alinhar colunas com as usadas no treino ---
    if feature_names is not None:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

    return df


def predict_churn_from_raw(df_raw: pd.DataFrame):
    """
    Usa os artifacts carregados (_CHURN_MODEL/_CHURN_SCALER/_CHURN_FEATURES)
    para prever churn a partir do df bruto (formulário ou CSV Telco).
    """
    if _CHURN_MODEL is None or _CHURN_SCALER is None or _CHURN_FEATURES is None:
        raise RuntimeError(_CHURN_ERROR or "Modelo de churn indisponível.")

    df_proc = preprocess_telco(
        df_raw,
        scaler=_CHURN_SCALER,
        feature_names=_CHURN_FEATURES
    )

    proba = _CHURN_MODEL.predict_proba(df_proc)[0, 1]
    pred = _CHURN_MODEL.predict(df_proc)[0]

    return int(pred), float(proba)


# ============================================================
# 🔹 DICIONÁRIO DE TRADUÇÃO DAS VARIÁVEIS IMPORTANTES DO AMES
# ============================================================

# ==========================
# Config Ames / Imóveis
# ==========================

COLUNAS_PROJETO = [
    "preco",
    "quartos",
    "banheiros",
    "area_habitavel",
    "area_lote",
    "andares",
    "area_acima_solo",
    "area_porao",
    "ano_construcao",
    "latitude",
    "longitude",
    "area_habitavel_viz",
    "area_lote_viz",
    "faixa_preco",
    "idade_imovel",
    "area_total",
    "densidade_construcao",
    "preco_m2",
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

# 🔹 Lista de variáveis que aparecerão no dropdown
VARIAVEIS_IMPORTANTES = list(NOMES_AMIGAVEIS.keys())

@lru_cache(maxsize=1)
def load_ames_data() -> pd.DataFrame:
    """Carrega o dataset de imóveis já tratado (colunas em português)."""
    csv_path = Path("data/ames.csv")
    if not csv_path.exists():
        raise FileNotFoundError("Ficheiro data/ames_tratado.csv não encontrado.")
    
    df = pd.read_csv(csv_path)

    # Garante que só usamos as colunas do projeto que existem no CSV
    colunas_validas = [c for c in COLUNAS_PROJETO if c in df.columns]
    df = df[colunas_validas]

    return df


def calcular_estatisticas_1d(serie: pd.Series) -> dict:
    """Calcula medidas descritivas para uma variável numérica."""
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

    # Assimetria e curtose
    assimetria = float(s.skew())
    curtose = float(s.kurtosis())

    # Teste de normalidade (Shapiro-Wilk) – usar amostra se tiver muitos dados
    sample = s
    if len(s) > 5000:
        sample = s.sample(5000, random_state=42)

    try:
        stat, p_valor = stats.shapiro(sample)
    except Exception:
        stat, p_valor = np.nan, np.nan

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
        "stat_shapiro": float(stat) if not np.isnan(stat) else None,
        "p_valor_shapiro": float(p_valor) if not np.isnan(p_valor) else None,
    }

def calcular_testes_adicionais(serie: pd.Series, df: pd.DataFrame, var: str) -> dict:
    """
    Calcula testes adicionais para a variável selecionada:
    - Teste de normalidade Jarque-Bera
    - Correlação com preço (preco)
    - Correlação com preço por m² (preco_m2)
    """
    resultados = {}
    s = serie.dropna()

    # --- Jarque-Bera (normalidade) ---
    try:
        jb_stat, jb_p = stats.jarque_bera(s)
        resultados["jb_stat"] = float(jb_stat)
        resultados["jb_p"] = float(jb_p)
    except Exception:
        resultados["jb_stat"] = None
        resultados["jb_p"] = None

    # --- Correlações com 'preco' e 'preco_m2' ---
    for alvo in ["preco", "preco_m2"]:
        r_key = f"corr_{alvo}_r"
        p_key = f"corr_{alvo}_p"

        if alvo in df.columns and var != alvo:
            s2 = df[alvo].dropna()
            conjunto = pd.concat([s, s2], axis=1).dropna()

            if len(conjunto) >= 3:
                r, p = stats.pearsonr(conjunto.iloc[:, 0], conjunto.iloc[:, 1])
                resultados[r_key] = float(r)
                resultados[p_key] = float(p)
            else:
                resultados[r_key] = None
                resultados[p_key] = None
        else:
            resultados[r_key] = None
            resultados[p_key] = None

    return resultados

def calcular_testes_adicionais(
    serie: pd.Series,
    df_filtrado: pd.DataFrame,
    var: str,
    df_completo: pd.DataFrame | None,
) -> dict:
    """
    Calcula testes adicionais para a variável selecionada:
    - Jarque–Bera (normalidade)
    - Correlações de Pearson com preco e preco_m2
    - Correlação de Spearman com preco
    - Kruskal–Wallis por faixa_preco (se df_completo não for None)
    - Regressão linear simples: preco ~ var
    """
    resultados: dict[str, float | None] = {}

    s = serie.dropna()

    # --- Jarque–Bera ---
    try:
        jb_stat, jb_p = stats.jarque_bera(s)
        resultados["jb_stat"] = float(jb_stat)
        resultados["jb_p"] = float(jb_p)
    except Exception:
        resultados["jb_stat"] = None
        resultados["jb_p"] = None

    # --- Correlações de Pearson com preco e preco_m2 ---
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

    # --- Correlação de Spearman com preco ---
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

    # --- Kruskal–Wallis por faixa_preco (usando df_completo, todas as faixas) ---
    if df_completo is not None and "faixa_preco" in df_completo.columns and var in df_completo.columns:
        grupos = []
        for faixa in sorted(df_completo["faixa_preco"].dropna().unique().tolist()):
            vals = df_completo.loc[df_completo["faixa_preco"] == faixa, var].dropna()
            if len(vals) >= 3:  # precisa de pelo menos alguns dados por grupo
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

    # --- Regressão linear simples: preco ~ var (usando df_filtrado) ---
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


# ======== Rotas ========

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/fx", methods=["GET", "POST"])
def fx():
    erro = None
    resultado = None

    if request.method == "POST":
        try:
            pair_name = request.form.get("pair")
            algoritmo = request.form.get("algoritmo")
            n_days_raw = request.form.get("n_days", "30")

            # Sanitizar nº de dias
            try:
                n_days = int(n_days_raw)
            except Exception:
                n_days = 30

            if n_days < 1:
                n_days = 1
            if n_days > 180:
                n_days = 180

            # Validar inputs
            if pair_name not in FX_PAIRS:
                raise ValueError("Par de moedas inválido.")
            if algoritmo not in ["rf", "arima", "prophet"]:
                raise ValueError("Algoritmo inválido.")

            # Série histórica (3 anos)
            ticker = FX_PAIRS[pair_name]
            series = fx_download_history(ticker, period="3y")

            # Treinar / prever conforme o algoritmo
            if algoritmo == "rf":
                metrics, forecast_df = fx_train_and_forecast_rf(series, n_days)
                alg_label = "Random Forest"
            elif algoritmo == "arima":
                metrics, forecast_df = fx_train_and_forecast_arima(series, n_days)
                alg_label = "ARIMA (1,1,1)"
            else:
                metrics, forecast_df = fx_train_and_forecast_prophet(series, n_days)
                alg_label = "Prophet"

            # Histórico do último ano (para deixar o gráfico mais limpo)
            history_last_year = series.last("365D")

            # ---------- Plotly: preparar dados explicitamente ----------
            # Converte os índices para datas “puras” e valores para float
            x_hist = [d.date() for d in history_last_year.index]
            y_hist = [float(v) for v in history_last_year.values]

            x_fore = [d.date() for d in forecast_df.index]
            y_fore = [float(v) for v in forecast_df["forecast_rate"].values]

            # Figura Plotly
            fig = go.Figure()

            # Série histórica (linha + marcadores, bem visível)
            fig.add_trace(
                go.Scatter(
                    x=x_hist,
                    y=y_hist,
                    mode="lines+markers",
                    name="Histórico",
                    line=dict(width=2)
                )
            )

            # Série de previsão (linha tracejada + marcadores)
            fig.add_trace(
                go.Scatter(
                    x=x_fore,
                    y=y_fore,
                    mode="lines+markers",
                    name="Previsão",
                    line=dict(width=2, dash="dash")
                )
            )

            fig.update_layout(
                title=f"Histórico e previsão — {pair_name}",
                xaxis_title="Data",
                yaxis_title=f"Taxa {pair_name}",
                margin=dict(l=40, r=20, t=40, b=40),
                template="plotly_dark",  # combina com o tema escuro
                height=400,
                hovermode="x unified"
            )

            # Formato de casas decimais no eixo Y
            fig.update_yaxes(tickformat=".3f")

            # Gera o HTML do gráfico (div) com JS do Plotly via CDN
            fx_plot_div = plot(fig, include_plotlyjs="cdn", output_type="div")

            # Monta o resultado enviado ao template
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

    return render_template(
        "fx.html",
        pairs=FX_PAIRS,
        erro=erro,
        resultado=resultado
    )



# ---------- Cotações ----------
# ---------- Cotações + Previsão ----------

@app.route("/quotes", methods=["GET", "POST"])
def quotes():
    # Defaults para formulário
    DEFAULT_TICKER = "AAPL"
    DEFAULT_ALGO = "rf"
    DEFAULT_N_DAYS = 30

    # --- Ler inputs do formulário/URL ---
    preset_ticker = request.values.get("preset_ticker")
    default_ticker = preset_ticker if preset_ticker else DEFAULT_TICKER

    raw_period   = request.values.get("period", "1y")
    raw_interval = request.values.get("interval", "1d")
    period = _coerce_period_month(raw_period)
    interval, period, notice = normalize_interval_and_period(raw_interval, period)

    algoritmo = request.values.get("algoritmo", DEFAULT_ALGO)
    n_days_raw = request.values.get("n_days", str(DEFAULT_N_DAYS))
    try:
        n_days = int(n_days_raw)
    except Exception:
        n_days = DEFAULT_N_DAYS
    if n_days < 1:
        n_days = 1
    if n_days > 180:
        n_days = 180

    # Form dict para o template
    form = {
        "ticker":  request.values.get("ticker", default_ticker).upper(),
        "period":  period,
        "interval": interval,
        "algoritmo": algoritmo,
        "n_days":  n_days,
    }

    chart_data = None
    error = None
    metrics = None
    forecast_df = None
    stock_plot_div = None
    alg_label = None

    try:
        # ---------- HISTÓRICO p/ Chart.js (como já tinhas) ----------
        df = fetch_history(form["ticker"], form["period"], form["interval"])
        intraday = form["interval"] in ["1m","2m","5m","15m","30m","60m","90m","1h"]
        date_fmt = "%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d"
        chart_data = {
            "labels": df["Date"].dt.strftime(date_fmt).tolist(),
            "close":  df["Close"].round(4).tolist(),
            "ohlc":   df[["Open", "High", "Low", "Close"]].round(4).values.tolist(),
        }

        # ---------- PREVISÃO (usando últimos 3 anos diários) ----------
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

        # Histórico recente (1 ano) para o gráfico de ML
        history_last_year = series.last("365D")

        x_hist = [d.date() for d in history_last_year.index]
        y_hist = [float(v) for v in history_last_year.values]

        x_fore = [d.date() for d in forecast_df.index]
        y_fore = [float(v) for v in forecast_df["forecast_rate"].values]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_hist,
                y=y_hist,
                mode="lines+markers",
                name="Histórico (fecho)",
                line=dict(width=2)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_fore,
                y=y_fore,
                mode="lines+markers",
                name="Previsão",
                line=dict(width=2, dash="dash")
            )
        )

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

    return render_template(
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

# ========= E-COMMERCE DASHBOARD (VENDAS) =========

# Caminho do CSV de e-commerce
ECOM_PATH = Path(__file__).resolve().parent / "data" / "e_commerce.csv"

ECOM_DF = None
ECOM_LOAD_ERROR = None

def _load_ecom_data():
    """
    Carrega e limpa o dataset de e-commerce.
    Espera um CSV com colunas: 
    InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
    """
    global ECOM_DF, ECOM_LOAD_ERROR

    try:
        df = pd.read_csv(ECOM_PATH, encoding="latin1")

        # Converter data (dataset é dd/mm/yyyy hh:mm)
        df["InvoiceDate"] = pd.to_datetime(
            df["InvoiceDate"],
            dayfirst=True,
            errors="coerce"
        )

        # Remover linhas sem data
        df = df.dropna(subset=["InvoiceDate"])

        # Filtrar vendas válidas (sem devoluções)
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

        # Valor total da linha
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

        # Garantir tipo numérico de CustomerID
        if "CustomerID" in df.columns:
            df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")

        ECOM_DF = df
        ECOM_LOAD_ERROR = None
    except Exception as e:
        ECOM_DF = None
        ECOM_LOAD_ERROR = f"Erro ao carregar dataset de e-commerce: {e}"

# carregar ao subir a app
_load_ecom_data()


# ---------- Meteo ----------
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
            "wind_unit": "km/h" if wind_unit == "kmh" else ("mph" if wind_unit == "mph" else wind_unit),
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

    return render_template("weather.html",
                           city=city, meta=meta, chart=chart, units=units, error=error)

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

# ---------- ML: Heart Disease ----------
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

    # defaults iniciais (usamos medianas quando possível)
    defaults = {
        "sexo":                          "1",
        "idade":                         _dflt("idade", 50),
        "cigarros_por_dia":              _dflt("cigarros_por_dia", 0),
        "uso_medicamento_pressao":       "0",
        "AVC":                           "0",
        "hipertensao":                   "0",
        "diabetes":                      "0",
        "colesterol_total":              _dflt("colesterol_total", 220),
        "pressao_arterial_sistolica":    _dflt("pressao_arterial_sistolica", 130),
        "pressao_arterial_diastolica":   _dflt("pressao_arterial_diastolica", 80),
        "IMC":                           _dflt("IMC", 27.0),
        "freq_cardiaca":                 _dflt("freq_cardiaca", 80),
        "glicemia":                      _dflt("glicemia", 90),
        "fumante":                       "0",
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
                p_hat = float(max(p_all))  # confiança na classe prevista
            result = "Risco ALTO (provável presença)" if y_hat == 1 else "Risco BAIXO (provável ausência)"
            prob = p_hat
            for k in _HEART_FEATURES:
                defaults[k] = str(vals[k])
        except Exception as e:
            error = f"Entrada inválida: {e}"

    heart_info = {"source": _HEART_SOURCE or "indisponível"}
    try:
        metas = sorted((ARTIF_DIR.glob("meta_*.joblib")), reverse=True)
        for mp in metas:
            meta = joblib_load(mp)
            if meta.get("task") == "heart_disease":
                heart_info["version"] = meta.get("version")
                break
    except Exception:
        pass

    return render_template(
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

# ---------- NLP: Supervised (modelo PT carregado) ----------
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

    return render_template(
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
    

# ---------- Dashboard de E-commerce (Vendas) ----------
@app.route("/ecom", methods=["GET"])
def ecom_dashboard():
    """
    Dashboard de vendas da loja online.
    Mostra KPIs + filtros + gráficos de receita mensal, top produtos e top países.
    """
    erro = ECOM_LOAD_ERROR
    kpis = None
    revenue_plot_div = None
    products_plot_div = None
    countries_plot_div = None
    top_products_table = None
    top_countries_table = None

    # Filtros vindos da query string (?start_date=...&end_date=...)
    filtros = {
        "start_date": request.args.get("start_date", "").strip(),
        "end_date": request.args.get("end_date", "").strip(),
        "country": request.args.get("country", "").strip(),
        "product": request.args.get("product", "").strip(),  # agora virá do <select>
    }

    # Listas para os <select>
    countries_list = []
    products_list = []  # NOVO

    if ECOM_DF is not None:
        countries_list = sorted(
            ECOM_DF["Country"].dropna().unique().tolist()
        )
        products_list = sorted(      # NOVO
            ECOM_DF["Description"].dropna().unique().tolist()
        )

    if ECOM_DF is not None and erro is None:
        df = ECOM_DF.copy()

        # ----- Aplicar filtros -----
        # Filtro de data inicial
        if filtros["start_date"]:
            try:
                dt_ini = pd.to_datetime(filtros["start_date"])
                df = df[df["InvoiceDate"] >= dt_ini]
            except Exception:
                pass

        # Filtro de data final
        if filtros["end_date"]:
            try:
                dt_fim = pd.to_datetime(filtros["end_date"])
                df = df[df["InvoiceDate"] <= dt_fim]
            except Exception:
                pass

        # Filtro de país
        if filtros["country"]:
            df = df[df["Country"] == filtros["country"]]

        # Filtro de produto (agora com igualdade, vindo do dropdown)
        if filtros["product"]:
            df = df[df["Description"] == filtros["product"]]

        if df.empty:
            erro = "Nenhum dado encontrado para os filtros selecionados."
        else:
            # ----- KPIs -----
            total_revenue = float(df["TotalPrice"].sum())
            num_orders = int(df["InvoiceNo"].nunique())
            num_customers = int(df["CustomerID"].nunique())
            total_qty = float(df["Quantity"].sum())  # total de itens

            avg_ticket = float(total_revenue / num_orders) if num_orders > 0 else 0.0
            avg_items_per_order = float(total_qty / num_orders) if num_orders > 0 else 0.0
            avg_revenue_per_customer = float(total_revenue / num_customers) if num_customers > 0 else 0.0

            # Período coberto pelos dados filtrados
            first_date = pd.to_datetime(df["InvoiceDate"].min())
            last_date = pd.to_datetime(df["InvoiceDate"].max())
            period_days = max((last_date - first_date).days + 1, 1)
            avg_daily_revenue = float(total_revenue / period_days) if period_days > 0 else 0.0

            kpis = {
                "total_revenue": total_revenue,
                "num_orders": num_orders,
                "num_customers": num_customers,
                "avg_ticket": avg_ticket,
                "avg_items_per_order": avg_items_per_order,
                "avg_revenue_per_customer": avg_revenue_per_customer,
                "period_days": period_days,
                "avg_daily_revenue": avg_daily_revenue,
            }

            # ----- Receita mensal -----
            df_month = (
                df.set_index("InvoiceDate")
                  .resample("M")["TotalPrice"]
                  .sum()
                  .reset_index()
            )
            df_month["MonthStr"] = df_month["InvoiceDate"].dt.strftime("%Y-%m")

            fig_rev = go.Figure()
            fig_rev.add_trace(
                go.Scatter(
                    x=df_month["MonthStr"],
                    y=df_month["TotalPrice"],
                    mode="lines+markers",
                    name="Receita mensal",
                    line=dict(width=2)
                )
            )
            fig_rev.update_layout(
                title="Receita mensal",
                xaxis_title="Mês",
                yaxis_title="Receita",
                margin=dict(l=40, r=20, t=40, b=40),
                template="plotly_dark",
                height=350,
                hovermode="x unified",
            )
            fig_rev.update_yaxes(tickformat=".2f")
            revenue_plot_div = plot(fig_rev, include_plotlyjs="cdn", output_type="div")

            # ----- Top 10 produtos por receita -----
            top_products = (
                df.groupby("Description", as_index=False)["TotalPrice"]
                  .sum()
                  .sort_values("TotalPrice", ascending=False)
                  .head(10)
            )

            fig_prod = go.Figure()
            fig_prod.add_trace(
                go.Bar(
                    x=top_products["TotalPrice"],
                    y=top_products["Description"],
                    orientation="h",
                    name="Receita por produto",
                )
            )
            fig_prod.update_layout(
                title="Top 10 produtos por receita",
                xaxis_title="Receita",
                yaxis_title="Produto",
                margin=dict(l=120, r=20, t=40, b=40),
                template="plotly_dark",
                height=450,
            )
            products_plot_div = plot(fig_prod, include_plotlyjs=False, output_type="div")

            # Tabela dos produtos
            top_products_table = top_products.to_dict(orient="records")

            # ----- Top 10 países por receita -----
            top_countries = (
                df.groupby("Country", as_index=False)["TotalPrice"]
                  .sum()
                  .sort_values("TotalPrice", ascending=False)
                  .head(10)
            )

            fig_ctry = go.Figure()
            fig_ctry.add_trace(
                go.Bar(
                    x=top_countries["Country"],
                    y=top_countries["TotalPrice"],
                    name="Receita por país",
                )
            )
            fig_ctry.update_layout(
                title="Top 10 países por receita",
                xaxis_title="País",
                yaxis_title="Receita",
                margin=dict(l=40, r=20, t=40, b=40),
                template="plotly_dark",
                height=350,
            )
            countries_plot_div = plot(fig_ctry, include_plotlyjs=False, output_type="div")

            # Tabela dos países
            top_countries_table = top_countries.to_dict(orient="records")

    return render_template(
        "ecom.html",
        erro=erro,
        kpis=kpis,
        revenue_plot_div=revenue_plot_div,
        products_plot_div=products_plot_div,
        countries_plot_div=countries_plot_div,
        top_products_table=top_products_table,
        top_countries_table=top_countries_table,
        filtros=filtros,
        countries=countries_list,
        products=products_list,   # NOVO
    )

# ---------- Dashboard de E-commerce: RFM / Segmentação ----------
@app.route("/ecom/rfm", methods=["GET"])
def ecom_rfm():
    """
    Análise RFM (Recency, Frequency, Monetary) por cliente
    + segmentação em grupos (VIP, Leal, etc.).
    """
    erro = ECOM_LOAD_ERROR
    rfm_table = None
    seg_summary = None
    seg_plot_div = None

    # Filtro simples por país
    filtros = {
        "country": request.args.get("country", "").strip(),
    }

    # Lista de países para o select
    countries_list = []
    if ECOM_DF is not None:
        countries_list = sorted(
            ECOM_DF["Country"].dropna().unique().tolist()
        )

    if ECOM_DF is not None and erro is None:
        df = ECOM_DF.copy()

        # Garante que temos InvoiceDate em datetime
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        # Aplica filtro de país (opcional)
        if filtros["country"]:
            df = df[df["Country"] == filtros["country"]]

        # Remove clientes sem ID
        df = df.dropna(subset=["CustomerID"])

        if df.empty:
            erro = "Nenhum dado encontrado para os filtros selecionados."
        else:
            # Data de referência = última data de compra do dataset filtrado
            ref_date = df["InvoiceDate"].max()

            # ---- Cálculo do RFM por cliente ----
            rfm = (
                df.groupby("CustomerID")
                  .agg(
                      Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
                      Frequency=("InvoiceNo", "nunique"),
                      Monetary=("TotalPrice", "sum"),
                  )
                  .reset_index()
            )

            if rfm.empty:
                erro = "Não foi possível calcular RFM (sem clientes válidos)."
            else:
                # ---- Pontuações R, F, M em 4 faixas (1–4) ----
                # Podem falhar se houver poucos valores distintos => tratamos com try/except.

                # Recency: quanto menor, melhor => labels 4 (mais recente) -> 1 (mais antigo)
                try:
                    rfm["R_score"] = pd.qcut(
                        rfm["Recency"],
                        4,
                        labels=[4, 3, 2, 1]
                    ).astype(int)
                except Exception:
                    rfm["R_score"] = 2

                # Frequency: quanto maior, melhor => labels 1 (menos freq) -> 4 (mais freq)
                try:
                    rfm["F_score"] = pd.qcut(
                        rfm["Frequency"],
                        4,
                        labels=[1, 2, 3, 4]
                    ).astype(int)
                except Exception:
                    rfm["F_score"] = 2

                # Monetary: quanto maior, melhor => labels 1 (menor) -> 4 (maior)
                try:
                    rfm["M_score"] = pd.qcut(
                        rfm["Monetary"],
                        4,
                        labels=[1, 2, 3, 4]
                    ).astype(int)
                except Exception:
                    rfm["M_score"] = 2

                # Score total
                rfm["RFM_score"] = (
                    rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
                )

                # ---- Segmentação simples baseada no score total ----
                def _segment(row):
                    s = row["RFM_score"]
                    if s >= 10:
                        return "Clientes VIP"
                    elif s >= 8:
                        return "Clientes Leais"
                    elif s >= 5:
                        return "Em crescimento"
                    else:
                        return "Em risco"

                rfm["Segment"] = rfm.apply(_segment, axis=1)

                # ---- Resumo por segmento ----
                seg_summary_df = (
                    rfm.groupby("Segment", as_index=False)
                       .agg(
                           num_customers=("CustomerID", "nunique"),
                           avg_recency=("Recency", "mean"),
                           avg_frequency=("Frequency", "mean"),
                           total_monetary=("Monetary", "sum"),
                       )
                       .sort_values("total_monetary", ascending=False)
                )

                seg_summary = seg_summary_df.to_dict(orient="records")

                # ---- Gráfico Plotly: nº de clientes por segmento ----
                fig_seg = go.Figure()
                fig_seg.add_trace(
                    go.Bar(
                        x=seg_summary_df["Segment"],
                        y=seg_summary_df["num_customers"],
                        name="Nº de clientes",
                    )
                )
                fig_seg.update_layout(
                    title="Distribuição de clientes por segmento RFM",
                    xaxis_title="Segmento",
                    yaxis_title="Nº de clientes",
                    margin=dict(l=40, r=20, t=40, b=40),
                    template="plotly_dark",
                    height=350,
                )
                seg_plot_div = plot(fig_seg, include_plotlyjs="cdn", output_type="div")

                # ---- Tabela de clientes (limitamos aos top 200 por score) ----
                rfm_table = (
                    rfm.sort_values(["RFM_score", "Monetary"], ascending=[False, False])
                       .head(200)
                       .to_dict(orient="records")
                )

    return render_template(
        "ecom_rfm.html",
        erro=erro,
        filtros=filtros,
        countries=countries_list,
        seg_plot_div=seg_plot_div,
        seg_summary=seg_summary,
        rfm_table=rfm_table,
    )

# ---------- Dashboard de E-commerce: Previsão de Receita ----------

@app.route("/ecom/forecast", methods=["GET"])
def ecom_forecast():
    """
    Previsão de receita mensal usando Prophet.
    Agrupa as vendas por mês, treina o modelo e prevê os próximos N meses.
    """
    erro = ECOM_LOAD_ERROR
    kpis = None
    forecast_plot_div = None

    # horizonte em meses (query string ?periods=12)
    periods_raw = request.args.get("periods", "12")
    try:
        periods = int(periods_raw)
    except Exception:
        periods = 12
    periods = max(1, min(36, periods))  # entre 1 e 36 meses

    if ECOM_DF is not None and erro is None:
        df = ECOM_DF.copy()

        # Garante datetime
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        # Receita mensal
        df_month = (
            df.set_index("InvoiceDate")
              .resample("M")["TotalPrice"]
              .sum()
              .reset_index()
        )

        if df_month.empty or len(df_month) < 6:
            erro = "Poucos dados mensais para treinar o modelo de previsão."
        else:
            # Prophet espera colunas ds (data) e y (valor)
            df_prophet = df_month.rename(columns={"InvoiceDate": "ds", "TotalPrice": "y"})

            # Treino
            m = Prophet()
            m.fit(df_prophet)

            # Futuro: próximos 'periods' meses
            future = m.make_future_dataframe(periods=periods, freq="M")
            forecast = m.predict(future)

            # Separa histórico vs futuro (baseado na última data real)
            last_ds = df_prophet["ds"].max()
            hist_mask = forecast["ds"] <= last_ds
            fut_mask = forecast["ds"] > last_ds

            hist = forecast.loc[hist_mask, ["ds", "yhat"]].copy()
            fut = forecast.loc[fut_mask, ["ds", "yhat"]].copy()

            hist["MonthStr"] = hist["ds"].dt.strftime("%Y-%m")
            fut["MonthStr"] = fut["ds"].dt.strftime("%Y-%m")

            # ---------- Gráfico Plotly ----------
            fig = go.Figure()

            # Receita real (barras)
            fig.add_trace(
                go.Bar(
                    x=df_prophet["ds"].dt.strftime("%Y-%m"),
                    y=df_prophet["y"],
                    name="Receita real (mensal)",
                )
            )

            # Previsão futura (linha tracejada)
            if not fut.empty:
                fig.add_trace(
                    go.Scatter(
                        x=fut["MonthStr"],
                        y=fut["yhat"],
                        mode="lines+markers",
                        name="Previsão (yhat)",
                        line=dict(width=2, dash="dash"),
                    )
                )

            fig.update_layout(
                title=f"Receita mensal e previsão ({periods} meses à frente)",
                xaxis_title="Mês",
                yaxis_title="Receita",
                margin=dict(l=40, r=20, t=40, b=40),
                template="plotly_dark",
                height=400,
                hovermode="x unified",
            )
            fig.update_yaxes(tickformat=".2f")

            forecast_plot_div = plot(fig, include_plotlyjs="cdn", output_type="div")

            # ---------- KPIs da previsão ----------
            df_sorted = df_prophet.sort_values("ds").reset_index(drop=True)

            # Receita dos últimos 12 meses (histórico)
            last12_hist = df_sorted.tail(12)
            last12_revenue = float(last12_hist["y"].sum())

            # Receita prevista para os próximos 12 meses (ou menos se periods < 12)
            next12_fut = fut.sort_values("ds").head(12)
            next12_revenue = float(next12_fut["yhat"].sum()) if not next12_fut.empty else None

            # CAGR aproximado (histórico)
            cagr = None
            try:
                first_val = float(df_sorted["y"].iloc[0])
                last_val = float(df_sorted["y"].iloc[-1])
                years = (df_sorted["ds"].iloc[-1] - df_sorted["ds"].iloc[0]).days / 365.25
                if years > 0 and first_val > 0:
                    cagr = (last_val / first_val) ** (1 / years) - 1
            except Exception:
                cagr = None

            avg_monthly_hist = float(df_sorted["y"].mean())
            avg_monthly_forecast = None
            try:
                if not next12_fut.empty:
                    avg_monthly_forecast = float(next12_fut["yhat"].mean())
            except Exception:
                avg_monthly_forecast = None

            kpis = {
                "periods": periods,
                "last12_revenue": last12_revenue,
                "next12_revenue": next12_revenue,
                "cagr": cagr,
                "avg_monthly_hist": avg_monthly_hist,
                "avg_monthly_forecast": avg_monthly_forecast,
            }

    return render_template(
        "ecom_forecast.html",
        erro=erro,
        kpis=kpis,
        periods=periods,
        forecast_plot_div=forecast_plot_div,
    )


# ---------- Dashboard de E-commerce: Clusters (K-Means sobre RFM) ----------
@app.route("/ecom/clusters", methods=["GET"])
def ecom_clusters():
    """
    Clusterização de clientes com K-Means usando RFM (Recency, Frequency, Monetary).
    Mostra:
      - Distribuição de clientes por cluster
      - Scatter plot (Frequency x Monetary) colorido por cluster
      - Tabela-resumo por cluster
      - Tabela de clientes (top 300)
    """
    erro = ECOM_LOAD_ERROR
    cluster_scatter_div = None
    cluster_bar_div = None
    cluster_summary = None
    cluster_table = None

    # Filtros
    filtros = {
        "country": request.args.get("country", "").strip(),
    }

    # Nº de clusters (k) com limites sensatos
    try:
        k = int(request.args.get("k", "4"))
    except Exception:
        k = 4
    if k < 2:
        k = 2
    if k > 8:
        k = 8

    # Lista de países para o <select>
    countries_list = []
    if ECOM_DF is not None:
        countries_list = sorted(
            ECOM_DF["Country"].dropna().unique().tolist()
        )

    if ECOM_DF is not None and erro is None:
        df = ECOM_DF.copy()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        # Filtro de país (opcional)
        if filtros["country"]:
            df = df[df["Country"] == filtros["country"]]

        # Remove clientes sem ID
        df = df.dropna(subset=["CustomerID"])

        if df.empty:
            erro = "Nenhum dado encontrado para os filtros selecionados."
        else:
            # Data de referência = última compra do dataset filtrado
            ref_date = df["InvoiceDate"].max()

            # ---- Cálculo do RFM por cliente ----
            rfm = (
                df.groupby("CustomerID")
                  .agg(
                      Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
                      Frequency=("InvoiceNo", "nunique"),
                      Monetary=("TotalPrice", "sum"),
                  )
                  .reset_index()
            )

            # Remove casos “quebrados” (sem gasto ou sem compras)
            rfm = rfm[(rfm["Monetary"] > 0) & (rfm["Frequency"] > 0)]

            if rfm.empty:
                erro = "Não foi possível calcular clusters (sem clientes válidos após filtragem)."
            else:
                # ---- Preparar dados para K-Means ----
                X = rfm[["Recency", "Frequency", "Monetary"]].copy()

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                km = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=10,
                )
                rfm["Cluster"] = km.fit_predict(X_scaled)

                # ---- Resumo por cluster ----
                cluster_summary_df = (
                    rfm.groupby("Cluster", as_index=False)
                       .agg(
                           num_customers=("CustomerID", "nunique"),
                           avg_recency=("Recency", "mean"),
                           avg_frequency=("Frequency", "mean"),
                           avg_monetary=("Monetary", "mean"),
                           total_monetary=("Monetary", "sum"),
                       )
                       .sort_values("total_monetary", ascending=False)
                )
                cluster_summary = cluster_summary_df.to_dict(orient="records")

                # ---- Gráfico: nº clientes por cluster ----
                fig_bar = go.Figure()
                fig_bar.add_trace(
                    go.Bar(
                        x=[f"Cluster {int(c)}" for c in cluster_summary_df["Cluster"]],
                        y=cluster_summary_df["num_customers"],
                        name="Nº de clientes",
                    )
                )
                fig_bar.update_layout(
                    title=f"Distribuição de clientes por cluster (k={k})",
                    xaxis_title="Cluster",
                    yaxis_title="Nº de clientes",
                    margin=dict(l=40, r=20, t=40, b=40),
                    template="plotly_dark",
                    height=350,
                )
                # Inclui JS do Plotly neste primeiro gráfico
                cluster_bar_div = plot(fig_bar, include_plotlyjs="cdn", output_type="div")

                # ---- Scatter: Frequency x Monetary colorido por cluster ----
                fig_scatter = go.Figure()
                fig_scatter.add_trace(
                    go.Scatter(
                        x=rfm["Frequency"],
                        y=rfm["Monetary"],
                        mode="markers",
                        text=[f"Cliente {cid}" for cid in rfm["CustomerID"]],
                        marker=dict(
                            size=7,
                            color=rfm["Cluster"],
                            colorscale="Viridis",
                            showscale=True,
                        ),
                    )
                )
                fig_scatter.update_layout(
                    title=f"Clusters de clientes (Frequency x Monetary) — k={k}",
                    xaxis_title="Frequency (nº de encomendas)",
                    yaxis_title="Monetary (receita total)",
                    margin=dict(l=40, r=20, t=40, b=40),
                    template="plotly_dark",
                    height=400,
                )
                # Não precisa repetir o JS do Plotly
                cluster_scatter_div = plot(fig_scatter, include_plotlyjs=False, output_type="div")

                # ---- Tabela de clientes (top 300) ----
                cluster_table = (
                    rfm.sort_values(["Cluster", "Monetary"], ascending=[True, False])
                       .head(300)
                       .to_dict(orient="records")
                )

    return render_template(
        "ecom_clusters.html",
        erro=erro,
        filtros=filtros,
        countries=countries_list,
        k=k,
        cluster_bar_div=cluster_bar_div,
        cluster_scatter_div=cluster_scatter_div,
        cluster_summary=cluster_summary,
        cluster_table=cluster_table,
    )


@app.route("/churn_xai", methods=["GET", "POST"])
def churn_xai_dashboard():
    """
    Formulário interativo para churn:
    - user preenche dados de um cliente (Telco)
    - modelo prevê probabilidade de churn
    """

    erro = _CHURN_ERROR
    result = None
    prob_pct = None

    # Defaults do formulário (podes ajustar à vontade)
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
                # Monta df_raw com as colunas do CSV original
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

    return render_template(
        "churn_xai.html",
        erro=erro,
        result=result,
        prob_pct=prob_pct,
        defaults=defaults,
    )

@app.route("/ames", methods=["GET", "POST"])
def ames_dashboard():
    # df completo, sem filtro (para algumas análises globais)
    df_completo = load_ames_data()
    df = df_completo.copy()

    # Variáveis numéricas do projeto (exceto faixa_preco)
    numeric_cols = [c for c in COLUNAS_PROJETO if c in df.columns and c != "faixa_preco"]

    # Variável seleccionada
    default_var = "preco" if "preco" in numeric_cols else numeric_cols[0]
    var = request.form.get("variavel", default_var)

    # Filtro de faixa de preço
    faixas_unicas = ["Todos"]
    if "faixa_preco" in df.columns:
        faixas_unicas += sorted(df["faixa_preco"].dropna().unique().tolist())

    faixa_selecionada = request.form.get("faixa_preco", "Todos")

    # Aplica filtro se não for "Todos"
    df_filtrado = df.copy()
    if faixa_selecionada != "Todos" and "faixa_preco" in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado["faixa_preco"] == faixa_selecionada]

    # Série da variável (já filtrada)
    if var not in df_filtrado.columns:
        # fallback de segurança
        var = default_var

    serie = df_filtrado[var].dropna()

    # Estatísticas descritivas
    stats_dict = calcular_estatisticas_1d(serie)

    # Testes adicionais
    # - df_filtrado: respeita filtro de faixa_preco
    # - df_completo: usado para Kruskal–Wallis entre faixas (todas as faixas)
    testes_extra = calcular_testes_adicionais(
        serie=serie,
        df_filtrado=df_filtrado,
        var=var,
        df_completo=df_completo if faixa_selecionada == "Todos" else None,
    )

    # Nome amigável da variável
    label = NOMES_AMIGAVEIS.get(var, var)

    # Gráficos (com df_filtrado)
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

    # Interpretação da normalidade (Shapiro)
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

    return render_template(
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




# Healthcheck
@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

