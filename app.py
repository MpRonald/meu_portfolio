import os
import json
import logging
from datetime import datetime
from pathlib import Path
from io import StringIO
from types import SimpleNamespace

import pandas as pd
import plotly
import plotly.express as px
from plotly.offline import plot as plotly_plot

from flask import Flask, render_template, jsonify, request, make_response
from werkzeug.middleware.proxy_fix import ProxyFix
from jinja2 import TemplateNotFound

from services import PortfolioService


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))).resolve()
ARTIF_DIR = Path(os.getenv("ARTIF_DIR", str(BASE_DIR / "artifacts"))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIF_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_DEFAULT_CITY = "Lisboa"


# =========================
# Ames constants (compat com teu template)
# =========================
COLUNAS_PROJETO = [
    "preco", "quartos", "banheiros", "area_habitavel", "area_lote",
    "andares", "area_acima_solo", "area_porao", "ano_construcao",
    "latitude", "longitude", "area_habitavel_viz", "area_lote_viz",
    "faixa_preco", "idade_imovel", "area_total", "densidade_construcao", "preco_m2",
]

NOMES_AMIGAVEIS = {
    "preco": "Property Price (€)",
    "quartos": "Bedrooms",
    "banheiros": "Bathrooms",
    "area_habitavel": "Living Area (m²)",
    "area_lote": "Lot Area (m²)",
    "andares": "Number of Floors",
    "area_acima_solo": "Above-Ground Area (m²)",
    "area_porao": "Basement Area (m²)",
    "ano_construcao": "Year Built",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "area_habitavel_viz": "Neighborhood Avg Living Area (m²)",
    "area_lote_viz": "Neighborhood Avg Lot Area (m²)",
    "faixa_preco": "Price Range",
    "idade_imovel": "Property Age (years)",
    "area_total": "Total Area (m²)",
    "densidade_construcao": "Build Density",
    "preco_m2": "Price per m² (€)",
}

NOMES_FAIXA = {
    "baixo": "Low Price",
    "medio": "Medium Price",
    "alto": "High Price",
    "muito_alto": "Very High Price",
}

PRICE_RANGE_EN = {
    "baixo": "Low",
    "medio": "Medium",
    "alto": "High",
    "muito_alto": "Very High",
}


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
        app.config.get("ARTIF_DIR"),
    )


def create_app() -> Flask:
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")
    env = (os.getenv("FLASK_ENV") or os.getenv("ENV") or "production").lower().strip()
    app.config["ENV_NAME"] = env
    debug_flag = (os.getenv("FLASK_DEBUG") or "").strip() == "1"
    app.config["DEBUG"] = True if env == "development" or debug_flag else False

    app.config["BASE_DIR"] = str(BASE_DIR)
    app.config["DATA_DIR"] = str(DATA_DIR)
    app.config["ARTIF_DIR"] = str(ARTIF_DIR)

    _configure_logging(app)

    # Service com acesso ao DATA_DIR
    service = PortfolioService(timeout=30, data_dir=app.config["DATA_DIR"])

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
    def not_found(_e):
        return jsonify({"error": "not_found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        env_name = (app.config.get("ENV_NAME") or "production").lower()
        is_prod = env_name == "production" or os.getenv("RAILWAY_ENVIRONMENT") is not None
        if is_prod:
            return jsonify({"error": "internal_server_error"}), 500
        return jsonify({"error": "internal_server_error", "detail": str(e)}), 500

    def safe_render(template_name: str, **ctx):
        try:
            return render_template(template_name, **ctx)
        except TemplateNotFound:
            return jsonify({
                "template_missing": template_name,
                "hint": "Template não encontrado (ok nesta fase).",
                "context_keys": sorted(list(ctx.keys()))
            }), 200

    @app.route("/healthz")
    def healthz():
        return jsonify({
            "status": "ok",
            "time": datetime.utcnow().isoformat() + "Z",
            "env": app.config.get("ENV_NAME"),
        }), 200

    @app.route("/")
    def index():
        return safe_render("index.html")

    # =========================
    # STUBS (para não quebrar o index.html)
    # =========================
    @app.route("/ecom")
    def ecom_dashboard():
        return safe_render("ecom.html", erro="(Em construção) A migrar para o novo app.")

    @app.route("/ecom/rfm")
    def ecom_rfm():
        return safe_render("ecom_rfm.html", erro="(Em construção) A migrar para o novo app.")

    @app.route("/ecom/forecast")
    def ecom_forecast():
        return safe_render("ecom_forecast.html", erro="(Em construção) A migrar para o novo app.")

    @app.route("/ecom/clusters")
    def ecom_clusters():
        return safe_render("ecom_clusters.html", erro="(Em construção) A migrar para o novo app.")

    @app.route("/quotes")
    def quotes():
        return safe_render("quotes.html", erro="(Em construção) A migrar para o novo app.")

    # =========================
    # FX (RF / ARIMA / Prophet)
    # =========================
    @app.route("/fx", methods=["GET", "POST"])
    def fx():
        pairs = dict(getattr(service, "FX_PAIRS", {}))  # existe no teu services.py

        if request.method == "GET":
            return safe_render("fx.html", pairs=pairs, resultado=None, erro=None)

        # POST
        try:
            pair_name = (request.form.get("pair") or "").strip()
            algoritmo = (request.form.get("algoritmo") or "rf").strip().lower()
            n_days = int(request.form.get("n_days", 30))

            if pair_name not in pairs:
                raise ValueError("Par de moedas inválido.")
            if n_days < 1 or n_days > 180:
                raise ValueError("n_days deve estar entre 1 e 180.")
            if algoritmo not in {"rf", "arima", "prophet"}:
                raise ValueError("Algoritmo inválido.")

            ticker = pairs[pair_name]
            series = service.fx_download_history(ticker=ticker, period="3y")

            # Treinar + prever
            if algoritmo == "rf":
                metrics, forecast_df = service.fx_train_and_forecast_rf(series, n_days)
                algoritmo_label = "Random Forest"
            elif algoritmo == "arima":
                metrics, forecast_df = service.fx_train_and_forecast_arima(series, n_days)
                algoritmo_label = "ARIMA"
            else:
                # Prophet pode não estar instalado -> mensagem amigável
                try:
                    metrics, forecast_df = service.fx_train_and_forecast_prophet(series, n_days)
                except ModuleNotFoundError:
                    raise RuntimeError("Prophet não está disponível neste deploy. Usa RF ou ARIMA.")
                algoritmo_label = "Prophet"

            # Gráfico: histórico recente + previsão
            hist = series.copy().dropna()
            hist_recent = hist.tail(180)  # deixa leve

            df_hist = pd.DataFrame({"date": hist_recent.index, "value": hist_recent.values, "type": "Histórico"})
            df_fc = pd.DataFrame({"date": forecast_df.index, "value": forecast_df["forecast_rate"].values, "type": "Previsão"})
            df_plot = pd.concat([df_hist, df_fc], ignore_index=True)

            fig = px.line(
                df_plot,
                x="date",
                y="value",
                color="type",
                title=f"{pair_name} — Histórico e Previsão ({algoritmo_label})"
            )
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            fx_plot_div = plotly_plot(fig, output_type="div", include_plotlyjs=False)

            # Montar objeto compatível com o template (acesso por ponto)
            resultado = SimpleNamespace(
                pair_name=pair_name,
                algoritmo=algoritmo,
                algoritmo_label=algoritmo_label,
                n_days=n_days,
                metrics=SimpleNamespace(**metrics),
                forecast=forecast_df,
                fx_plot_div=fx_plot_div,
            )

            return safe_render("fx.html", pairs=pairs, resultado=resultado, erro=None)

        except Exception as e:
            return safe_render("fx.html", pairs=pairs, resultado=None, erro=str(e))

    @app.route("/ml/heart")
    def ml_heart():
        return safe_render("ml_heart.html", erro="(Em construção) A migrar para o novo app.")

    @app.route("/nlp/supervised")
    def nlp_supervised():
        return safe_render("nlp_supervised.html", erro="(Em construção) A migrar para o novo app.")

    @app.route("/loan")
    def loan_form():
        return safe_render("loan.html", erro="(Em construção) A migrar para o novo app.")

    @app.route("/churn/xai")
    def churn_xai_dashboard():
        return safe_render("churn_xai.html", erro="(Em construção) A migrar para o novo app.")

    # =========================
    # WEATHER
    # =========================
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
            loc = service.geocode_city(city or WEATHER_DEFAULT_CITY, count=1, lang="pt")
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

            js = service.fetch_weather_forecast(
                loc["latitude"], loc["longitude"], days, temp_unit, wind_unit, lang="pt"
            )
            daily = js.get("daily") or {}
            dates = daily.get("time") or []
            tmax = daily.get("temperature_2m_max") or []
            tmin = daily.get("temperature_2m_min") or []
            rain = daily.get("precipitation_sum") or []
            wmax = daily.get("wind_speed_10m_max") or []

            if not dates:
                raise RuntimeError("Sem dados de previsão para este local.")

            prob = [None] * len(dates)

            chart = {
                "labels": dates,
                "tmax": [round(x, 2) if x is not None else None for x in tmax],
                "tmin": [round(x, 2) if x is not None else None for x in tmin],
                "rain": [round(x, 2) if x is not None else None for x in rain],
                "wmax": [round(x, 2) if x is not None else None for x in wmax],
                "prob": prob,
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

        loc = service.geocode_city(city or WEATHER_DEFAULT_CITY, count=1, lang="pt")
        if not loc:
            return make_response("Cidade não encontrada", 400)

        js = service.fetch_weather_forecast(loc["latitude"], loc["longitude"], days, temp_unit, wind_unit, lang="pt")
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

    # =========================
    # AMES
    # =========================
    @app.route("/ames", methods=["GET", "POST"])
    def ames_dashboard():
        df_completo = service.load_ames_data()
        df = df_completo.copy()

        numeric_cols = [c for c in COLUNAS_PROJETO if c in df.columns and c != "faixa_preco"]
        if not numeric_cols:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            return jsonify({"error": "No numeric columns available in the Ames dataset."}), 500

        nomes_amig = dict(NOMES_AMIGAVEIS)
        for c in numeric_cols:
            nomes_amig.setdefault(c, c)

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

        serie = pd.to_numeric(df_filtrado[var], errors="coerce").dropna()
        stats_dict = service.calcular_estatisticas_1d(serie)

        testes_extra = service.calcular_testes_adicionais(
            serie=serie,
            df_filtrado=df_filtrado,
            var=var,
            df_completo=df_completo if faixa_selecionada == "Todos" else None,
        )

        label = nomes_amig.get(var, var)

        df_plot = df_filtrado.copy()
        if "faixa_preco" in df_plot.columns:
            df_plot["price_range_en"] = df_plot["faixa_preco"].map(PRICE_RANGE_EN).fillna(df_plot["faixa_preco"])

        color_col = "price_range_en" if "price_range_en" in df_plot.columns else None

        fig_hist = px.histogram(
            df_plot,
            x=var,
            nbins=40,
            marginal="box",
            title=f"Distribution of {label}",
            labels={var: label},
        )

        fig_box = px.box(
            df_plot,
            y=var,
            points="outliers",
            title=f"Boxplot of {label}",
            labels={var: label},
        )

        graph_hist_json = json.dumps(fig_hist, cls=plotly.utils.PlotlyJSONEncoder)
        graph_box_json = json.dumps(fig_box, cls=plotly.utils.PlotlyJSONEncoder)

        graph_scatter_json = None
        if "preco" in df_plot.columns and var in df_plot.columns and var != "preco":
            fig_scatter = px.scatter(
                df_plot,
                x=var,
                y="preco",
                color=color_col,
                title=f"Price vs {label}",
                labels={
                    var: label,
                    "preco": "Price (€)",
                    "price_range_en": "Price range",
                },
            )
            graph_scatter_json = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)

        graph_box_faixa_json = None
        if "price_range_en" in df_plot.columns:
            fig_box_faixa = px.box(
                df_plot,
                x="price_range_en",
                y=var,
                title=f"{label} by price range",
                labels={"price_range_en": "Price range", var: label},
            )
            graph_box_faixa_json = json.dumps(fig_box_faixa, cls=plotly.utils.PlotlyJSONEncoder)

        graph_preco_ano_json = None
        if "preco" in df_plot.columns and "ano_construcao" in df_plot.columns:
            fig_preco_ano = px.scatter(
                df_plot,
                x="ano_construcao",
                y="preco",
                color=color_col,
                title="Price vs Year Built",
                labels={
                    "ano_construcao": "Year Built",
                    "preco": "Price (€)",
                    "price_range_en": "Price range",
                },
            )
            graph_preco_ano_json = json.dumps(fig_preco_ano, cls=plotly.utils.PlotlyJSONEncoder)

        graph_heatmap_json = None
        corr_cols = [c for c in ["preco", "preco_m2", "area_habitavel", "area_total", "quartos", "banheiros"] if c in df_plot.columns]
        if len(corr_cols) >= 2:
            corr = df_plot[corr_cols].corr(numeric_only=True)
            fig_heat = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation matrix (Pearson)",
            )
            graph_heatmap_json = json.dumps(fig_heat, cls=plotly.utils.PlotlyJSONEncoder)

        graph_map_json = None
        if "latitude" in df_plot.columns and "longitude" in df_plot.columns:
            df_map = df_plot.dropna(subset=["latitude", "longitude"]).copy()
            MAX_PONTOS = 2000
            if len(df_map) > MAX_PONTOS:
                df_map = df_map.sample(MAX_PONTOS, random_state=42)

            if len(df_map) > 0:
                fig_map = px.scatter_mapbox(
                    df_map,
                    lat="latitude",
                    lon="longitude",
                    color=color_col,
                    zoom=9,
                    height=550,
                    title=f"Properties map (sample up to {MAX_PONTOS} properties)",
                    labels={"price_range_en": "Price range"},
                )
                fig_map.update_layout(
                    mapbox_style="open-street-map",
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                fig_map.update_traces(marker=dict(size=6, opacity=0.6))
                graph_map_json = json.dumps(fig_map, cls=plotly.utils.PlotlyJSONEncoder)

        graph_box_bairro_json = None
        graph_bar_bairro_json = None
        bairro_col = "bairro" if "bairro" in df_plot.columns else ("Neighborhood" if "Neighborhood" in df_plot.columns else None)

        if bairro_col and "preco" in df_plot.columns:
            top = (
                df_plot[[bairro_col, "preco"]]
                .dropna()
                .groupby(bairro_col)["preco"]
                .mean()
                .sort_values(ascending=False)
                .head(20)
                .index
                .tolist()
            )
            df_bairro = df_plot[df_plot[bairro_col].isin(top)].copy()

            if len(df_bairro) > 0:
                fig_box_bairro = px.box(
                    df_bairro,
                    x=bairro_col,
                    y="preco",
                    title="Price distribution by neighborhood (Top 20)",
                    labels={bairro_col: "Neighborhood", "preco": "Price (€)"},
                )
                fig_box_bairro.update_layout(xaxis_tickangle=-45)
                graph_box_bairro_json = json.dumps(fig_box_bairro, cls=plotly.utils.PlotlyJSONEncoder)

                fig_bar_bairro = px.bar(
                    df_bairro.groupby(bairro_col, as_index=False)["preco"].mean().sort_values("preco", ascending=False),
                    x=bairro_col,
                    y="preco",
                    title="Average price by neighborhood (Top 20)",
                    labels={bairro_col: "Neighborhood", "preco": "Average price (€)"},
                )
                fig_bar_bairro.update_layout(xaxis_tickangle=-45)
                graph_bar_bairro_json = json.dumps(fig_bar_bairro, cls=plotly.utils.PlotlyJSONEncoder)

        interpretacao_normalidade = None
        if stats_dict.get("p_valor_shapiro") is not None:
            alpha = 0.05
            if stats_dict["p_valor_shapiro"] < alpha:
                interpretacao_normalidade = (
                    "p < 0.05 ⇒ reject the null hypothesis of normality "
                    "(the distribution is not approximately normal)."
                )
            else:
                interpretacao_normalidade = (
                    "p ≥ 0.05 ⇒ fail to reject the null hypothesis of normality "
                    "(the distribution can be considered approximately normal)."
                )

        return safe_render(
            "ames.html",
            variavel_selecionada=var,
            variaveis=numeric_cols,
            faixa_selecionada=faixa_selecionada,
            faixas=faixas_unicas,
            estatisticas=stats_dict,
            testes_extra=testes_extra,
            interpretacao_normalidade=interpretacao_normalidade,
            nomes_amigaveis=nomes_amig,
            nomes_faixa=NOMES_FAIXA,
            graph_hist_json=graph_hist_json,
            graph_box_json=graph_box_json,
            graph_scatter_json=graph_scatter_json,
            graph_box_faixa_json=graph_box_faixa_json,
            graph_preco_ano_json=graph_preco_ano_json,
            graph_heatmap_json=graph_heatmap_json,
            graph_map_json=graph_map_json,
            graph_box_bairro_json=graph_box_bairro_json,
            graph_bar_bairro_json=graph_bar_bairro_json,
        )

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=app.config.get("DEBUG", False))
