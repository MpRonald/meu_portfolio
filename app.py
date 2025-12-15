import os
import json
import logging
from datetime import datetime
from pathlib import Path
from io import StringIO

import pandas as pd
import plotly
import plotly.express as px

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
    "preco": "Preço do Imóvel (€)",
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
    "preco_m2": "Preço por m² (€)",
}

NOMES_FAIXA = {
    "baixo": "Preço Baixo",
    "medio": "Preço Médio",
    "alto": "Preço Alto",
    "muito_alto": "Preço Muito Alto",
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

    @app.route("/fx")
    def fx():
        return safe_render("fx.html", erro="(Em construção) A migrar para o novo app.")

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

            # ⚠️ teu weather.html usa chart.prob, mas a API não devolve probabilidade -> não quebrar JS
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

        # variáveis numéricas preferidas
        numeric_cols = [c for c in COLUNAS_PROJETO if c in df.columns and c != "faixa_preco"]

        # fallback: quaisquer numéricas do CSV
        if not numeric_cols:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            return jsonify({"error": "Sem colunas numéricas disponíveis no dataset Ames."}), 500

        # garantir nomes amigáveis para todas as variáveis listadas
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

        # --- gráficos base (sempre) ---
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

        # --- extras (opcionais) ---
        graph_scatter_json = None
        if "preco" in df_filtrado.columns and var in df_filtrado.columns and var != "preco":
            fig_scatter = px.scatter(
                df_filtrado,
                x=var,
                y="preco",
                color="faixa_preco" if "faixa_preco" in df_filtrado.columns else None,
                title=f"Preço vs {label}",
                labels={var: label, "preco": "Preço (€)"},
            )
            graph_scatter_json = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)

        graph_box_faixa_json = None
        if "faixa_preco" in df_filtrado.columns:
            fig_box_faixa = px.box(
                df_filtrado,
                x="faixa_preco",
                y=var,
                title=f"{label} por faixa de preço",
                labels={"faixa_preco": "Faixa de preço", var: label},
            )
            graph_box_faixa_json = json.dumps(fig_box_faixa, cls=plotly.utils.PlotlyJSONEncoder)

        graph_preco_ano_json = None
        if "preco" in df_filtrado.columns and "ano_construcao" in df_filtrado.columns:
            fig_preco_ano = px.scatter(
                df_filtrado,
                x="ano_construcao",
                y="preco",
                color="faixa_preco" if "faixa_preco" in df_filtrado.columns else None,
                title="Preço vs Ano de Construção",
                labels={"ano_construcao": "Ano de Construção", "preco": "Preço (€)"},
            )
            graph_preco_ano_json = json.dumps(fig_preco_ano, cls=plotly.utils.PlotlyJSONEncoder)

        graph_heatmap_json = None
        corr_cols = [c for c in ["preco", "preco_m2", "area_habitavel", "area_total", "quartos", "banheiros"] if c in df_filtrado.columns]
        if len(corr_cols) >= 2:
            corr = df_filtrado[corr_cols].corr(numeric_only=True)
            fig_heat = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Matriz de correlação (Pearson)",
            )
            graph_heatmap_json = json.dumps(fig_heat, cls=plotly.utils.PlotlyJSONEncoder)

        graph_map_json = None
        if "latitude" in df_filtrado.columns and "longitude" in df_filtrado.columns:
            df_map = df_filtrado.dropna(subset=["latitude", "longitude"]).copy()

            # Performance: limitar pontos
            MAX_PONTOS = 2000
            if len(df_map) > MAX_PONTOS:
                df_map = df_map.sample(MAX_PONTOS, random_state=42)

            if len(df_map) > 0:
                fig_map = px.scatter_mapbox(
                    df_map,
                    lat="latitude",
                    lon="longitude",
                    color="faixa_preco" if "faixa_preco" in df_map.columns else None,
                    zoom=9,
                    height=550,
                    title=f"Mapa de imóveis (amostra até {MAX_PONTOS} imóveis)",
                )

        # 👉 OpenStreetMap (não precisa token)
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=40, b=0)
        )

        # Pontos pequenos e leves
        fig_map.update_traces(
            marker=dict(size=6, opacity=0.6)
        )

        graph_map_json = json.dumps(fig_map, cls=plotly.utils.PlotlyJSONEncoder)


        # bairro (opcional)
        graph_box_bairro_json = None
        graph_bar_bairro_json = None
        bairro_col = "bairro" if "bairro" in df_filtrado.columns else ("Neighborhood" if "Neighborhood" in df_filtrado.columns else None)

        if bairro_col and "preco" in df_filtrado.columns:
            top = (
                df_filtrado[[bairro_col, "preco"]]
                .dropna()
                .groupby(bairro_col)["preco"]
                .mean()
                .sort_values(ascending=False)
                .head(20)
                .index
                .tolist()
            )
            df_bairro = df_filtrado[df_filtrado[bairro_col].isin(top)].copy()

            if len(df_bairro) > 0:
                fig_box_bairro = px.box(
                    df_bairro,
                    x=bairro_col,
                    y="preco",
                    title="Distribuição de preço por bairro (Top 20)",
                    labels={bairro_col: "Bairro", "preco": "Preço (€)"},
                )
                fig_box_bairro.update_layout(xaxis_tickangle=-45)
                graph_box_bairro_json = json.dumps(fig_box_bairro, cls=plotly.utils.PlotlyJSONEncoder)

                fig_bar_bairro = px.bar(
                    df_bairro.groupby(bairro_col, as_index=False)["preco"].mean().sort_values("preco", ascending=False),
                    x=bairro_col,
                    y="preco",
                    title="Preço médio por bairro (Top 20)",
                    labels={bairro_col: "Bairro", "preco": "Preço médio (€)"},
                )
                fig_bar_bairro.update_layout(xaxis_tickangle=-45)
                graph_bar_bairro_json = json.dumps(fig_bar_bairro, cls=plotly.utils.PlotlyJSONEncoder)

        interpretacao_normalidade = None
        if stats_dict.get("p_valor_shapiro") is not None:
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
