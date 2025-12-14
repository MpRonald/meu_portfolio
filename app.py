import os
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
from jinja2 import TemplateNotFound

from io import StringIO
import pandas as pd
from flask import request, make_response
from services import PortfolioService



BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))).resolve()
ARTIF_DIR = Path(os.getenv("ARTIF_DIR", str(BASE_DIR / "artifacts"))).resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIF_DIR.mkdir(parents=True, exist_ok=True)


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
        # Isto vai funcionar porque tens templates/index.html e base.html
        return safe_render("index.html")

    # ---------- STUBS para não rebentar o index.html ----------
    # (depois substituímos pela lógica real, aos poucos)
    @app.route("/ecom")
    def ecom_dashboard():
        return safe_render("ecom.html", erro="(Em construção) A migrar para o novo app.", kpis=None)

    @app.route("/ecom/rfm")
    def ecom_rfm():
        return safe_render("ecom_rfm.html", erro="(Em construção) A migrar para o novo app.", rfm_table=None)

    @app.route("/ecom/forecast")
    def ecom_forecast():
        return safe_render("ecom_forecast.html", erro="(Em construção) A migrar para o novo app.", kpis=None)

    @app.route("/ecom/clusters")
    def ecom_clusters():
        return safe_render("ecom_clusters.html", erro="(Em construção) A migrar para o novo app.", cluster_summary=None)

    return app

service = PortfolioService(timeout=30)
WEATHER_DEFAULT_CITY = "Lisboa"


app = create_app()

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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=app.config.get("DEBUG", False))
