import os
import json
import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import plotly
import plotly.express as px

from flask import Flask, render_template, jsonify, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge
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

# =========================
# Cleaner settings
# =========================
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
DEMO_FILENAME = os.getenv("CLEANER_DEMO_FILE", "dirty_demo_dataset.csv")
DEMO_PATH = DATA_DIR / DEMO_FILENAME

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
        "App a iniciar | ENV=%s | DEBUG=%s | DATA_DIR=%s | ARTIF_DIR=%s | MAX_UPLOAD_MB=%s",
        app.config.get("ENV_NAME"),
        app.config.get("DEBUG"),
        app.config.get("DATA_DIR"),
        app.config.get("ARTIF_DIR"),
        app.config.get("MAX_UPLOAD_MB"),
    )


def create_app() -> Flask:
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

    # ✅ Inject current year into all templates (base.html footer)
    @app.context_processor
    def inject_current_year():
        return {"current_year": datetime.now().year}

    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")
    env = (os.getenv("FLASK_ENV") or os.getenv("ENV") or "production").lower().strip()
    app.config["ENV_NAME"] = env
    debug_flag = (os.getenv("FLASK_DEBUG") or "").strip() == "1"
    app.config["DEBUG"] = True if env == "development" or debug_flag else False

    app.config["BASE_DIR"] = str(BASE_DIR)
    app.config["DATA_DIR"] = str(DATA_DIR)
    app.config["ARTIF_DIR"] = str(ARTIF_DIR)

    # ✅ Upload cap (server-side)
    app.config["MAX_UPLOAD_MB"] = MAX_UPLOAD_MB
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES  # Flask will raise 413

    _configure_logging(app)

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

    # ✅ Friendly 413 page for the cleaner
    @app.errorhandler(RequestEntityTooLarge)
    def handle_413(_e):
        demo = _load_demo_payload()
        return safe_render(
            "cleaner.html",
            result=None,
            error=f"File too large. Max allowed is {app.config.get('MAX_UPLOAD_MB', 5)}MB.",
            demo=demo,
            max_upload_mb=app.config.get("MAX_UPLOAD_MB", 5),
        ), 413

    def safe_render(template_name: str, **ctx):
        try:
            return render_template(template_name, **ctx)
        except TemplateNotFound:
            return jsonify({
                "template_missing": template_name,
                "hint": "Template não encontrado.",
                "context_keys": sorted(list(ctx.keys()))
            }), 200

    def _load_demo_payload():
        """
        Loads demo dataset from data/dirty_demo_dataset.csv.
        Returns payload with meta + table preview (first 20 rows).
        If missing, returns None (template handles it safely).
        """
        path = DEMO_PATH
        if not path.exists():
            return None

        try:
            # Robust delimiter sniffing
            try:
                df = pd.read_csv(path, sep=None, engine="python")
            except Exception:
                df = pd.read_csv(path, sep=",", engine="python", encoding="latin-1")

            meta = {
                "filename": path.name,
                "ext": path.suffix.lower().lstrip("."),
                "rows": int(len(df)),
                "cols": int(df.shape[1]),
            }

            preview = df.head(20).copy()
            # avoid NaN in template
            preview = preview.fillna("")

            table = {
                "columns": preview.columns.tolist(),
                "rows": preview.values.tolist(),
            }
            return {"meta": meta, "table": table}
        except Exception as e:
            app.logger.warning("Failed loading demo dataset: %s", str(e))
            return None

    @app.route("/healthz")
    def healthz():
        app.logger.info("healthz ping")
        return jsonify({
            "status": "ok",
            "time": datetime.utcnow().isoformat() + "Z",
            "env": app.config.get("ENV_NAME"),
        }), 200


    @app.route("/")
    def index():
        return safe_render("index.html")

    # =========================
    # AMES (EXECUTIVE)
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
        if var not in numeric_cols:
            var = default_var

        faixas_unicas = ["Todos"]
        if "faixa_preco" in df.columns:
            faixas_unicas += sorted(df["faixa_preco"].dropna().unique().tolist())

        faixa_selecionada = request.form.get("faixa_preco", "Todos")
        if faixa_selecionada not in faixas_unicas:
            faixa_selecionada = "Todos"

        payload = service.build_ames_dashboard_payload(
            df_full=df_completo,
            variavel=var,
            faixa_preco=faixa_selecionada,
        )

        stats_dict = payload.get("estatisticas") or {}
        testes_extra = payload.get("testes_extra") or {}

        interpretacao_normalidade = None
        p_shapiro = stats_dict.get("p_valor_shapiro")
        if p_shapiro is not None:
            alpha = 0.05
            if p_shapiro < alpha:
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
            graph_hist_json=payload.get("graph_hist_json"),
            graph_box_json=payload.get("graph_box_json"),
            graph_scatter_json=payload.get("graph_scatter_json"),
            graph_box_faixa_json=payload.get("graph_box_faixa_json"),
            graph_preco_ano_json=payload.get("graph_preco_ano_json"),
            graph_heatmap_json=payload.get("graph_heatmap_json"),
            graph_map_json=payload.get("graph_map_json"),
            graph_box_bairro_json=payload.get("graph_box_bairro_json"),
            graph_bar_bairro_json=payload.get("graph_bar_bairro_json"),
        )

    # =========================
    # SALES (Walmart)
    # =========================
    @app.route("/sales", methods=["GET"])
    def sales_dashboard():
        warning = None

        df_full = service.load_walmart_data()
        df = df_full.copy()

        filters = {
            "date_start": (request.args.get("date_start") or "").strip() or None,
            "date_end": (request.args.get("date_end") or "").strip() or None,
            "category": (request.args.get("category") or "All").strip(),
            "store_location": (request.args.get("store_location") or "All").strip(),
            "product_name": (request.args.get("product_name") or "All").strip(),
            "promotion_applied": (request.args.get("promotion_applied") or "All").strip(),
            "loyalty": (request.args.get("loyalty") or "All").strip(),
            "payment_method": (request.args.get("payment_method") or "All").strip(),
            "store_a": (request.args.get("store_a") or "None").strip(),
            "store_b": (request.args.get("store_b") or "None").strip(),
        }

        options = service.get_walmart_options(df_full)
        df_f = df.copy()

        if filters["date_start"]:
            try:
                ds = pd.to_datetime(filters["date_start"], errors="raise")
                df_f = df_f[df_f["transaction_date"] >= ds]
            except Exception:
                warning = "Invalid start date. Ignoring date_start."
        if filters["date_end"]:
            try:
                de = pd.to_datetime(filters["date_end"], errors="raise")
                df_f = df_f[df_f["transaction_date"] <= de]
            except Exception:
                warning = "Invalid end date. Ignoring date_end."

        if filters["category"] != "All":
            df_f = df_f[df_f["category"] == filters["category"]]
        if filters["store_location"] != "All":
            df_f = df_f[df_f["store_location"] == filters["store_location"]]
        if filters["product_name"] != "All":
            df_f = df_f[df_f["product_name"] == filters["product_name"]]
        if filters["promotion_applied"] in {"true", "false"}:
            want = filters["promotion_applied"] == "true"
            df_f = df_f[df_f["promotion_applied"] == want]
        if filters["loyalty"] != "All":
            df_f = df_f[df_f["customer_loyalty_level"] == filters["loyalty"]]
        if filters["payment_method"] != "All":
            df_f = df_f[df_f["payment_method"] == filters["payment_method"]]

        if df_f.empty:
            warning = warning or "No data for the selected filters. Showing empty charts."
            df_f = df_f.head(0)

        meta = service.get_walmart_meta(df_full)
        meta_selected = service.get_walmart_meta(df_f, fallback_full=meta)

        kpis = service.compute_sales_kpis(df_f)
        charts = service.build_sales_charts(df_f, df_full)

        compare = None
        store_a = filters["store_a"]
        store_b = filters["store_b"]

        if store_a != "None" and store_b != "None" and store_a != store_b:
            df_a = df_f[df_f["store_location"] == store_a].copy()
            df_b = df_f[df_f["store_location"] == store_b].copy()
            compare = service.build_store_comparison(df_a, df_b, store_a, store_b)

        preview_cols = [
            "transaction_id", "transaction_date", "store_location",
            "category", "product_name", "quantity_sold", "unit_price", "revenue",
            "promotion_applied", "stockout_indicator",
            "forecasted_demand", "actual_demand"
        ]
        preview_cols = [c for c in preview_cols if c in df_f.columns]
        table_df = df_f[preview_cols].head(12).copy()

        if "transaction_date" in table_df.columns:
            table_df["transaction_date"] = table_df["transaction_date"].dt.strftime("%Y-%m-%d")
        if "revenue" in table_df.columns:
            table_df["revenue"] = table_df["revenue"].map(lambda x: f"{x:,.2f}")

        table = {
            "columns": table_df.columns.tolist(),
            "rows": table_df.values.tolist(),
        }

        return safe_render(
            "sales.html",
            meta=meta_selected,
            filters=filters,
            options=options,
            warning=warning,
            kpis=kpis,
            charts=charts,
            table=table,
            compare=compare,
        )

    # =========================
    # DATA CLEANER (CSV/Excel → Clean + Report → Excel/PDF)
    # =========================
    @app.route("/cleaner", methods=["GET", "POST"])
    def data_cleaner():
        result = None
        error = None
        demo = _load_demo_payload()

        if request.method == "POST":
            # Manual check (in addition to MAX_CONTENT_LENGTH)
            if request.content_length and request.content_length > MAX_UPLOAD_BYTES:
                error = f"File too large. Max allowed is {MAX_UPLOAD_MB}MB."
                return safe_render(
                    "cleaner.html",
                    result=None,
                    error=error,
                    demo=demo,
                    max_upload_mb=MAX_UPLOAD_MB,
                )

            f = request.files.get("file")
            if not f or not f.filename:
                error = "Please select a CSV or Excel file."
            else:
                try:
                    file_id = uuid4().hex[:12]
                    payload = service.clean_uploaded_file(
                        file_storage=f,
                        artifacts_dir=Path(app.config["ARTIF_DIR"]),
                        file_id=file_id,
                    )
                    result = payload
                except Exception as e:
                    error = str(e)

        return safe_render(
            "cleaner.html",
            result=result,
            error=error,
            demo=demo,
            max_upload_mb=MAX_UPLOAD_MB,
        )

    @app.route("/cleaner/demo-download", methods=["GET"])
    def cleaner_demo_download():
        if not DEMO_PATH.exists():
            return jsonify({"error": "demo_not_found", "hint": f"Missing {DEMO_PATH.name} in data/"}), 404

        # Let Flask infer a safe mimetype
        return send_file(
            DEMO_PATH,
            as_attachment=True,
            download_name=DEMO_PATH.name,
            mimetype="text/csv",
        )

    @app.route("/cleaner/download/<file_id>/<kind>", methods=["GET"])
    def cleaner_download(file_id: str, kind: str):
        artifacts_dir = Path(app.config["ARTIF_DIR"])
        if kind == "excel":
            path = artifacts_dir / f"cleaned_{file_id}.xlsx"
            mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            download_name = f"cleaned_{file_id}.xlsx"
        elif kind == "pdf":
            path = artifacts_dir / f"report_{file_id}.pdf"
            mimetype = "application/pdf"
            download_name = f"report_{file_id}.pdf"
        else:
            return jsonify({"error": "invalid_kind"}), 400

        if not path.exists():
            return jsonify({"error": "file_not_found"}), 404

        return send_file(
            path,
            mimetype=mimetype,
            as_attachment=True,
            download_name=download_name
        )

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=app.config.get("DEBUG", False))
