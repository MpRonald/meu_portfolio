import os
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly
import plotly.express as px

from flask import Flask, render_template, jsonify, request
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


def _load_walmart_sales_csv(data_dir: Path) -> pd.DataFrame:
    """
    Procura Walmart.csv / walmart.csv dentro de data_dir.
    """
    candidates = [
        data_dir / "Walmart.csv",
        data_dir / "walmart.csv",
    ]
    csv_path = None
    for p in candidates:
        if p.exists():
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError(
            f"CSV do Walmart não encontrado em {data_dir}. "
            f"Coloca o ficheiro em: data/Walmart.csv"
        )

    df = pd.read_csv(csv_path)

    # Normaliza datas
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    # Medidas úteis
    if {"quantity_sold", "unit_price"}.issubset(df.columns):
        df["revenue"] = pd.to_numeric(df["quantity_sold"], errors="coerce") * pd.to_numeric(df["unit_price"], errors="coerce")

    # Booleans
    for bcol in ["promotion_applied", "holiday_indicator", "stockout_indicator"]:
        if bcol in df.columns:
            if df[bcol].dtype == object:
                df[bcol] = df[bcol].astype(str).str.strip().str.lower().map(
                    {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
                ).fillna(df[bcol])
    return df


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

    _configure_logging(app)

    # Service com acesso ao DATA_DIR (Ames)
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
    # SALES (WALMART)
    # =========================
    @app.route("/sales", methods=["GET"])
    def sales_dashboard():
        """
        Dashboard de BI com filtros via querystring:
          - start=YYYY-MM-DD
          - end=YYYY-MM-DD
          - category=...
          - store=...
          - promo=true/false/all
        """
        df = _load_walmart_sales_csv(DATA_DIR)

        # Filtros (GET)
        start = (request.args.get("start") or "").strip()
        end = (request.args.get("end") or "").strip()
        category = (request.args.get("category") or "").strip()
        store = (request.args.get("store") or "").strip()
        promo = (request.args.get("promo") or "all").strip().lower()

        df_f = df.copy()

        # date range
        if "transaction_date" in df_f.columns:
            if start:
                dt_start = pd.to_datetime(start, errors="coerce")
                if pd.notna(dt_start):
                    df_f = df_f[df_f["transaction_date"] >= dt_start]
            if end:
                dt_end = pd.to_datetime(end, errors="coerce")
                if pd.notna(dt_end):
                    # inclui o dia inteiro
                    df_f = df_f[df_f["transaction_date"] < (dt_end + pd.Timedelta(days=1))]

        # category
        if category and "category" in df_f.columns:
            df_f = df_f[df_f["category"].astype(str) == category]

        # store
        if store and "store_location" in df_f.columns:
            df_f = df_f[df_f["store_location"].astype(str) == store]

        # promo
        if promo in {"true", "false"} and "promotion_applied" in df_f.columns:
            df_f = df_f[df_f["promotion_applied"] == (promo == "true")]

        # Listas para dropdowns (sem depender do filtro atual)
        categories = sorted(df["category"].dropna().astype(str).unique().tolist()) if "category" in df.columns else []
        stores = sorted(df["store_location"].dropna().astype(str).unique().tolist()) if "store_location" in df.columns else []

        # KPIs
        total_revenue = float(df_f["revenue"].sum()) if "revenue" in df_f.columns else 0.0
        total_units = int(pd.to_numeric(df_f.get("quantity_sold"), errors="coerce").fillna(0).sum()) if "quantity_sold" in df_f.columns else 0
        orders = int(df_f["transaction_id"].nunique()) if "transaction_id" in df_f.columns else int(len(df_f))
        customers = int(df_f["customer_id"].nunique()) if "customer_id" in df_f.columns else 0
        aov = float(total_revenue / orders) if orders > 0 else 0.0

        promo_share = None
        if "promotion_applied" in df_f.columns and len(df_f) > 0:
            promo_share = float(df_f["promotion_applied"].mean() * 100.0)

        stockout_rate = None
        if "stockout_indicator" in df_f.columns and len(df_f) > 0:
            stockout_rate = float(df_f["stockout_indicator"].mean() * 100.0)

        # ================
        # Charts (Plotly)
        # ================
        graph_rev_ts_json = None
        if "transaction_date" in df_f.columns and "revenue" in df_f.columns:
            ts = (
                df_f.dropna(subset=["transaction_date"])
                .assign(day=lambda x: x["transaction_date"].dt.date)
                .groupby("day", as_index=False)["revenue"].sum()
                .sort_values("day")
            )
            if len(ts) > 0:
                fig = px.line(ts, x="day", y="revenue", title="Revenue over time (daily)")
                fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
                graph_rev_ts_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        graph_cat_json = None
        if "category" in df_f.columns and "revenue" in df_f.columns:
            cat = (
                df_f.groupby("category", as_index=False)["revenue"].sum()
                .sort_values("revenue", ascending=False)
            )
            if len(cat) > 0:
                fig = px.bar(cat, x="category", y="revenue", title="Revenue by category")
                fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=-20)
                graph_cat_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        graph_store_json = None
        if "store_location" in df_f.columns and "revenue" in df_f.columns:
            st = (
                df_f.groupby("store_location", as_index=False)["revenue"].sum()
                .sort_values("revenue", ascending=False)
                .head(15)
            )
            if len(st) > 0:
                fig = px.bar(st, x="store_location", y="revenue", title="Top stores by revenue (Top 15)")
                fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=-25)
                graph_store_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        graph_pay_json = None
        if "payment_method" in df_f.columns and "revenue" in df_f.columns:
            pm = (
                df_f.groupby("payment_method", as_index=False)["revenue"].sum()
                .sort_values("revenue", ascending=False)
            )
            if len(pm) > 0:
                fig = px.pie(pm, names="payment_method", values="revenue", title="Revenue share by payment method")
                fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
                graph_pay_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        graph_demand_json = None
        if {"forecasted_demand", "actual_demand"}.issubset(df_f.columns):
            dd = df_f[["forecasted_demand", "actual_demand"]].dropna()
            if len(dd) > 0:
                fig = px.scatter(
                    dd,
                    x="forecasted_demand",
                    y="actual_demand",
                    title="Forecasted vs Actual demand",
                    trendline="ols" if len(dd) >= 30 else None,
                )
                fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
                graph_demand_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Top products table
        top_products = []
        if {"product_name", "revenue"}.issubset(df_f.columns):
            tp = (
                df_f.groupby("product_name", as_index=False)["revenue"].sum()
                .sort_values("revenue", ascending=False)
                .head(10)
            )
            top_products = tp.to_dict(orient="records")

        # Preview table (últimas 15 transações)
        preview_rows = []
        preview_cols = []
        if len(df_f) > 0:
            cols = [c for c in ["transaction_date", "store_location", "category", "product_name", "quantity_sold", "unit_price", "revenue", "promotion_applied"] if c in df_f.columns]
            preview_cols = cols
            prev = df_f.sort_values("transaction_date" if "transaction_date" in df_f.columns else df_f.index, ascending=False).head(15)
            prev = prev[cols].copy() if cols else prev.head(15)
            # serialização simples
            if "transaction_date" in prev.columns:
                prev["transaction_date"] = prev["transaction_date"].dt.strftime("%Y-%m-%d %H:%M")
            preview_rows = prev.to_dict(orient="records")

        ctx = dict(
            # filtros
            start=start,
            end=end,
            category=category,
            store=store,
            promo=promo,
            categories=categories,
            stores=stores,

            # kpis
            total_revenue=total_revenue,
            total_units=total_units,
            orders=orders,
            customers=customers,
            aov=aov,
            promo_share=promo_share,
            stockout_rate=stockout_rate,

            # charts
            graph_rev_ts_json=graph_rev_ts_json,
            graph_cat_json=graph_cat_json,
            graph_store_json=graph_store_json,
            graph_pay_json=graph_pay_json,
            graph_demand_json=graph_demand_json,

            # tables
            top_products=top_products,
            preview_cols=preview_cols,
            preview_rows=preview_rows,
        )

        return safe_render("sales.html", **ctx)

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
