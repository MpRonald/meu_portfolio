import os
import json
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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


def _empty_fig(title: str = "") -> dict:
    fig = px.scatter(pd.DataFrame({"x": [], "y": []}), x="x", y="y", title=title)
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text="No data for the selected filters",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False
        )],
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _load_walmart_csv(data_dir: Path) -> pd.DataFrame:
    candidates = [data_dir / "Walmart.csv", data_dir / "walmart.csv"]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"CSV do Walmart não encontrado em {data_dir}. "
            f"Coloca o ficheiro em: data/Walmart.csv"
        )

    df = pd.read_csv(csv_path)

    # Tipos / normalização básica
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    for c in ["quantity_sold", "unit_price", "forecasted_demand", "actual_demand"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Revenue: tenta inferir de quantity * unit_price
    if "revenue" not in df.columns and {"quantity_sold", "unit_price"}.issubset(df.columns):
        df["revenue"] = df["quantity_sold"] * df["unit_price"]
    if "revenue" in df.columns:
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    # Booleans
    bool_map = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    for bcol in ["promotion_applied", "holiday_indicator", "stockout_indicator"]:
        if bcol in df.columns and df[bcol].dtype == object:
            df[bcol] = (
                df[bcol].astype(str).str.strip().str.lower()
                .map(bool_map)
                .where(df[bcol].notna(), df[bcol])
            )

    # Garante strings
    for s in ["category", "store_location", "payment_method", "loyalty_level", "product_name"]:
        if s in df.columns:
            df[s] = df[s].astype(str)

    return df


def _fmt_money(x: float) -> str:
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)


def _fmt_int(x: float) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def _fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.1f}%"


def create_app() -> Flask:
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

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

    # ======================================================
    # SALES DASHBOARD (WALMART)  ✅ compatível com seu sales.html
    # ======================================================
    @app.route("/sales", methods=["GET"])
    def sales_dashboard():
        warning = None
        try:
            df_all = _load_walmart_csv(DATA_DIR)
        except Exception as e:
            warning = str(e)
            df_all = pd.DataFrame()

        # -------- Options (dropdowns) --------
        options = {
            "categories": sorted(df_all["category"].dropna().astype(str).unique().tolist()) if "category" in df_all.columns else [],
            "stores": sorted(df_all["store_location"].dropna().astype(str).unique().tolist()) if "store_location" in df_all.columns else [],
            "loyalty_levels": sorted(df_all["loyalty_level"].dropna().astype(str).unique().tolist()) if "loyalty_level" in df_all.columns else [],
            "payment_methods": sorted(df_all["payment_method"].dropna().astype(str).unique().tolist()) if "payment_method" in df_all.columns else [],
        }

        # -------- Filters (GET params) --------
        date_start = (request.args.get("date_start") or "").strip()
        date_end = (request.args.get("date_end") or "").strip()
        category = (request.args.get("category") or "All").strip()
        store_location = (request.args.get("store_location") or "All").strip()
        promotion_applied = (request.args.get("promotion_applied") or "All").strip()
        loyalty = (request.args.get("loyalty") or "All").strip()
        payment_method = (request.args.get("payment_method") or "All").strip()

        filters = {
            "date_start": date_start or None,
            "date_end": date_end or None,
            "category": category if category != "" else "All",
            "store_location": store_location if store_location != "" else "All",
            "promotion_applied": promotion_applied if promotion_applied != "" else "All",
            "loyalty": loyalty if loyalty != "" else "All",
            "payment_method": payment_method if payment_method != "" else "All",
        }

        df = df_all.copy()

        # -------- Apply filters --------
        if "transaction_date" in df.columns:
            if date_start:
                dt = pd.to_datetime(date_start, errors="coerce")
                if pd.notna(dt):
                    df = df[df["transaction_date"] >= dt]
            if date_end:
                dt = pd.to_datetime(date_end, errors="coerce")
                if pd.notna(dt):
                    df = df[df["transaction_date"] < (dt + pd.Timedelta(days=1))]

        if category != "All" and "category" in df.columns:
            df = df[df["category"].astype(str) == category]

        if store_location != "All" and "store_location" in df.columns:
            df = df[df["store_location"].astype(str) == store_location]

        if promotion_applied in {"true", "false"} and "promotion_applied" in df.columns:
            df = df[df["promotion_applied"] == (promotion_applied == "true")]

        if loyalty != "All" and "loyalty_level" in df.columns:
            df = df[df["loyalty_level"].astype(str) == loyalty]

        if payment_method != "All" and "payment_method" in df.columns:
            df = df[df["payment_method"].astype(str) == payment_method]

        if df.empty:
            warning = warning or "No rows for the selected filters."

        # -------- Meta (top chips) --------
        date_min = None
        date_max = None
        if "transaction_date" in df_all.columns and df_all["transaction_date"].notna().any():
            date_min = df_all["transaction_date"].min()
            date_max = df_all["transaction_date"].max()

        meta = {
            "n_rows": int(len(df_all)) if not df_all.empty else 0,
            "date_min": date_min.strftime("%Y-%m-%d") if isinstance(date_min, pd.Timestamp) else "—",
            "date_max": date_max.strftime("%Y-%m-%d") if isinstance(date_max, pd.Timestamp) else "—",
            "n_stores": int(df_all["store_location"].nunique()) if "store_location" in df_all.columns else 0,
            "n_categories": int(df_all["category"].nunique()) if "category" in df_all.columns else 0,
        }

        # -------- KPIs --------
        revenue = float(df["revenue"].sum()) if "revenue" in df.columns else 0.0
        transactions = int(df["transaction_id"].nunique()) if "transaction_id" in df.columns else int(len(df))
        units = float(df["quantity_sold"].sum()) if "quantity_sold" in df.columns else 0.0
        aov = (revenue / transactions) if transactions > 0 else 0.0

        promo_share = None
        if "promotion_applied" in df.columns and len(df) > 0:
            promo_share = float(df["promotion_applied"].mean() * 100.0)

        stockout_rate = None
        if "stockout_indicator" in df.columns and len(df) > 0:
            stockout_rate = float(df["stockout_indicator"].mean() * 100.0)

        avg_unit_price = float(df["unit_price"].mean()) if "unit_price" in df.columns and len(df) > 0 else None

        # MAPE (forecast vs actual demand)
        mape_val = None
        if {"forecasted_demand", "actual_demand"}.issubset(df.columns):
            dd = df[["forecasted_demand", "actual_demand"]].dropna()
            if len(dd) > 0:
                denom = dd["actual_demand"].abs().clip(lower=1e-8)
                mape_val = float(((dd["forecasted_demand"] - dd["actual_demand"]).abs() / denom).mean() * 100.0)

        kpis = {
            "revenue": _fmt_money(revenue),
            "transactions": _fmt_int(transactions),
            "units": _fmt_int(units),
            "aov": _fmt_money(aov),
            "promo_share": _fmt_pct(promo_share),
            "stockout_rate": _fmt_pct(stockout_rate),
            "mape": _fmt_pct(mape_val),
            "avg_unit_price": _fmt_money(avg_unit_price) if avg_unit_price is not None else "—",
        }

        # -------- Charts --------
        # Revenue trend (monthly)
        if not df.empty and "transaction_date" in df.columns and "revenue" in df.columns:
            tmp = df.dropna(subset=["transaction_date"]).copy()
            tmp["month"] = tmp["transaction_date"].dt.to_period("M").dt.to_timestamp()
            ts = tmp.groupby("month", as_index=False)["revenue"].sum().sort_values("month")
            fig_rev_trend = px.line(ts, x="month", y="revenue", title="Revenue trend (monthly)")
            fig_rev_trend.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        else:
            fig_rev_trend = _empty_fig("Revenue trend (monthly)")

        # Revenue by category
        if not df.empty and "category" in df.columns and "revenue" in df.columns:
            cat = df.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
            fig_rev_cat = px.bar(cat, x="category", y="revenue", title="Revenue by category")
            fig_rev_cat.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=-20)
        else:
            fig_rev_cat = _empty_fig("Revenue by category")

        # Top products (bar)
        if not df.empty and {"product_name", "revenue"}.issubset(df.columns):
            tp = (
                df.groupby("product_name", as_index=False)["revenue"].sum()
                .sort_values("revenue", ascending=False).head(12)
            )
            fig_top_products = px.bar(tp, x="product_name", y="revenue", title="Top products (revenue)")
            fig_top_products.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=-25)
        else:
            fig_top_products = _empty_fig("Top products (revenue)")

        # Top stores
        if not df.empty and {"store_location", "revenue"}.issubset(df.columns):
            st = (
                df.groupby("store_location", as_index=False)["revenue"].sum()
                .sort_values("revenue", ascending=False).head(15)
            )
            fig_top_stores = px.bar(st, x="store_location", y="revenue", title="Top stores by revenue")
            fig_top_stores.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=-25)
        else:
            fig_top_stores = _empty_fig("Top stores by revenue")

        # Payment mix (pie)
        if not df.empty and {"payment_method", "revenue"}.issubset(df.columns):
            pm = df.groupby("payment_method", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
            fig_payment = px.pie(pm, names="payment_method", values="revenue", title="Payment mix (revenue)")
            fig_payment.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        else:
            fig_payment = _empty_fig("Payment mix (revenue)")

        # Forecast vs actual scatter
        if not df.empty and {"forecasted_demand", "actual_demand"}.issubset(df.columns):
            dd = df[["forecasted_demand", "actual_demand"]].dropna()
            fig_fc = px.scatter(dd, x="forecasted_demand", y="actual_demand", title="Forecast vs actual demand")
            fig_fc.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        else:
            fig_fc = _empty_fig("Forecast vs actual demand")

        charts = {
            "rev_trend_json": json.dumps(fig_rev_trend, cls=plotly.utils.PlotlyJSONEncoder),
            "rev_category_json": json.dumps(fig_rev_cat, cls=plotly.utils.PlotlyJSONEncoder),
            "top_products_json": json.dumps(fig_top_products, cls=plotly.utils.PlotlyJSONEncoder),
            "top_stores_json": json.dumps(fig_top_stores, cls=plotly.utils.PlotlyJSONEncoder),
            "payment_mix_json": json.dumps(fig_payment, cls=plotly.utils.PlotlyJSONEncoder),
            "forecast_scatter_json": json.dumps(fig_fc, cls=plotly.utils.PlotlyJSONEncoder),
        }

        # -------- Table preview (table.columns + table.rows) --------
        preview_cols = [c for c in [
            "transaction_date", "store_location", "category", "product_name",
            "quantity_sold", "unit_price", "revenue", "promotion_applied",
            "loyalty_level", "payment_method", "stockout_indicator"
        ] if c in df.columns]

        df_prev = df.copy()
        if "transaction_date" in df_prev.columns:
            df_prev = df_prev.sort_values("transaction_date", ascending=False)

        df_prev = df_prev[preview_cols].head(15) if preview_cols else df_prev.head(15)

        # serializa cells (principalmente datas)
        rows = []
        if not df_prev.empty:
            for _, r in df_prev.iterrows():
                row = []
                for c in df_prev.columns:
                    v = r[c]
                    if isinstance(v, pd.Timestamp):
                        row.append(v.strftime("%Y-%m-%d %H:%M"))
                    elif pd.isna(v):
                        row.append("")
                    else:
                        row.append(v)
                rows.append(row)

        table = {
            "columns": list(df_prev.columns),
            "rows": rows,
        }

        return safe_render(
            "sales.html",
            meta=meta,
            filters=filters,
            options=options,
            kpis=kpis,
            charts=charts,
            table=table,
            warning=warning,
        )

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
            df_plot, x=var, nbins=40, marginal="box",
            title=f"Distribution of {label}", labels={var: label},
        )
        fig_box = px.box(
            df_plot, y=var, points="outliers",
            title=f"Boxplot of {label}", labels={var: label},
        )

        graph_hist_json = json.dumps(fig_hist, cls=plotly.utils.PlotlyJSONEncoder)
        graph_box_json = json.dumps(fig_box, cls=plotly.utils.PlotlyJSONEncoder)

        graph_scatter_json = None
        if "preco" in df_plot.columns and var in df_plot.columns and var != "preco":
            fig_scatter = px.scatter(
                df_plot, x=var, y="preco", color=color_col,
                title=f"Price vs {label}",
                labels={var: label, "preco": "Price (€)", "price_range_en": "Price range"},
            )
            graph_scatter_json = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)

        graph_box_faixa_json = None
        if "price_range_en" in df_plot.columns:
            fig_box_faixa = px.box(
                df_plot, x="price_range_en", y=var,
                title=f"{label} by price range",
                labels={"price_range_en": "Price range", var: label},
            )
            graph_box_faixa_json = json.dumps(fig_box_faixa, cls=plotly.utils.PlotlyJSONEncoder)

        graph_preco_ano_json = None
        if "preco" in df_plot.columns and "ano_construcao" in df_plot.columns:
            fig_preco_ano = px.scatter(
                df_plot, x="ano_construcao", y="preco", color=color_col,
                title="Price vs Year Built",
                labels={"ano_construcao": "Year Built", "preco": "Price (€)", "price_range_en": "Price range"},
            )
            graph_preco_ano_json = json.dumps(fig_preco_ano, cls=plotly.utils.PlotlyJSONEncoder)

        graph_heatmap_json = None
        corr_cols = [c for c in ["preco", "preco_m2", "area_habitavel", "area_total", "quartos", "banheiros"] if c in df_plot.columns]
        if len(corr_cols) >= 2:
            corr = df_plot[corr_cols].corr(numeric_only=True)
            fig_heat = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix (Pearson)")
            graph_heatmap_json = json.dumps(fig_heat, cls=plotly.utils.PlotlyJSONEncoder)

        graph_map_json = None
        if "latitude" in df_plot.columns and "longitude" in df_plot.columns:
            df_map = df_plot.dropna(subset=["latitude", "longitude"]).copy()
            MAX_PONTOS = 2000
            if len(df_map) > MAX_PONTOS:
                df_map = df_map.sample(MAX_PONTOS, random_state=42)
            if len(df_map) > 0:
                fig_map = px.scatter_mapbox(
                    df_map, lat="latitude", lon="longitude", color=color_col,
                    zoom=9, height=550,
                    title=f"Properties map (sample up to {MAX_PONTOS} properties)",
                    labels={"price_range_en": "Price range"},
                )
                fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))
                fig_map.update_traces(marker=dict(size=6, opacity=0.6))
                graph_map_json = json.dumps(fig_map, cls=plotly.utils.PlotlyJSONEncoder)

        graph_box_bairro_json = None
        graph_bar_bairro_json = None
        bairro_col = "bairro" if "bairro" in df_plot.columns else ("Neighborhood" if "Neighborhood" in df_plot.columns else None)
        if bairro_col and "preco" in df_plot.columns:
            top = (
                df_plot[[bairro_col, "preco"]].dropna()
                .groupby(bairro_col)["preco"].mean()
                .sort_values(ascending=False).head(20).index.tolist()
            )
            df_bairro = df_plot[df_plot[bairro_col].isin(top)].copy()
            if len(df_bairro) > 0:
                fig_box_bairro = px.box(
                    df_bairro, x=bairro_col, y="preco",
                    title="Price distribution by neighborhood (Top 20)",
                    labels={bairro_col: "Neighborhood", "preco": "Price (€)"},
                )
                fig_box_bairro.update_layout(xaxis_tickangle=-45)
                graph_box_bairro_json = json.dumps(fig_box_bairro, cls=plotly.utils.PlotlyJSONEncoder)

                fig_bar_bairro = px.bar(
                    df_bairro.groupby(bairro_col, as_index=False)["preco"].mean().sort_values("preco", ascending=False),
                    x=bairro_col, y="preco",
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
