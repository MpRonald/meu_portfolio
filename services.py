from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats

import plotly
import plotly.express as px


class PortfolioService:
    def __init__(self, timeout: int = 30, data_dir: Optional[str] = None):
        self.timeout = timeout
        self.data_dir = Path(data_dir) if data_dir else None

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

    # =========================
    # Walmart Sales
    # =========================
    @lru_cache(maxsize=1)
    def load_walmart_data(self) -> pd.DataFrame:
        if not self.data_dir:
            raise RuntimeError("data_dir não configurado no PortfolioService.")

        csv_path = self.data_dir / "Walmart.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Ficheiro {csv_path} não encontrado (esperado: data/Walmart.csv).")

        df = pd.read_csv(csv_path)

        # Parse date
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

        # Numerics
        for c in ["quantity_sold", "unit_price", "forecasted_demand", "actual_demand"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Boolean normalization
        if "promotion_applied" in df.columns:
            df["promotion_applied"] = df["promotion_applied"].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
            )
            df["promotion_applied"] = df["promotion_applied"].fillna(False)

        if "stockout_indicator" in df.columns:
            df["stockout_indicator"] = df["stockout_indicator"].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
            )
            df["stockout_indicator"] = df["stockout_indicator"].fillna(False)

        # Revenue
        df["revenue"] = (df.get("quantity_sold", 0) * df.get("unit_price", 0)).astype(float)

        # Clean a few strings
        for c in ["category", "store_location", "product_name", "payment_method", "customer_loyalty_level"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

        df = df.dropna(subset=["transaction_date"])  # keep rows with valid dates
        df = df.sort_values("transaction_date")

        return df

    def get_walmart_options(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        def uniq(col: str) -> List[str]:
            if col not in df.columns:
                return []
            vals = df[col].dropna().astype(str).str.strip()
            vals = vals[vals != ""].unique().tolist()
            return sorted(vals)

        return {
            "categories": uniq("category"),
            "stores": uniq("store_location"),
            "products": uniq("product_name"),
            "payment_methods": uniq("payment_method"),
            "loyalty_levels": uniq("customer_loyalty_level"),
        }

    def get_walmart_meta(self, df: pd.DataFrame, fallback_full: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if df is None or df.empty:
            return fallback_full or {
                "n_rows": 0,
                "date_min": "—",
                "date_max": "—",
                "n_stores": 0,
                "n_categories": 0,
            }

        date_min = df["transaction_date"].min()
        date_max = df["transaction_date"].max()

        return {
            "n_rows": int(len(df)),
            "date_min": date_min.strftime("%Y-%m-%d") if pd.notna(date_min) else "—",
            "date_max": date_max.strftime("%Y-%m-%d") if pd.notna(date_max) else "—",
            "n_stores": int(df["store_location"].nunique()) if "store_location" in df.columns else 0,
            "n_categories": int(df["category"].nunique()) if "category" in df.columns else 0,
        }

    def compute_sales_kpis(self, df: pd.DataFrame) -> Dict[str, str]:
        if df is None or df.empty:
            return {
                "revenue": "0.00",
                "transactions": "0",
                "units": "0",
                "aov": "0.00",
                "promo_share": "0.0%",
                "stockout_rate": "0.0%",
                "mape": "—",
                "avg_unit_price": "0.00",
            }

        revenue = float(df["revenue"].sum()) if "revenue" in df.columns else 0.0
        transactions = int(len(df))
        units = float(df["quantity_sold"].sum()) if "quantity_sold" in df.columns else 0.0
        aov = revenue / transactions if transactions > 0 else 0.0

        promo_share = None
        if "promotion_applied" in df.columns and transactions > 0:
            promo_share = float(df["promotion_applied"].mean()) * 100.0
        promo_share_str = f"{promo_share:.1f}%" if promo_share is not None else "—"

        stockout_rate = None
        if "stockout_indicator" in df.columns and transactions > 0:
            stockout_rate = float(df["stockout_indicator"].mean()) * 100.0
        stockout_rate_str = f"{stockout_rate:.1f}%" if stockout_rate is not None else "—"

        # MAPE (forecasted_demand vs actual_demand)
        mape_str = "—"
        if "forecasted_demand" in df.columns and "actual_demand" in df.columns:
            sub = df[["forecasted_demand", "actual_demand"]].dropna()
            if len(sub) > 0:
                denom = sub["actual_demand"].abs().replace(0, np.nan)
                mape = (sub["forecasted_demand"] - sub["actual_demand"]).abs().div(denom).mean() * 100.0
                if pd.notna(mape):
                    mape_str = f"{float(mape):.1f}%"

        avg_unit_price = float(df["unit_price"].mean()) if "unit_price" in df.columns else 0.0

        return {
            "revenue": f"{revenue:,.2f}",
            "transactions": f"{transactions:,}",
            "units": f"{units:,.0f}",
            "aov": f"{aov:,.2f}",
            "promo_share": promo_share_str,
            "stockout_rate": stockout_rate_str,
            "mape": mape_str,
            "avg_unit_price": f"{avg_unit_price:,.2f}",
        }

    def _to_plotly_json(self, fig) -> str:
        return plotly.io.to_json(fig, validate=False)

    def build_sales_charts(self, df: pd.DataFrame, df_full: pd.DataFrame) -> Dict[str, str]:
        # Ensure even if empty
        if df is None or df.empty:
            # Create minimal empty charts
            empty = px.scatter(pd.DataFrame({"x": [], "y": []}), x="x", y="y", title="No data")
            js = self._to_plotly_json(empty)
            return {
                "rev_trend_json": js,
                "rev_category_json": js,
                "top_products_json": js,
                "top_stores_json": js,
                "payment_mix_json": js,
                "forecast_scatter_json": js,
            }

        d = df.copy()

        # Revenue trend (monthly)
        d["month"] = d["transaction_date"].dt.to_period("M").dt.to_timestamp()
        rev_m = d.groupby("month", as_index=False)["revenue"].sum()

        fig_trend = px.line(
            rev_m, x="month", y="revenue",
            title="Revenue trend",
            labels={"month": "Month", "revenue": "Revenue"}
        )
        fig_trend.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)

        # Revenue by category
        if "category" in d.columns:
            rev_cat = d.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
            fig_cat = px.bar(
                rev_cat.head(12),
                x="category", y="revenue",
                title="Revenue by category",
                labels={"category": "Category", "revenue": "Revenue"}
            )
            fig_cat.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
        else:
            fig_cat = px.bar(pd.DataFrame({"category": [], "revenue": []}), x="category", y="revenue", title="Revenue by category")

        # Top products
        if "product_name" in d.columns:
            top_p = d.groupby("product_name", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False).head(12)
            fig_top_p = px.bar(
                top_p,
                x="revenue", y="product_name",
                orientation="h",
                title="Top products (revenue)",
                labels={"product_name": "Product", "revenue": "Revenue"}
            )
            fig_top_p.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
        else:
            fig_top_p = px.bar(pd.DataFrame({"product_name": [], "revenue": []}), x="revenue", y="product_name", title="Top products")

        # Top stores
        if "store_location" in d.columns:
            top_s = d.groupby("store_location", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False).head(10)
            fig_top_s = px.bar(
                top_s,
                x="store_location", y="revenue",
                title="Top stores (revenue)",
                labels={"store_location": "Store", "revenue": "Revenue"}
            )
            fig_top_s.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
        else:
            fig_top_s = px.bar(pd.DataFrame({"store_location": [], "revenue": []}), x="store_location", y="revenue", title="Top stores")

        # Payment mix
        if "payment_method" in d.columns:
            pay = d.groupby("payment_method", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
            fig_pay = px.pie(pay, names="payment_method", values="revenue", title="Payment mix (revenue)")
            fig_pay.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
        else:
            fig_pay = px.pie(pd.DataFrame({"payment_method": [], "revenue": []}), names="payment_method", values="revenue", title="Payment mix")

        # Forecast vs actual
        if "forecasted_demand" in d.columns and "actual_demand" in d.columns:
            sub = d[["forecasted_demand", "actual_demand", "store_location"]].dropna()
            sub = sub.head(4000)  # limit
            fig_fc = px.scatter(
                sub,
                x="forecasted_demand",
                y="actual_demand",
                title="Forecast vs actual demand",
                labels={"forecasted_demand": "Forecasted", "actual_demand": "Actual"},
            )
            fig_fc.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
        else:
            fig_fc = px.scatter(pd.DataFrame({"forecasted_demand": [], "actual_demand": []}), x="forecasted_demand", y="actual_demand", title="Forecast vs actual")

        return {
            "rev_trend_json": self._to_plotly_json(fig_trend),
            "rev_category_json": self._to_plotly_json(fig_cat),
            "top_products_json": self._to_plotly_json(fig_top_p),
            "top_stores_json": self._to_plotly_json(fig_top_s),
            "payment_mix_json": self._to_plotly_json(fig_pay),
            "forecast_scatter_json": self._to_plotly_json(fig_fc),
        }

    def build_store_comparison(self, df_a: pd.DataFrame, df_b: pd.DataFrame, store_a: str, store_b: str) -> Dict[str, Any]:
        kpi_a = self.compute_sales_kpis(df_a)
        kpi_b = self.compute_sales_kpis(df_b)

        # Category compare
        def cat_rev(d: pd.DataFrame, label: str) -> pd.DataFrame:
            if d.empty or "category" not in d.columns:
                return pd.DataFrame({"category": [], "revenue": [], "store": []})
            tmp = d.groupby("category", as_index=False)["revenue"].sum()
            tmp["store"] = label
            return tmp

        ca = cat_rev(df_a, store_a)
        cb = cat_rev(df_b, store_b)
        cat = pd.concat([ca, cb], ignore_index=True)
        if not cat.empty:
            cat = cat.sort_values("revenue", ascending=False)
        fig_cat_cmp = px.bar(
            cat,
            x="category",
            y="revenue",
            color="store",
            barmode="group",
            title="Revenue by category (Store A vs Store B)",
            labels={"category": "Category", "revenue": "Revenue", "store": "Store"}
        )
        fig_cat_cmp.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)

        # Trend compare
        def monthly(d: pd.DataFrame, label: str) -> pd.DataFrame:
            if d.empty:
                return pd.DataFrame({"month": [], "revenue": [], "store": []})
            x = d.copy()
            x["month"] = x["transaction_date"].dt.to_period("M").dt.to_timestamp()
            tmp = x.groupby("month", as_index=False)["revenue"].sum()
            tmp["store"] = label
            return tmp

        ta = monthly(df_a, store_a)
        tb = monthly(df_b, store_b)
        t = pd.concat([ta, tb], ignore_index=True)
        fig_trend_cmp = px.line(
            t,
            x="month",
            y="revenue",
            color="store",
            title="Revenue trend (Store A vs Store B)",
            labels={"month": "Month", "revenue": "Revenue", "store": "Store"}
        )
        fig_trend_cmp.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)

        return {
            "store_a": store_a,
            "store_b": store_b,
            "kpi_a": kpi_a,
            "kpi_b": kpi_b,
            "cat_compare_json": plotly.io.to_json(fig_cat_cmp, validate=False),
            "trend_compare_json": plotly.io.to_json(fig_trend_cmp, validate=False),
        }
