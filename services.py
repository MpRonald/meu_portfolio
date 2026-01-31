from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import io
import re
import csv
import numpy as np
import pandas as pd
from scipy import stats

import plotly
import plotly.express as px
import plotly.graph_objects as go

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


class PortfolioService:
    def __init__(self, timeout: int = 30, data_dir: Optional[str] = None):
        self.timeout = timeout
        self.data_dir = Path(data_dir) if data_dir else None

    # ==========================================================
    # Helpers (format / safety)
    # ==========================================================
    def _fmt_money(self, v: float, currency: str = "$") -> str:
        """Executive-friendly money formatting: 1,234 -> $1,234 | 1.2M -> $1.2M"""
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return f"{currency}0"
            v = float(v)
        except Exception:
            return f"{currency}0"

        sign = "-" if v < 0 else ""
        v_abs = abs(v)

        if v_abs >= 1_000_000_000:
            return f"{sign}{currency}{v_abs/1_000_000_000:.1f}B"
        if v_abs >= 1_000_000:
            return f"{sign}{currency}{v_abs/1_000_000:.1f}M"
        if v_abs >= 1_000:
            return f"{sign}{currency}{v_abs:,.0f}"
        return f"{sign}{currency}{v_abs:.2f}"

    def _fmt_int(self, v: float) -> str:
        try:
            return f"{int(round(float(v))):,}"
        except Exception:
            return "0"

    def _fmt_pct(self, v: float, digits: int = 1) -> str:
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            return f"{float(v):.{digits}f}%"
        except Exception:
            return "—"

    def _safe_to_datetime(self, s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, errors="coerce")

    def _to_plotly_json(self, fig) -> str:
        return plotly.io.to_json(fig, validate=False)

    def _pick_first_existing_col(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _empty_fig_json(self, title: str = "No data") -> str:
        fig = px.scatter(pd.DataFrame({"x": [], "y": []}), x="x", y="y", title=title)
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=420)
        return self._to_plotly_json(fig)

    # ==========================================================
    # Ames (dataset + stats)
    # ==========================================================
    @lru_cache(maxsize=1)
    def load_ames_data(self) -> pd.DataFrame:
        if not self.data_dir:
            raise RuntimeError("data_dir não configurado no PortfolioService.")

        csv_path = self.data_dir / "ames.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Ficheiro {csv_path} não encontrado.")

        df = pd.read_csv(csv_path)

        # attempt numeric conversion for object columns (except known categoricals)
        for c in df.columns:
            if c in {"faixa_preco", "bairro", "Neighborhood"}:
                continue
            if df[c].dtype == object:
                converted = pd.to_numeric(df[c], errors="coerce")
                if converted.notna().sum() > 0:
                    df[c] = converted

        # normalize common important numeric cols if present
        for c in ["preco", "preco_m2", "YearBuilt", "Year_Built", "year_built", "latitude", "lat", "longitude", "lon"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

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

        # Pearson with targets
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

        # Spearman with price
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

        # Kruskal–Wallis by faixa_preco
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

        # Simple linear regression: preco ~ var
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

    def build_ames_dashboard_payload(
        self,
        df_full: pd.DataFrame,
        variavel: str,
        faixa_preco: str = "Todos",
        max_heatmap_cols: int = 14,
    ) -> Dict[str, Any]:
        if df_full is None or df_full.empty:
            return {
                "df_filtrado": pd.DataFrame(),
                "estatisticas": self.calcular_estatisticas_1d(pd.Series(dtype=float)),
                "testes_extra": {},
                "graph_hist_json": self._empty_fig_json("No data"),
                "graph_box_json": self._empty_fig_json("No data"),
                "graph_scatter_json": None,
                "graph_box_faixa_json": None,
                "graph_preco_ano_json": None,
                "graph_box_bairro_json": None,
                "graph_bar_bairro_json": None,
                "graph_heatmap_json": None,
                "graph_map_json": None,
            }

        df = df_full.copy()

        if faixa_preco and faixa_preco != "Todos" and "faixa_preco" in df.columns:
            df = df[df["faixa_preco"].astype(str) == str(faixa_preco)].copy()

        if variavel not in df.columns:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            variavel = num_cols[0] if num_cols else variavel

        serie = pd.to_numeric(df.get(variavel, pd.Series(dtype=float)), errors="coerce")

        estatisticas = self.calcular_estatisticas_1d(serie)
        testes_extra = self.calcular_testes_adicionais(serie, df, variavel, df_full)

        fig_hist = px.histogram(
            df.assign(_driver=serie),
            x="_driver",
            nbins=40,
            title=f"Distribution — {variavel}",
            labels={"_driver": variavel},
        )
        fig_hist.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=420)

        fig_box = px.box(
            df.assign(_driver=serie),
            y="_driver",
            title="Outliers and spread",
            labels={"_driver": variavel},
        )
        fig_box.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=420)

        graph_scatter_json = None
        if "preco" in df.columns and variavel in df.columns and variavel != "preco":
            tmp = df[[variavel, "preco"]].copy()
            tmp[variavel] = pd.to_numeric(tmp[variavel], errors="coerce")
            tmp["preco"] = pd.to_numeric(tmp["preco"], errors="coerce")
            tmp = tmp.dropna()
            if len(tmp) > 0:
                tmp = tmp.head(6000)
                fig_sc = px.scatter(
                    tmp,
                    x=variavel,
                    y="preco",
                    title=f"Price vs {variavel}",
                    labels={variavel: variavel, "preco": "Price"},
                    trendline="ols",
                )
                fig_sc.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=420)
                fig_sc.update_yaxes(tickprefix="$")
                graph_scatter_json = self._to_plotly_json(fig_sc)

        graph_box_faixa_json = None
        if "faixa_preco" in df_full.columns and variavel in df_full.columns:
            tmp = df_full[[variavel, "faixa_preco"]].copy()
            tmp[variavel] = pd.to_numeric(tmp[variavel], errors="coerce")
            tmp["faixa_preco"] = tmp["faixa_preco"].astype(str)
            tmp = tmp.dropna(subset=[variavel, "faixa_preco"])
            if len(tmp) > 0:
                fig_bf = px.box(
                    tmp,
                    x="faixa_preco",
                    y=variavel,
                    title=f"{variavel} by price segment",
                    labels={"faixa_preco": "Price segment", variavel: variavel},
                )
                fig_bf.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=420)
                graph_box_faixa_json = self._to_plotly_json(fig_bf)

        graph_preco_ano_json = None
        year_col = self._pick_first_existing_col(df, ["YearBuilt", "Year_Built", "year_built", "ano_construcao"])
        if "preco" in df.columns and year_col is not None:
            tmp = df[[year_col, "preco"]].copy()
            tmp[year_col] = pd.to_numeric(tmp[year_col], errors="coerce")
            tmp["preco"] = pd.to_numeric(tmp["preco"], errors="coerce")
            tmp = tmp.dropna()
            if len(tmp) > 0:
                tmp = tmp[(tmp[year_col] >= 1800) & (tmp[year_col] <= 2100)].copy()
                tmp = tmp.head(8000)
                fig_y = px.scatter(
                    tmp,
                    x=year_col,
                    y="preco",
                    title="Price vs Year Built",
                    labels={year_col: "Year Built", "preco": "Price"},
                )
                fig_y.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=520)
                fig_y.update_yaxes(tickprefix="$")
                graph_preco_ano_json = self._to_plotly_json(fig_y)

        graph_box_bairro_json = None
        graph_bar_bairro_json = None
        neigh_col = self._pick_first_existing_col(df_full, ["bairro", "Neighborhood", "neighborhood"])
        if neigh_col is not None and "preco" in df_full.columns:
            tmp = df_full[[neigh_col, "preco"]].copy()
            tmp[neigh_col] = tmp[neigh_col].astype(str).str.strip()
            tmp["preco"] = pd.to_numeric(tmp["preco"], errors="coerce")
            tmp = tmp.dropna(subset=[neigh_col, "preco"])
            tmp = tmp[tmp[neigh_col] != ""]
            if len(tmp) > 0:
                counts = tmp[neigh_col].value_counts().head(25).index.tolist()
                tmp2 = tmp[tmp[neigh_col].isin(counts)].copy()

                fig_nb = px.box(
                    tmp2,
                    x=neigh_col,
                    y="preco",
                    title="Price distribution by neighborhood",
                    labels={neigh_col: "Neighborhood", "preco": "Price"},
                )
                fig_nb.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=560)
                fig_nb.update_yaxes(tickprefix="$")
                fig_nb.update_xaxes(tickangle=45)
                graph_box_bairro_json = self._to_plotly_json(fig_nb)

                avg = (
                    tmp.groupby(neigh_col, as_index=False)["preco"]
                    .mean()
                    .sort_values("preco", ascending=False)
                    .head(20)
                )
                fig_bar = px.bar(
                    avg,
                    x=neigh_col,
                    y="preco",
                    title="Average price ranking (Top 20)",
                    labels={neigh_col: "Neighborhood", "preco": "Avg Price"},
                )
                fig_bar.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=560)
                fig_bar.update_yaxes(tickprefix="$")
                fig_bar.update_xaxes(tickangle=45)
                graph_bar_bairro_json = self._to_plotly_json(fig_bar)

        graph_heatmap_json = None
        num_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
        preferred = [c for c in ["preco", "preco_m2", "GrLivArea", "TotalBsmtSF", "GarageCars", "GarageArea", "OverallQual", "YearBuilt"] if c in num_cols]
        cols = preferred if len(preferred) >= 4 else num_cols[:max_heatmap_cols]
        cols = cols[:max_heatmap_cols]

        if len(cols) >= 3:
            corr = df_full[cols].corr(method="pearson")
            fig_h = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation matrix",
            )
            fig_h.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=620)
            graph_heatmap_json = self._to_plotly_json(fig_h)

        graph_map_json = None
        lat_col = self._pick_first_existing_col(df, ["latitude", "lat", "Latitude"])
        lon_col = self._pick_first_existing_col(df, ["longitude", "lon", "Longitude"])
        if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
            tmp = df[[lat_col, lon_col]].copy()
            tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors="coerce")
            tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors="coerce")
            tmp = tmp.dropna()
            if len(tmp) > 0:
                tmp = tmp.head(8000)
                fig_map = px.scatter_mapbox(
                    tmp,
                    lat=lat_col,
                    lon=lon_col,
                    zoom=10,
                    height=520,
                    title="Geographic distribution",
                )
                fig_map.update_layout(
                    mapbox_style="open-street-map",
                    margin=dict(l=10, r=10, t=55, b=10),
                )
                graph_map_json = self._to_plotly_json(fig_map)

        return {
            "df_filtrado": df,
            "estatisticas": estatisticas,
            "testes_extra": testes_extra,
            "graph_hist_json": self._to_plotly_json(fig_hist),
            "graph_box_json": self._to_plotly_json(fig_box),
            "graph_scatter_json": graph_scatter_json,
            "graph_box_faixa_json": graph_box_faixa_json,
            "graph_preco_ano_json": graph_preco_ano_json,
            "graph_box_bairro_json": graph_box_bairro_json,
            "graph_bar_bairro_json": graph_bar_bairro_json,
            "graph_heatmap_json": graph_heatmap_json,
            "graph_map_json": graph_map_json,
        }

    # ==========================================================
    # Walmart Sales
    # ==========================================================
    @lru_cache(maxsize=1)
    def load_walmart_data(self) -> pd.DataFrame:
        if not self.data_dir:
            raise RuntimeError("data_dir não configurado no PortfolioService.")

        csv_path = self.data_dir / "Walmart.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Ficheiro {csv_path} não encontrado (esperado: data/Walmart.csv).")

        df = pd.read_csv(csv_path)

        if "transaction_date" in df.columns:
            df["transaction_date"] = self._safe_to_datetime(df["transaction_date"])

        for c in ["quantity_sold", "unit_price", "forecasted_demand", "actual_demand"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        def to_bool(col: str) -> None:
            if col not in df.columns:
                return
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
            )
            df[col] = df[col].fillna(False)

        to_bool("promotion_applied")
        to_bool("stockout_indicator")

        q = df["quantity_sold"] if "quantity_sold" in df.columns else 0
        p = df["unit_price"] if "unit_price" in df.columns else 0
        df["revenue"] = (q * p).astype(float)

        for c in ["category", "store_location", "product_name", "payment_method", "customer_loyalty_level"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

        if "transaction_date" in df.columns:
            df = df.dropna(subset=["transaction_date"])
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

        date_min = df["transaction_date"].min() if "transaction_date" in df.columns else None
        date_max = df["transaction_date"].max() if "transaction_date" in df.columns else None

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
                "revenue": self._fmt_money(0.0),
                "transactions": "0",
                "units": "0",
                "aov": self._fmt_money(0.0),
                "promo_share": "0.0%",
                "stockout_rate": "0.0%",
                "mape": "—",
                "avg_unit_price": self._fmt_money(0.0),
            }

        revenue = float(df["revenue"].sum()) if "revenue" in df.columns else 0.0
        transactions = int(len(df))
        units = float(df["quantity_sold"].sum()) if "quantity_sold" in df.columns else 0.0
        aov = revenue / transactions if transactions > 0 else 0.0

        promo_share = None
        if "promotion_applied" in df.columns and transactions > 0:
            promo_share = float(df["promotion_applied"].mean()) * 100.0

        stockout_rate = None
        if "stockout_indicator" in df.columns and transactions > 0:
            stockout_rate = float(df["stockout_indicator"].mean()) * 100.0

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
            "revenue": self._fmt_money(revenue),
            "transactions": self._fmt_int(transactions),
            "units": self._fmt_int(units),
            "aov": self._fmt_money(aov),
            "promo_share": self._fmt_pct(promo_share, digits=1) if promo_share is not None else "—",
            "stockout_rate": self._fmt_pct(stockout_rate, digits=1) if stockout_rate is not None else "—",
            "mape": mape_str,
            "avg_unit_price": self._fmt_money(avg_unit_price),
        }

    def build_sales_charts(self, df: pd.DataFrame, df_full: pd.DataFrame) -> Dict[str, str]:
        if df is None or df.empty:
            empty = px.scatter(pd.DataFrame({"x": [], "y": []}), x="x", y="y", title="No data for current filters")
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

        d["month"] = d["transaction_date"].dt.to_period("M").dt.to_timestamp()
        rev_m = d.groupby("month", as_index=False)["revenue"].sum()

        fig_trend = px.line(
            rev_m,
            x="month",
            y="revenue",
            title="Revenue trend",
            labels={"month": "Month", "revenue": "Revenue"},
        )
        fig_trend.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=420)
        fig_trend.update_yaxes(tickprefix="$")

        if "category" in d.columns:
            rev_cat = (
                d.groupby("category", as_index=False)["revenue"]
                .sum()
                .sort_values("revenue", ascending=False)
            )
            fig_cat = px.bar(
                rev_cat.head(12),
                x="category",
                y="revenue",
                title="Revenue by category",
                labels={"category": "Category", "revenue": "Revenue"},
            )
            fig_cat.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=420)
            fig_cat.update_yaxes(tickprefix="$")
        else:
            fig_cat = px.bar(pd.DataFrame({"category": [], "revenue": []}), x="category", y="revenue", title="Revenue by category")

        if "product_name" in d.columns:
            top_p = (
                d.groupby("product_name", as_index=False)["revenue"]
                .sum()
                .sort_values("revenue", ascending=False)
                .head(12)
            )
            fig_top_p = px.bar(
                top_p,
                x="revenue",
                y="product_name",
                orientation="h",
                title="Top products (revenue)",
                labels={"product_name": "Product", "revenue": "Revenue"},
            )
            fig_top_p.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=420)
            fig_top_p.update_xaxes(tickprefix="$")
        else:
            fig_top_p = px.bar(pd.DataFrame({"product_name": [], "revenue": []}), x="revenue", y="product_name", title="Top products")

        if "store_location" in d.columns:
            top_s = (
                d.groupby("store_location", as_index=False)["revenue"]
                .sum()
                .sort_values("revenue", ascending=False)
                .head(10)
            )
            fig_top_s = px.bar(
                top_s,
                x="store_location",
                y="revenue",
                title="Top stores (revenue)",
                labels={"store_location": "Store", "revenue": "Revenue"},
            )
            fig_top_s.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=420)
            fig_top_s.update_yaxes(tickprefix="$")
        else:
            fig_top_s = px.bar(pd.DataFrame({"store_location": [], "revenue": []}), x="store_location", y="revenue", title="Top stores")

        if "payment_method" in d.columns:
            pay = (
                d.groupby("payment_method", as_index=False)["revenue"]
                .sum()
                .sort_values("revenue", ascending=False)
            )
            fig_pay = px.pie(pay, names="payment_method", values="revenue", title="Payment mix (revenue)")
            fig_pay.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=420)
        else:
            fig_pay = px.pie(pd.DataFrame({"payment_method": [], "revenue": []}), names="payment_method", values="revenue", title="Payment mix")

        if "forecasted_demand" in d.columns and "actual_demand" in d.columns:
            sub = d[["forecasted_demand", "actual_demand"]].dropna().head(4000)

            fig_fc = px.scatter(
                sub,
                x="forecasted_demand",
                y="actual_demand",
                title="Forecast vs actual demand",
                labels={"forecasted_demand": "Forecasted", "actual_demand": "Actual"},
            )
            fig_fc.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=420)

            if len(sub) > 0:
                mn = float(np.nanmin([sub["forecasted_demand"].min(), sub["actual_demand"].min()]))
                mx = float(np.nanmax([sub["forecasted_demand"].max(), sub["actual_demand"].max()]))
                fig_fc.add_trace(
                    go.Scatter(
                        x=[mn, mx],
                        y=[mn, mx],
                        mode="lines",
                        name="Ideal (y = x)",
                        line=dict(dash="dash"),
                        showlegend=True,
                    )
                )
        else:
            fig_fc = px.scatter(
                pd.DataFrame({"forecasted_demand": [], "actual_demand": []}),
                x="forecasted_demand",
                y="actual_demand",
                title="Forecast vs actual demand",
            )

        return {
            "rev_trend_json": self._to_plotly_json(fig_trend),
            "rev_category_json": self._to_plotly_json(fig_cat),
            "top_products_json": self._to_plotly_json(fig_top_p),
            "top_stores_json": self._to_plotly_json(fig_top_s),
            "payment_mix_json": self._to_plotly_json(fig_pay),
            "forecast_scatter_json": self._to_plotly_json(fig_fc),
        }

    def build_store_comparison(self, df_a: pd.DataFrame, df_b: pd.DataFrame, store_a: str, store_b: str) -> Dict[str, Any]:
        df_a = df_a if df_a is not None else pd.DataFrame()
        df_b = df_b if df_b is not None else pd.DataFrame()

        kpi_a = self.compute_sales_kpis(df_a)
        kpi_b = self.compute_sales_kpis(df_b)

        def cat_rev(d: pd.DataFrame, label: str) -> pd.DataFrame:
            if d.empty or "category" not in d.columns or "revenue" not in d.columns:
                return pd.DataFrame({"category": [], "revenue": [], "store": []})
            tmp = d.groupby("category", as_index=False)["revenue"].sum()
            tmp["store"] = label
            return tmp

        ca = cat_rev(df_a, store_a)
        cb = cat_rev(df_b, store_b)
        cat = pd.concat([ca, cb], ignore_index=True)

        if not cat.empty:
            top = (
                cat.groupby("category", as_index=False)["revenue"]
                .sum()
                .sort_values("revenue", ascending=False)
                .head(12)["category"]
                .tolist()
            )
            cat = cat[cat["category"].isin(top)].copy()

        fig_cat_cmp = px.bar(
            cat,
            x="category",
            y="revenue",
            color="store",
            barmode="group",
            title="Revenue by category (Store A vs Store B)",
            labels={"category": "Category", "revenue": "Revenue", "store": "Store"},
        )
        fig_cat_cmp.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=420)
        fig_cat_cmp.update_yaxes(tickprefix="$")

        def monthly(d: pd.DataFrame, label: str) -> pd.DataFrame:
            if d.empty or "transaction_date" not in d.columns or "revenue" not in d.columns:
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
            labels={"month": "Month", "revenue": "Revenue", "store": "Store"},
        )
        fig_trend_cmp.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=420)
        fig_trend_cmp.update_yaxes(tickprefix="$")

        return {
            "store_a": store_a,
            "store_b": store_b,
            "kpi_a": kpi_a,
            "kpi_b": kpi_b,
            "cat_compare_json": self._to_plotly_json(fig_cat_cmp),
            "trend_compare_json": self._to_plotly_json(fig_trend_cmp),
        }

    # ==========================================================
    # ✅ Data Cleaner & Quality Report
    # - Detects CSV delimiter automatically ( ; , \t | )
    # - Keeps your robust "repair rows" fallback when needed
    # ==========================================================
    def read_uploaded_dataset(self, file_storage) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        filename = (file_storage.filename or "").strip()
        if not filename:
            raise ValueError("Empty filename.")

        ext = filename.lower().split(".")[-1]
        content = file_storage.read()
        if not content:
            raise ValueError("Uploaded file is empty.")

        meta = {"filename": filename, "ext": ext}

        # Excel
        if ext in {"xlsx", "xls"}:
            bio = io.BytesIO(content)
            df = pd.read_excel(bio)
            meta.update({"rows": int(len(df)), "cols": int(df.shape[1])})
            return df, meta

        if ext not in {"csv", "txt"}:
            raise ValueError("Unsupported file type. Please upload CSV or Excel (xlsx/xls).")

        # Decode
        text = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                text = content.decode(enc)
                break
            except Exception:
                continue
        if text is None:
            text = content.decode("latin-1", errors="replace")

        # Remove empty lines
        lines = [ln.strip("\n\r") for ln in text.splitlines() if ln.strip("\n\r").strip() != ""]
        if len(lines) < 2:
            raise ValueError("CSV has no data rows.")

        # ----------------------------
        # 1) Detect delimiter
        # ----------------------------
        sample_block = "\n".join(lines[:50])
        delimiter = ","
        try:
            dialect = csv.Sniffer().sniff(sample_block, delimiters=[",", ";", "\t", "|"])
            delimiter = dialect.delimiter
        except Exception:
            # fallback heuristic
            head = lines[0]
            counts = {
                ",": head.count(","),
                ";": head.count(";"),
                "\t": head.count("\t"),
                "|": head.count("|"),
            }
            delimiter = max(counts, key=counts.get) if max(counts.values()) > 0 else ","

        meta["detected_delimiter"] = delimiter

        # ----------------------------
        # 2) Fast path: pandas read_csv with detected delimiter
        # ----------------------------
        try:
            bio = io.StringIO("\n".join(lines))
            df = pd.read_csv(bio, sep=delimiter, engine="python")
            meta.update({"rows": int(len(df)), "cols": int(df.shape[1])})
            return df, meta
        except Exception:
            # fall back to robust reader below
            pass

        # ----------------------------
        # 3) Robust fallback: csv.reader + repair rows
        # ----------------------------
        header_raw = next(csv.reader([lines[0]], delimiter=delimiter, skipinitialspace=True))
        header = [h.strip() for h in header_raw]
        n_cols = len(header)

        # Identify "money-like" columns by header name (where decimal-comma is common)
        money_idx = []
        for i, h in enumerate(header):
            h_low = h.lower()
            if any(k in h_low for k in ["€", "eur", "price", "amount", "spend", "total", "valor", "preço", "preco", "custo", "revenue"]):
                money_idx.append(i)

        def is_int_token(tok: str) -> bool:
            t = (tok or "").strip()
            return bool(re.fullmatch(r"\d+", t))

        def is_thousand_token(tok: str) -> bool:
            t = (tok or "").strip()
            return bool(re.fullmatch(r"\d{1,3}(\.\d{3})+", t))  # 1.234 or 12.345.678

        def is_decimal_token(tok: str) -> bool:
            t = (tok or "").strip()
            return bool(re.fullmatch(r"\d{1,2}", t))  # 50 (cents) / 7 etc.

        def repair_row(parts: List[str]) -> List[str]:
            parts = [p.strip() for p in parts]

            # If too many columns: try merge likely numeric splits in money columns
            while len(parts) > n_cols:
                merged = False

                # Fix decimal comma split: "1.234" ; "50" -> "1.234,50"
                for j in money_idx:
                    if j < len(parts) - 1 and len(parts) > n_cols:
                        if (is_thousand_token(parts[j]) or is_int_token(parts[j])) and is_decimal_token(parts[j + 1]):
                            parts[j] = f"{parts[j]},{parts[j+1]}"
                            del parts[j + 1]
                            merged = True
                            break

                if merged:
                    continue

                # Fix comma thousand split: "2" ; "450.75" -> "2,450.75"
                for j in money_idx:
                    if j < len(parts) - 1 and len(parts) > n_cols:
                        if is_int_token(parts[j]) and bool(re.fullmatch(r"\d+\.\d+", (parts[j + 1] or "").strip())):
                            parts[j] = f"{parts[j]},{parts[j+1]}"
                            del parts[j + 1]
                            merged = True
                            break

                if merged:
                    continue

                # Fallback: merge last two tokens using comma (safer than shifting columns)
                parts[-2] = f"{parts[-2]},{parts[-1]}"
                parts = parts[:-1]

            # If too few columns: pad
            if len(parts) < n_cols:
                parts = parts + [""] * (n_cols - len(parts))

            return parts[:n_cols]

        rows = []
        for ln in lines[1:]:
            parts = next(csv.reader([ln], delimiter=delimiter, skipinitialspace=True))
            parts = repair_row(parts)
            rows.append(parts)

        df = pd.DataFrame(rows, columns=header)
        meta.update({"rows": int(len(df)), "cols": int(df.shape[1])})
        return df, meta

    def _normalize_colname(self, c: str) -> str:
        c0 = str(c).strip()
        c0 = re.sub(r"\s+", " ", c0)
        c1 = c0.lower()
        c1 = re.sub(r"[^\w\s]", "", c1)        # remove punctuation
        c1 = re.sub(r"\s+", "_", c1)           # spaces -> underscore
        c1 = re.sub(r"_+", "_", c1).strip("_")
        return c1 if c1 else "col"

    def _parse_mixed_number(self, s: pd.Series) -> pd.Series:
        """
        Converts common PT/EN number formats safely:
        - "1.234,50" -> 1234.50
        - "1,234.50" -> 1234.50
        - "845.00" -> 845.00
        - "2,450.75" -> 2450.75
        """
        if s is None:
            return s

        x = s.astype(str).str.strip()
        x = x.replace({"": np.nan, "nan": np.nan, "None": np.nan, "none": np.nan, "null": np.nan, "NULL": np.nan})

        def conv_one(v: str):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            t = str(v).strip()
            if t == "":
                return np.nan

            # keep only digits, dot, comma, minus
            t = re.sub(r"[^\d\-,\.]", "", t)

            # Case A: contains both comma and dot
            if "," in t and "." in t:
                # Decide decimal by last separator
                if t.rfind(",") > t.rfind("."):
                    # "1.234,50" -> remove dots (thousands), comma -> dot
                    t2 = t.replace(".", "").replace(",", ".")
                else:
                    # "1,234.50" -> remove commas (thousands)
                    t2 = t.replace(",", "")
                try:
                    return float(t2)
                except Exception:
                    return np.nan

            # Case B: only comma => assume decimal comma (or thousand)
            if "," in t and "." not in t:
                if re.fullmatch(r"-?\d+,\d{1,2}", t):
                    t2 = t.replace(",", ".")
                else:
                    t2 = t.replace(",", "")
                try:
                    return float(t2)
                except Exception:
                    return np.nan

            # Case C: only dot => standard float (or thousand groups)
            if "." in t and "," not in t:
                if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", t):
                    t2 = t.replace(".", "")
                else:
                    t2 = t
                try:
                    return float(t2)
                except Exception:
                    return np.nan

            # Case D: digits only
            if re.fullmatch(r"-?\d+", t):
                try:
                    return float(t)
                except Exception:
                    return np.nan

            return np.nan

        return x.map(conv_one)

    def clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Opinionated cleaning:
        - trim strings
        - normalize column names
        - drop fully empty rows/cols (AFTER normalizing empties)
        - convert numeric-like strings (robust mixed separators)
        - parse datetime-like columns (robust: format="mixed")
        - remove duplicates (exact duplicates)
        """
        if df is None or df.empty:
            return df, {"note": "empty_df"}

        original = df.copy()

        # 1) Drop fully empty rows/cols (initial pass)
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

        # 2) Normalize col names (and keep uniqueness)
        new_cols = [self._normalize_colname(c) for c in df.columns]
        seen = {}
        unique_cols = []
        for c in new_cols:
            if c not in seen:
                seen[c] = 0
                unique_cols.append(c)
            else:
                seen[c] += 1
                unique_cols.append(f"{c}_{seen[c]}")
        col_map = dict(zip(df.columns.tolist(), unique_cols))
        df = df.rename(columns=col_map)

        # 3) Trim strings + normalize null-like tokens
        null_tokens = {"", "nan", "none", "null", "NaN", "NULL", "None", "N/A", "n/a"}
        for c in df.columns:
            if df[c].dtype == object:
                s = df[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
                s = s.replace(list(null_tokens), np.nan)
                df[c] = s

        # 3b) NOW drop rows/cols that became fully empty after cleaning strings
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

        # 4) Convert numeric-like strings (robust)
        for c in df.columns:
            if df[c].dtype == object:
                s = df[c].dropna()
                if len(s) == 0:
                    continue

                sample = s.head(200).astype(str).str.strip()
                # "numeric-ish" if mostly composed by digits/.,,- and spaces/currency
                cleaned = sample.str.replace(r"[^\d\.,\-\s]", "", regex=True)
                looks_num = cleaned.str.match(r"^[\d\.,\-\s]+$").mean()

                if looks_num >= 0.70:
                    df[c] = self._parse_mixed_number(df[c])

        # 5) Datetime parsing (robust: handles mixed formats)
        for c in df.columns:
            if df[c].dtype == object:
                s = df[c].dropna().astype(str)
                if len(s) == 0:
                    continue

                sample = s.head(200)

                p_false = pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=False)
                p_true = pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=True)

                r_false = p_false.notna().mean()
                r_true = p_true.notna().mean()

                best_dayfirst = True if r_true > r_false else False
                best_ratio = max(r_false, r_true)

                if best_ratio >= 0.60:
                    df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed", dayfirst=best_dayfirst)

        # 6) Remove duplicates (exact rows)
        before_dups = len(df)
        df = df.drop_duplicates()
        removed_dups = before_dups - len(df)

        df = df.reset_index(drop=True)

        report = self.build_data_quality_report(original, df, removed_dups)
        return df, report

    def build_data_quality_report(self, df_raw: pd.DataFrame, df_clean: pd.DataFrame, removed_dups: int) -> Dict[str, Any]:
        def missing_pct(d: pd.DataFrame) -> float:
            if d is None or d.empty:
                return 0.0
            total = d.shape[0] * d.shape[1]
            if total == 0:
                return 0.0
            return float(d.isna().sum().sum()) / float(total) * 100.0

        raw_rows, raw_cols = (int(df_raw.shape[0]), int(df_raw.shape[1])) if df_raw is not None else (0, 0)
        clean_rows, clean_cols = (int(df_clean.shape[0]), int(df_clean.shape[1])) if df_clean is not None else (0, 0)

        raw_missing = missing_pct(df_raw)
        clean_missing = missing_pct(df_clean)

        dtypes = {}
        if df_clean is not None and not df_clean.empty:
            for k, v in df_clean.dtypes.items():
                dtypes[str(v)] = dtypes.get(str(v), 0) + 1

        top_missing_cols = []
        if df_clean is not None and not df_clean.empty:
            miss = (df_clean.isna().mean() * 100.0).sort_values(ascending=False).head(8)
            top_missing_cols = [{"col": idx, "missing_pct": float(val)} for idx, val in miss.items() if val > 0]

        return {
            "raw_rows": raw_rows,
            "raw_cols": raw_cols,
            "clean_rows": clean_rows,
            "clean_cols": clean_cols,
            "raw_missing_pct": float(raw_missing),
            "clean_missing_pct": float(clean_missing),
            "removed_duplicates": int(removed_dups),
            "dtype_summary": dtypes,
            "top_missing_cols": top_missing_cols,
        }

    def export_excel_cleaned(self, df_clean: pd.DataFrame, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df_clean.to_excel(writer, sheet_name="cleaned_data", index=False)

            # Apply date formats (Excel-friendly)
            ws = writer.sheets["cleaned_data"]
            dtypes = df_clean.dtypes.to_list()

            for j, dtype in enumerate(dtypes, start=1):
                if np.issubdtype(dtype, np.datetime64):
                    for row in range(2, 2 + len(df_clean)):
                        cell = ws.cell(row=row, column=j)
                        cell.number_format = "yyyy-mm-dd"

    def export_pdf_report(self, report: Dict[str, Any], meta: Dict[str, Any], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        c = canvas.Canvas(str(out_path), pagesize=A4)
        width, height = A4

        x = 2.0 * cm
        y = height - 2.0 * cm

        def line(txt: str, dy: float = 0.65 * cm, size: int = 11, bold: bool = False):
            nonlocal y
            c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
            c.drawString(x, y, txt)
            y -= dy
            if y < 2.0 * cm:
                c.showPage()
                y = height - 2.0 * cm

        line("Data Cleaner — Quality Report", dy=0.9 * cm, size=16, bold=True)
        line(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", size=10)
        line("")

        line("File metadata", bold=True)
        line(f"Filename: {meta.get('filename', '—')}")
        line(f"Type: {meta.get('ext', '—')}")
        if "detected_delimiter" in meta:
            line(f"CSV delimiter detected: {repr(meta.get('detected_delimiter'))}")
        line("")

        line("Shape & completeness", bold=True)
        line(f"Raw rows/cols: {report.get('raw_rows', 0)} / {report.get('raw_cols', 0)}")
        line(f"Clean rows/cols: {report.get('clean_rows', 0)} / {report.get('clean_cols', 0)}")
        line(f"Removed duplicates: {report.get('removed_duplicates', 0)}")
        line(f"Missing values (raw): {report.get('raw_missing_pct', 0.0):.1f}%")
        line(f"Missing values (clean): {report.get('clean_missing_pct', 0.0):.1f}%")
        line("")

        line("Column dtypes (clean)", bold=True)
        dts = report.get("dtype_summary") or {}
        if dts:
            for k, v in dts.items():
                line(f"{k}: {v}")
        else:
            line("—")
        line("")

        line("Top missing columns (clean)", bold=True)
        tmc = report.get("top_missing_cols") or []
        if tmc:
            for item in tmc:
                line(f"{item['col']}: {item['missing_pct']:.1f}%")
        else:
            line("—")

        c.save()

    def clean_uploaded_file(self, file_storage, artifacts_dir: Path, file_id: str) -> Dict[str, Any]:
        df_raw, meta = self.read_uploaded_dataset(file_storage)
        df_clean, report = self.clean_dataframe(df_raw)

        excel_path = artifacts_dir / f"cleaned_{file_id}.xlsx"
        pdf_path = artifacts_dir / f"report_{file_id}.pdf"
        self.export_excel_cleaned(df_clean, excel_path)
        self.export_pdf_report(report, meta, pdf_path)

        preview = df_clean.head(20).copy()
        for c in preview.columns:
            if np.issubdtype(preview[c].dtype, np.datetime64):
                preview[c] = preview[c].dt.strftime("%Y-%m-%d %H:%M:%S")

        table = {
            "columns": preview.columns.tolist(),
            "rows": preview.fillna("").values.tolist(),
        }

        return {
            "file_id": file_id,
            "meta": meta,
            "report": report,
            "table": table,
            "download_excel_url": f"/cleaner/download/{file_id}/excel",
            "download_pdf_url": f"/cleaner/download/{file_id}/pdf",
        }
