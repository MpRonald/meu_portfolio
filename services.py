from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


class PortfolioService:
    def __init__(self, timeout: int = 30, data_dir: Optional[str] = None):
        self.timeout = timeout
        self.data_dir = Path(data_dir) if data_dir else None

    # ======================================================
    # ======================= AMES =========================
    # ======================================================
    @lru_cache(maxsize=1)
    def load_ames_data(self) -> pd.DataFrame:
        if not self.data_dir:
            raise RuntimeError("data_dir não configurado no PortfolioService.")

        csv_path = self.data_dir / "ames.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Ficheiro {csv_path} não encontrado.")

        df = pd.read_csv(csv_path)

        # Garantir tipos numéricos sempre que possível
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

        # Jarque–Bera
        try:
            jb_stat, jb_p = stats.jarque_bera(s)
            resultados["jb_stat"] = float(jb_stat)
            resultados["jb_p"] = float(jb_p)
        except Exception:
            resultados["jb_stat"] = None
            resultados["jb_p"] = None

        # Pearson (preco / preco_m2)
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

        # Spearman
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

        # Kruskal–Wallis
        if df_completo is not None and "faixa_preco" in df_completo.columns:
            grupos = []
            for faixa in df_completo["faixa_preco"].dropna().unique():
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

        # Regressão linear simples
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

    # ======================================================
    # =================== WALMART SALES ====================
    # ======================================================
    @lru_cache(maxsize=1)
    def load_walmart_sales_data(self) -> pd.DataFrame:
        if not self.data_dir:
            raise RuntimeError("data_dir não configurado no PortfolioService.")

        csv_path = self.data_dir / "walmart.csv"
        if not csv_path.exists():
            alt = self.data_dir / "Walmart.csv"
            if alt.exists():
                csv_path = alt
            else:
                raise FileNotFoundError("walmart.csv não encontrado em data/")

        df = pd.read_csv(csv_path)

        # Datas
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df["year"] = df["transaction_date"].dt.year
        df["month"] = df["transaction_date"].dt.to_period("M").astype(str)
        df["weekday"] = df["transaction_date"].dt.day_name()

        # Booleanos
        for col in ["promotion_applied", "holiday_indicator", "stockout_indicator"]:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})

        # Numéricos
        for col in [
            "quantity_sold", "unit_price", "forecasted_demand",
            "actual_demand", "inventory_level"
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Métricas derivadas
        df["revenue"] = df["quantity_sold"] * df["unit_price"]
        df["forecast_error"] = df["actual_demand"] - df["forecasted_demand"]
        df["forecast_ape"] = (
            df["forecast_error"].abs() / df["actual_demand"].replace(0, np.nan)
        ) * 100

        return df
