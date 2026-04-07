"""
data_processor.py
-----------------
Data Engineering pipeline for the Vivo Competitive Pricing Dashboard.

Responsibilities:
  - Load and sanitize the raw scraped CSV.
  - Engineer derived features (GB_Base, Has_Bonus, Preco_Por_GB).
  - Apply rule-based strategic clustering (Strategy_Cluster).

All heavy work is wrapped with @st.cache_data so Streamlit only executes
the pipeline once per unique filepath argument, then serves from memory.
"""

import re
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading and processing data...")
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw Vivo CSV, clean it, and return an enriched DataFrame.

    Pipeline steps
    --------------
    1. Load CSV with the correct delimiter (;) and encoding (utf-8).
    2. Coerce numeric columns to proper dtypes.
    3. Extract GB_Base  – integer gigabyte value from Filtro_Plano.
    4. Create Has_Bonus – True when the Bonus field contains any text.
    5. Compute Preco_Por_GB – monthly price divided by base gigabytes.
    6. Assign Strategy_Cluster via ordered business rules.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the semicolon-delimited CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-enriched DataFrame ready for visualisation.
    """

    # ------------------------------------------------------------------
    # 1. LOAD RAW DATA
    # ------------------------------------------------------------------
    df = pd.read_csv(
        filepath,
        sep=";",
        encoding="utf-8",
        # Treat truly empty cells as NaN so we can detect them uniformly.
        keep_default_na=True,
    )

    # Defensive strip: remove accidental leading/trailing whitespace from
    # all string columns introduced during scraping.
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # ------------------------------------------------------------------
    # 2. COERCE NUMERIC COLUMNS
    #    Preco_Mensal and Preco_Total may arrive as strings (e.g. "45,00")
    #    depending on locale.  We normalise the decimal separator and cast.
    # ------------------------------------------------------------------
    for num_col in ["Preco_Mensal", "Preco_Total", "Parcelas"]:
        df[num_col] = (
            df[num_col]
            .astype(str)
            .str.replace(",", ".", regex=False)   # handle BR locale decimals
            .pipe(pd.to_numeric, errors="coerce") # non-parseable → NaN
        )

    # ------------------------------------------------------------------
    # 3. FEATURE: GB_Base
    #    Filtro_Plano values look like "20GB (Req: Sim)".
    #    We extract the leading integer, which is the base plan size in GB.
    #    Rows that cannot be parsed (e.g. malformed scrape) are set to NaN
    #    and then filtered out so they don't corrupt downstream metrics.
    # ------------------------------------------------------------------
    def _extract_gb(value: str) -> float:
        """
        Pull the first integer token from a string such as '20GB (Req: Sim)'.
        Returns float so that NaN (from failed matches) is representable.
        """
        if pd.isna(value):
            return float("nan")
        match = re.match(r"(\d+)\s*GB", str(value), flags=re.IGNORECASE)
        return float(match.group(1)) if match else float("nan")

    df["GB_Base"] = df["Filtro_Plano"].apply(_extract_gb)

    # Cast to nullable integer (Int64) to keep NaN support while using int
    # semantics everywhere else (avoids the float "20.0" display issue).
    df["GB_Base"] = df["GB_Base"].astype("Int64")

    # Drop rows where GB_Base is 0 or NaN — zero-GB rows are placeholder
    # artefacts from the scraper and would produce division-by-zero below.
    df = df[df["GB_Base"].notna() & (df["GB_Base"] > 0)].copy()

    # ------------------------------------------------------------------
    # 4. FEATURE: Has_Bonus
    #    True when the Bonus column contains any non-whitespace text.
    #    An empty string after stripping is treated the same as NaN.
    # ------------------------------------------------------------------
    df["Has_Bonus"] = (
        df["Bonus"]
        .fillna("")          # normalise NaN → empty string
        .str.strip()         # remove invisible whitespace
        .str.len()           # length 0 ⟹ no bonus
        .gt(0)               # boolean: True when there IS bonus text
    )

    # ------------------------------------------------------------------
    # 5. FEATURE: Preco_Por_GB  (price efficiency metric)
    #    = monthly price / base GB
    #    Lower value  → cheaper per GB  → more aggressive/competitive offer.
    #    Round to 4 decimal places to avoid floating-point noise in
    #    percentile comparisons later.
    # ------------------------------------------------------------------
    df["Preco_Por_GB"] = (
        df["Preco_Mensal"] / df["GB_Base"].astype(float)
    ).round(4)

    # Drop rows where the price metric couldn't be computed (NaN price).
    df = df[df["Preco_Por_GB"].notna()].copy()

    # ------------------------------------------------------------------
    # 6. STRATEGY CLUSTERING
    #    Rules are applied in priority order via numpy.select-style logic.
    #    We compute dataset-wide thresholds ONCE so every rule uses the
    #    same reference distribution.
    # ------------------------------------------------------------------
    df = _assign_strategy_cluster(df)

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _assign_strategy_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a Strategy_Cluster label to every row using ordered business rules.

    Cluster definitions
    -------------------
    Aggressive Acquisition
        The competitor is willing to subsidise margin with bonus data to
        win new customers at the cheapest price points.
        Condition: Has_Bonus = True  AND  Preco_Por_GB ≤ 30th percentile.

    Premium Lock-in
        Annual commitment + no bonus + premium pricing = retention play
        targeting customers who have already decided to stay.
        Condition: Tipo_Oferta = "Anual"  AND  Has_Bonus = False
                   AND  Preco_Por_GB > dataset median.

    Battleground
        Bonus is offered but price is not a differentiator — these markets
        are contested on features/brand, not price.
        Condition: Has_Bonus = True  AND  Preco_Por_GB > 30th percentile.
        (i.e. bonus present but NOT cheap enough for Aggressive Acquisition)

    Standard
        Catch-all for offers that do not match any strategic pattern above.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already contains Has_Bonus, Preco_Por_GB, Tipo_Oferta.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with a new categorical column Strategy_Cluster.
    """

    prices = df["Preco_Por_GB"]

    # Pre-compute thresholds once — avoids recalculating per-row.
    p30    = prices.quantile(0.30)  # bottom 30 % ceiling
    median = prices.median()        # 50th percentile

    # --- condition masks (evaluated in priority order) -------------------

    # Rule 1 – Aggressive Acquisition
    # Cheap AND has bonus: the strongest competitive threat signal.
    mask_aggressive = (
        df["Has_Bonus"] &
        (prices <= p30)
    )

    # Rule 2 – Premium Lock-in
    # Annual plan, no bonus, above-median price: monetising loyal base.
    mask_premium = (
        (df["Tipo_Oferta"] == "Anual") &
        (~df["Has_Bonus"]) &
        (prices > median)
    )

    # Rule 3 – Battleground
    # Bonus present but price is NOT in the bottom 30 % — competing on
    # features/brand rather than pure price.
    mask_battleground = (
        df["Has_Bonus"] &
        (prices > p30)
    )

    # Rule 4 – Standard (default — no explicit mask needed)

    # --- apply labels in priority order ----------------------------------
    # np.select respects first-match semantics: the first True condition
    # wins; later rules are never evaluated for that row.
    df["Strategy_Cluster"] = np.select(
        condlist=[
            mask_aggressive,
            mask_premium,
            mask_battleground,
        ],
        choicelist=[
            "Aggressive Acquisition",
            "Premium Lock-in",
            "Battleground",
        ],
        default="Standard",  # Rule 4
    )

    # Convert to ordered Categorical for efficient groupby/sort operations
    # in Streamlit and Pandas downstream.
    cluster_order = [
        "Aggressive Acquisition",
        "Premium Lock-in",
        "Battleground",
        "Standard",
    ]
    df["Strategy_Cluster"] = pd.Categorical(
        df["Strategy_Cluster"],
        categories=cluster_order,
        ordered=True,
    )

    return df
