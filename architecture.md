# Vivo Competitive Pricing Intelligence — Architecture Document

## Project Overview

This tool is an executive-grade competitive intelligence dashboard built in Streamlit. It ingests pricing data scraped from Vivo's public mobile plan catalogue and transforms it into strategic signals: which markets are being subsidised with bonus data, where Vivo is pushing annual lock-ins, and how aggressively they are pricing per gigabyte across Brazilian states and area codes.

The primary audience is a commercial or strategy team that needs to answer questions like:
- *Where is Vivo flooding the market with bonus offers to acquire customers?*
- *Which area codes are receiving premium lock-in treatment vs. discounted monthly plans?*
- *Which states have the highest promotional pressure?*

---

## Data Pipeline (`data_processor.py`)

### Input

`result.csv` — semicolon-delimited, UTF-8 encoded. Columns:

| Column | Type | Description |
|---|---|---|
| `Cidade` | string | Municipality name |
| `UF` | string | State abbreviation (e.g. SP, RJ) |
| `DDD` | integer | Telephone area code |
| `Filtro_Plano` | string | Plan label e.g. `"20GB (Req: Sim)"` |
| `Tipo_Oferta` | string | `"Mensal"` (monthly) or `"Anual"` (annual) |
| `Preco_Mensal` | float | Monthly price in BRL |
| `Parcelas` | integer | Number of instalments |
| `Preco_Total` | float | Total contract price in BRL |
| `Descricao` | string | Human-readable plan description |
| `Bonus` | string | Bonus description (empty if none) |
| `Proposta_Ativa` | string | Whether the offer is active |

### Pipeline Steps

All steps are executed inside `load_and_clean_data(filepath)`, which is wrapped with `@st.cache_data` so the entire pipeline runs exactly once per Streamlit session.

**1. Load & sanitise**
CSV is loaded with `sep=";"`, `encoding="utf-8"`. All string columns are stripped of leading/trailing whitespace introduced by the scraper. Numeric columns (`Preco_Mensal`, `Preco_Total`, `Parcelas`) are coerced from string to float, handling Brazilian locale decimal commas.

**2. `GB_Base` extraction**
A regex (`r"(\d+)\s*GB"`) pulls the integer gigabyte value from `Filtro_Plano`. The result is cast to nullable `Int64`. Rows with zero or unparseable GB values are dropped — they are scraper artefacts and would produce division-by-zero errors downstream.

**3. `Has_Bonus` flag**
`True` when the `Bonus` column contains any non-whitespace text after stripping. Both `NaN` and empty strings evaluate to `False`.

**4. `Preco_Por_GB` metric**
```
Preco_Por_GB = Preco_Mensal / GB_Base
```
Rounded to 4 decimal places to suppress floating-point noise in percentile comparisons. Rows where the metric cannot be computed (NaN price) are dropped.

**5. `Strategy_Cluster` classification**

Thresholds are computed once on the full dataset distribution:

| Threshold | Value |
|---|---|
| `p30` | 30th percentile of `Preco_Por_GB` |
| `median` | 50th percentile of `Preco_Por_GB` |

Rules are applied in strict priority order via `numpy.select`:

| Priority | Cluster | Condition |
|---|---|---|
| 1 | **Aggressive Acquisition** | `Has_Bonus = True` AND `Preco_Por_GB ≤ P30` |
| 2 | **Premium Lock-in** | `Tipo_Oferta = "Anual"` AND `Has_Bonus = False` AND `Preco_Por_GB > median` |
| 3 | **Battleground** | `Has_Bonus = True` AND `Preco_Por_GB > P30` |
| 4 | **Standard** | All remaining rows |

The column is stored as an ordered `pd.Categorical` for efficient downstream groupby and sort operations.

---

## Application Structure (`app.py`)

### Filter System (Sidebar)

Filters are applied in two tiers:

**Geographic cascade** — each level restricts the options available to the next:
```
UF (State)  →  DDD (Area Code)  →  Filtro_Plano (Plan Size)  →  Cidade (City)
```

**Independent filters** — applied only at the final DataFrame construction, they do not affect the geographic option lists:
- **Offer Type** (`Tipo_Oferta`): multiselect for Mensal / Anual
- **Monthly Price** (`Preco_Mensal`): dual-ended slider with BRL range from the full dataset

Empty multiselect = "select all" for that dimension.

### KPI Row

| Metric | Calculation | Delta baseline |
|---|---|---|
| Municipalities Covered | `df["Cidade"].nunique()` | National total |
| Avg Price / GB | `df["Preco_Por_GB"].mean()` | National mean |
| Bonus Penetration | `df["Has_Bonus"].mean() × 100` | National % |
| Annual vs Monthly Δ | `mean(Anual) − mean(Mensal)` on `Preco_Mensal` | — |

Deltas use `delta_color="inverse"` for price and bonus metrics (higher = worse competitive pressure).

### Visualisations

**Chart 1 — Strategy Breakdown (Donut)**
- Answers: *What proportion of Vivo's current offers fall into each strategic archetype?*
- Data: `value_counts()` of `Strategy_Cluster` on the filtered view.
- Design: Donut with `hole=0.52`, percent labels on slices, cluster colour palette. Legend suppressed — slice labels are self-explanatory.

**Chart 2 — Avg Monthly Price: Annual vs Monthly (Grouped Bar)**
- Answers: *How steep is the annual discount in the highest-volume area codes?*
- Data: Top 5 DDDs by row count; grouped by `Tipo_Oferta`; mean `Preco_Mensal` per group.
- Design: Side-by-side bars with price labels above each bar. Teal = monthly, violet = annual. Sorted by DDD volume so the most commercially relevant markets appear first.

**Chart 3 — Bonus Landscape (Horizontal Bar)**
- Answers: *Which states are receiving the heaviest promotional subsidy?*
- Data: Per-UF `Has_Bonus` penetration rate; top 10 states sorted descending (chart sorted ascending for readability — highest at top).
- Design: Horizontal bars with a continuous colour scale (light grey → amber → red) that independently encodes intensity. No colour legend needed — the X-axis value already carries the meaning.

### Data Table

Full filtered dataset exposed via `st.dataframe` with `column_config` for typed formatting (currency, checkboxes, integer GB). Sorted by `UF` then `Cidade` by default; any column is user-sortable.

---

## Tech Stack

| Library | Version constraint | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.35 | Application framework, UI components, caching |
| `pandas` | ≥ 2.1 | Data ingestion, cleaning, feature engineering |
| `plotly` | ≥ 5.22 | Interactive Plotly Express charts |
| `numpy` | (installed with pandas) | `np.select` for vectorised cluster assignment |

---

## File Structure

```
vivo-easy-agent/
├── result.csv            # Raw scraped data (source of truth)
├── data_processor.py     # ETL pipeline + clustering logic
├── app.py                # Streamlit dashboard entry point
├── requirements.txt      # Pinned library versions
└── architecture.md       # This document
```

## Running the Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.
