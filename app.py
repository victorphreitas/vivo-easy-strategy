"""
app.py
------
Vivo Competitive Pricing – Strategic Executive Dashboard.

Entry point for the Streamlit application.

Structure
---------
1. Page config & global style constants.
2. Data load (cached pipeline in data_processor.py).
3. Sidebar: 6 cascading/independent filters.
4. Filtered DataFrame derivation.
5. Top KPI row (4 executive metrics).
6. Visualisation layer: 3 executive charts in a 2+1 grid.
7. Detailed sortable data table.
"""

import os
import json

import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv

from data_processor import load_and_clean_data

# ---------------------------------------------------------------------------
# API KEY BOOTSTRAP
#   Priority: .env file (local dev)  →  Streamlit secrets (cloud deploy).
#   The OpenAI client is instantiated once at module level so it is reused
#   across button clicks without re-reading the environment each time.
# ---------------------------------------------------------------------------
load_dotenv()   # no-op if .env does not exist

_openai_api_key: str = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
_openai_client: OpenAI | None = OpenAI(api_key=_openai_api_key) if _openai_api_key else None

# ---------------------------------------------------------------------------
# 1. PAGE CONFIG  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Vivo · Competitive Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 2. GLOBAL STYLE CONSTANTS
# ---------------------------------------------------------------------------
DATA_PATH = "result.csv"

# Cluster colour palette — shared by KPIs, charts, and the data table.
CLUSTER_PALETTE = {
    "Aggressive Acquisition": "#EF553B",
    "Premium Lock-in":        "#636EFA",
    "Battleground":           "#FFA15A",
    "Standard":               "#B0B7C3",
}

OFFER_PALETTE = {
    "Mensal": "#00B5D8",
    "Anual":  "#805AD5",
}

PLOTLY_TEMPLATE = "plotly_white"

# Axis style token — applied uniformly to every chart.
AXIS_STYLE = dict(
    showgrid=True,
    gridcolor="#F0F2F5",
    gridwidth=1,
    linecolor="#E2E8F0",
    zeroline=False,
    tickfont=dict(size=11, color="#4A5568"),
    title_font=dict(size=12, color="#2D3748", family="Inter, sans-serif"),
)


def _apply_base_layout(fig, *, show_legend: bool = False) -> None:
    """
    Stamp a consistent, print-safe layout onto a Plotly figure in-place.
    Centralises all theme choices so individual chart blocks stay concise.
    """
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=36, b=8),
        font=dict(family="Inter, sans-serif", size=12, color="#4A5568"),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left",   x=0,
            title_text="",
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#E2E8F0",
            font_size=12,
            font_family="Inter, sans-serif",
        ),
    )


# ---------------------------------------------------------------------------
# AI LAYER — DATA CONTEXT + EMAIL GENERATION
# ---------------------------------------------------------------------------

def generate_data_context(df: pd.DataFrame) -> str:
    """
    Summarise the filtered DataFrame into a compact, token-efficient JSON
    string safe to pass to the LLM.

    Captures:
      - Total municipalities in the filtered view.
      - Strategy_Cluster distribution as percentages.
      - Average Preco_Mensal split by Tipo_Oferta (Mensal / Anual).
      - Top 3 most-common bonus descriptions currently active.
      - Average Preco_Por_GB for the selection.

    Returns a formatted JSON string so the LLM receives structured data
    rather than a raw table dump (cheaper on tokens, easier to reason over).
    """
    total_municipalities = int(df["Cidade"].nunique())
    total_offers         = len(df)

    # --- Strategy cluster distribution --------------------------------------
    cluster_counts = df["Strategy_Cluster"].value_counts()
    cluster_pct = {
        cluster: round(count / total_offers * 100, 1)
        for cluster, count in cluster_counts.items()
    }

    # --- Average price by offer type ----------------------------------------
    price_by_type = df.groupby("Tipo_Oferta")["Preco_Mensal"].mean().round(2)
    avg_price_mensal = float(price_by_type.get("Mensal", float("nan")))
    avg_price_anual  = float(price_by_type.get("Anual",  float("nan")))

    # Replace NaN with None so JSON serialises cleanly.
    avg_price_mensal = None if pd.isna(avg_price_mensal) else avg_price_mensal
    avg_price_anual  = None if pd.isna(avg_price_anual)  else avg_price_anual

    # --- Top 3 bonus offers -------------------------------------------------
    # Only consider rows where a bonus is actually present.
    bonus_series = df[df["Has_Bonus"]]["Bonus"].dropna()
    top_bonuses  = bonus_series.value_counts().head(3).index.tolist()

    # --- Average price per GB -----------------------------------------------
    avg_price_per_gb = round(float(df["Preco_Por_GB"].mean()), 4)

    context = {
        "total_municipalities":    total_municipalities,
        "total_offers_in_view":    total_offers,
        "avg_price_per_gb_brl":    avg_price_per_gb,
        "strategy_cluster_distribution_pct": cluster_pct,
        "avg_monthly_price_brl": {
            "mensal_plan": avg_price_mensal,
            "anual_plan":  avg_price_anual,
        },
        "top_3_active_bonus_offers": top_bonuses,
    }

    return json.dumps(context, ensure_ascii=False, indent=2)


_SYSTEM_PROMPT = (
    "Você é um Consultor de Estratégia C-Level especializado no mercado de Telecom brasileiro. "
    "Vou fornecer um resumo de dados sobre a estratégia atual de preços e produtos de um concorrente (Vivo), "
    "com base em filtros geográficos e faixas de preço específicos. "
    "Sua tarefa é fornecer uma explicação lógica e conceitual dessa estratégia. "
    "O público-alvo é o seu Diretor, que precisa entender a mecânica e os 'porquês' por trás desses números "
    "para apresentar ao Conselho Executivo. "
    "Não escreva um e-mail. Não use jargões vazios. "
    "Estruture sua resposta estritamente nos seguintes blocos: "
    "1. A Lógica dos Dados (O que os números filtrados revelam), "
    "2. Desconstrução da Estratégia (A tese de mercado deles: ex: forçando fidelização anual via spread de preços, "
    "subsídio agressivo em áreas de alta concorrência), "
    "e 3. Implicações Estratégicas. "
    "Seja impecavelmente lógico, denso em valor e escreva obrigatoriamente em Português do Brasil (PT-BR)."
)


def generate_strategic_analysis(context_string: str, filter_summary: str) -> str:
    """
    Call the OpenAI Chat Completions API and return a structured strategic
    analysis in Brazilian Portuguese aimed at a Telecom Director audience.

    Parameters
    ----------
    context_string : str
        JSON output of generate_data_context() for the current filtered view.
    filter_summary : str
        Human-readable description of the active sidebar filters, included
        so the LLM can ground its analysis in the specific geographic scope.

    Returns
    -------
    str
        The full structured analysis as returned by the model.

    Raises
    ------
    RuntimeError
        If the OpenAI client was not initialised (missing API key).
    openai.APIError
        Propagated directly so the Streamlit caller can display it.
    """
    if _openai_client is None:
        raise RuntimeError(
            "OpenAI API key not found. Add OPENAI_API_KEY to your .env file "
            "or to Streamlit secrets (st.secrets['OPENAI_API_KEY'])."
        )

    user_message = (
        f"Active dashboard filters: {filter_summary}\n\n"
        f"Competitor data summary (JSON):\n{context_string}"
    )

    response = _openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.4,      # low temperature = factual, grounded tone
        max_tokens=900,       # ≈ 650 words — tight executive email length
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 3. LOAD DATA
# ---------------------------------------------------------------------------
df_full: pd.DataFrame = load_and_clean_data(DATA_PATH)

# ---------------------------------------------------------------------------
# 4. SIDEBAR — CASCADING & INDEPENDENT FILTERS
#
#    Cascade chain (geographic):
#        UF  →  DDD  →  Filtro_Plano  →  Cidade
#
#    Each level's available options derive from the slice produced by all
#    upstream levels, making an impossible filter combination structurally
#    impossible to reach.
#
#    Independent filters (applied only at the final df construction):
#        Tipo_Oferta  |  Preco_Mensal range slider
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 Filters")
    st.markdown("---")

    # ── LEVEL 1: UF (State) ─────────────────────────────────────────────────
    all_ufs = sorted(df_full["UF"].dropna().unique().tolist())

    selected_ufs = st.multiselect(
        "State (UF)",
        options=all_ufs,
        default=[],
        placeholder="All states",
    )
    active_ufs = selected_ufs or all_ufs

    # ── LEVEL 2: DDD (Area Code) ────────────────────────────────────────────
    # Options constrained to the UF slice.
    _slice_uf = df_full[df_full["UF"].isin(active_ufs)]

    all_ddds = sorted(
        _slice_uf["DDD"].dropna().astype(str).unique().tolist()
    )

    selected_ddds = st.multiselect(
        "Area Code (DDD)",
        options=all_ddds,
        default=[],
        placeholder="All DDDs",
    )
    active_ddds = selected_ddds or all_ddds

    # ── LEVEL 3: Filtro_Plano (Plan Size) ───────────────────────────────────
    # Options constrained to UF + DDD slice.
    _slice_ddd = _slice_uf[_slice_uf["DDD"].astype(str).isin(active_ddds)]

    all_plans = sorted(
        _slice_ddd["Filtro_Plano"].dropna().unique().tolist(),
        key=lambda x: int(x.split("GB")[0]) if x.split("GB")[0].isdigit() else 0,
    )

    selected_plans = st.multiselect(
        "Plan Size (Filtro_Plano)",
        options=all_plans,
        default=[],
        placeholder="All plans",
    )
    active_plans = selected_plans or all_plans

    # ── LEVEL 4: Cidade (City) ───────────────────────────────────────────────
    # Options constrained to UF + DDD + Plan slice.
    _slice_plan = _slice_ddd[_slice_ddd["Filtro_Plano"].isin(active_plans)]

    all_cities = sorted(_slice_plan["Cidade"].dropna().unique().tolist())

    selected_cities = st.multiselect(
        "City (Cidade)",
        options=all_cities,
        default=[],
        placeholder="All cities",
    )
    active_cities = selected_cities or all_cities

    st.markdown("---")

    # ── INDEPENDENT: Offer Type (Tipo_Oferta) ────────────────────────────────
    # Does not restrict geographic options — applied at final filter only.
    all_offer_types = sorted(df_full["Tipo_Oferta"].dropna().unique().tolist())

    selected_offer_types = st.multiselect(
        "Offer Type",
        options=all_offer_types,
        default=[],
        placeholder="Monthly & Annual",
    )
    active_offer_types = selected_offer_types or all_offer_types

    # ── INDEPENDENT: Monthly Price Range slider ───────────────────────────────
    price_min_global = float(df_full["Preco_Mensal"].min())
    price_max_global = float(df_full["Preco_Mensal"].max())

    price_range = st.slider(
        "Monthly Price (R$)",
        min_value=price_min_global,
        max_value=price_max_global,
        value=(price_min_global, price_max_global),
        step=1.0,
        format="R$ %.0f",
    )

    st.markdown("---")
    st.caption("Geographic filters cascade top-to-bottom. Offer Type and Price are independent.")

# ---------------------------------------------------------------------------
# 5. FILTERED DATAFRAME
#    Apply all six active filter dimensions in a single boolean mask.
# ---------------------------------------------------------------------------
df: pd.DataFrame = df_full[
    df_full["UF"].isin(active_ufs) &
    df_full["DDD"].astype(str).isin(active_ddds) &
    df_full["Filtro_Plano"].isin(active_plans) &
    df_full["Cidade"].isin(active_cities) &
    df_full["Tipo_Oferta"].isin(active_offer_types) &
    df_full["Preco_Mensal"].between(price_range[0], price_range[1])
].copy()

# Guard: prevent downstream errors on an empty selection.
if df.empty:
    st.warning("No data matches the current filter selection. Please broaden your criteria.")
    st.stop()

# Remove ghost categories carried over from the full dataset slice.
df["Strategy_Cluster"] = df["Strategy_Cluster"].cat.remove_unused_categories()

# ---------------------------------------------------------------------------
# 6. DASHBOARD HEADER
# ---------------------------------------------------------------------------
st.title("📡 Vivo · Competitive Pricing Intelligence")
st.markdown(
    "Executive view of geographic pricing strategies scraped from Vivo's public offer catalogue. "
    "Use the sidebar to drill into specific states, area codes, or plan sizes."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# 7. KPI ROW
#    Four headline metrics, each shown with a delta vs. the national baseline.
# ---------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    n_cities = df["Cidade"].nunique()
    n_cities_full = df_full["Cidade"].nunique()
    st.metric(
        "🏙️ Municipalities Covered",
        f"{n_cities:,}",
        delta=f"{n_cities - n_cities_full:+,} vs. national",
        delta_color="off",
        help="Unique cities in the current filter selection.",
    )

with kpi2:
    avg_ppgb = df["Preco_Por_GB"].mean()
    avg_ppgb_full = df_full["Preco_Por_GB"].mean()
    st.metric(
        "💰 Avg Price / GB",
        f"R$ {avg_ppgb:.2f}",
        delta=f"R$ {avg_ppgb - avg_ppgb_full:+.2f} vs. national",
        delta_color="inverse",
        help="Mean of (Preco_Mensal ÷ GB_Base) across filtered rows.",
    )

with kpi3:
    pct_bonus = df["Has_Bonus"].mean() * 100
    pct_bonus_full = df_full["Has_Bonus"].mean() * 100
    st.metric(
        "🎁 Bonus Penetration",
        f"{pct_bonus:.1f}%",
        delta=f"{pct_bonus - pct_bonus_full:+.1f}pp vs. national",
        delta_color="inverse",
        help="Share of offers in the selection that include a bonus.",
    )

with kpi4:
    price_by_type = df.groupby("Tipo_Oferta")["Preco_Mensal"].mean()
    avg_annual  = price_by_type.get("Anual",  pd.NA)
    avg_monthly = price_by_type.get("Mensal", pd.NA)

    if pd.notna(avg_annual) and pd.notna(avg_monthly):
        delta_val = avg_annual - avg_monthly
        st.metric(
            "📅 Annual vs Monthly Δ",
            f"R$ {delta_val:+.2f}",
            help="Mean monthly price (Anual) minus (Mensal). Negative = annual is cheaper.",
        )
    else:
        st.metric("📅 Annual vs Monthly Δ", "N/A",
                  help="Both offer types must be present in the selection.")

st.markdown("---")

# ---------------------------------------------------------------------------
# 8. VISUALISATION LAYER
#
#    Layout: Row 1 = Chart 1 (donut, narrower) + Chart 2 (grouped bar, wider)
#            Row 2 = Chart 3 (horizontal bar, full width)
# ---------------------------------------------------------------------------

# ── ROW 1 ───────────────────────────────────────────────────────────────────
col_donut, col_bar = st.columns([1, 1.6], gap="large")

# ── Chart 1: Strategy Breakdown — Donut ─────────────────────────────────────
with col_donut:
    st.subheader("Strategy Breakdown")
    st.caption("Share of total offers assigned to each strategic cluster in the current selection.")

    # Aggregate: count offers (rows) per cluster.
    donut_df = (
        df["Strategy_Cluster"]
        .value_counts()
        .reset_index(name="Count")
        .rename(columns={"index": "Strategy_Cluster"})
    )
    # value_counts already names the column correctly in pandas >= 2.0
    if "Strategy_Cluster" not in donut_df.columns:
        donut_df.columns = ["Strategy_Cluster", "Count"]

    fig1 = px.pie(
        donut_df,
        names="Strategy_Cluster",
        values="Count",
        hole=0.52,                          # donut shape
        color="Strategy_Cluster",
        color_discrete_map=CLUSTER_PALETTE,
        category_orders={"Strategy_Cluster": list(CLUSTER_PALETTE.keys())},
    )

    _apply_base_layout(fig1, show_legend=True)

    fig1.update_traces(
        textposition="outside",
        textinfo="percent+label",
        textfont=dict(size=11, family="Inter, sans-serif"),
        hovertemplate="<b>%{label}</b><br>Offers: %{value:,}<br>Share: %{percent}<extra></extra>",
        pull=[0.03] * len(donut_df),        # slight separation on all slices
    )

    # Suppress the default legend — labels on slices are self-explanatory.
    fig1.update_layout(
        showlegend=False,
        margin=dict(l=8, r=8, t=36, b=8),
    )

    st.plotly_chart(fig1, use_container_width=True)


# ── Chart 2: Price Comparison — Grouped Bar (Top 5 DDDs) ────────────────────
with col_bar:
    st.subheader("Avg Monthly Price: Annual vs Monthly")
    st.caption("Top 5 area codes by offer volume. Grouped bars reveal the monthly-price discount Vivo uses to push 12-month lock-ins.")

    # Identify top 5 DDDs by row count in the filtered view.
    top5_ddds = (
        df["DDD"].value_counts()
        .head(5)
        .index
        .astype(str)
        .tolist()
    )

    chart2_df = (
        df[df["DDD"].astype(str).isin(top5_ddds)]
        .groupby(["DDD", "Tipo_Oferta"], observed=True)["Preco_Mensal"]
        .mean()
        .round(2)
        .reset_index(name="Avg_Price")
    )
    chart2_df["DDD"] = chart2_df["DDD"].astype(str)

    # Sort DDDs by the same volume order as selected above.
    fig2 = px.bar(
        chart2_df,
        x="DDD",
        y="Avg_Price",
        color="Tipo_Oferta",
        color_discrete_map=OFFER_PALETTE,
        barmode="group",
        category_orders={
            "DDD": top5_ddds,
            "Tipo_Oferta": ["Mensal", "Anual"],
        },
        text="Avg_Price",
        labels={
            "DDD":        "Area Code (DDD)",
            "Avg_Price":  "Avg Monthly Price (R$)",
            "Tipo_Oferta": "Offer Type",
        },
    )

    _apply_base_layout(fig2, show_legend=True)

    fig2.update_traces(
        texttemplate="R$ %{text:.0f}",
        textposition="outside",
        textfont=dict(size=10),
        marker_line_width=0,
    )

    fig2.update_xaxes(**AXIS_STYLE, title_text="Area Code (DDD)", type="category")
    fig2.update_yaxes(**AXIS_STYLE, title_text="Avg Monthly Price (R$)")

    fig2.update_layout(
        uniformtext_minsize=9,
        uniformtext_mode="hide",
    )

    st.plotly_chart(fig2, use_container_width=True)


# ── ROW 2: Chart 3 — Bonus Landscape (Horizontal Bar, full width) ────────────
st.subheader("Bonus Landscape — Top 10 States by Promotional Coverage")
st.caption("Percentage of plans in each state that include a bonus offer. Higher coverage signals heavier competitive subsidisation in that market.")

# Compute per-UF bonus penetration, take top 10 by %.
chart3_df = (
    df.groupby("UF", observed=True)["Has_Bonus"]
    .agg(["sum", "count"])
    .reset_index()
    .rename(columns={"sum": "With_Bonus", "count": "Total"})
)
chart3_df["Bonus_Pct"] = (chart3_df["With_Bonus"] / chart3_df["Total"] * 100).round(1)

# Sort descending and take top 10; reverse for horizontal bar readability
# (highest value appears at the top of the chart).
chart3_df = chart3_df.nlargest(10, "Bonus_Pct").sort_values("Bonus_Pct", ascending=True)

fig3 = px.bar(
    chart3_df,
    x="Bonus_Pct",
    y="UF",
    orientation="h",
    text="Bonus_Pct",
    color="Bonus_Pct",
    color_continuous_scale=[
        [0.0, "#EDF2F7"],    # light grey  – low bonus coverage
        [0.5, "#F6AD55"],    # amber       – moderate
        [1.0, "#EF553B"],    # red         – high / aggressive
    ],
    labels={
        "Bonus_Pct": "% Plans with Bonus",
        "UF":        "State",
    },
    hover_data={
        "With_Bonus": True,
        "Total":      True,
        "Bonus_Pct":  ":.1f",
    },
)

_apply_base_layout(fig3, show_legend=False)

fig3.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
    textfont=dict(size=11),
    marker_line_width=0,
)

fig3.update_xaxes(
    **AXIS_STYLE,
    title_text="% of Plans with Bonus",
    range=[0, chart3_df["Bonus_Pct"].max() * 1.15],  # headroom for labels
)
fig3.update_yaxes(**AXIS_STYLE, title_text="")
fig3.update_coloraxes(showscale=False)   # colour encodes magnitude; scale bar is redundant

fig3.update_layout(height=380)

st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# 9. DETAILED DATA TABLE
#    Sortable by any column for city-level deep-dives.
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📋 Detailed Offer Explorer")
st.caption(
    f"Showing {len(df):,} offers across {df['Cidade'].nunique():,} cities. "
    "Click any column header to sort."
)

st.dataframe(
    df[[
        "Cidade", "UF", "DDD",
        "Filtro_Plano", "GB_Base",
        "Tipo_Oferta", "Preco_Mensal", "Preco_Total",
        "Has_Bonus", "Preco_Por_GB",
        "Strategy_Cluster",
    ]].sort_values(["UF", "Cidade"]),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Cidade":           st.column_config.TextColumn("City"),
        "UF":               st.column_config.TextColumn("State"),
        "DDD":              st.column_config.NumberColumn("DDD",          format="%d"),
        "Filtro_Plano":     st.column_config.TextColumn("Plan"),
        "GB_Base":          st.column_config.NumberColumn("GB",           format="%d GB"),
        "Tipo_Oferta":      st.column_config.TextColumn("Type"),
        "Preco_Mensal":     st.column_config.NumberColumn("Monthly (R$)", format="R$ %.2f"),
        "Preco_Total":      st.column_config.NumberColumn("Total (R$)",   format="R$ %.2f"),
        "Has_Bonus":        st.column_config.CheckboxColumn("Bonus?"),
        "Preco_Por_GB":     st.column_config.NumberColumn("R$/GB",        format="R$ %.4f"),
        "Strategy_Cluster": st.column_config.TextColumn("Cluster"),
    },
)

# ---------------------------------------------------------------------------
# 10. AI STRATEGY ANALYST
#
#     On button click:
#       1. Build a human-readable filter summary (shown in the prompt header).
#       2. Compress the filtered DataFrame into a token-safe JSON context.
#       3. Call GPT-4o with the system prompt + context.
#       4. Store the result in st.session_state so it survives reruns without
#          re-calling the API (Streamlit re-executes the full script on every
#          widget interaction).
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🤖 Analista IA de Inteligência Competitiva")
st.caption(
    "Gera uma análise estratégica estruturada com base no **recorte atualmente filtrado** do dataset. "
    "Ajuste os filtros na barra lateral para delimitar o escopo antes de gerar a análise."
)

# Warn immediately if the API key is missing so the executive doesn't click
# the button and see a cryptic error only after a 5-second wait.
if _openai_client is None:
    st.warning(
        "**OpenAI API key not configured.** "
        "Add `OPENAI_API_KEY` to a `.env` file in the project root or to "
        "`st.secrets` to enable this feature.",
        icon="🔑",
    )

# --- Build a human-readable summary of the active filters ------------------
# This string is included in the LLM user message so it can open the email
# with the correct geographic/price scope.
def _build_filter_summary() -> str:
    parts: list[str] = []

    # Geographic filters — only mention when narrowed from the full set.
    if selected_ufs:
        parts.append(f"States: {', '.join(sorted(selected_ufs))}")
    if selected_ddds:
        parts.append(f"DDDs: {', '.join(sorted(selected_ddds))}")
    if selected_plans:
        parts.append(f"Plans: {', '.join(sorted(selected_plans))}")
    if selected_cities:
        parts.append(f"Cities: {', '.join(sorted(selected_cities))}")

    # Independent filters.
    if selected_offer_types:
        parts.append(f"Offer type: {', '.join(sorted(selected_offer_types))}")

    # Always show price range so the LLM knows the price scope.
    parts.append(
        f"Monthly price: R$ {price_range[0]:.0f} – R$ {price_range[1]:.0f}"
    )

    if not parts:
        return "No filters active — full national dataset."

    return " | ".join(parts)


filter_summary = _build_filter_summary()

# Show the executive what the AI will analyse so they can adjust if needed.
st.info(f"**Current analysis scope:** {filter_summary}", icon="🔎")

# --- Button + generation logic ----------------------------------------------
# st.session_state persists the email text across reruns so the executive can
# read it without triggering another API call every time they scroll.
if "ai_email_text" not in st.session_state:
    st.session_state["ai_email_text"] = None

# Store the filter state that produced the current cached email so we can
# detect when the filters have changed and prompt the user to regenerate.
if "ai_email_filter_key" not in st.session_state:
    st.session_state["ai_email_filter_key"] = None

current_filter_key = filter_summary   # cheap string comparison as cache key

if st.button(
    "Desconstruir Estratégia (Contexto Atual)",
    type="primary",
    disabled=(_openai_client is None),
):
    with st.spinner("Analisando padrões geográficos e construindo tese estratégica..."):
        try:
            context_str    = generate_data_context(df)
            analysis_text  = generate_strategic_analysis(context_str, filter_summary)

            st.session_state["ai_email_text"]       = analysis_text
            st.session_state["ai_email_filter_key"] = current_filter_key

        except RuntimeError as exc:
            # Missing API key — already warned above, but surface cleanly.
            st.error(str(exc))
        except Exception as exc:
            # Network errors, rate limits, model errors from OpenAI.
            st.error(f"OpenAI API error: {exc}")

# --- Render cached email, if available --------------------------------------
if st.session_state["ai_email_text"]:
    # Warn if the filters changed since the last generation — the cached
    # email may no longer match what is displayed in the charts above.
    if st.session_state["ai_email_filter_key"] != current_filter_key:
        st.warning(
            "The dashboard filters have changed since this brief was generated. "
            "Click **Generate Executive Brief** again to refresh the analysis.",
            icon="⚠️",
        )

    st.markdown(
        f"""
<div style="
    background: #F7FAFC;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #636EFA;
    border-radius: 6px;
    padding: 1.5rem 1.75rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    line-height: 1.7;
    color: #2D3748;
    white-space: pre-wrap;
">
{st.session_state["ai_email_text"]}
</div>
""",
        unsafe_allow_html=True,
    )
