import calendar
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Investor Insight Studio",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

MONTH_ORDER = list(calendar.month_name[1:])
DATA_PATH = Path(__file__).resolve().parent / "clean_sales_data.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Some exports wrap each CSV row in quotes, which makes pandas ingest the
    # entire row as a single column. If that happens, split the row manually.
    if len(df.columns) == 1:
        raw_rows = pd.Series([df.columns[0], *df.iloc[:, 0].astype(str).tolist()])
        split_rows = raw_rows.str.replace('"', "", regex=False).str.split(",", expand=True)
        split_rows.columns = split_rows.iloc[0].str.strip()
        df = split_rows.iloc[1:].reset_index(drop=True)

    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.replace('"', "", regex=False)
    )

    required_columns = [
        "Sales Person",
        "Country",
        "Product",
        "Date",
        "Amount",
        "Boxes Shipped",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    df = df.dropna(subset=["Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Boxes Shipped"] = pd.to_numeric(df["Boxes Shipped"], errors="coerce")
    if "Revenue_per_Box" in df.columns:
        df["Revenue_per_Box"] = pd.to_numeric(df["Revenue_per_Box"], errors="coerce")
    df = df.dropna(subset=["Amount", "Boxes Shipped"]).copy()

    if "Month" not in df.columns:
        df["Month"] = df["Date"].dt.month
    else:
        df["Month"] = pd.to_numeric(df["Month"], errors="coerce").fillna(df["Date"].dt.month)

    if "Month_Name" not in df.columns:
        df["Month_Name"] = df["Date"].dt.month_name()

    if "Revenue_per_Box" not in df.columns:
        df["Revenue_per_Box"] = df["Amount"] / df["Boxes Shipped"].replace(0, pd.NA)

    df["Month_Name"] = pd.Categorical(df["Month_Name"], categories=MONTH_ORDER, ordered=True)
    return df


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def concentration_label(top_share: float) -> str:
    if top_share >= 0.4:
        return "High concentration"
    if top_share >= 0.25:
        return "Moderate concentration"
    return "Diversified mix"


def summarize_case(revenue: float, ebitda: float, multiple: float, label: str) -> dict:
    return {
        "Case": label,
        "Annual Revenue": revenue,
        "EBITDA": ebitda,
        "Revenue Valuation": revenue * multiple,
    }


df = load_data()

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(191, 219, 254, 0.6), transparent 28%),
            radial-gradient(circle at top right, rgba(253, 230, 138, 0.5), transparent 24%),
            linear-gradient(180deg, #f8fafc 0%, #eff6ff 45%, #f8fafc 100%);
    }
    .hero-card, .insight-card {
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }
    .hero-kicker {
        color: #1d4ed8;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .hero-title {
        color: #0f172a;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin-top: 0.25rem;
    }
    .hero-copy {
        color: #334155;
        font-size: 1rem;
        margin-top: 0.7rem;
    }
    .insight-title {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .insight-copy {
        color: #475569;
        font-size: 0.95rem;
        margin-bottom: 0;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stMetricLabel"] {
        color: #475569;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a;
    }
    div[data-testid="stMetricDelta"] {
        color: #1d4ed8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">Investor Decision Support</div>
        <div class="hero-title">Investor Insight Studio</div>
        <div class="hero-copy">
            Explore commercial traction, estimate earnings power, and pressure-test valuation assumptions
            without sending the business to a consultancy firm first.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Filters")
    countries = st.multiselect(
        "Country",
        options=sorted(df["Country"].unique()),
        default=sorted(df["Country"].unique()),
    )
    products = st.multiselect(
        "Product",
        options=sorted(df["Product"].unique()),
        default=sorted(df["Product"].unique()),
    )
    sales_people = st.multiselect(
        "Sales Person",
        options=sorted(df["Sales Person"].unique()),
        default=sorted(df["Sales Person"].unique()),
    )
    min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.header("Investor Assumptions")
    gross_margin = st.slider("Estimated Gross Margin", 0.10, 0.95, 0.58, 0.01)
    opex_ratio = st.slider("Operating Expense Ratio", 0.05, 0.70, 0.27, 0.01)
    revenue_multiple = st.slider("Revenue Multiple", 0.5, 12.0, 3.0, 0.1)
    growth_uplift = st.slider("Upside Growth Scenario", 0.00, 1.00, 0.20, 0.01)

start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

filtered_df = df[
    df["Country"].isin(countries)
    & df["Product"].isin(products)
    & df["Sales Person"].isin(sales_people)
    & df["Date"].between(start_date, end_date)
].copy()

if filtered_df.empty:
    st.warning("No records match the current filters. Adjust the filters to continue.")
    st.stop()

period_months = filtered_df["Date"].dt.to_period("M").nunique()
observed_months = max(period_months, 1)
total_revenue = filtered_df["Amount"].sum()
total_boxes = filtered_df["Boxes Shipped"].sum()
avg_order_value = filtered_df["Amount"].mean()
avg_revenue_per_box = safe_divide(total_revenue, total_boxes)
annualized_revenue = total_revenue * (12 / observed_months)
estimated_ebitda_margin = gross_margin - opex_ratio
estimated_ebitda = annualized_revenue * estimated_ebitda_margin

monthly = (
    filtered_df.assign(Month_Start=filtered_df["Date"].dt.to_period("M").dt.to_timestamp())
    .groupby("Month_Start", as_index=False)["Amount"]
    .sum()
    .sort_values("Month_Start")
)
monthly["Month_Label"] = monthly["Month_Start"].dt.strftime("%b %Y")
monthly["Previous Revenue"] = monthly["Amount"].shift(1)
monthly["Growth Rate"] = (
    (monthly["Amount"] - monthly["Previous Revenue"]) / monthly["Previous Revenue"]
)
latest_month_revenue = monthly["Amount"].iloc[-1]
previous_month_revenue = monthly["Previous Revenue"].iloc[-1] if len(monthly) > 1 else pd.NA
run_rate_growth = (
    safe_divide(latest_month_revenue - previous_month_revenue, previous_month_revenue)
    if pd.notna(previous_month_revenue)
    else 0.0
)

country_mix = (
    filtered_df.groupby("Country")["Amount"]
    .sum()
    .reset_index()
    .sort_values("Amount", ascending=False)
)
country_mix["Share"] = country_mix["Amount"] / total_revenue
top_country_share = country_mix["Share"].iloc[0]
hhi = (country_mix["Share"] ** 2).sum()

product_mix = (
    filtered_df.groupby("Product")
    .agg(
        Revenue=("Amount", "sum"),
        Boxes=("Boxes Shipped", "sum"),
    )
    .reset_index()
    .sort_values("Revenue", ascending=False)
)
product_mix["Revenue_per_Box"] = product_mix.apply(
    lambda row: safe_divide(row["Revenue"], row["Boxes"]), axis=1
)
top_product_share = product_mix["Revenue"].iloc[0] / total_revenue

sales_team = (
    filtered_df.groupby("Sales Person")
    .agg(Revenue=("Amount", "sum"), Deals=("Amount", "count"))
    .reset_index()
    .sort_values("Revenue", ascending=False)
)
sales_team["Avg Deal Size"] = sales_team["Revenue"] / sales_team["Deals"]

base_case = summarize_case(
    annualized_revenue,
    estimated_ebitda,
    revenue_multiple,
    "Base",
)
bull_case = summarize_case(
    annualized_revenue * (1 + growth_uplift),
    annualized_revenue * (1 + growth_uplift) * estimated_ebitda_margin,
    revenue_multiple + 1,
    "Bull",
)
bear_case = summarize_case(
    annualized_revenue * 0.9,
    annualized_revenue * 0.9 * max(estimated_ebitda_margin - 0.05, -0.25),
    max(revenue_multiple - 1, 0.5),
    "Bear",
)
scenario_df = pd.DataFrame([bear_case, base_case, bull_case])

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Annualized Revenue",
    format_currency(annualized_revenue),
    delta=f"{observed_months} months observed",
)
col2.metric(
    "Estimated EBITDA",
    format_currency(estimated_ebitda),
    delta=f"at {format_percent(estimated_ebitda_margin)} margin",
)
col3.metric("Avg Revenue / Box", format_currency(avg_revenue_per_box))
col4.metric(
    "Monthly Growth",
    format_percent(run_rate_growth),
    delta=(
        f"vs {monthly['Month_Label'].iloc[-2]}"
        if len(monthly) > 1
        else "single-month selection"
    ),
)

col5, col6, col7, col8 = st.columns(4)
col5.metric("Total Revenue", format_currency(total_revenue))
col6.metric("Orders", f"{len(filtered_df):,}")
col7.metric("Avg Order Value", format_currency(avg_order_value))
col8.metric("EBITDA Margin", format_percent(estimated_ebitda_margin))

insight_left, insight_mid, insight_right = st.columns(3)
insight_left.markdown(
    f"""
    <div class="insight-card">
        <div class="insight-title">Market Mix</div>
        <p class="insight-copy">
            Top country contributes {format_percent(top_country_share)} of filtered revenue,
            indicating {concentration_label(top_country_share).lower()}.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
insight_mid.markdown(
    f"""
    <div class="insight-card">
        <div class="insight-title">Product Dependence</div>
        <p class="insight-copy">
            The leading product accounts for {format_percent(top_product_share)} of revenue,
            which helps gauge how resilient the portfolio is to demand shifts.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
insight_right.markdown(
    f"""
    <div class="insight-card">
        <div class="insight-title">Portfolio Concentration</div>
        <p class="insight-copy">
            Revenue concentration index (HHI) is {hhi:.2f}. Lower values suggest a healthier spread
            across markets and less single-market exposure.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, operations_tab, valuation_tab = st.tabs(
    ["Performance", "Operations", "Valuation"]
)

with overview_tab:
    chart_col, mix_col = st.columns((1.4, 1))

    with chart_col:
        fig_revenue = px.line(
            monthly,
            x="Month_Start",
            y="Amount",
            markers=True,
            title="Monthly Revenue Trend",
            labels={"Amount": "Revenue", "Month_Start": "Month"},
        )
        fig_revenue.update_traces(line_color="#1d4ed8", marker_color="#ea580c")
        fig_revenue.update_layout(
            yaxis_tickprefix="$",
            hovermode="x unified",
            xaxis=dict(tickformat="%b %Y"),
        )
        st.plotly_chart(fig_revenue, use_container_width=True)

        if len(monthly) > 1:
            best_month = monthly.loc[monthly["Amount"].idxmax()]
            weakest_month = monthly.loc[monthly["Amount"].idxmin()]
            st.caption(
                f"Interpretation: this line now shows actual calendar months. "
                f"Best month was {best_month['Month_Label']} at {format_currency(best_month['Amount'])}, "
                f"while the weakest month was {weakest_month['Month_Label']} at {format_currency(weakest_month['Amount'])}. "
                f"The latest month, {monthly['Month_Label'].iloc[-1]}, changed by {format_percent(run_rate_growth)} "
                f"against the previous active month."
            )
        else:
            st.caption(
                "Interpretation: the current filters leave only one active month, so there is no trend yet to compare across time."
            )

    with mix_col:
        fig_country = px.pie(
            country_mix,
            names="Country",
            values="Amount",
            title="Revenue Mix by Country",
            hole=0.55,
        )
        fig_country.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_country, use_container_width=True)

    fig_product = px.bar(
        product_mix.head(10),
        x="Revenue",
        y="Product",
        orientation="h",
        title="Top Products by Revenue",
        labels={"Revenue": "Revenue", "Product": ""},
    )
    fig_product.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_tickprefix="$")
    fig_product.update_traces(marker_color="#0f766e")
    st.plotly_chart(fig_product, use_container_width=True)

with operations_tab:
    ops_col1, ops_col2 = st.columns((1.2, 1))

    with ops_col1:
        fig_efficiency = px.scatter(
            product_mix,
            x="Boxes",
            y="Revenue_per_Box",
            size="Revenue",
            color="Revenue",
            hover_name="Product",
            title="Product Efficiency Map",
            labels={
                "Boxes": "Boxes Shipped",
                "Revenue_per_Box": "Average Revenue per Box",
            },
            color_continuous_scale="Tealgrn",
        )
        fig_efficiency.update_layout(yaxis_tickprefix="$")
        st.plotly_chart(fig_efficiency, use_container_width=True)

    with ops_col2:
        fig_team = px.bar(
            sales_team.head(10),
            x="Sales Person",
            y="Revenue",
            color="Avg Deal Size",
            title="Top Sales Team Contribution",
            color_continuous_scale="Blues",
        )
        fig_team.update_layout(xaxis_tickangle=-35, yaxis_tickprefix="$")
        st.plotly_chart(fig_team, use_container_width=True)

    st.dataframe(
        product_mix[["Product", "Revenue", "Boxes", "Revenue_per_Box"]]
        .head(12)
        .style.format(
            {
                "Revenue": "${:,.0f}",
                "Boxes": "{:,.0f}",
                "Revenue_per_Box": "${:,.2f}",
            }
        ),
        use_container_width=True,
    )

with valuation_tab:
    val_col1, val_col2 = st.columns((1.15, 1))

    with val_col1:
        fig_scenario = px.bar(
            scenario_df,
            x="Case",
            y="Revenue Valuation",
            color="Case",
            title="Scenario-Based Revenue Valuation",
            text="Revenue Valuation",
        )
        fig_scenario.update_traces(texttemplate="$%{text:,.0f}")
        fig_scenario.update_layout(showlegend=False, yaxis_tickprefix="$")
        st.plotly_chart(fig_scenario, use_container_width=True)

    with val_col2:
        st.subheader("Investment Memo Snapshot")
        st.markdown(
            f"""
            - Observed period covers **{observed_months} months** and implies an annualized revenue run-rate of **{format_currency(annualized_revenue)}**.
            - Using a gross margin assumption of **{format_percent(gross_margin)}** and operating expenses of **{format_percent(opex_ratio)}**, the estimated EBITDA margin is **{format_percent(estimated_ebitda_margin)}**.
            - Geographic exposure is led by **{country_mix.iloc[0]['Country']}**, which contributes **{format_percent(top_country_share)}** of revenue.
            - Commercial upside exists if the current mix can scale without increasing concentration faster than margin expansion.
            """
        )

    st.dataframe(
        scenario_df.style.format(
            {
                "Annual Revenue": "${:,.0f}",
                "EBITDA": "${:,.0f}",
                "Revenue Valuation": "${:,.0f}",
            }
        ),
        use_container_width=True,
    )

st.caption(
    "Assumption note: this app estimates investor-style metrics from sales data only. "
    "Valuation outputs are directional and should be refined with COGS, opex, cash flow, and balance sheet data."
)
