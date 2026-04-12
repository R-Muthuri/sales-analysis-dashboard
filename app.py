"""
Chocolate Sales Analysis Dashboard
====================================
Drop-in replacement for the Jupyter notebook analysis.
Robust CSV loader handles BOM, Windows line-endings, and quoted-row exports.
"""

import calendar
import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chocolate Sales Dashboard",
    page_icon="🍫",
    layout="wide",
)

MONTH_ORDER = list(calendar.month_name[1:])
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "clean_sales_data.csv"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Cormorant Garamond', serif; }

.stApp {
    background: linear-gradient(160deg, #1a0a00 0%, #2d1206 40%, #1a0a00 100%);
    color: #f5e6d3;
}

section[data-testid="stSidebar"] {
    background: rgba(30, 10, 0, 0.95) !important;
    border-right: 1px solid rgba(200, 105, 42, 0.25);
}
section[data-testid="stSidebar"] * { color: #f5e6d3 !important; }

button[data-baseweb="tab"] {
    color: #c8a07a !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #f5c97a !important;
    border-bottom: 2px solid #c8692a !important;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(200,105,42,0.3);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
div[data-testid="stMetricLabel"] p { color: #c8a07a !important; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }
div[data-testid="stMetricValue"] { color: #f5c97a !important; font-family: 'Cormorant Garamond', serif; font-size: 1.9rem !important; }
div[data-testid="stMetricDelta"] { color: #a3d9a5 !important; }

.kicker { font-size:0.75rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#c8692a; }
.hero-title { font-family:'Cormorant Garamond',serif; font-size:2.6rem; font-weight:800; color:#f5e6d3; line-height:1.1; margin:0.2rem 0 0.5rem; }
.hero-sub { color:#c8a07a; font-size:0.97rem; }

.insight-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(200,105,42,0.25);
    border-left: 4px solid #c8692a;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.1rem;
    margin-top: 0.5rem;
}
.insight-title { font-weight:700; color:#f5c97a; font-size:0.95rem; margin-bottom:0.3rem; }
.insight-copy { color:#c8a07a; font-size:0.88rem; line-height:1.5; }

.section-head {
    font-family:'Cormorant Garamond',serif;
    font-size:1.55rem;
    color:#f5c97a;
    border-bottom: 2px solid rgba(200,105,42,0.4);
    padding-bottom:0.3rem;
    margin: 1.8rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(26,10,0,0)",
    plot_bgcolor="rgba(255,255,255,0.04)",
    font=dict(family="DM Sans", color="#f5e6d3"),
    title_font=dict(family="Cormorant Garamond", color="#f5c97a", size=18),
    colorway=["#c8692a", "#f5c97a", "#e6a95a", "#a05020", "#6b2f1a", "#f5e6d3"],
    xaxis=dict(gridcolor="rgba(200,105,42,0.15)", zerolinecolor="rgba(200,105,42,0.2)"),
    yaxis=dict(gridcolor="rgba(200,105,42,0.15)", zerolinecolor="rgba(200,105,42,0.2)"),
)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def money_m(value: float) -> str:
    return f"${value / 1e6:.2f}M"


def concentration_note(share: float) -> str:
    if share >= 0.35:
        return "high geographic concentration"
    if share >= 0.20:
        return "moderate geographic concentration"
    return "healthy geographic diversification"


def team_balance_note(cv: float) -> str:
    if cv < 0.10:
        return "very balanced"
    if cv < 0.25:
        return "fairly balanced"
    return "top-heavy"


def lr_quality_note(score: float) -> str:
    if score < 0:
        return "performs worse than a simple average benchmark"
    if score < 0.1:
        return "has almost no explanatory power"
    if score < 0.4:
        return "captures only a weak relationship"
    return "shows a meaningful directional signal"


def trend_note(growth: float) -> str:
    if growth > 0.05:
        return "accelerated strongly"
    if growth > 0:
        return "improved modestly"
    if growth > -0.05:
        return "softened slightly"
    return "declined materially"


# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(raw_source: bytes) -> pd.DataFrame:
    raw = raw_source.decode("utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        lines.append(line)

    df = pd.read_csv(io.StringIO("\n".join(lines)))
    df.columns = df.columns.str.strip().str.replace('"', "", regex=False)

    expected = ["Sales Person", "Country", "Product", "Date", "Amount", "Boxes Shipped"]
    col_map = {}
    for exp in expected:
        for actual in df.columns:
            if actual.lower().strip() == exp.lower() and actual != exp:
                col_map[actual] = exp
    df = df.rename(columns=col_map)

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    df["Amount"] = pd.to_numeric(
        df["Amount"].astype(str).str.replace(r"[\$,]", "", regex=True), errors="coerce"
    )
    df["Boxes Shipped"] = pd.to_numeric(df["Boxes Shipped"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Amount", "Boxes Shipped"]).copy()

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%B")
    df["Day"] = df["Date"].dt.day
    df["Revenue_per_Box"] = df["Amount"] / df["Boxes Shipped"].replace(0, pd.NA)
    df["Month_Name"] = pd.Categorical(df["Month_Name"], categories=MONTH_ORDER, ordered=True)
    return df


@st.cache_resource
def train_rf(model_df: pd.DataFrame):
    features = ["Boxes Shipped", "Month", "Product", "Country"]
    target = "Revenue_per_Box"

    X = model_df[features]
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pre = ColumnTransformer([
        ("num", "passthrough", ["Boxes Shipped", "Month"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Product", "Country"]),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    yp = pipe.predict(X_test)
    return pipe, r2_score(y_test, yp), mean_absolute_error(y_test, yp)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='kicker'>Global Sales Analysis · 2022-2024</div>
<div class='hero-title'>🍫 Chocolate Sales Dashboard</div>
<div class='hero-sub'>Commercial traction · Product efficiency · Revenue prediction</div>
<hr style='border:1px solid rgba(200,105,42,0.3); margin: 0.8rem 0 1.5rem;'>
""", unsafe_allow_html=True)

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload Chocolate Sales CSV", type=["csv"])
if uploaded is None:
    st.caption(f"Using bundled dataset: `{DEFAULT_DATA_PATH.name}`. Upload another CSV to replace it.")
    raw_source = DEFAULT_DATA_PATH.read_bytes()
else:
    raw_source = uploaded.getvalue()

try:
    df = load_data(raw_source)
except Exception as e:
    st.error(f"❌ Could not load file: {e}")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔎 Filters")
    countries = st.multiselect("Country", sorted(df["Country"].unique()), default=sorted(df["Country"].unique()))
    products = st.multiselect("Product", sorted(df["Product"].unique()), default=sorted(df["Product"].unique()))
    sp_list = st.multiselect("Sales Person", sorted(df["Sales Person"].unique()), default=sorted(df["Sales Person"].unique()))
    st.markdown("---")
    st.markdown("### 📊 Investor Assumptions")
    gross_margin = st.slider("Gross Margin", 0.10, 0.95, 0.58, 0.01)
    opex_ratio = st.slider("OpEx Ratio", 0.05, 0.70, 0.27, 0.01)
    rev_multiple = st.slider("Revenue Multiple", 0.5, 12.0, 3.0, 0.1)
    growth_uplift = st.slider("Bull Growth Uplift", 0.00, 1.00, 0.20, 0.01)

filt = df[
    df["Country"].isin(countries) &
    df["Product"].isin(products) &
    df["Sales Person"].isin(sp_list)
].copy()

if filt.empty:
    st.warning("No records match the current filters.")
    st.stop()

# ── Derived metrics ───────────────────────────────────────────────────────────
total_revenue = filt["Amount"].sum()
total_boxes = filt["Boxes Shipped"].sum()
avg_rpb = safe_divide(total_revenue, total_boxes)
observed_months = max(filt["Date"].dt.to_period("M").nunique(), 1)
annualized_rev = total_revenue * (12 / observed_months)
ebitda_margin = gross_margin - opex_ratio
ebitda = annualized_rev * ebitda_margin

monthly_sales = (
    filt.assign(Month_Start=filt["Date"].dt.to_period("M").dt.to_timestamp())
    .groupby("Month_Start", as_index=False)["Amount"]
    .sum()
    .sort_values("Month_Start")
)
monthly_sales["Month_Label"] = monthly_sales["Month_Start"].dt.strftime("%b %Y")
monthly_sales["Previous_Amount"] = monthly_sales["Amount"].shift(1)
monthly_sales["Growth"] = (
    (monthly_sales["Amount"] - monthly_sales["Previous_Amount"]) / monthly_sales["Previous_Amount"]
)

country_sales = (
    filt.groupby("Country")["Amount"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
country_sales["Share"] = country_sales["Amount"] / total_revenue
hhi = (country_sales["Share"] ** 2).sum()

top_products = (
    filt.groupby("Product")
    .agg(Revenue=("Amount", "sum"), Boxes=("Boxes Shipped", "sum"))
    .reset_index()
    .sort_values("Revenue", ascending=False)
)
top_products["RPB"] = top_products.apply(
    lambda row: safe_divide(row["Revenue"], row["Boxes"]), axis=1
)
efficiency = (
    top_products[["Product", "RPB"]]
    .rename(columns={"RPB": "Revenue_per_Box"})
    .sort_values("Revenue_per_Box", ascending=False)
    .reset_index(drop=True)
)

top_sp = (
    filt.groupby("Sales Person")["Amount"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

latest_growth = (
    monthly_sales["Growth"].iloc[-1]
    if len(monthly_sales) > 1 and pd.notna(monthly_sales["Growth"].iloc[-1])
    else 0.0
)
top_country = country_sales.iloc[0]
best_month = monthly_sales.loc[monthly_sales["Amount"].idxmax()]
worst_month = monthly_sales.loc[monthly_sales["Amount"].idxmin()]
top_vol = top_products.iloc[0]
top_eff = efficiency.iloc[0]
top_person = top_sp.iloc[0]
spread = safe_divide(top_sp.head(5)["Amount"].std(), top_sp.head(5)["Amount"].mean())

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Revenue", money_m(total_revenue))
c2.metric("Annualized Revenue", money_m(annualized_rev), f"{observed_months} months")
c3.metric("Estimated EBITDA", money_m(ebitda), f"{ebitda_margin * 100:.1f}% margin")
c4.metric("Boxes Shipped", f"{total_boxes:,}")
c5.metric("Avg Revenue / Box", f"${avg_rpb:.2f}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Monthly Trend",
    "🌍 Geography",
    "📦 Products",
    "👥 Sales Team",
    "🤖 ML Predictor",
])

# ══ TAB 1 — Monthly Trend ════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-head'>Monthly Sales Analysis</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.line(
            monthly_sales,
            x="Month_Start",
            y="Amount",
            markers=True,
            title="Revenue Trend by Month",
            labels={"Month_Start": "Month", "Amount": "Revenue ($)"},
        )
        fig.update_traces(line=dict(color="#c8692a", width=3), marker=dict(color="#f5c97a", size=9))
        fig.add_traces(go.Scatter(
            x=monthly_sales["Month_Start"],
            y=monthly_sales["Amount"],
            fill="tozeroy",
            fillcolor="rgba(200,105,42,0.1)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.update_layout(**PLOT_LAYOUT, xaxis_tickangle=-30)
        fig.update_layout(yaxis_tickprefix="$", xaxis=dict(tickformat="%b %Y"))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        monthly_rank = monthly_sales.sort_values("Amount", ascending=False)
        fig2 = px.bar(
            monthly_rank,
            x="Month_Label",
            y="Amount",
            title="Monthly Sales Ranking",
            labels={"Month_Label": "Month", "Amount": "Revenue ($)"},
            color="Amount",
            color_continuous_scale=["#6b2f1a", "#c8692a", "#f5c97a"],
        )
        fig2.update_layout(**PLOT_LAYOUT, xaxis_tickangle=-30, coloraxis_showscale=False)
        fig2.update_layout(yaxis_tickprefix="$")
        st.plotly_chart(fig2, use_container_width=True)

    i1, i2, i3 = st.columns(3)
    i1.markdown(f"""<div class='insight-card'><div class='insight-title'>Peak Month</div>
    <div class='insight-copy'><strong>{best_month['Month_Label']}</strong> leads at <strong>{money_m(best_month['Amount'])}</strong>, making it the strongest calendar month in the filtered dataset.</div></div>""", unsafe_allow_html=True)
    i2.markdown(f"""<div class='insight-card'><div class='insight-title'>Weakest Month</div>
    <div class='insight-copy'><strong>{worst_month['Month_Label']}</strong> falls to <strong>{money_m(worst_month['Amount'])}</strong>, which marks the low point of the period currently in view.</div></div>""", unsafe_allow_html=True)
    if len(monthly_sales) > 1:
        previous_label = monthly_sales["Month_Label"].iloc[-2]
        pattern_copy = (
            f"The latest active month <strong>{trend_note(latest_growth)}</strong> by "
            f"<strong>{latest_growth * 100:.1f}%</strong> versus <strong>{previous_label}</strong>. "
            f"Use this chart as a true month-by-month timeline, not a pooled January-vs-February comparison."
        )
    else:
        pattern_copy = "Only one active month is available under the current filters, so a trend pattern cannot yet be inferred."
    i3.markdown(f"""<div class='insight-card'><div class='insight-title'>Pattern</div>
    <div class='insight-copy'>{pattern_copy}</div></div>""", unsafe_allow_html=True)

# ══ TAB 2 — Geography ════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-head'>Revenue by Country</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.bar(
            country_sales.sort_values("Amount"),
            x="Amount",
            y="Country",
            orientation="h",
            title="Sales by Country",
            labels={"Amount": "Revenue ($)", "Country": ""},
            color="Amount",
            color_continuous_scale=["#6b2f1a", "#c8692a", "#f5c97a"],
        )
        fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, xaxis_tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.pie(
            country_sales,
            names="Country",
            values="Amount",
            title="Revenue Distribution by Country",
            hole=0.52,
            color_discrete_sequence=["#c8692a", "#e6a95a", "#f5c97a", "#a05020", "#6b2f1a", "#f5e6d3"],
        )
        fig2.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#1a0a00", width=2)),
        )
        fig2.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""<div class='insight-card'><div class='insight-title'>📌 Market Insight</div>
    <div class='insight-copy'><strong>{top_country['Country']}</strong> leads at <strong>{money_m(top_country['Amount'])}</strong>
    with a <strong>{top_country['Share'] * 100:.1f}%</strong> share of filtered revenue. The current footprint shows
    <strong>{concentration_note(top_country['Share'])}</strong>, with an HHI of <strong>{hhi:.2f}</strong>.
    </div></div>""", unsafe_allow_html=True)

# ══ TAB 3 — Products ═════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-head'>Product Performance</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.bar(
            top_products.head(10).sort_values("Revenue"),
            x="Revenue",
            y="Product",
            orientation="h",
            title="Top 10 Products by Revenue",
            color="Revenue",
            color_continuous_scale=["#6b2f1a", "#c8692a", "#f5c97a"],
        )
        fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, xaxis_tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.bar(
            efficiency.head(10).sort_values("Revenue_per_Box"),
            x="Revenue_per_Box",
            y="Product",
            orientation="h",
            title="Top 10 Products: Revenue per Box",
            color="Revenue_per_Box",
            color_continuous_scale=["#6b2f1a", "#c8692a", "#f5c97a"],
        )
        fig2.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, xaxis_tickprefix="$")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        top_products,
        x="Boxes",
        y="RPB",
        size="Revenue",
        color="Revenue",
        hover_name="Product",
        title="Product Efficiency Map — Volume vs Revenue per Box",
        labels={"Boxes": "Boxes Shipped", "RPB": "Avg Revenue / Box ($)", "Revenue": "Total Revenue"},
        color_continuous_scale=["#6b2f1a", "#c8692a", "#f5c97a"],
    )
    fig3.update_layout(**PLOT_LAYOUT, yaxis_tickprefix="$")
    st.plotly_chart(fig3, use_container_width=True)

    i1, i2 = st.columns(2)
    i1.markdown(f"""<div class='insight-card'><div class='insight-title'>Top Revenue Product</div>
    <div class='insight-copy'><strong>{top_vol['Product']}</strong> generates the highest total revenue at <strong>${top_vol['Revenue'] / 1e3:.0f}K</strong>.</div></div>""", unsafe_allow_html=True)
    i2.markdown(f"""<div class='insight-card'><div class='insight-title'>Most Efficient Product</div>
    <div class='insight-copy'><strong>{top_eff['Product']}</strong> earns <strong>${top_eff['Revenue_per_Box']:.2f}/box</strong> based on weighted product revenue and volume, making it the strongest unit-economics product in the current filter.</div></div>""", unsafe_allow_html=True)

# ══ TAB 4 — Sales Team ═══════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-head'>Salesperson Performance</div>", unsafe_allow_html=True)

    fig = px.bar(
        top_sp.head(10),
        x="Sales Person",
        y="Amount",
        title="Top 10 Salespersons by Revenue",
        color="Amount",
        color_continuous_scale=["#6b2f1a", "#c8692a", "#f5c97a"],
        labels={"Amount": "Revenue ($)", "Sales Person": ""},
    )
    fig.update_layout(**PLOT_LAYOUT, xaxis_tickangle=-30, coloraxis_showscale=False, yaxis_tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""<div class='insight-card'><div class='insight-title'>📌 Team Insight</div>
    <div class='insight-copy'>
    <strong>{top_person['Sales Person']}</strong> leads at <strong>{money_m(top_person['Amount'])}</strong>.
    The top-five revenue spread has a coefficient of variation of <strong>{spread:.2f}</strong>, which suggests a
    <strong>{team_balance_note(spread)}</strong> sales team rather than one dominated by a single seller.
    </div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-head'>Full Leaderboard</div>", unsafe_allow_html=True)
    full_sp = filt.groupby("Sales Person").agg(
        Revenue=("Amount", "sum"),
        Orders=("Amount", "count"),
        Avg_Deal=("Amount", "mean"),
    ).reset_index().sort_values("Revenue", ascending=False)
    st.dataframe(
        full_sp.style.format({"Revenue": "${:,.0f}", "Avg_Deal": "${:,.0f}"}),
        use_container_width=True,
        height=380,
    )

# ══ TAB 5 — ML Predictor ═════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-head'>Machine Learning: Revenue per Box Predictor</div>", unsafe_allow_html=True)

    # Linear regression baseline
    X_lr = filt[["Boxes Shipped"]]
    y_lr = filt["Amount"]
    X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(
        X_lr, y_lr, test_size=0.2, random_state=42
    )
    lr = LinearRegression().fit(X_lr_train, y_lr_train)
    lr_pred = lr.predict(X_lr_test)
    lr_r2 = r2_score(y_lr_test, lr_pred)
    lr_mae = mean_absolute_error(y_lr_test, lr_pred)

    c1, c2 = st.columns(2)
    c1.markdown(f"""<div class='insight-card'><div class='insight-title'>Linear Regression Baseline</div>
    <div class='insight-copy'>
    Equation: <strong>Revenue = {lr.coef_[0]:.2f} × Boxes + {lr.intercept_:.0f}</strong><br>
    Holdout R² = <strong>{lr_r2:.5f}</strong> &nbsp;|&nbsp; MAE = <strong>${lr_mae:,.2f}</strong><br>
    This baseline <strong>{lr_quality_note(lr_r2)}</strong>, which means shipment volume alone is not enough to explain revenue.
    </div></div>""", unsafe_allow_html=True)

    fig_lr = px.scatter(
        filt.sample(min(500, len(filt)), random_state=42),
        x="Boxes Shipped",
        y="Amount",
        title="Boxes Shipped vs Revenue (with Regression Line)",
        labels={"Boxes Shipped": "Boxes Shipped", "Amount": "Revenue ($)"},
        opacity=0.4,
        color_discrete_sequence=["#c8692a"],
    )
    x_range = [filt["Boxes Shipped"].min(), filt["Boxes Shipped"].max()]
    line_frame = pd.DataFrame({"Boxes Shipped": x_range})
    y_range = lr.predict(line_frame)
    fig_lr.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode="lines",
        line=dict(color="#f5c97a", width=2.5),
        name="Regression Line",
    ))
    fig_lr.update_layout(**PLOT_LAYOUT, yaxis_tickprefix="$")
    st.plotly_chart(fig_lr, use_container_width=True)

    features = ["Boxes Shipped", "Month", "Product", "Country"]
    target = "Revenue_per_Box"
    model_df = filt[features + [target]].dropna().copy()

    if len(model_df) < 50 or model_df["Product"].nunique() < 2 or model_df["Country"].nunique() < 2:
        st.warning("Broaden the filters a bit to train the Random Forest reliably.")
    else:
        with st.spinner("Training Random Forest…"):
            rf_pipe, rf_r2, rf_mae = train_rf(model_df)

        c2.markdown(f"""<div class='insight-card'><div class='insight-title'>Random Forest Results</div>
        <div class='insight-copy'>
        Holdout R² = <strong>{rf_r2:.4f}</strong> &nbsp;|&nbsp; MAE = <strong>${rf_mae:.2f}</strong><br>
        Product, country, and seasonality explain revenue per box far better than raw shipment volume alone.
        </div></div>""", unsafe_allow_html=True)

        feat_names = rf_pipe.named_steps["pre"].get_feature_names_out()
        importances = rf_pipe.named_steps["model"].feature_importances_
        fi_df = (
            pd.DataFrame({"Feature": feat_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(12)
        )
        fig_fi = px.bar(
            fi_df.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Feature Importances (Random Forest)",
            color="Importance",
            color_continuous_scale=["#6b2f1a", "#c8692a", "#f5c97a"],
        )
        fig_fi.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("<div class='section-head'>🔮 Live Revenue per Box Predictor</div>", unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        pred_boxes = p1.number_input("Boxes Shipped", 1, 1000, 50)
        pred_product = p2.selectbox("Product", sorted(df["Product"].unique()))
        pred_country = p3.selectbox("Country", sorted(df["Country"].unique()))
        pred_month = p4.slider("Month", 1, 12, 6)

        if st.button("🍫 Predict Revenue per Box", use_container_width=True):
            new_data = pd.DataFrame({
                "Boxes Shipped": [pred_boxes],
                "Product": [pred_product],
                "Country": [pred_country],
                "Month": [pred_month],
            })
            rpb = rf_pipe.predict(new_data)[0]
            total = rpb * pred_boxes
            st.success(f"**Revenue per Box:** ${rpb:.2f}  ·  **Estimated Total Revenue:** ${total:,.2f}")

        st.markdown("<div class='section-head'>💼 Investor Valuation Scenarios</div>", unsafe_allow_html=True)
        scenarios = pd.DataFrame([
            {"Case": "Bear", "Annual Revenue": annualized_rev * 0.90, "EBITDA": annualized_rev * 0.90 * max(ebitda_margin - 0.05, -0.25), "Valuation": annualized_rev * 0.90 * max(rev_multiple - 1, 0.5)},
            {"Case": "Base", "Annual Revenue": annualized_rev, "EBITDA": ebitda, "Valuation": annualized_rev * rev_multiple},
            {"Case": "Bull", "Annual Revenue": annualized_rev * (1 + growth_uplift), "EBITDA": annualized_rev * (1 + growth_uplift) * ebitda_margin, "Valuation": annualized_rev * (1 + growth_uplift) * (rev_multiple + 1)},
        ])

        v1, v2 = st.columns((1.2, 1))
        with v1:
            fig_val = px.bar(
                scenarios,
                x="Case",
                y="Valuation",
                color="Case",
                text="Valuation",
                title="Scenario-Based Revenue Valuation",
                color_discrete_map={"Bear": "#6b2f1a", "Base": "#c8692a", "Bull": "#f5c97a"},
            )
            fig_val.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig_val.update_layout(**PLOT_LAYOUT, showlegend=False, yaxis_tickprefix="$")
            st.plotly_chart(fig_val, use_container_width=True)

        with v2:
            st.dataframe(
                scenarios.style.format({"Annual Revenue": "${:,.0f}", "EBITDA": "${:,.0f}", "Valuation": "${:,.0f}"}),
                use_container_width=True,
            )
            st.markdown(f"""<div class='insight-card'><div class='insight-title'>Investment Memo</div>
            <div class='insight-copy'>
            {observed_months}-month run-rate implies annualized revenue of <strong>{money_m(annualized_rev)}</strong>.
            At a {ebitda_margin * 100:.1f}% EBITDA margin and {rev_multiple}× multiple, base-case valuation is
            <strong>{money_m(annualized_rev * rev_multiple)}</strong>.
            The current top country share is {top_country['Share'] * 100:.1f}% and the strongest product yields
            ${top_eff['Revenue_per_Box']:.2f} per box, showing where margin expansion could come from.
            </div></div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border:1px solid rgba(200,105,42,0.3); margin-top:2rem;'>
<p style='text-align:center; color:#6b4020; font-size:0.82rem;'>
🍫 Chocolate Sales Dashboard · 2022-2024 · Built with Streamlit
</p>""", unsafe_allow_html=True)
