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
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
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
    colorway=["#c8692a","#f5c97a","#e6a95a","#a05020","#6b2f1a","#f5e6d3"],
    xaxis=dict(gridcolor="rgba(200,105,42,0.15)", zerolinecolor="rgba(200,105,42,0.2)"),
    yaxis=dict(gridcolor="rgba(200,105,42,0.15)", zerolinecolor="rgba(200,105,42,0.2)"),
)


# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file) -> pd.DataFrame:
    raw = file.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        lines.append(line)

    df = pd.read_csv(io.StringIO("\n".join(lines)))
    df.columns = df.columns.str.strip().str.replace('"', '', regex=False)

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
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "Amount", "Boxes Shipped"]).copy()

    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%B")
    df["Day"]        = df["Date"].dt.day
    df["Revenue_per_Box"] = df["Amount"] / df["Boxes Shipped"].replace(0, pd.NA)
    df["Month_Name"] = pd.Categorical(df["Month_Name"], categories=MONTH_ORDER, ordered=True)
    return df


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='kicker'>Global Sales Analysis · 2022</div>
<div class='hero-title'>🍫 Chocolate Sales Dashboard</div>
<div class='hero-sub'>Commercial traction · Product efficiency · Revenue prediction</div>
<hr style='border:1px solid rgba(200,105,42,0.3); margin: 0.8rem 0 1.5rem;'>
""", unsafe_allow_html=True)

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload Chocolate Sales CSV", type=["csv"])
if uploaded is None:
    st.info("👆 Upload the Chocolate Sales CSV to begin.")
    st.stop()

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"❌ Could not load file: {e}")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔎 Filters")
    countries = st.multiselect("Country",      sorted(df["Country"].unique()),      default=sorted(df["Country"].unique()))
    products  = st.multiselect("Product",      sorted(df["Product"].unique()),      default=sorted(df["Product"].unique()))
    sp_list   = st.multiselect("Sales Person", sorted(df["Sales Person"].unique()), default=sorted(df["Sales Person"].unique()))
    st.markdown("---")
    st.markdown("### 📊 Investor Assumptions")
    gross_margin  = st.slider("Gross Margin",       0.10, 0.95, 0.58, 0.01)
    opex_ratio    = st.slider("OpEx Ratio",         0.05, 0.70, 0.27, 0.01)
    rev_multiple  = st.slider("Revenue Multiple",   0.5,  12.0, 3.0,  0.1)
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
total_revenue   = filt["Amount"].sum()
total_boxes     = filt["Boxes Shipped"].sum()
avg_rpb         = filt["Revenue_per_Box"].mean()
observed_months = max(filt["Date"].dt.to_period("M").nunique(), 1)
annualized_rev  = total_revenue * (12 / observed_months)
ebitda_margin   = gross_margin - opex_ratio
ebitda          = annualized_rev * ebitda_margin

monthly_sales = (
    filt.groupby(["Month", "Month_Name"])["Amount"]
    .sum().reset_index().sort_values("Month")
)
country_sales = filt.groupby("Country")["Amount"].sum().sort_values(ascending=False).reset_index()
top_products  = (
    filt.groupby("Product")
    .agg(Revenue=("Amount","sum"), Boxes=("Boxes Shipped","sum"), RPB=("Revenue_per_Box","mean"))
    .reset_index().sort_values("Revenue", ascending=False)
)
top_sp     = filt.groupby("Sales Person")["Amount"].sum().sort_values(ascending=False).reset_index()
efficiency = filt.groupby("Product")["Revenue_per_Box"].mean().sort_values(ascending=False).reset_index()

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Revenue",      f"${total_revenue/1e6:.2f}M")
c2.metric("Annualized Revenue", f"${annualized_rev/1e6:.2f}M", f"{observed_months} months")
c3.metric("Estimated EBITDA",   f"${ebitda/1e6:.2f}M",         f"{ebitda_margin*100:.1f}% margin")
c4.metric("Boxes Shipped",      f"{total_boxes:,}")
c5.metric("Avg Revenue / Box",  f"${avg_rpb:.2f}")

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
            monthly_sales, x="Month_Name", y="Amount", markers=True,
            title="Revenue Trend by Month",
            labels={"Month_Name":"Month","Amount":"Revenue ($)"},
        )
        fig.update_traces(line=dict(color="#c8692a", width=3), marker=dict(color="#f5c97a", size=9))
        fig.add_traces(go.Scatter(
            x=monthly_sales["Month_Name"], y=monthly_sales["Amount"],
            fill="tozeroy", fillcolor="rgba(200,105,42,0.1)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.update_layout(**PLOT_LAYOUT, xaxis_tickangle=-30)
        fig.update_layout(yaxis_tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        monthly_rank = monthly_sales.sort_values("Amount", ascending=False)
        fig2 = px.bar(
            monthly_rank, x="Month_Name", y="Amount",
            title="Monthly Sales Ranking",
            labels={"Month_Name":"Month","Amount":"Revenue ($)"},
            color="Amount", color_continuous_scale=["#6b2f1a","#c8692a","#f5c97a"],
        )
        fig2.update_layout(**PLOT_LAYOUT, xaxis_tickangle=-30, coloraxis_showscale=False)
        fig2.update_layout(yaxis_tickprefix="$")
        st.plotly_chart(fig2, use_container_width=True)

    best  = monthly_sales.loc[monthly_sales["Amount"].idxmax()]
    worst = monthly_sales.loc[monthly_sales["Amount"].idxmin()]
    i1, i2, i3 = st.columns(3)
    i1.markdown(f"""<div class='insight-card'><div class='insight-title'>Peak Month</div>
    <div class='insight-copy'><strong>{best['Month_Name']}</strong> leads at <strong>${best['Amount']/1e6:.2f}M</strong> — strong early-year momentum.</div></div>""", unsafe_allow_html=True)
    i2.markdown(f"""<div class='insight-card'><div class='insight-title'>Weakest Month</div>
    <div class='insight-copy'><strong>{worst['Month_Name']}</strong> dips to <strong>${worst['Amount']/1e6:.2f}M</strong> — a mid-quarter slowdown worth investigating.</div></div>""", unsafe_allow_html=True)
    i3.markdown(f"""<div class='insight-card'><div class='insight-title'>Pattern</div>
    <div class='insight-copy'>Cyclical trend: early-year peak → mid-year trough → summer recovery. Suggests seasonal demand dynamics.</div></div>""", unsafe_allow_html=True)

# ══ TAB 2 — Geography ════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-head'>Revenue by Country</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.bar(
            country_sales.sort_values("Amount"),
            x="Amount", y="Country", orientation="h",
            title="Sales by Country",
            labels={"Amount":"Revenue ($)","Country":""},
            color="Amount", color_continuous_scale=["#6b2f1a","#c8692a","#f5c97a"],
        )
        fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, xaxis_tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.pie(
            country_sales, names="Country", values="Amount",
            title="Revenue Distribution by Country", hole=0.52,
            color_discrete_sequence=["#c8692a","#e6a95a","#f5c97a","#a05020","#6b2f1a","#f5e6d3"],
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label",
                           marker=dict(line=dict(color="#1a0a00", width=2)))
        fig2.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    top_c = country_sales.iloc[0]
    st.markdown(f"""<div class='insight-card'><div class='insight-title'>📌 Market Insight</div>
    <div class='insight-copy'><strong>{top_c['Country']}</strong> leads at <strong>${top_c['Amount']/1e6:.2f}M</strong>
    ({top_c['Amount']/total_revenue*100:.1f}% share). Revenue is well spread across all markets —
    no single country exceeds 19%, indicating a healthy, diversified portfolio with low geographic concentration risk.
    </div></div>""", unsafe_allow_html=True)

# ══ TAB 3 — Products ═════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-head'>Product Performance</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.bar(
            top_products.head(10).sort_values("Revenue"),
            x="Revenue", y="Product", orientation="h",
            title="Top 10 Products by Revenue",
            color="Revenue", color_continuous_scale=["#6b2f1a","#c8692a","#f5c97a"],
        )
        fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, xaxis_tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.bar(
            efficiency.head(10).sort_values("Revenue_per_Box"),
            x="Revenue_per_Box", y="Product", orientation="h",
            title="Top 10 Products: Revenue per Box",
            color="Revenue_per_Box", color_continuous_scale=["#6b2f1a","#c8692a","#f5c97a"],
        )
        fig2.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, xaxis_tickprefix="$")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        top_products, x="Boxes", y="RPB", size="Revenue",
        color="Revenue", hover_name="Product",
        title="Product Efficiency Map — Volume vs Revenue per Box",
        labels={"Boxes":"Boxes Shipped","RPB":"Avg Revenue / Box ($)","Revenue":"Total Revenue"},
        color_continuous_scale=["#6b2f1a","#c8692a","#f5c97a"],
    )
    fig3.update_layout(**PLOT_LAYOUT, yaxis_tickprefix="$")
    st.plotly_chart(fig3, use_container_width=True)

    top_vol = top_products.iloc[0]
    top_eff = efficiency.iloc[0]
    i1, i2 = st.columns(2)
    i1.markdown(f"""<div class='insight-card'><div class='insight-title'>Top Revenue Product</div>
    <div class='insight-copy'><strong>{top_vol['Product']}</strong> generates the highest total revenue at <strong>${top_vol['Revenue']/1e3:.0f}K</strong>.</div></div>""", unsafe_allow_html=True)
    i2.markdown(f"""<div class='insight-card'><div class='insight-title'>Most Efficient Product</div>
    <div class='insight-copy'><strong>{top_eff['Product']}</strong> earns <strong>${top_eff['Revenue_per_Box']:.2f}/box</strong> — the highest value per unit. Prioritizing this product mix could significantly lift margins.</div></div>""", unsafe_allow_html=True)

# ══ TAB 4 — Sales Team ═══════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-head'>Salesperson Performance</div>", unsafe_allow_html=True)

    fig = px.bar(
        top_sp.head(10),
        x="Sales Person", y="Amount",
        title="Top 10 Salespersons by Revenue",
        color="Amount", color_continuous_scale=["#6b2f1a","#c8692a","#f5c97a"],
        labels={"Amount":"Revenue ($)","Sales Person":""},
    )
    fig.update_layout(**PLOT_LAYOUT, xaxis_tickangle=-30, coloraxis_showscale=False, yaxis_tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

    top_person = top_sp.iloc[0]
    spread = top_sp.head(5)["Amount"].std() / top_sp.head(5)["Amount"].mean()
    st.markdown(f"""<div class='insight-card'><div class='insight-title'>📌 Team Insight</div>
    <div class='insight-copy'>
    <strong>{top_person['Sales Person']}</strong> leads at <strong>${top_person['Amount']/1e6:.2f}M</strong>.
    Performance is tightly clustered across the top 10 (CV = {spread:.2f}), indicating a stable and balanced
    sales force with no over-reliance on any single individual.
    </div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-head'>Full Leaderboard</div>", unsafe_allow_html=True)
    full_sp = filt.groupby("Sales Person").agg(
        Revenue=("Amount","sum"),
        Orders=("Amount","count"),
        Avg_Deal=("Amount","mean"),
    ).reset_index().sort_values("Revenue", ascending=False)
    st.dataframe(
        full_sp.style.format({"Revenue":"${:,.0f}","Avg_Deal":"${:,.0f}"}),
        use_container_width=True, height=380,
    )

# ══ TAB 5 — ML Predictor ═════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-head'>Machine Learning: Revenue per Box Predictor</div>", unsafe_allow_html=True)

    # Linear regression baseline
    X_lr = filt[["Boxes Shipped"]]
    y_lr = filt["Amount"]
    lr   = LinearRegression().fit(X_lr, y_lr)
    lr_r2 = r2_score(y_lr, lr.predict(X_lr))

    c1, c2 = st.columns(2)
    c1.markdown(f"""<div class='insight-card'><div class='insight-title'>Linear Regression Baseline</div>
    <div class='insight-copy'>
    Equation: <strong>Revenue = {lr.coef_[0]:.2f} × Boxes + {lr.intercept_:.0f}</strong><br>
    R² = <strong>{lr_r2:.5f}</strong> — effectively zero. Volume alone cannot predict revenue.
    Pricing and product mix are the true drivers.
    </div></div>""", unsafe_allow_html=True)

    # Scatter: boxes vs revenue
    fig_lr = px.scatter(
        filt.sample(min(500, len(filt)), random_state=42),
        x="Boxes Shipped", y="Amount",
        title="Boxes Shipped vs Revenue (with Regression Line)",
        labels={"Boxes Shipped":"Boxes Shipped","Amount":"Revenue ($)"},
        opacity=0.4, color_discrete_sequence=["#c8692a"],
    )
    x_range = [filt["Boxes Shipped"].min(), filt["Boxes Shipped"].max()]
    y_range = [lr.predict([[x]])[0] for x in x_range]
    fig_lr.add_trace(go.Scatter(x=x_range, y=y_range, mode="lines",
                                line=dict(color="#f5c97a", width=2.5), name="Regression Line"))
    fig_lr.update_layout(**PLOT_LAYOUT, yaxis_tickprefix="$")
    st.plotly_chart(fig_lr, use_container_width=True)

    # Train Random Forest
    @st.cache_resource
    def train_rf(n_rows):
        features = ["Boxes Shipped", "Month", "Product", "Country"]
        target   = "Revenue_per_Box"
        model_df = filt[features + [target]].dropna()
        X = model_df[features]; y = model_df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pre = ColumnTransformer([
            ("num", "passthrough", ["Boxes Shipped", "Month"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Product", "Country"]),
        ])
        pipe = Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=100, random_state=42))])
        pipe.fit(X_train, y_train)
        yp = pipe.predict(X_test)
        return pipe, r2_score(y_test, yp), mean_absolute_error(y_test, yp)

    with st.spinner("Training Random Forest…"):
        rf_pipe, rf_r2, rf_mae = train_rf(len(filt))

    c2.markdown(f"""<div class='insight-card'><div class='insight-title'>Random Forest Results</div>
    <div class='insight-copy'>
    R² = <strong>{rf_r2:.4f}</strong> &nbsp;|&nbsp; MAE = <strong>${rf_mae:.2f}</strong><br>
    A massive improvement over linear regression — product type and country are the dominant revenue drivers,
    not shipment volume.
    </div></div>""", unsafe_allow_html=True)

    # Feature importance
    feat_names  = rf_pipe.named_steps["pre"].get_feature_names_out()
    importances = rf_pipe.named_steps["model"].feature_importances_
    fi_df = (
        pd.DataFrame({"Feature": feat_names, "Importance": importances})
        .sort_values("Importance", ascending=False).head(12)
    )
    fig_fi = px.bar(
        fi_df.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
        title="Top Feature Importances (Random Forest)",
        color="Importance", color_continuous_scale=["#6b2f1a","#c8692a","#f5c97a"],
    )
    fig_fi.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Live predictor
    st.markdown("<div class='section-head'>🔮 Live Revenue per Box Predictor</div>", unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    pred_boxes   = p1.number_input("Boxes Shipped", 1, 1000, 50)
    pred_product = p2.selectbox("Product", sorted(df["Product"].unique()))
    pred_country = p3.selectbox("Country", sorted(df["Country"].unique()))
    pred_month   = p4.slider("Month", 1, 12, 6)

    if st.button("🍫 Predict Revenue per Box", use_container_width=True):
        new_data = pd.DataFrame({
            "Boxes Shipped": [pred_boxes],
            "Product":       [pred_product],
            "Country":       [pred_country],
            "Month":         [pred_month],
        })
        rpb   = rf_pipe.predict(new_data)[0]
        total = rpb * pred_boxes
        st.success(f"**Revenue per Box:** ${rpb:.2f}  ·  **Estimated Total Revenue:** ${total:,.2f}")

    # Valuation scenarios
    st.markdown("<div class='section-head'>💼 Investor Valuation Scenarios</div>", unsafe_allow_html=True)
    scenarios = pd.DataFrame([
        {"Case":"Bear", "Annual Revenue": annualized_rev*0.90,           "EBITDA": annualized_rev*0.90*max(ebitda_margin-0.05,-0.25), "Valuation": annualized_rev*0.90*max(rev_multiple-1,0.5)},
        {"Case":"Base", "Annual Revenue": annualized_rev,                "EBITDA": ebitda,                                             "Valuation": annualized_rev*rev_multiple},
        {"Case":"Bull", "Annual Revenue": annualized_rev*(1+growth_uplift), "EBITDA": annualized_rev*(1+growth_uplift)*ebitda_margin,  "Valuation": annualized_rev*(1+growth_uplift)*(rev_multiple+1)},
    ])

    v1, v2 = st.columns((1.2, 1))
    with v1:
        fig_val = px.bar(
            scenarios, x="Case", y="Valuation", color="Case",
            text="Valuation", title="Scenario-Based Revenue Valuation",
            color_discrete_map={"Bear":"#6b2f1a","Base":"#c8692a","Bull":"#f5c97a"},
        )
        fig_val.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_val.update_layout(**PLOT_LAYOUT, showlegend=False, yaxis_tickprefix="$")
        st.plotly_chart(fig_val, use_container_width=True)

    with v2:
        st.dataframe(
            scenarios.style.format({"Annual Revenue":"${:,.0f}","EBITDA":"${:,.0f}","Valuation":"${:,.0f}"}),
            use_container_width=True,
        )
        st.markdown(f"""<div class='insight-card'><div class='insight-title'>Investment Memo</div>
        <div class='insight-copy'>
        {observed_months}-month run-rate implies annualized revenue of <strong>${annualized_rev/1e6:.2f}M</strong>.
        At a {ebitda_margin*100:.1f}% EBITDA margin and {rev_multiple}× multiple, base-case valuation is
        <strong>${annualized_rev*rev_multiple/1e6:.2f}M</strong>.
        The efficiency gap between top and bottom products represents a clear margin expansion opportunity.
        </div></div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border:1px solid rgba(200,105,42,0.3); margin-top:2rem;'>
<p style='text-align:center; color:#6b4020; font-size:0.82rem;'>
🍫 Chocolate Sales Dashboard · 2022 · Built with Streamlit
</p>""", unsafe_allow_html=True)
