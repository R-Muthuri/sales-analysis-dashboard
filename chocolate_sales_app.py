import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chocolate Sales Dashboard",
    page_icon="🍫",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}
.metric-card {
    background: linear-gradient(135deg, #3d1a0e 0%, #6b2f1a 100%);
    border-radius: 12px;
    padding: 20px 24px;
    color: #f5e6d3;
    margin-bottom: 8px;
}
.metric-card .label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    opacity: 0.75;
    margin-bottom: 4px;
}
.metric-card .value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f5c97a;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #3d1a0e;
    border-bottom: 3px solid #c8692a;
    padding-bottom: 6px;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.insight-box {
    background: #fdf6ee;
    border-left: 4px solid #c8692a;
    padding: 14px 18px;
    border-radius: 0 8px 8px 0;
    color: #3d1a0e;
    font-size: 0.92rem;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib theme ──────────────────────────────────────────────────────────
CHOCO_PALETTE = ["#6b2f1a", "#c8692a", "#e6a95a", "#f5c97a", "#f5e6d3", "#3d1a0e"]
plt.rcParams.update({
    "axes.facecolor": "#fdf6ee",
    "figure.facecolor": "#fdf6ee",
    "axes.edgecolor": "#c8692a",
    "axes.labelcolor": "#3d1a0e",
    "xtick.color": "#3d1a0e",
    "ytick.color": "#3d1a0e",
    "text.color": "#3d1a0e",
    "axes.titlecolor": "#3d1a0e",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "font.family": "serif",
})

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-family:Playfair Display,serif; font-size:2.6rem; color:#3d1a0e; margin-bottom:0;'>
    🍫 Chocolate Sales Dashboard
</h1>
<p style='color:#6b2f1a; font-size:1rem; margin-top:4px;'>
    Global Sales Analysis · 2022
</p>
<hr style='border:1px solid #c8692a; margin-bottom:1.5rem;'>
""", unsafe_allow_html=True)

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your Chocolate Sales CSV file",
    type=["csv"],
    help="Upload the 'Chocolate Sales.csv' file to load the dashboard."
)

if uploaded_file is None:
    st.info("👆 Please upload the Chocolate Sales CSV file to get started.")
    st.stop()

# ── Load & Clean Data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Normalize column names: strip whitespace and fix casing
    df.columns = df.columns.str.strip()
    # Flexible column rename — match regardless of case
    col_map = {c: c for c in df.columns}
    expected = ['Sales Person', 'Country', 'Product', 'Date', 'Amount', 'Boxes Shipped']
    for exp in expected:
        for actual in df.columns:
            if actual.lower() == exp.lower() and actual != exp:
                col_map[actual] = exp
    df = df.rename(columns=col_map)
    df['Amount'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.drop_duplicates()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%B')
    df['Day'] = df['Date'].dt.day
    df['Revenue_per_Box'] = df['Amount'] / df['Boxes Shipped']
    return df

try:
    df = load_data(uploaded_file)
except KeyError as e:
    st.error(f"Column not found: {e}. Please check your CSV has these columns: Sales Person, Country, Product, Date, Amount, Boxes Shipped")
    st.stop()
except Exception as e:
    st.error(f"Could not load file: {e}")
    st.stop()

# ── Sidebar Filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔎 Filters")
    countries = st.multiselect(
        "Country",
        options=sorted(df['Country'].unique()),
        default=sorted(df['Country'].unique())
    )
    products = st.multiselect(
        "Product",
        options=sorted(df['Product'].unique()),
        default=sorted(df['Product'].unique())
    )
    st.markdown("---")
    st.markdown("*Use filters to drill down into specific markets or products.*")

filtered = df[df['Country'].isin(countries) & df['Product'].isin(products)]

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Total Revenue</div>
        <div class="value">${filtered['Amount'].sum()/1e6:.2f}M</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Total Transactions</div>
        <div class="value">{len(filtered):,}</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Boxes Shipped</div>
        <div class="value">{filtered['Boxes Shipped'].sum():,}</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Avg Revenue / Box</div>
        <div class="value">${filtered['Revenue_per_Box'].mean():.2f}</div>
    </div>""", unsafe_allow_html=True)

# ── Monthly Trend ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Monthly Sales Trend</div>', unsafe_allow_html=True)

monthly_sales = (
    filtered.groupby(['Month', 'Month_Name'])['Amount']
    .sum().reset_index().sort_values('Month')
)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(monthly_sales['Month_Name'], monthly_sales['Amount'] / 1e6,
            marker='o', color=CHOCO_PALETTE[0], linewidth=2.5, markersize=7,
            markerfacecolor=CHOCO_PALETTE[2])
    ax.fill_between(range(len(monthly_sales)), monthly_sales['Amount'] / 1e6,
                    alpha=0.12, color=CHOCO_PALETTE[0])
    ax.set_title("Revenue Trend by Month")
    ax.set_xlabel("Month"); ax.set_ylabel("Revenue (M)")
    ax.set_xticks(range(len(monthly_sales)))
    ax.set_xticklabels(monthly_sales['Month_Name'], rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col2:
    monthly_rank = (
        filtered.groupby('Month_Name')['Amount']
        .sum().sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(monthly_rank.index, monthly_rank.values / 1e6,
                  color=CHOCO_PALETTE[:len(monthly_rank)])
    ax.set_title("Monthly Sales Ranking")
    ax.set_xlabel("Month"); ax.set_ylabel("Revenue (M)")
    ax.set_xticklabels(monthly_rank.index, rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

st.markdown("""<div class="insight-box">
📌 <strong>Insight:</strong> Sales peak in <strong>January</strong> (~2.87M) before declining to a trough in April (~2.16M),
then recover through June–July. This cyclical pattern suggests early-year momentum followed by a mid-quarter slowdown.
</div>""", unsafe_allow_html=True)

# ── Geography ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Revenue by Country</div>', unsafe_allow_html=True)

country_sales = filtered.groupby('Country')['Amount'].sum().sort_values(ascending=False)

col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(country_sales.index[::-1], country_sales.values[::-1] / 1e6,
            color=CHOCO_PALETTE[0])
    ax.set_title("Sales by Country"); ax.set_xlabel("Revenue (M)")
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col4:
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        country_sales.values,
        labels=country_sales.index,
        autopct='%1.1f%%',
        colors=CHOCO_PALETTE,
        startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("Revenue Distribution by Country")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

st.markdown("""<div class="insight-box">
📌 <strong>Insight:</strong> <strong>Australia</strong> leads at ~3.65M, followed closely by UK, India, and USA.
Revenue is well diversified — no single market exceeds 19% of total sales.
</div>""", unsafe_allow_html=True)

# ── Products ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Product Performance</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    top_products = (
        filtered.groupby('Product')['Amount'].sum()
        .sort_values(ascending=False).head(10)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(top_products.index[::-1], top_products.values[::-1] / 1e3,
            color=CHOCO_PALETTE[1])
    ax.set_title("Top 10 Products by Revenue"); ax.set_xlabel("Revenue (K)")
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col6:
    efficiency = (
        filtered.groupby('Product')['Revenue_per_Box'].mean()
        .sort_values(ascending=False).head(10)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(efficiency.index[::-1], efficiency.values[::-1],
            color=CHOCO_PALETTE[2])
    ax.set_title("Top 10 Products: Revenue per Box"); ax.set_xlabel("$/Box")
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

st.markdown("""<div class="insight-box">
📌 <strong>Insight:</strong> <strong>White Choc</strong> leads efficiency at ~$246/box — nearly 4× more efficient than the lowest-ranked products.
High-volume products don't always mean high efficiency; a product mix strategy focusing on value-dense items could significantly boost margins.
</div>""", unsafe_allow_html=True)

# ── Salesperson Performance ───────────────────────────────────────────────────
st.markdown('<div class="section-title">Salesperson Performance</div>', unsafe_allow_html=True)

top_sp = (
    filtered.groupby('Sales Person')['Amount'].sum()
    .sort_values(ascending=False).head(10)
)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(top_sp.index, top_sp.values / 1e3,
       color=CHOCO_PALETTE[0])
ax.set_title("Top 10 Salespersons by Revenue")
ax.set_xlabel("Sales Person"); ax.set_ylabel("Revenue (K)")
ax.set_xticklabels(top_sp.index, rotation=35, ha='right', fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
st.pyplot(fig); plt.close()

st.markdown("""<div class="insight-box">
📌 <strong>Insight:</strong> <strong>Ches Bonnell</strong> leads at ~1.02M, but performance is tightly clustered among the top 10,
suggesting a stable and balanced sales team with no single over-reliance.
</div>""", unsafe_allow_html=True)

# ── Volume vs Revenue ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Volume vs Revenue Analysis</div>', unsafe_allow_html=True)

col7, col8 = st.columns([2, 1])

with col7:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.regplot(data=filtered, x='Boxes Shipped', y='Amount',
                ax=ax,
                scatter_kws={'alpha': 0.25, 'color': CHOCO_PALETTE[0], 's': 15},
                line_kws={'color': CHOCO_PALETTE[2], 'linewidth': 2})
    ax.set_title("Boxes Shipped vs Revenue")
    ax.set_xlabel("Boxes Shipped"); ax.set_ylabel("Revenue ($)")
    ax.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col8:
    X_lr = filtered[['Boxes Shipped']]
    y_lr = filtered['Amount']
    lr = LinearRegression().fit(X_lr, y_lr)
    r2 = r2_score(y_lr, lr.predict(X_lr))
    st.markdown(f"""
    <div class="metric-card" style="margin-top:30px;">
        <div class="label">Linear Regression</div>
        <div class="value" style="font-size:1.1rem;">Revenue = {lr.coef_[0]:.2f} × Boxes + {lr.intercept_:.0f}</div>
    </div>
    <div class="metric-card">
        <div class="label">R² Score</div>
        <div class="value">{r2:.5f}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""<div class="insight-box">
    📌 R² ≈ 0 means <strong>volume has virtually no impact on revenue</strong>.
    Pricing and product mix are the true revenue drivers.
    </div>""", unsafe_allow_html=True)

# ── Random Forest Model ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">🤖 Machine Learning: Revenue Predictor</div>', unsafe_allow_html=True)

@st.cache_resource
def train_rf(data):
    features = ['Boxes Shipped', 'Month', 'Product', 'Country']
    target = 'Revenue_per_Box'
    model_df = data[features + [target]].dropna()
    X = model_df[features]
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer(transformers=[
        ('num', 'passthrough', ['Boxes Shipped', 'Month']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Product', 'Country'])
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return pipeline, r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)

with st.spinner("Training Random Forest model..."):
    rf_pipeline, rf_r2, rf_mae = train_rf(filtered)

m1, m2 = st.columns(2)
with m1:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Random Forest R²</div>
        <div class="value">{rf_r2:.4f}</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Mean Absolute Error</div>
        <div class="value">${rf_mae:.2f}</div>
    </div>""", unsafe_allow_html=True)

# Feature importance
feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()
importances = rf_pipeline.named_steps['model'].feature_importances_
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
fi_df = fi_df.sort_values('Importance', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1], color=CHOCO_PALETTE[0])
ax.set_title("Top 10 Feature Importances (Random Forest)")
ax.set_xlabel("Importance Score")
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Predictor Widget ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔮 Predict Revenue per Box</div>', unsafe_allow_html=True)

p1, p2, p3, p4 = st.columns(4)
with p1:
    pred_boxes = st.number_input("Boxes Shipped", min_value=1, max_value=1000, value=50)
with p2:
    pred_product = st.selectbox("Product", sorted(df['Product'].unique()))
with p3:
    pred_country = st.selectbox("Country", sorted(df['Country'].unique()))
with p4:
    pred_month = st.slider("Month", 1, 12, 6)

if st.button("🍫 Predict", use_container_width=True):
    new_data = pd.DataFrame({
        'Boxes Shipped': [pred_boxes],
        'Product': [pred_product],
        'Country': [pred_country],
        'Month': [pred_month]
    })
    prediction = rf_pipeline.predict(new_data)[0]
    total_est = prediction * pred_boxes
    st.success(f"**Predicted Revenue per Box:** ${prediction:.2f}  |  **Estimated Total Revenue:** ${total_est:,.2f}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border:1px solid #c8692a; margin-top:2rem;'>
<p style='text-align:center; color:#6b2f1a; font-size:0.82rem;'>
    🍫 Chocolate Sales Analysis · 2022 · Built with Streamlit
</p>
""", unsafe_allow_html=True)
