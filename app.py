import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv("data/clean_sales_data.csv")

# Convert date
df['Date'] = pd.to_datetime(df['Date'])
df['Month_Name'] = df['Date'].dt.month_name()

# Create revenue per box
df['Revenue_per_Box'] = df['Amount'] / df['Boxes Shipped']

# Title
st.title("Sales Performance Dashboard")

# KPIs
total_revenue = df['Amount'].sum()
total_boxes = df['Boxes Shipped'].sum()

col1, col2 = st.columns(2)
col1.metric("Total Revenue", f"{total_revenue:,.0f}")
col2.metric("Total Boxes", f"{total_boxes:,.0f}")

# Sidebar filters
st.sidebar.header("Filters")

country = st.sidebar.multiselect(
    "Country", df['Country'].unique(), default=df['Country'].unique()
)

product = st.sidebar.multiselect(
    "Product", df['Product'].unique(), default=df['Product'].unique()
)

filtered_df = df[
    (df['Country'].isin(country)) &
    (df['Product'].isin(product))
]

# Monthly Trend
monthly = filtered_df.groupby('Month_Name')['Amount'].sum().reset_index()

fig1 = px.line(monthly, x='Month_Name', y='Amount', title="Sales Trend")
st.plotly_chart(fig1)

# Country Revenue
country_sales = filtered_df.groupby('Country')['Amount'].sum().reset_index()

fig2 = px.bar(country_sales, x='Country', y='Amount', title="Revenue by Country")
st.plotly_chart(fig2)

# Top Products
top_products = (
    filtered_df.groupby('Product')['Amount']
    .sum()
    .reset_index()
    .sort_values(by='Amount', ascending=False)
    .head(10)
)

fig3 = px.bar(top_products, x='Product', y='Amount', title="Top Products")
st.plotly_chart(fig3)

# Efficiency
efficiency = (
    filtered_df.groupby('Product')['Revenue_per_Box']
    .mean()
    .reset_index()
    .sort_values(by='Revenue_per_Box', ascending=False)
    .head(10)
)

fig4 = px.bar(efficiency, x='Product', y='Revenue_per_Box', title="Revenue per Box")
st.plotly_chart(fig4)