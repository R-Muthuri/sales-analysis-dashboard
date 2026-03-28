import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_excel("clean_sales_data.xls")

# Ensure correct types
df['Date'] = pd.to_datetime(df['Date'])
df['Month_Name'] = df['Date'].dt.month_name()

# KPIs
total_revenue = df['Amount'].sum()
total_boxes = df['Boxes Shipped'].sum()

st.title("Sales Performance Dashboard")

# KPIs display
col1, col2 = st.columns(2)
col1.metric("Total Revenue", f"{total_revenue:,.0f}")
col2.metric("Total Boxes", f"{total_boxes:,.0f}")

# Filters
st.sidebar.header("Filters")
selected_country = st.sidebar.multiselect(
    "Select Country", df['Country'].unique(), default=df['Country'].unique()
)
selected_product = st.sidebar.multiselect(
    "Select Product", df['Product'].unique(), default=df['Product'].unique()
)

# Apply filters
filtered_df = df[
    (df['Country'].isin(selected_country)) &
    (df['Product'].isin(selected_product))
]

# Monthly Trend (sorted chronologically)
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
monthly = filtered_df.groupby('Month_Name')['Amount'].sum().reset_index()
monthly['Month_Name'] = pd.Categorical(monthly['Month_Name'], categories=month_order, ordered=True)
monthly = monthly.sort_values('Month_Name')
fig1 = px.line(monthly, x='Month_Name', y='Amount', title="Sales Trend")
st.plotly_chart(fig1)

# Country Revenue
country = filtered_df.groupby('Country')['Amount'].sum().reset_index()
fig2 = px.bar(country, x='Country', y='Amount', title="Revenue by Country")
st.plotly_chart(fig2)

# Top Products
product = (filtered_df.groupby('Product')['Amount'].sum()
           .reset_index()
           .sort_values(by='Amount', ascending=False)
           .head(10))
fig3 = px.bar(product, x='Product', y='Amount', title="Top Products")
st.plotly_chart(fig3)

# Efficiency
filtered_df = filtered_df.copy()
filtered_df['Revenue_per_Box'] = filtered_df['Amount'] / filtered_df['Boxes Shipped']
eff = (filtered_df.groupby('Product')['Revenue_per_Box'].mean()
       .reset_index()
       .sort_values(by='Revenue_per_Box', ascending=False)
       .head(10))
fig4 = px.bar(eff, x='Product', y='Revenue_per_Box', title="Revenue per Box")
st.plotly_chart(fig4)
