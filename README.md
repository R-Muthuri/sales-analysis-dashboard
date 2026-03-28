# Sales Analysis and Predictive Modeling Dashboard

## Overview
This project analyzes sales data to identify key revenue drivers and build a predictive model using machine learning.

## Key Insights
- Revenue is driven by product characteristics, not volume
- Strong variation exists in revenue per box
- Sales performance is diversified across countries and products

## Machine Learning
- Model: Random Forest Regressor
- R² Score: 0.94
- MAE: 31

## Dashboard
An interactive dashboard was built using Streamlit to visualize:
- Sales trends
- Revenue by country and product
- Efficiency (revenue per box)

## Tools Used
- Python (Pandas, Scikit-learn)
- Plotly
- Streamlit

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
