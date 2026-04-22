# Geopolitical Risk & Global Shipping Analysis Dashboard

## Overview
This Dash-Plotly interactive dashboard analyzes the relationship between geopolitical events and global shipping patterns to answer two key research questions:

1. **Key Question 1:** Do geopolitical events affect global shipping?
2. **Key Question 2:** What is the Pearson coefficient between geopolitical events and lead times?

## Data Sources
- **Trade Data:** TradeMerchTotal_og.csv - Global merchandise trade volumes
- **Geopolitical Risk Data:** data_gpr_export_og.csv - GPR High Index (GPRHI) by country and month
- **Combined Dataset:** trademerch_gpr.csv - Merged data for analysis

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup Steps

1. **Install dependencies:**
```bash
pip install -r requirements_dashboard.txt
```

2. **Navigate to the project directory:**
```bash
cd "c:\Users\Pixel-Tone\Documents\GitHub\ThirdYearProjects\second_semester\data_viz\finals\final_project"
```

3. **Run the dashboard:**
```bash
python dashboard.py
```

4. **Access the dashboard:**
Open your browser and navigate to: `http://localhost:8050/`

## Dashboard Features

### Tab 1: Key Question 1 - Do Geopolitical Events Affect Global Shipping?
- **Time Series Plot:** Shows GPR High Index and trade volume trends over time
- **Correlation Scatter Plot:** Visualizes the relationship between geopolitical risk and trade volumes with trend line
- **Pearson Correlation:** Displays the global correlation coefficient

### Tab 2: Key Question 2 - Pearson Coefficient Analysis
- **Country-Level Correlations:** Bar chart showing top 15 countries ranked by correlation strength
- **Correlation Heatmap:** Summary statistics (avg GPR, avg trade volume, correlation) for top countries

### Tab 3: Detailed Analysis
- **GPR Trend:** Line chart of geopolitical risk over time
- **Trade Volume Trend:** Line chart of global trade volumes
- **Top Countries by Trade Volume:** Bar chart of highest trading nations
- **Trade Flow Distribution:** Pie chart showing imports vs exports distribution

### Tab 4: Insights & Findings
- **Summary Statistics:** Key findings and interpretations
- **Correlation Interpretation:** Statistical significance analysis
- **Data Overview:** Analysis period, country count, and average metrics

## Key Metrics

The dashboard displays three main KPIs:
1. **Pearson Correlation:** The strength of the relationship between geopolitical risk and trade
2. **Average GPR Index:** Mean geopolitical risk across the analysis period
3. **Total Trade Volume:** Aggregate global trade in billions USD

## Data Visualization Components

- **Dual-axis time series:** Tracks GPR and trade trends simultaneously
- **Interactive charts:** Hover for detailed values, click legend items to toggle visibility
- **Color coding:** Red for geopolitical risk, Blue for trade volumes, Green/Red for positive/negative correlations
- **Bootstrap cards:** Clean, responsive layout

## Technical Stack

- **Dash:** Interactive web framework
- **Plotly:** Data visualization library
- **Pandas:** Data manipulation and analysis
- **Scipy:** Statistical correlation calculations
- **Dash Bootstrap Components:** UI styling

## Interpretation Guide

### Pearson Correlation Coefficient
- **Range:** -1.0 to 1.0
- **0.7 to 1.0 or -0.7 to -1.0:** Very strong correlation
- **0.5 to 0.7 or -0.5 to -0.7:** Strong correlation
- **0.3 to 0.5 or -0.3 to -0.5:** Moderate correlation
- **0.0 to 0.3 or -0.0 to -0.3:** Weak correlation

### P-Value
- **P < 0.05:** Statistically significant relationship
- **P ≥ 0.05:** Relationship may be due to chance

## Usage Tips

1. Use the tabs to explore different analytical perspectives
2. Hover over charts for detailed tooltips
3. Click legend items to show/hide data series
4. The dashboard auto-updates based on the underlying data

## Files Structure

```
final_project/
├── dashboard.py                 # Main Dash application
├── requirements_dashboard.txt   # Python dependencies
├── README_DASHBOARD.md         # This file
├── trademerch_gpr.csv          # Combined analysis dataset
├── TradeMerchTotal_og.csv      # Trade merchandise data
└── data_gpr_export_og.csv      # Geopolitical risk data
```

## Troubleshooting

### Port Already in Use
If port 8050 is already in use, modify the last line in `dashboard.py`:
```python
app.run_server(debug=True, port=8051)  # Change 8050 to another port
```

### Missing Data
Ensure all three CSV files are in the same directory as `dashboard.py`

### Import Errors
Reinstall dependencies:
```bash
pip install --upgrade -r requirements_dashboard.txt
```

## Research Questions Addressed

### Q1: Do Geopolitical Events Affect Global Shipping?
The dashboard shows the time series relationship and calculates statistical correlation. If the Pearson coefficient is significantly different from zero (p < 0.05), it indicates geopolitical events do affect shipping.

### Q2: Pearson Coefficient Between Geopolitical Events & Lead Times
The dashboard calculates:
- **Global Pearson Coefficient:** Overall correlation between GPRHI and trade volumes
- **Country-specific Correlations:** Individual country relationships
- **Statistical Significance:** P-values for hypothesis testing

## Notes

- Higher GPRHI values indicate greater geopolitical risk
- Trade volumes include both imports and exports
- Analysis covers years 2021-2025 with 44 countries
- Lead times are inferred from trade volume changes associated with geopolitical events

---
**Author:** Data Visualization Project  
**Created:** 2024
