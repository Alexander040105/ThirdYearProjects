import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings('ignore')

# Load data
df_combined = pd.read_csv('trademerch_gpr.csv')
trade_df = pd.read_csv('TradeMerchTotal_og.csv')
gpr_df = pd.read_csv('data_gpr_export_og.csv')

# Prepare data for analysis
df_combined['Year'] = df_combined['Year'].astype(int)
df_combined = df_combined.sort_values('Year')

# Convert to numeric and handle missing values
df_combined['Trade_Value'] = pd.to_numeric(df_combined['Trade_Value'], errors='coerce')
df_combined['GPRHI'] = pd.to_numeric(df_combined['GPRHI'], errors='coerce')

# Calculate aggregated metrics by year
yearly_data = df_combined.groupby('Year').agg({
    'Trade_Value': 'sum',
    'GPRHI': 'mean'
}).reset_index()

# Calculate Pearson correlation
valid_data = yearly_data.dropna(subset=['Trade_Value', 'GPRHI'])
if len(valid_data) > 1:
    pearson_coeff, p_value = pearsonr(valid_data['GPRHI'], valid_data['Trade_Value'])
else:
    pearson_coeff = 0
    p_value = 1

# Calculate correlation by country
country_corr = []
for country in df_combined['Country'].unique():
    country_data = df_combined[df_combined['Country'] == country].sort_values('Year')
    country_data = country_data.dropna(subset=['Trade_Value', 'GPRHI'])
    if len(country_data) > 2:
        try:
            corr, pval = pearsonr(country_data['GPRHI'], country_data['Trade_Value'])
            country_corr.append({
                'Country': country,
                'Correlation': corr,
                'P-Value': pval,
                'Sample_Size': len(country_data)
            })
        except:
            pass

country_corr_df = pd.DataFrame(country_corr).sort_values('Correlation', ascending=False)

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define color scheme
color_gpr = '#E74C3C'  # Red for GPR
color_trade = '#3498DB'  # Blue for Trade

# Create the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Global Shipping & Geopolitical Risk Analysis Dashboard", 
                   className="text-center mt-4 mb-2"),
            html.H5("Analytical Framework: Impact of Geopolitical Events on Global Shipping", 
                   className="text-center text-muted mb-4")
        ])
    ]),
    
    # Key Metrics Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Pearson Correlation", className="card-title"),
                    html.H3(f"{pearson_coeff:.4f}", className="text-primary"),
                    html.P(f"P-Value: {p_value:.4f}", className="text-muted small")
                ])
            ])
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Average GPR Index", className="card-title"),
                    html.H3(f"{yearly_data['GPRHI'].mean():.3f}", className="text-danger"),
                    html.P(f"Range: {yearly_data['GPRHI'].min():.3f} - {yearly_data['GPRHI'].max():.3f}", 
                          className="text-muted small")
                ])
            ])
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Total Trade Volume", className="card-title"),
                    html.H3(f"${yearly_data['Trade_Value'].sum()/1e6:.1f}B", className="text-info"),
                    html.P(f"Period: {yearly_data['Year'].min()} - {yearly_data['Year'].max()}", 
                          className="text-muted small")
                ])
            ])
        ], md=4),
    ], className="mb-4"),
    
    # Tabs for different views
    dbc.Tabs([
        # Tab 1: Time Series Analysis
        dbc.Tab(label="Key Question 1: Do Geopolitical Events Affect Global Shipping?", 
               children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='timeseries-plot')
                ])
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='correlation-scatter')
                ])
            ])
        ]),
        
        # Tab 2: Pearson Coefficient Analysis
        dbc.Tab(label="Key Question 2: Pearson Coefficient Between Events & Lead Times", 
               children=[
            dbc.Row([
                dbc.Col([
                    html.H5("Country-Level Pearson Correlations", className="mt-3 mb-3"),
                    dcc.Graph(id='country-correlation-chart')
                ])
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='correlation-heatmap')
                ])
            ])
        ]),
        
        # Tab 3: Detailed Analysis
        dbc.Tab(label="Detailed Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.H5("GPR Index Trend", className="mt-3 mb-3"),
                    dcc.Graph(id='gpr-trend')
                ], md=6),
                dbc.Col([
                    html.H5("Trade Volume Trend", className="mt-3 mb-3"),
                    dcc.Graph(id='trade-trend')
                ], md=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("Top Countries by Trade Volume", className="mt-3 mb-3"),
                    dcc.Graph(id='top-countries-trade')
                ], md=6),
                dbc.Col([
                    html.H5("Trade Flow Distribution", className="mt-3 mb-3"),
                    dcc.Graph(id='trade-flow-dist')
                ], md=6)
            ])
        ]),
        
        # Tab 4: Summary & Insights
        dbc.Tab(label="Insights & Findings", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Key Findings", className="card-title"),
                            html.Hr(),
                            html.H6("Q1: Do Geopolitical Events Affect Global Shipping?", className="fw-bold mt-3"),
                            html.P(f"Pearson Correlation: {pearson_coeff:.4f} with p-value {p_value:.4f}"),
                            html.P(
                                f"Interpretation: The correlation coefficient of {pearson_coeff:.4f} indicates a "
                                f"{'moderate positive' if pearson_coeff > 0.3 else 'weak positive' if pearson_coeff > 0 else 'negative'} "
                                f"relationship between geopolitical risk and global trade volumes. "
                                f"{'This is statistically significant.' if p_value < 0.05 else 'This relationship is not statistically significant.'}"
                            ),
                            html.Hr(),
                            html.H6("Q2: Pearson Coefficient Between Geopolitical Events & Lead Times", className="fw-bold mt-3"),
                            html.P(f"Global Pearson Coefficient: {pearson_coeff:.4f}"),
                            html.P(
                                "The GPR High Index (GPRHI) measures geopolitical risk, and its relationship with "
                                "trade volumes can serve as a proxy for supply chain disruptions. A higher correlation "
                                "suggests that geopolitical events have a measurable impact on shipping patterns."
                            ),
                            html.Hr(),
                            html.H6("Data Observations", className="fw-bold mt-3"),
                            html.Ul([
                                html.Li(f"Analysis period: {yearly_data['Year'].min()} - {yearly_data['Year'].max()}"),
                                html.Li(f"Number of countries analyzed: {df_combined['Country'].nunique()}"),
                                html.Li(f"Total observations: {len(df_combined)}"),
                                html.Li(f"Average GPR Index: {yearly_data['GPRHI'].mean():.3f}")
                            ])
                        ])
                    ], className="mt-4")
                ])
            ])
        ])
    ], id="tabs"),
    
], fluid=True, className="mb-5")

# Callback for timeseries plot
@app.callback(
    Output('timeseries-plot', 'figure'),
    Input('tabs', 'active_tab')
)
def update_timeseries(tab):
    if tab != 'tab-0':
        return go.Figure()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add GPRHI trace
    fig.add_trace(
        go.Scatter(
            x=yearly_data['Year'],
            y=yearly_data['GPRHI'],
            name='GPR High Index (Geopolitical Risk)',
            line=dict(color=color_gpr, width=3),
            marker=dict(size=8)
        ),
        secondary_y=False
    )
    
    # Add Trade Value trace
    fig.add_trace(
        go.Scatter(
            x=yearly_data['Year'],
            y=yearly_data['Trade_Value']/1e6,  # Convert to billions
            name='Total Trade Volume (Billions USD)',
            line=dict(color=color_trade, width=3),
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="<b>GPR High Index</b>", secondary_y=False, title_font=dict(color=color_gpr))
    fig.update_yaxes(title_text="<b>Trade Volume (Billions USD)</b>", secondary_y=True, title_font=dict(color=color_trade))
    
    fig.update_layout(
        title="<b>Geopolitical Risk vs Global Shipping Volume Over Time</b>",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig

# Callback for correlation scatter plot
@app.callback(
    Output('correlation-scatter', 'figure'),
    Input('tabs', 'active_tab')
)
def update_scatter(tab):
    if tab != 'tab-0':
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly_data['GPRHI'],
        y=yearly_data['Trade_Value']/1e6,
        mode='markers+lines',
        name='Annual Data',
        marker=dict(
            size=12,
            color=yearly_data['Year'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Year")
        ),
        line=dict(color='rgba(31, 119, 180, 0.3)')
    ))
    
    # Add trend line
    z = np.polyfit(yearly_data['GPRHI'], yearly_data['Trade_Value']/1e6, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(yearly_data['GPRHI'].min(), yearly_data['GPRHI'].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_trend,
        y=p(x_trend),
        name=f'Trend (r={pearson_coeff:.3f})',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title=f"<b>Correlation Analysis: Geopolitical Risk vs Trade Volume</b><br><sub>Pearson r = {pearson_coeff:.4f}, p-value = {p_value:.4f}</sub>",
        xaxis_title="GPR High Index (Geopolitical Risk)",
        yaxis_title="Trade Volume (Billions USD)",
        height=500,
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig

# Callback for country correlation chart
@app.callback(
    Output('country-correlation-chart', 'figure'),
    Input('tabs', 'active_tab')
)
def update_country_corr(tab):
    if tab != 'tab-1':
        return go.Figure()
    
    # Top 15 countries by correlation
    top_countries = country_corr_df.head(15)
    
    colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in top_countries['Correlation']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_countries['Correlation'],
            y=top_countries['Country'],
            orientation='h',
            marker=dict(color=colors),
            text=top_countries['Correlation'].round(3),
            textposition='auto',
            name='Pearson Correlation'
        )
    ])
    
    fig.update_layout(
        title="<b>Top 15 Countries: Pearson Correlation (Geopolitical Risk vs Trade)</b>",
        xaxis_title="Pearson Correlation Coefficient",
        yaxis_title="Country",
        height=500,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

# Callback for correlation heatmap
@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('tabs', 'active_tab')
)
def update_heatmap(tab):
    if tab != 'tab-1':
        return go.Figure()
    
    # Create correlation matrix for top countries
    top_15_countries = country_corr_df.head(15)['Country'].tolist()
    
    # Prepare data matrix
    corr_data = []
    for country in top_15_countries:
        country_vals = df_combined[df_combined['Country'] == country][['GPRHI', 'Trade_Value']].dropna()
        if len(country_vals) > 1:
            corr_data.append(country_vals.values)
    
    # Create summary stats for display
    summary_matrix = []
    for country in top_15_countries:
        country_data = df_combined[df_combined['Country'] == country]
        summary_matrix.append([
            country_data['GPRHI'].mean(),
            country_data['Trade_Value'].mean()/1e6,
            country_corr_df[country_corr_df['Country'] == country]['Correlation'].values[0] if country in country_corr_df['Country'].values else 0
        ])
    
    summary_df = pd.DataFrame(summary_matrix, 
                             columns=['Avg GPR Index', 'Avg Trade Vol (B$)', 'Correlation'],
                             index=top_15_countries)
    
    fig = go.Figure(data=go.Heatmap(
        z=summary_df.values,
        x=summary_df.columns,
        y=summary_df.index,
        colorscale='RdBu',
        text=np.round(summary_df.values, 3),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="<b>Country-Level Summary: GPR, Trade Volume & Correlation</b>",
        height=500,
        template='plotly_white'
    )
    
    return fig

# Callback for GPR trend
@app.callback(
    Output('gpr-trend', 'figure'),
    Input('tabs', 'active_tab')
)
def update_gpr_trend(tab):
    if tab != 'tab-2':
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly_data['Year'],
        y=yearly_data['GPRHI'],
        mode='lines+markers',
        name='GPR Index',
        line=dict(color=color_gpr, width=3),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="<b>Geopolitical Risk High Index Trend</b>",
        xaxis_title="Year",
        yaxis_title="GPR High Index",
        height=400,
        template='plotly_white'
    )
    
    return fig

# Callback for Trade trend
@app.callback(
    Output('trade-trend', 'figure'),
    Input('tabs', 'active_tab')
)
def update_trade_trend(tab):
    if tab != 'tab-2':
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly_data['Year'],
        y=yearly_data['Trade_Value']/1e6,
        mode='lines+markers',
        name='Trade Volume',
        line=dict(color=color_trade, width=3),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="<b>Global Trade Volume Trend</b>",
        xaxis_title="Year",
        yaxis_title="Trade Volume (Billions USD)",
        height=400,
        template='plotly_white'
    )
    
    return fig

# Callback for top countries trade
@app.callback(
    Output('top-countries-trade', 'figure'),
    Input('tabs', 'active_tab')
)
def update_top_countries(tab):
    if tab != 'tab-2':
        return go.Figure()
    
    top_countries_trade = df_combined.groupby('Country')['Trade_Value'].sum().sort_values(ascending=False).head(15)
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_countries_trade.index,
            x=top_countries_trade.values/1e6,
            orientation='h',
            marker=dict(color=color_trade)
        )
    ])
    
    fig.update_layout(
        title="<b>Top 15 Countries by Total Trade Volume</b>",
        xaxis_title="Trade Volume (Billions USD)",
        yaxis_title="Country",
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

# Callback for trade flow distribution
@app.callback(
    Output('trade-flow-dist', 'figure'),
    Input('tabs', 'active_tab')
)
def update_trade_flow(tab):
    if tab != 'tab-2':
        return go.Figure()
    
    trade_flow = df_combined.groupby('Trade_Flow')['Trade_Value'].sum()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=trade_flow.index,
            values=trade_flow.values,
            hole=0.3
        )
    ])
    
    fig.update_layout(
        title="<b>Trade Flow Distribution</b>",
        height=400,
        template='plotly_white'
    )
    
    return fig

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Dashboard Running at: http://localhost:8050/")
    print("="*60 + "\n")
    app.run_server(debug=True, port=8050)
