from pathlib import Path
import warnings

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "trademerch_gpr.csv"

REGION_MAP = {
    "Argentina": "Latin America",
    "Australia": "Asia-Pacific",
    "Belgium": "Europe",
    "Brazil": "Latin America",
    "Canada": "North America",
    "Chile": "Latin America",
    "China": "Asia-Pacific",
    "Colombia": "Latin America",
    "Denmark": "Europe",
    "Egypt": "Middle East & Africa",
    "Finland": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Hong_Kong": "Asia-Pacific",
    "Hungary": "Europe",
    "India": "Asia-Pacific",
    "Indonesia": "Asia-Pacific",
    "Israel": "Middle East & Africa",
    "Italy": "Europe",
    "Japan": "Asia-Pacific",
    "Malaysia": "Asia-Pacific",
    "Mexico": "North America",
    "Netherlands": "Europe",
    "Norway": "Europe",
    "Peru": "Latin America",
    "Philippines": "Asia-Pacific",
    "Poland": "Europe",
    "Portugal": "Europe",
    "Russia": "Europe",
    "Saudi_Arabia": "Middle East & Africa",
    "South_Africa": "Middle East & Africa",
    "South_Korea": "Asia-Pacific",
    "Spain": "Europe",
    "Sweden": "Europe",
    "Switzerland": "Europe",
    "Taiwan": "Asia-Pacific",
    "Thailand": "Asia-Pacific",
    "Tunisia": "Middle East & Africa",
    "Turkiye": "Middle East & Africa",
    "Ukraine": "Europe",
    "United_Kingdom": "Europe",
    "United_States": "North America",
    "Venezuela": "Latin America",
    "Viet_Nam": "Asia-Pacific",
}

COLORS = {
    "ink": "#132A3A",
    "muted": "#5D7285",
    "paper": "#F6F1EB",
    "surface": "#FFFDF9",
    "border": "#D8CFC4",
    "risk": "#B54738",
    "trade": "#1F5E7A",
    "accent": "#D28B36",
    "positive": "#2D7F5E",
    "negative": "#C84C36",
}


def load_data():
    df = pd.read_csv(DATA_FILE)
    unnamed_columns = [column for column in df.columns if str(column).startswith("Unnamed")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Trade_Value"] = pd.to_numeric(df["Trade_Value"], errors="coerce")
    df["GPRHI"] = pd.to_numeric(df["GPRHI"], errors="coerce")
    df = df.dropna(subset=["Year", "Country", "Trade_Flow", "Trade_Value", "GPRHI"]).copy()

    df["Year"] = df["Year"].astype(int)
    df["Country_Label"] = df["Country"].str.replace("_", " ", regex=False)
    df["Region"] = df["Country"].map(REGION_MAP).fillna("Other")
    df["Trade_Value_B"] = df["Trade_Value"] / 1000.0
    return df.sort_values(["Year", "Country", "Trade_Flow"])


df_combined = load_data()
YEAR_MIN = int(df_combined["Year"].min())
YEAR_MAX = int(df_combined["Year"].max())
YEARS = list(range(YEAR_MIN, YEAR_MAX + 1))
REGIONS = sorted(df_combined["Region"].unique())
COUNTRIES = sorted(df_combined["Country_Label"].unique())


def aggregate_country_year(dataframe):
    group_columns = ["Year", "Country", "Country_Label", "Region"]
    return (
        dataframe.groupby(group_columns, as_index=False)
        .agg(Trade_Value=("Trade_Value", "sum"), Trade_Value_B=("Trade_Value_B", "sum"), GPRHI=("GPRHI", "mean"))
    )


def safe_pearson(x_values, y_values):
    if len(x_values) < 2:
        return np.nan, np.nan
    if np.isclose(np.std(x_values), 0) or np.isclose(np.std(y_values), 0):
        return np.nan, np.nan
    return pearsonr(x_values, y_values)


def interpret_strength(value):
    if pd.isna(value):
        return "not enough variation"
    absolute = abs(value)
    if absolute >= 0.7:
        return "very strong"
    if absolute >= 0.5:
        return "strong"
    if absolute >= 0.3:
        return "moderate"
    if absolute > 0:
        return "weak"
    return "no clear"


def interpret_direction(value):
    if pd.isna(value):
        return "relationship"
    if value > 0:
        return "positive relationship"
    if value < 0:
        return "negative relationship"
    return "relationship"


def metric_card(title, value_id, note_id):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-uppercase small", style={"letterSpacing": "0.08em", "color": COLORS["muted"]}),
                html.H3(id=value_id, className="mt-2 mb-1", style={"color": COLORS["ink"], "fontWeight": "700"}),
                html.Div(id=note_id, style={"color": COLORS["muted"], "fontSize": "0.95rem"}),
            ]
        ),
        style={
            "backgroundColor": COLORS["surface"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "18px",
            "boxShadow": "0 12px 30px rgba(19, 42, 58, 0.06)",
        },
        className="h-100",
    )


def story_card(title, body_id):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(title, style={"color": COLORS["ink"], "fontWeight": "700"}),
                html.Div(id=body_id, style={"color": COLORS["muted"], "lineHeight": "1.6"}),
            ]
        ),
        style={
            "backgroundColor": COLORS["surface"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "18px",
        },
        className="h-100",
    )


def framework_card(title, body):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, style={"color": COLORS["ink"], "fontWeight": "700", "fontSize": "1rem"}),
                html.P(body, className="mb-0 mt-2", style={"color": COLORS["muted"], "lineHeight": "1.55"}),
            ]
        ),
        style={
            "backgroundColor": COLORS["surface"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "18px",
        },
        className="h-100",
    )


def style_figure(fig, height=430):
    fig.update_layout(
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["ink"], family="Segoe UI"),
        margin=dict(l=55, r=30, t=70, b=45),
        height=height,
        hoverlabel=dict(bgcolor="#FFF8F0", font_color=COLORS["ink"]),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=COLORS["border"])
    fig.update_yaxes(gridcolor="#E9E1D8", zeroline=False, linecolor=COLORS["border"])
    return fig


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color=COLORS["muted"]),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_figure(fig, height=350)


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Global Supply Chain Resilience Architect"

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div("DV-2026-GROUP-01", style={"letterSpacing": "0.18em", "textTransform": "uppercase", "color": COLORS["accent"], "fontWeight": "700"}),
                        html.H1(
                            "Global Supply Chain Resilience Architect",
                            className="mt-2",
                            style={"color": COLORS["ink"], "fontWeight": "800"},
                        ),
                        html.P(
                            "A narrative dashboard that tests whether geopolitical risk changes global trade behavior from 2021 to 2025. "
                            "Because the dataset does not contain direct shipping lead-time records, merchandise trade volume is used as the dashboard's supply-chain stress proxy.",
                            style={"color": COLORS["muted"], "fontSize": "1.05rem", "maxWidth": "900px"},
                        ),
                    ]
                )
            ],
            className="pt-4 pb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Analytical Framework", style={"color": COLORS["ink"], "fontWeight": "700"}),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            framework_card(
                                                "Key Question 1",
                                                "Do geopolitical events affect global shipping behavior? "
                                                "This dashboard answers that by comparing changes in GPRHI and total trade volume over time.",
                                            ),
                                            md=4,
                                        ),
                                        dbc.Col(
                                            framework_card(
                                                "Key Question 2",
                                                "What is the Pearson coefficient between geopolitical risk and our supply-chain proxy? "
                                                "Because no direct lead-time field exists, merchandise trade volume is used as the measurable proxy variable.",
                                            ),
                                            md=4,
                                        ),
                                        dbc.Col(
                                            framework_card(
                                                "Analytical Logic",
                                                "The dashboard moves from overall relationship, to statistical test, to country and regional breakdown so the report can connect evidence, interpretation, and business relevance.",
                                            ),
                                            md=4,
                                        ),
                                    ],
                                    className="g-3 mt-1",
                                ),
                            ]
                        ),
                        style={
                            "backgroundColor": COLORS["surface"],
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "22px",
                            "boxShadow": "0 14px 35px rgba(19, 42, 58, 0.08)",
                        },
                    ),
                    width=12,
                )
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Interactive Filters", style={"color": COLORS["ink"], "fontWeight": "700"}),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Year Range", style={"fontWeight": "600", "color": COLORS["ink"]}),
                                                dcc.RangeSlider(
                                                    id="year-range",
                                                    min=YEAR_MIN,
                                                    max=YEAR_MAX,
                                                    value=[YEAR_MIN, YEAR_MAX],
                                                    marks={year: str(year) for year in YEARS},
                                                    allowCross=False,
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Region", style={"fontWeight": "600", "color": COLORS["ink"]}),
                                                dcc.Dropdown(
                                                    id="region-filter",
                                                    options=[{"label": "All Regions", "value": "All"}] + [{"label": region, "value": region} for region in REGIONS],
                                                    value="All",
                                                    clearable=False,
                                                ),
                                            ],
                                            md=2,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Trade Flow", style={"fontWeight": "600", "color": COLORS["ink"]}),
                                                dcc.Dropdown(
                                                    id="flow-filter",
                                                    options=[
                                                        {"label": "All Trade Flows", "value": "All"},
                                                        {"label": "Exports", "value": "Exports"},
                                                        {"label": "Imports", "value": "Imports"},
                                                    ],
                                                    value="All",
                                                    clearable=False,
                                                ),
                                            ],
                                            md=2,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Countries", style={"fontWeight": "600", "color": COLORS["ink"]}),
                                                dcc.Dropdown(
                                                    id="country-filter",
                                                    options=[{"label": country, "value": country} for country in COUNTRIES],
                                                    value=[],
                                                    multi=True,
                                                    placeholder="Leave blank to keep all countries in scope",
                                                ),
                                            ],
                                            md=4,
                                        ),
                                    ],
                                    className="g-3 mt-1",
                                ),
                            ]
                        ),
                        style={
                            "backgroundColor": COLORS["surface"],
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "22px",
                            "boxShadow": "0 14px 35px rgba(19, 42, 58, 0.08)",
                        },
                    ),
                    width=12,
                )
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(metric_card("Pearson Correlation", "metric-corr", "metric-corr-note"), md=4),
                dbc.Col(metric_card("Trade Volume In Scope", "metric-trade", "metric-trade-note"), md=4),
                dbc.Col(metric_card("Most Exposed View", "metric-exposure", "metric-exposure-note"), md=4),
            ],
            className="g-4 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Hero Visual 1: Risk and Trade Through Time", style={"color": COLORS["ink"], "fontWeight": "700"}),
                        dcc.Graph(id="hero-timeseries"),
                    ],
                    md=7,
                ),
                dbc.Col(
                    [
                        html.H4("Hero Visual 2: Statistical Relationship", style={"color": COLORS["ink"], "fontWeight": "700"}),
                        dcc.Graph(id="hero-scatter"),
                    ],
                    md=5,
                ),
            ],
            className="g-4 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(story_card("What This View Says", "story-main"), md=6),
                dbc.Col(story_card("Why This Matters For The Report", "story-report"), md=6),
            ],
            className="g-4 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Hero Visual 3: Country Sensitivity Ranking", style={"color": COLORS["ink"], "fontWeight": "700"}),
                        dcc.Graph(id="country-correlation"),
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        html.H4("Hero Visual 4: Regional Trade Heatmap", style={"color": COLORS["ink"], "fontWeight": "700"}),
                        dcc.Graph(id="region-heatmap"),
                    ],
                    md=6,
                ),
            ],
            className="g-4 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Supporting Visual 5: Trade Flow Mix", style={"color": COLORS["ink"], "fontWeight": "700"}),
                        dcc.Graph(id="flow-mix"),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.H4("Supporting Visual 6: Highest-Volume Countries", style={"color": COLORS["ink"], "fontWeight": "700"}),
                        dcc.Graph(id="top-countries"),
                    ],
                    md=8,
                ),
            ],
            className="g-4 pb-5",
        ),
    ],
    fluid=True,
    style={"backgroundColor": COLORS["paper"], "minHeight": "100vh", "paddingLeft": "28px", "paddingRight": "28px"},
)


@app.callback(
    Output("metric-corr", "children"),
    Output("metric-corr-note", "children"),
    Output("metric-trade", "children"),
    Output("metric-trade-note", "children"),
    Output("metric-exposure", "children"),
    Output("metric-exposure-note", "children"),
    Output("hero-timeseries", "figure"),
    Output("hero-scatter", "figure"),
    Output("story-main", "children"),
    Output("story-report", "children"),
    Output("country-correlation", "figure"),
    Output("region-heatmap", "figure"),
    Output("flow-mix", "figure"),
    Output("top-countries", "figure"),
    Input("year-range", "value"),
    Input("region-filter", "value"),
    Input("flow-filter", "value"),
    Input("country-filter", "value"),
)
def update_dashboard(year_range, region_value, flow_value, country_values):
    start_year, end_year = year_range
    filtered = df_combined[(df_combined["Year"] >= start_year) & (df_combined["Year"] <= end_year)].copy()

    if region_value != "All":
        filtered = filtered[filtered["Region"] == region_value]

    if flow_value != "All":
        filtered = filtered[filtered["Trade_Flow"] == flow_value]

    if country_values:
        normalized_country_values = {country.replace("_", " ") for country in country_values}
        filtered = filtered[filtered["Country_Label"].isin(normalized_country_values)]

    if filtered.empty:
        no_data_message = "No observations match the current filter selection."
        return (
            "N/A",
            no_data_message,
            "0.0B USD",
            "No data selected",
            "N/A",
            "No exposure signal available",
            empty_figure(no_data_message),
            empty_figure(no_data_message),
            no_data_message,
            "The report needs enough observations to justify the analysis.",
            empty_figure(no_data_message),
            empty_figure(no_data_message),
            empty_figure(no_data_message),
            empty_figure(no_data_message),
        )

    country_year = aggregate_country_year(filtered)
    corr_value, p_value = safe_pearson(country_year["GPRHI"], country_year["Trade_Value"])

    yearly_trade = filtered.groupby("Year", as_index=False)["Trade_Value_B"].sum()
    yearly_gpr = filtered.drop_duplicates(["Year", "Country"]).groupby("Year", as_index=False)["GPRHI"].mean()
    yearly_view = yearly_trade.merge(yearly_gpr, on="Year", how="inner").sort_values("Year")

    corr_label = f"{corr_value:.3f}" if pd.notna(corr_value) else "N/A"
    corr_note = (
        f"Framework answer to Q2: {interpret_strength(corr_value).capitalize()} "
        f"{interpret_direction(corr_value)} across {len(country_year)} country-year observations."
    )
    trade_label = f"{filtered['Trade_Value_B'].sum():,.1f}B USD"
    trade_note = f"{filtered['Country_Label'].nunique()} countries, {len(filtered)} rows, {start_year}-{end_year}."

    country_corr_rows = []
    for country_name, country_frame in country_year.groupby("Country_Label"):
        country_corr, country_p = safe_pearson(country_frame["GPRHI"], country_frame["Trade_Value"])
        if pd.notna(country_corr):
            country_corr_rows.append(
                {
                    "Country_Label": country_name,
                    "Region": country_frame["Region"].iloc[0],
                    "Correlation": country_corr,
                    "PValue": country_p,
                    "Trade_Value_B": country_frame["Trade_Value_B"].sum(),
                }
            )

    country_corr_df = pd.DataFrame(country_corr_rows)
    if not country_corr_df.empty:
        country_corr_df["AbsCorrelation"] = country_corr_df["Correlation"].abs()
        country_corr_df = country_corr_df.sort_values(["AbsCorrelation", "Trade_Value_B"], ascending=[False, False])
        top_exposed = country_corr_df.iloc[0]
        exposure_label = top_exposed["Country_Label"]
        exposure_note = f"r = {top_exposed['Correlation']:.3f} in {top_exposed['Region']}."
    else:
        exposure_label = "Not enough data"
        exposure_note = "Country-level Pearson values need at least two observations with variation."

    highest_risk_year = yearly_view.loc[yearly_view["GPRHI"].idxmax()]
    lowest_trade_year = yearly_view.loc[yearly_view["Trade_Value_B"].idxmin()]

    time_fig = make_subplots(specs=[[{"secondary_y": True}]])
    time_fig.add_trace(
        go.Scatter(
            x=yearly_view["Year"],
            y=yearly_view["GPRHI"],
            mode="lines+markers",
            name="Average GPR Index",
            line=dict(color=COLORS["risk"], width=4),
            marker=dict(size=10),
            hovertemplate="Year %{x}<br>GPRHI %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    time_fig.add_trace(
        go.Scatter(
            x=yearly_view["Year"],
            y=yearly_view["Trade_Value_B"],
            mode="lines+markers",
            name="Trade Volume",
            line=dict(color=COLORS["trade"], width=4),
            marker=dict(size=10),
            hovertemplate="Year %{x}<br>Trade %{y:,.1f}B USD<extra></extra>",
        ),
        secondary_y=True,
    )
    time_fig.add_annotation(
        x=highest_risk_year["Year"],
        y=highest_risk_year["GPRHI"],
        text=f"Highest risk: {int(highest_risk_year['Year'])}",
        showarrow=True,
        arrowcolor=COLORS["risk"],
        bgcolor="#FFF7F2",
        bordercolor=COLORS["risk"],
    )
    time_fig.add_annotation(
        x=lowest_trade_year["Year"],
        y=lowest_trade_year["Trade_Value_B"],
        text=f"Lowest trade: {int(lowest_trade_year['Year'])}",
        showarrow=True,
        arrowcolor=COLORS["trade"],
        bgcolor="#F4FBFF",
        bordercolor=COLORS["trade"],
        yref="y2",
    )
    time_fig.update_layout(title="Do trade volumes move when geopolitical risk rises?")
    time_fig.update_xaxes(title_text="Year")
    time_fig.update_yaxes(title_text="Average GPRHI", secondary_y=False)
    time_fig.update_yaxes(title_text="Trade Volume (Billion USD)", secondary_y=True)
    time_fig = style_figure(time_fig, height=470)

    scatter_fig = go.Figure()
    scatter_fig.add_trace(
        go.Scatter(
            x=country_year["GPRHI"],
            y=country_year["Trade_Value_B"],
            mode="markers",
            marker=dict(
                size=np.clip(country_year["Trade_Value_B"] / country_year["Trade_Value_B"].max() * 36, 12, 36),
                color=country_year["Year"],
                colorscale=[[0.0, "#E7B96C"], [0.5, "#83A9A1"], [1.0, "#1F5E7A"]],
                line=dict(color="#FFFDF9", width=1),
                showscale=True,
                colorbar=dict(title="Year"),
            ),
            customdata=np.stack([country_year["Country_Label"], country_year["Region"]], axis=-1),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>GPRHI %{x:.2f}<br>Trade %{y:,.1f}B USD<extra></extra>",
            name="Country-year observation",
        )
    )
    if pd.notna(corr_value) and len(country_year) >= 2:
        trend_coeffs = np.polyfit(country_year["GPRHI"], country_year["Trade_Value_B"], 1)
        x_line = np.linspace(country_year["GPRHI"].min(), country_year["GPRHI"].max(), 100)
        y_line = np.poly1d(trend_coeffs)(x_line)
        scatter_fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Trend line",
                line=dict(color=COLORS["risk"], dash="dash", width=3),
                hoverinfo="skip",
            )
        )
    scatter_fig.update_layout(
        title=f"Pearson test on selected observations: r = {corr_label}",
        xaxis_title="Geopolitical Risk Index (GPRHI)",
        yaxis_title="Trade Volume (Billion USD)",
    )
    scatter_fig = style_figure(scatter_fig, height=470)

    main_story = (
        f"For the selected scope, the dashboard answers Key Question 1 by showing a {interpret_strength(corr_value)} "
        f"{interpret_direction(corr_value)} between geopolitical risk and trade volume. "
        f"The time-series chart shows whether spikes in risk line up with changes in trade, while the scatter plot supplies the Pearson-based statistical evidence for that claim."
    )
    report_story = (
        "These visuals align with the analytical framework because they move from question, to evidence, to interpretation. "
        "They first test whether geopolitical events affect shipping behavior, then quantify that relationship with Pearson correlation, "
        "and finally show where that relationship appears strongest across countries and regions."
    )

    if country_corr_df.empty:
        country_corr_fig = empty_figure("Not enough country-level variation for ranking.")
    else:
        top_country_corr = country_corr_df.head(12).sort_values("Correlation")
        country_corr_fig = go.Figure(
            go.Bar(
                x=top_country_corr["Correlation"],
                y=top_country_corr["Country_Label"],
                orientation="h",
                marker=dict(color=[COLORS["positive"] if value >= 0 else COLORS["negative"] for value in top_country_corr["Correlation"]]),
                text=[f"{value:.2f}" for value in top_country_corr["Correlation"]],
                textposition="outside",
                hovertemplate="%{y}<br>Pearson r %{x:.3f}<extra></extra>",
            )
        )
        country_corr_fig.update_layout(
            title="Countries ranked by correlation strength",
            xaxis_title="Pearson Correlation Coefficient",
            yaxis_title="Country",
        )
        country_corr_fig = style_figure(country_corr_fig, height=470)

    regions_in_scope = sorted(filtered["Region"].unique())
    region_year = (
        filtered.groupby(["Region", "Year"], as_index=False)["Trade_Value_B"]
        .sum()
        .pivot(index="Region", columns="Year", values="Trade_Value_B")
        .reindex(regions_in_scope)
        .fillna(0)
    )
    heatmap_fig = go.Figure(
        go.Heatmap(
            z=region_year.values,
            x=region_year.columns,
            y=region_year.index,
            colorscale=[[0.0, "#F7E5D0"], [0.5, "#E0B071"], [1.0, "#1F5E7A"]],
            text=np.round(region_year.values, 1),
            texttemplate="%{text}",
            hovertemplate="Region %{y}<br>Year %{x}<br>Trade %{z:,.1f}B USD<extra></extra>",
        )
    )
    heatmap_fig.update_layout(
        title="Where trade volume concentrates over time",
        xaxis_title="Year",
        yaxis_title="Region",
    )
    heatmap_fig = style_figure(heatmap_fig, height=470)

    flow_mix = filtered.groupby("Trade_Flow", as_index=False)["Trade_Value_B"].sum()
    flow_fig = go.Figure(
        go.Pie(
            labels=flow_mix["Trade_Flow"],
            values=flow_mix["Trade_Value_B"],
            hole=0.52,
            marker=dict(colors=[COLORS["trade"], COLORS["accent"]]),
            textinfo="label+percent",
            hovertemplate="%{label}<br>%{value:,.1f}B USD<extra></extra>",
        )
    )
    flow_fig.update_layout(title="Imports versus exports inside the selected scope")
    flow_fig = style_figure(flow_fig, height=410)

    top_trade = (
        filtered.groupby("Country_Label", as_index=False)["Trade_Value_B"]
        .sum()
        .sort_values("Trade_Value_B", ascending=False)
        .head(10)
        .sort_values("Trade_Value_B")
    )
    top_trade_fig = go.Figure(
        go.Bar(
            x=top_trade["Trade_Value_B"],
            y=top_trade["Country_Label"],
            orientation="h",
            marker=dict(color=COLORS["trade"]),
            text=[f"{value:,.0f}B" for value in top_trade["Trade_Value_B"]],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:,.1f}B USD<extra></extra>",
        )
    )
    top_trade_fig.update_layout(
        title="Highest-volume countries in the current view",
        xaxis_title="Trade Volume (Billion USD)",
        yaxis_title="Country",
    )
    top_trade_fig = style_figure(top_trade_fig, height=410)

    return (
        corr_label,
        corr_note,
        trade_label,
        trade_note,
        exposure_label,
        exposure_note,
        time_fig,
        scatter_fig,
        main_story,
        report_story,
        country_corr_fig,
        heatmap_fig,
        flow_fig,
        top_trade_fig,
    )


if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("Global Supply Chain Resilience Architect dashboard: http://localhost:8050/")
    print("=" * 72 + "\n")
    app.run_server(debug=True, port=8050)
