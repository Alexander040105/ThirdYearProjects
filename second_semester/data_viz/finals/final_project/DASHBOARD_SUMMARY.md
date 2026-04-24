# Dashboard Summary

## What I Changed

I improved the dashboard by turning it into a cleaner narrative dashboard instead of a basic tabbed dashboard. The updated version now has interactive filters for year, region, trade flow, and country so the user can explore the data by date, region, or category as required in the workflow. It also now uses a stronger visual hierarchy with KPI cards, clearer section titles, better colors, and six focused visuals instead of many loosely connected charts. I also added an explicit analytical framework section so the dashboard clearly shows the two research questions and how each visual supports them.

## How I Did It

I cleaned the dataset inside `dashboard.py`, removed unnamed columns, converted the important fields to numeric values, and added a region grouping so the dashboard could support regional filtering. I also converted `Trade_Value` into billions of USD for easier reading on charts. After that, I rebuilt the layout with:

- KPI cards for correlation, p-value, total trade volume, and the most exposed country view
- A dual-axis line chart to compare geopolitical risk and trade over time
- A scatter plot with a Pearson correlation trend line
- A country ranking chart for correlation strength
- A regional heatmap to show where trade is concentrated over time
- Supporting visuals for trade flow mix and top trading countries

## Why I Did It

I made these changes to follow the milestone instructions more closely. The dashboard now focuses on the best charts that directly support the analytical point, increases the data-ink ratio, uses a more professional palette, and tells a clearer story. I also added short narrative cards so the dashboard explains what the visuals mean instead of only showing charts. Most importantly, the dashboard is now more aligned with the analytical framework because it first asks whether geopolitical events affect shipping behavior, then quantifies the relationship using Pearson correlation, and then breaks the pattern down by country and region. This makes the dashboard better for both class presentation and report writing.

## Important Note

The dataset does not contain direct shipping lead-time values, so the dashboard uses merchandise trade volume as a supply-chain stress proxy. I made this explicit in the dashboard so the interpretation stays honest and academically defensible.
