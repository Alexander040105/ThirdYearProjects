# Technical Analysis Explained

## Why I Chose These Analyses

I chose the analysis in this dashboard to directly answer the analytical framework of the project. The first question asks whether geopolitical events affect global shipping behavior, while the second asks for the Pearson coefficient of that relationship. Since the available dataset combines `GPRHI` and trade values across countries and years, the most appropriate first step was to test their relationship using Pearson correlation. This gives a simple and accepted way to measure whether higher geopolitical risk tends to move together with changes in trade activity.

## What I Was Thinking While Building It

My main thinking was that the dashboard should not only look good, it should also defend the argument of the report. Because of that, I focused on visuals that do three jobs:

1. show the pattern over time
2. show the statistical relationship clearly
3. show where the relationship is strongest

The dual-axis time-series chart was included because it helps the viewer quickly compare changes in average geopolitical risk with changes in total trade volume. The scatter plot was included because it is the clearest visual for Pearson correlation and gives direct support for the statistical claim. The country correlation ranking was added because the global view alone can hide differences between countries. The regional heatmap was added to show where trade concentration sits across the years, which helps explain why some regions may be more exposed than others.

## Why These Choices Make Sense

These analyses fit the structure of your project because they move from broad to specific:

- first, the dashboard shows the overall relationship between risk and trade
- second, it tests that relationship statistically
- third, it breaks the result down by country and region

This is useful because supply-chain disruption is rarely uniform. Some countries and regions are more exposed to geopolitical shocks than others, so the dashboard needs both a global answer and a segmented answer.

This also keeps the dashboard aligned to the analytical framework because each major section has a role:

- the time-series chart supports Key Question 1
- the scatter plot and KPI cards support Key Question 2
- the country and regional visuals provide the deeper explanation needed for the report discussion

## Why This Should Be In The Report

These analyses should be included in the report because they help transform the project from a descriptive dashboard into an evidence-based one. Instead of only showing trade values, the report can explain how geopolitical risk relates to those values, which countries appear most sensitive, and why that matters for supply-chain resilience. This strengthens the report because it adds interpretation, analytical depth, and a clear link between the dashboard visuals and the research questions.

## Important Method Note

The dataset does not include direct lead-time measurements. Because of that, I treated merchandise trade volume as a proxy for supply-chain stress and responsiveness. I included this note because it is important to be transparent about what the data can and cannot prove. This honesty improves the credibility of both the dashboard and the written report.
