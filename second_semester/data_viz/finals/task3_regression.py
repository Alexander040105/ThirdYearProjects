# ============================================================
# Task 3 – Regression Analysis & Prediction
# Data Visualization – Lesson 7
# ============================================================
# Requirements:  pip install matplotlib numpy pandas
# Run with:      python task3_regression.py
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 1. REGRESSION EQUATION CONSTANTS ────────────────────────

# Given line of best fit:  y = 15.5x + 120
slope     = 15.5   # average revenue added per customer ($)
intercept = 120    # fixed base revenue regardless of customers ($)

# ── 2. KNOWN DATA ───────────────────────────────────────────

df = pd.DataFrame({
    "day":       ["Monday", "Tuesday"],
    "customers": [40,        60],         # x – number of customers
    "actual":    [750,       1050],        # y – real observed revenue ($)
})

# ── 3. COMPUTED COLUMNS ─────────────────────────────────────

# Apply the equation to the entire customers column at once (pandas vectorization)
df["predicted"] = slope * df["customers"] + intercept

# Residual = how far off the model was (positive = underestimate)
df["residual"]  = df["actual"] - df["predicted"]

# ── 4. SATURDAY PREDICTION ──────────────────────────────────

saturday_customers = 100
saturday_predicted = slope * saturday_customers + intercept  # 15.5×100 + 120 = 1670

# ── 5. PRINT RESULTS ────────────────────────────────────────

print("=" * 58)
print("  TASK 3 – REGRESSION ANALYSIS & PREDICTION")
print("=" * 58)
print(f"\n  Equation: y = {slope}x + {intercept}")
print()
print("  Deliverable 1 – Residual Analysis:")
print()

for _, row in df.iterrows():
    sign = "+" if row["residual"] >= 0 else ""
    print(f"    {row['day']}")
    print(f"      customers (x)     = {int(row['customers'])}")
    print(f"      predicted revenue = {slope} × {int(row['customers'])} + {intercept} = ${row['predicted']:.2f}")
    print(f"      actual revenue    = ${row['actual']:.2f}")
    print(f"      residual          = ${sign}{row['residual']:.2f}  "
            f"({'underestimate' if row['residual'] > 0 else 'perfect fit' if row['residual'] == 0 else 'overestimate'})")
    print()

print("  Deliverable 2 – Saturday Prediction:")
print(f"    customers          = {saturday_customers}")
print(f"    predicted revenue  = {slope} × {saturday_customers} + {intercept} = ${saturday_predicted:,.1f}")
print()
print("  Deliverable 3 – Slope Interpretation:")
print(f"    The slope ({slope}) means each additional customer")
print(f"    contributes ${slope:.2f} to daily revenue on average.")
print(f"    The intercept ({intercept}) is the fixed base revenue")
print(f"    (e.g. catering orders) earned regardless of foot traffic.")
print("=" * 58)

# ── 6. VISUALIZATION ────────────────────────────────────────

fig = plt.figure(figsize=(13, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

# ── Left panel: main regression chart ──
ax1 = fig.add_subplot(gs[0])

# Draw the full regression line from x=0 to x=110
x_line = np.linspace(0, 110, 300)  # 300 smooth points
y_line = slope * x_line + intercept

ax1.plot(
    x_line, y_line,
    color="#1D9E75",    # green line
    lw=2,
    label="y = 15.5x + 120  (regression line)",
    zorder=1
)

# Plot actual revenue points (blue circles)
ax1.scatter(
    df["customers"], df["actual"],
    color="#378ADD", s=90, zorder=5,
    label="Actual revenue"
)

# Plot predicted revenue points (orange X marks)
ax1.scatter(
    df["customers"], df["predicted"],
    color="#D85A30", s=90, zorder=5,
    marker="x", linewidths=2.5,
    label="Predicted revenue"
)

# Draw vertical residual lines (the gap = residual)
for _, row in df.iterrows():
    if row["residual"] != 0:
        ax1.vlines(
            x=row["customers"],
            ymin=min(row["predicted"], row["actual"]),
            ymax=max(row["predicted"], row["actual"]),
            color="gray", lw=1.2, ls="dotted",
            label="Residual" if _ == 0 else "_nolegend_"
        )
        ax1.annotate(
            f"  residual\n  = ${row['residual']:+.0f}",
            xy=(row["customers"], (row["predicted"] + row["actual"]) / 2),
            fontsize=8, color="gray"
        )

# Plot Saturday as a purple star
ax1.scatter(
    [saturday_customers], [saturday_predicted],
    color="#534AB7", s=180, zorder=6,
    marker="*",
    label=f"Saturday prediction  ${saturday_predicted:,.0f}"
)
ax1.annotate(
    f"  $1,670\n  (100 customers)",
    xy=(saturday_customers, saturday_predicted),
    xytext=(-68, 10),
    textcoords="offset points",
    fontsize=8.5, color="#534AB7"
)

# Annotate Monday and Tuesday dots
for _, row in df.iterrows():
    ax1.annotate(
        f"  {row['day']}\n  actual=${int(row['actual'])}",
        xy=(row["customers"], row["actual"]),
        xytext=(6, -20),
        textcoords="offset points",
        fontsize=8, color="#185FA5"
    )

ax1.set_xlabel("Number of customers (x)")
ax1.set_ylabel("Daily revenue in $ (y)")
ax1.set_title("Regression Line: Revenue vs Customers")
ax1.legend(fontsize=8.5, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 115)
ax1.set_ylim(0, 1900)

# ── Right panel: residual bar chart ──
ax2 = fig.add_subplot(gs[1])

days      = df["day"].tolist() + ["Saturday\n(prediction)"]
actuals   = df["actual"].tolist() + [None]
predicteds= df["predicted"].tolist() + [saturday_predicted]

bar_w = 0.3
x_pos = range(len(days))

# Actual bars (blue) – only for Mon & Tue
ax2.bar(
    [p - bar_w/2 for p in x_pos[:2]],
    df["actual"], width=bar_w,
    color="#378ADD", label="Actual", alpha=0.9
)

# Predicted bars (orange) for all three
ax2.bar(
    [p + bar_w/2 for p in x_pos[:2]] + [x_pos[2]],
    df["predicted"].tolist() + [saturday_predicted],
    width=bar_w,
    color="#D85A30",
    label="Predicted / Forecasted", alpha=0.85
)

# Annotate bar values
for i, (act, pred) in enumerate(zip(df["actual"], df["predicted"])):
    ax2.text(i - bar_w/2, act + 10, f"${int(act)}", ha="center", fontsize=8, color="#185FA5")
    ax2.text(i + bar_w/2, pred + 10, f"${int(pred)}", ha="center", fontsize=8, color="#993C1D")

ax2.text(2, saturday_predicted + 10, f"${int(saturday_predicted)}", ha="center", fontsize=8, color="#993C1D")

# Shade the Saturday bar area to indicate it's a forecast
ax2.axvspan(1.6, 2.4, color="#EEEDFE", alpha=0.4, label="Forecast zone")

ax2.set_xticks(list(x_pos))
ax2.set_xticklabels(days, fontsize=9)
ax2.set_ylabel("Revenue ($)")
ax2.set_title("Actual vs Predicted Revenue")
ax2.legend(fontsize=8.5)
ax2.grid(True, axis="y", alpha=0.3)
ax2.set_ylim(0, 2000)

plt.suptitle("Task 3 – Regression Analysis & Prediction", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("task3_output.png", dpi=150, bbox_inches="tight")
print("\n  Chart saved as task3_output.png")
plt.show()
