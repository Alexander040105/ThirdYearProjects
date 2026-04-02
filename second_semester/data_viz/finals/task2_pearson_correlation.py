# ============================================================
# Task 2 – Pearson Correlation Coefficient
# Data Visualization – Lesson 7
# ============================================================
# Requirements:  pip install matplotlib numpy pandas
# Run with:      python task2_pearson_correlation.py
# ============================================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 1. RAW DATA ─────────────────────────────────────────────

students     = ["A", "B", "C", "D", "E"]
study_hours  = [2,   4,   6,   8,   10]   # x – independent variable
exam_scores  = [70,  75,  85,  90,  95]   # y – dependent variable

# ── 2. BUILD DATAFRAME (pandas version) ─────────────────────

df = pd.DataFrame({
    "student":    students,
    "x":          study_hours,
    "y":          exam_scores,
})

# Add computed columns element-wise (no loop needed)
df["x2"] = df["x"] ** 2          # square of each x value
df["y2"] = df["y"] ** 2          # square of each y value
df["xy"] = df["x"] * df["y"]     # product of x and y for each row

# ── 3. COMPUTE SUMS ─────────────────────────────────────────

n      = len(df)
sum_x  = df["x"].sum()
sum_y  = df["y"].sum()
sum_x2 = df["x2"].sum()
sum_y2 = df["y2"].sum()
sum_xy = df["xy"].sum()

# ── 4. PEARSON FORMULA ──────────────────────────────────────

#         n(Σxy) − (Σx)(Σy)
# r = ─────────────────────────────────────────────
#      √[(n·Σx² − (Σx)²)(n·Σy² − (Σy)²)]

numerator   = n * sum_xy - sum_x * sum_y
denominator = math.sqrt(
    (n * sum_x2 - sum_x ** 2) *
    (n * sum_y2 - sum_y ** 2)
)
r = numerator / denominator

# Also compute using pandas built-in (for verification)
r_builtin = df["x"].corr(df["y"])

# ── 5. INTERPRET RESULT ─────────────────────────────────────

if   abs(r) >= 0.9: strength = "very strong"
elif abs(r) >= 0.7: strength = "strong"
elif abs(r) >= 0.4: strength = "moderate"
else:               strength = "weak"

direction = "positive" if r > 0 else "negative"

# ── 6. PRINT RESULTS ────────────────────────────────────────

print("=" * 52)
print("  TASK 2 – PEARSON CORRELATION COEFFICIENT")
print("=" * 52)
print()
print(df[["student","x","y","x2","y2","xy"]].to_string(index=False))
print()
print(f"  n       = {n}")
print(f"  Σx      = {sum_x}")
print(f"  Σy      = {sum_y}")
print(f"  Σx²     = {sum_x2}")
print(f"  Σy²     = {sum_y2}")
print(f"  Σxy     = {sum_xy}")
print()
print(f"  Numerator   = {n}×{sum_xy} − {sum_x}×{sum_y} = {numerator}")
print(f"  Denominator = {denominator:.4f}")
print(f"  r           = {r:.4f}  ({strength} {direction} correlation)")
print(f"  r (pandas)  = {r_builtin:.4f}  ← built-in verification")
print("=" * 52)

# ── 7. VISUALIZATION ────────────────────────────────────────

fig = plt.figure(figsize=(12, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# ── Left panel: scatter + trend line ──
ax1 = fig.add_subplot(gs[0])

# Plot each student as a dot
ax1.scatter(
    df["x"], df["y"],
    color="#378ADD",   # blue dots
    s=100,             # dot size (in points²)
    zorder=5,          # draw on top of the trend line
    label="Students"
)

# Label each dot with the student letter
for _, row in df.iterrows():
    ax1.annotate(
        row["student"],
        (row["x"], row["y"]),
        textcoords="offset points",
        xytext=(7, 4),        # shift 7px right, 4px up so text doesn't overlap dot
        fontsize=10
    )

# Compute and draw the trend line using numpy polyfit
slope_trend, intercept_trend = np.polyfit(df["x"], df["y"], 1)  # degree-1 = straight line
x_line = np.linspace(df["x"].min() - 1, df["x"].max() + 1, 200) # 200 smooth points
y_line = slope_trend * x_line + intercept_trend

ax1.plot(
    x_line, y_line,
    color="#D85A30",   # orange dashed line
    lw=1.8,
    ls="--",
    label=f"Trend line (slope={slope_trend:.2f})"
)

# Annotate the r value directly on the chart
ax1.text(
    2, 93,
    f"r = {r:.4f}\n({strength} {direction})",
    fontsize=9,
    color="#534AB7",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEEDFE", edgecolor="#AFA9EC")
)

ax1.set_xlabel("Study Hours (x)")
ax1.set_ylabel("Exam Score (y)")
ax1.set_title("Scatter Plot: Study Hours vs Exam Score")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── Right panel: bar chart of x² y² xy for each student ──
ax2 = fig.add_subplot(gs[1])

bar_width = 0.25
x_positions = range(len(students))

bars_x2 = ax2.bar(
    [p - bar_width for p in x_positions],
    df["x2"], width=bar_width,
    color="#85B7EB", label="x²"
)
bars_y2 = ax2.bar(
    x_positions,
    df["y2"], width=bar_width,
    color="#F0997B", label="y²",
    alpha=0.85
)
bars_xy = ax2.bar(
    [p + bar_width for p in x_positions],
    df["xy"], width=bar_width,
    color="#9FE1CB", label="xy"
)

ax2.set_xticks(x_positions)
ax2.set_xticklabels([f"Student {s}" for s in students])
ax2.set_ylabel("Value")
ax2.set_title("Computed columns: x², y², xy per student")
ax2.legend(fontsize=9)
ax2.grid(True, axis="y", alpha=0.3)

plt.suptitle("Task 2 – Pearson Correlation Analysis", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("task2_output.png", dpi=150, bbox_inches="tight")
print("\n  Chart saved as task2_output.png")
plt.show()
