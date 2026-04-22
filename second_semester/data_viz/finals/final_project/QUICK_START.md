# 🚀 QUICK START GUIDE - Dashboard Launch

## ⚡ Fastest Way (Windows)
```
1. Navigate to: C:\Users\Pixel-Tone\Documents\GitHub\ThirdYearProjects\second_semester\data_viz\finals\final_project
2. Double-click: run_dashboard.bat
3. Browser opens automatically to: http://localhost:8050/
```

---

## 📋 Alternative: Command Line

### Step 1: Install Dependencies
```bash
cd C:\Users\Pixel-Tone\Documents\GitHub\ThirdYearProjects\second_semester\data_viz\finals\final_project
pip install -r requirements_dashboard.txt
```

### Step 2: Run Dashboard
```bash
python dashboard.py
```

### Step 3: Open Browser
```
http://localhost:8050/
```

---

## 📊 Dashboard Structure

| Tab | Purpose | Answer |
|-----|---------|--------|
| **Tab 1** | Time series & correlation scatter | Q1: Do geopolitical events affect shipping? |
| **Tab 2** | Country correlations & heatmap | Q2: What is the Pearson coefficient? |
| **Tab 3** | Detailed trends & rankings | Granular analysis |
| **Tab 4** | Insights & findings | Summary & interpretation |

---

## 🎯 Key Metrics Displayed

- **Pearson Correlation Coefficient** (Global)
- **Average GPR Index** (Geopolitical Risk)
- **Total Trade Volume** (in Billions USD)

---

## 📈 What Each Tab Shows

### Tab 1: Geopolitical Events & Shipping
- Dual-axis time series (GPR + Trade)
- Scatter plot with trend line
- Correlation strength (r-value)
- Statistical significance (p-value)

### Tab 2: Pearson Coefficient Analysis
- Top 15 countries by correlation
- Summary heatmap
- Country-level statistics

### Tab 3: Detailed Analysis
- GPR trend over time
- Trade volume trend
- Top trading countries
- Import/Export distribution

### Tab 4: Summary & Interpretation
- Research question answers
- Statistical interpretation guide
- Data coverage statistics

---

## 🔍 Understanding the Results

### Correlation Coefficient (r):
- **+1.0 to +0.7** = Strong positive (events reduce trade)
- **+0.7 to +0.3** = Moderate positive
- **+0.3 to 0.0** = Weak positive
- **0.0 to -0.3** = Weak negative
- **-0.3 to -0.7** = Moderate negative (events increase trade)
- **-0.7 to -1.0** = Strong negative

### P-Value:
- **< 0.05** = Statistically significant ✅
- **≥ 0.05** = Not statistically significant ❌

---

## 📁 Files Included

| File | Purpose |
|------|---------|
| `dashboard.py` | Main application |
| `requirements_dashboard.txt` | Python packages needed |
| `run_dashboard.bat` | One-click launcher |
| `README_DASHBOARD.md` | Full documentation |
| `DASHBOARD_SUMMARY.md` | Detailed feature guide |
| `QUICK_START.md` | This file |
| `trademerch_gpr.csv` | Data file (required) |

---

## ✅ Checklist

- [ ] Navigate to final_project folder
- [ ] Ensure all CSV files are present
- [ ] Run dashboard (BAT file or python command)
- [ ] Browser opens to http://localhost:8050/
- [ ] Explore all 4 tabs
- [ ] Export insights for your report

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 8050 in use | Edit dashboard.py, change port=8051 |
| Missing modules | Run: `pip install -r requirements_dashboard.txt` |
| No data shown | Check CSV files are in same folder |
| Slow performance | Close other browser tabs, restart dashboard |
| Can't find folder | Paste path in Windows Explorer: C:\Users\Pixel-Tone\Documents\GitHub\ThirdYearProjects\second_semester\data_viz\finals\final_project |

---

## 💾 Your Data

**Combined Dataset:** trademerch_gpr.csv
- 440 records across 2021-2025
- 44 countries analyzed
- Trade values in millions USD
- GPRHI (Geopolitical Risk High Index)

---

## 📞 Need Help?

1. Check **README_DASHBOARD.md** for full documentation
2. Review **DASHBOARD_SUMMARY.md** for features
3. See **Troubleshooting** section above

---

**Ready? Double-click run_dashboard.bat or run: `python dashboard.py`**

Enjoy your analysis! 📊✨
