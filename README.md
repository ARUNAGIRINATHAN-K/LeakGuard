---
title: LeakGuard
emoji: 🛡️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.5.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Detect silent data leakage risks before model training
---

# 🛡️ LeakGuard

**Detect silent data leakage risks BEFORE model training**

LeakGuard analyzes CSV datasets to detect four critical types of data leakage that commonly cause models to fail in production.

## 🎯 What It Detects

| Type | Detection Method | Risk Indicators |
|------|-----------------|-----------------|
| **Target Leakage** | Mutual Information, Pearson & Spearman correlation | Features containing direct/indirect target information |
| **Time Leakage** | Correlation drift, rolling window analysis | Future information leaking into past samples |
| **Duplicate Leakage** | Row hashing, entity ID overlap | Same samples appearing across splits |
| **Proxy Leakage** | Feature importance instability | Hidden proxies acting as target substitutes |

## Quick Start

1. Upload your CSV dataset
2. Select target column (required)
3. Select time & entity ID columns (optional)
4. Click **Analyze** to get instant results

## What You Get

- **Feature Risk Table** - Detailed risk assessment with MI, Pearson, Spearman scores
- **Visual Analytics** - 5 interactive charts showing leakage patterns
- **Risk Summary** - Overall leakage risk across all categories

## Tech Stack

- **Frontend**: Gradio
- **Data Processing**: Pandas, NumPy
- **ML Detection**: Scikit-learn (Random Forest)
- **Statistics**: SciPy (Spearman, MI)
- **Visualization**: Matplotlib

## Features

✅ CPU-only (no GPU required)  
✅ Explainable results with statistical basis  
✅ Fast analysis (seconds for typical datasets)  
✅ Production-ready architecture  

## Links

- **Live Demo**: [Hugging Face Space](https://huggingface.co/spaces/ARUNAGIRINATHAN/leakguard)
- **GitHub**: [Source Code](https://github.com/ARUNAGIRINATHAN-K/LeakGuard)

## 📝 License

Apache 2.0

---

**Built for Kaggle & Hugging Face Spaces** | © 2026
