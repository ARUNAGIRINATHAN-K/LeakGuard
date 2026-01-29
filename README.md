---
title: LeakGuard
emoji: 🛡️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "6.5.0"
app_file: app.py
pinned: false
---

<div text-align="center">

#  LeakGuard

*A web app that analyzes a CSV dataset BEFORE model training and detects silent data leakage risks that commonly cause models to fail in production.*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF6F00?style=flat&logo=gradio&logoColor=white)](https://gradio.app/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow?style=for-the-badge)](https://huggingface.co/spaces/ARUNAGIRINATHAN/leakguard)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/ARUNAGIRINATHAN-K/LeakGuard)

</div>

## What It Detects

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
