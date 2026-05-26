# 🪔 Pitambari Brass — Analytics Dashboard

**Data-Driven Decision Making Platform for a Premium Brass Lifestyle Brand**  
*Moradabad Heritage × Wellness × Premium Living*

---

## 📦 What's Inside

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application — all 6 tabs |
| `utils.py` | Preprocessing, ordinal maps, feature engineering |
| `Pitambari_Brass_Dataset_2000.csv` | Synthetic survey dataset (2000 respondents, 29 columns) |
| `requirements.txt` | All Python dependencies |

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Community Cloud

1. Push all files to a **public GitHub repository** (no sub-folders — all files in root)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repo → branch → `app.py` as the main file
5. Click **Deploy** — done in ~2 minutes

---

## 📊 Dashboard Tabs

| Tab | Analysis Type | Algorithms |
|-----|--------------|------------|
| 1 — Market Overview | Descriptive | Charts, KPIs, Funnel |
| 2 — Customer Profiles | Descriptive + Diagnostic | RFM, Correlation, Filters |
| 3 — Segment Explorer | Predictive | K-Means, PCA, Radar |
| 4 — Product Intelligence | Predictive + Prescriptive | Apriori ARM (support, confidence, lift) |
| 5 — Revenue Predictor | Predictive | Random Forest (accuracy, precision, recall, F1, ROC), XGBoost Regression |
| 6 — Score New Leads | Prescriptive | All models applied to new CSV upload |

---

## 🎯 Five Customer Personas

| Persona | Core Insight |
|---------|-------------|
| Daily Devotee | High puja frequency → pooja articles buyer |
| Wellness Seeker | Health-motivated → Kansa utensils |
| Gifting Executive | Business owner → corporate hampers |
| Heritage Collector | Traditional craft owner → décor buyer |
| Casual Explorer | Price-sensitive → entry-level products |

---

## 🛠️ Tech Stack

- **Streamlit** — dashboard framework  
- **Scikit-Learn** — Random Forest, K-Means, PCA, Ridge  
- **XGBoost** — spend regression  
- **MLxtend** — Apriori association rule mining  
- **Plotly** — all interactive visualizations  
- **Pandas / NumPy** — data processing  

---

*Built for Pitambari Brass, Moradabad, India*
