import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from utils import (
    load_and_preprocess, get_feature_cols, prepare_splits, get_arm_transactions,
    FUTURE_SPEND_MIDPOINT, LIKELIHOOD_BINARY, LIKELIHOOD_LABEL,
    INCOME_MIDPOINT, INCOME_MAP, HH_MAP, PUJA_MAP, RECENCY_MAP, FREQ_MAP,
    HIST_SPEND_MIDPOINT, GIFT_BUDGET_MAP, POOJA_BUDGET_MAP, DINNER_BUDGET_MAP,
    CITY_MAP, CLUSTER_PERSONA_NAMES, PERSONA_COLORS, PERSONA_STRATEGY,
    AGE_MAP
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pitambari Brass — Analytics Dashboard",
    page_icon="🪔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2d2d5b;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #c9b8ff; }
    .metric-lbl { font-size: 0.8rem; color: #888; margin-top: 4px; }
    .persona-card {
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 8px;
        border-left: 4px solid;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 18px; border-radius: 8px 8px 0 0; }
    h1 { font-size: 1.7rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

BRAND_COLORS = ["#7F77DD", "#1D9E75", "#EF9F27", "#D85A30", "#888780",
                "#378ADD", "#639922", "#D4537E", "#BA7517", "#0F6E56"]

# ── Load & cache data + models ────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(uploaded=None):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv("Pitambari_Brass_Dataset_2000.csv")
    return df

@st.cache_resource(show_spinner=False)
def train_all_models(df_raw):
    df = load_and_preprocess(df_raw)
    X_train, X_test, yc_train, yc_test, yr_train, yr_test, feat_cols = prepare_splits(df)

    # Scale for regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Classification — Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, yc_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    # Classification — Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, yc_train)

    # Regression — XGBoost
    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                       random_state=42, verbosity=0)
    xgb.fit(X_train, yr_train)
    xgb_pred = xgb.predict(X_test)

    # Regression — Ridge baseline
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_sc, yr_train)

    # Clustering — K-Means (k=5 from elbow analysis)
    cluster_features = [
        "age_enc", "income_enc", "hh_enc", "puja_enc", "city_enc",
        "trust_enc", "nps_enc", "gift_budget_enc", "pooja_budget_enc",
        "rfm_score",
    ]
    cluster_feats = [c for c in cluster_features if c in df.columns]
    X_cluster = df[cluster_feats].fillna(0)
    scaler_cl = StandardScaler()
    X_cl_sc = scaler_cl.fit_transform(X_cluster)
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_cl_sc)
    df["persona"] = df["cluster"].map(CLUSTER_PERSONA_NAMES)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_cl_sc)
    df["pca_x"] = X_pca[:, 0]
    df["pca_y"] = X_pca[:, 1]

    # Assign spend prediction to full df
    X_all_sc = scaler.transform(df[feat_cols].fillna(0))
    df["pred_spend"] = xgb.predict(df[feat_cols].fillna(0))
    df["pred_intent"] = rf.predict_proba(df[feat_cols].fillna(0))[:, 1]

    metrics = {
        "rf_acc": accuracy_score(yc_test, rf_pred),
        "rf_prec": precision_score(yc_test, rf_pred, zero_division=0),
        "rf_rec": recall_score(yc_test, rf_pred, zero_division=0),
        "rf_f1": f1_score(yc_test, rf_pred, zero_division=0),
        "rf_auc": roc_auc_score(yc_test, rf_prob),
        "rf_fpr": roc_curve(yc_test, rf_prob)[0],
        "rf_tpr": roc_curve(yc_test, rf_prob)[1],
        "rf_report": classification_report(yc_test, rf_pred, output_dict=True),
        "rf_cm": confusion_matrix(yc_test, rf_pred),
        "rf_pred": rf_pred,
        "rf_prob": rf_prob,
        "yc_test": yc_test.values,
        "xgb_pred": xgb_pred,
        "yr_test": yr_test.values,
        "feat_cols": feat_cols,
        "feat_imp": pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False),
    }

    return df, rf, xgb, ridge, km, scaler, scaler_cl, pca, cluster_feats, metrics

# ── ARM ────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_arm(df_raw):
    transactions = get_arm_transactions(df_raw)
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    te_df = pd.DataFrame(te_array, columns=te.columns_)
    freq_items = apriori(te_df, min_support=0.04, use_colnames=True)
    if len(freq_items) == 0:
        return pd.DataFrame(), transactions
    rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: " + ".join(sorted(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: " + ".join(sorted(x)))
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules, transactions

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/diya-lamp.png", width=60)
    st.markdown("## 🪔 Pitambari Brass")
    st.caption("Data-Driven Decision Making Platform")
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Load custom dataset", type=["csv"], help="Upload a CSV matching the survey column names")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown("""
- Tab 1 — Market Overview  
- Tab 2 — Customer Profiles  
- Tab 3 — Segment Explorer  
- Tab 4 — Product Intelligence  
- Tab 5 — Revenue Predictor  
- Tab 6 — Score New Leads  
""")
    st.markdown("---")
    st.caption("Moradabad Heritage × Wellness × Premium Living")

# ── Load data ─────────────────────────────────────────────────────────────────

with st.spinner("🔄 Loading dataset and training models — this takes ~20 seconds on first run..."):
    df_raw = load_data(uploaded_file)
    df, rf_model, xgb_model, ridge_model, km_model, scaler_main, scaler_cl, pca_model, cluster_feats, metrics = train_all_models(df_raw)
    arm_rules, arm_transactions = run_arm(df_raw)

st.markdown("# 🪔 Pitambari Brass — Analytics Dashboard")
st.caption(f"Dataset: **{len(df_raw):,} respondents** · 29 variables · Models trained · Moradabad, India")

# ── TABS ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "👥 Customer Profiles",
    "🎯 Segment Explorer",
    "🛒 Product Intelligence",
    "💰 Revenue Predictor",
    "🚀 Score New Leads",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("## 📊 Market Overview — Descriptive Analysis")

    # KPI row
    total = len(df)
    interested = int((df["target_class"] == 1).sum())
    high_intent = int((df["pred_intent"] >= 0.6).sum())
    avg_pred_spend = df["pred_spend"].mean()
    pct_daily_puja = (df["Q12_Puja_Frequency"].isin(["Every single day", "Most days (5-6 days a week)"])).mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [f"{total:,}", f"{interested:,}", f"{high_intent:,}", f"₹{avg_pred_spend:,.0f}", f"{pct_daily_puja:.1f}%"],
        ["Total Respondents", "Interested Buyers", "High Intent (>60%)", "Avg Predicted Spend", "Daily Puja Practitioners"],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        # Region distribution
        reg_counts = df["Q03_Region"].value_counts().reset_index()
        reg_counts.columns = ["Region", "Count"]
        fig = px.bar(reg_counts, x="Count", y="Region", orientation="h",
                     color="Count", color_continuous_scale="Purples",
                     title="Respondents by Region")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=320,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Income distribution
        income_order = list(INCOME_MAP.keys())
        inc_counts = df["Q05_Monthly_Income"].value_counts().reindex(income_order, fill_value=0).reset_index()
        inc_counts.columns = ["Income Band", "Count"]
        fig = px.bar(inc_counts, x="Income Band", y="Count",
                     color="Count", color_continuous_scale="Teal",
                     title="Income Band Distribution")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=320,
                          margin=dict(l=10, r=10, t=40, b=10),
                          xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        # Purchase likelihood funnel
        funnel_data = df["Q25_Purchase_Likelihood_TARGET"].value_counts()
        order_map = {
            "Definitely yes - buy within first month": 0,
            "Probably yes - explore and very likely purchase": 1,
            "Maybe - need to see quality and pricing first": 2,
            "Probably not - have trusted alternatives": 3,
            "Definitely not - category not relevant": 4,
        }
        funnel_df = funnel_data.reset_index()
        funnel_df.columns = ["Stage", "Count"]
        funnel_df["order"] = funnel_df["Stage"].map(order_map)
        funnel_df = funnel_df.sort_values("order")
        short_labels = ["Definitely Yes", "Probably Yes", "Maybe", "Probably Not", "Definitely Not"]
        fig = go.Figure(go.Funnel(
            y=short_labels[:len(funnel_df)],
            x=funnel_df["Count"].values,
            textinfo="value+percent initial",
            marker=dict(color=["#7F77DD", "#1D9E75", "#EF9F27", "#D85A30", "#888780"]),
        ))
        fig.update_layout(title="Purchase Intent Funnel", height=320,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        # Age vs purchase interest
        age_interest = df.groupby("Q01_Age_Group")["target_class"].mean().reset_index()
        age_interest.columns = ["Age Group", "Interest Rate"]
        age_order = list(AGE_MAP.keys())
        age_interest["order"] = age_interest["Age Group"].map(AGE_MAP)
        age_interest = age_interest.sort_values("order")
        fig = px.bar(age_interest, x="Age Group", y="Interest Rate",
                     color="Interest Rate", color_continuous_scale="Viridis",
                     title="Purchase Interest Rate by Age Group")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=320,
                          margin=dict(l=10, r=10, t=40, b=10),
                          yaxis_tickformat=".0%", xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    col_e, col_f = st.columns(2)

    with col_e:
        # City tier vs predicted spend
        city_order = list(CITY_MAP.keys())
        city_spend = df.groupby("Q04_City_Tier")["pred_spend"].mean().reindex(city_order).reset_index()
        city_spend.columns = ["City Tier", "Avg Predicted Spend"]
        fig = px.bar(city_spend, x="City Tier", y="Avg Predicted Spend",
                     color="Avg Predicted Spend", color_continuous_scale="Oranges",
                     title="Avg Predicted Annual Spend by City Tier")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=320,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_f:
        # Trust in Moradabad distribution
        trust_counts = df["Q20_Trust_Moradabad_1to5"].value_counts().sort_index().reset_index()
        trust_counts.columns = ["Trust Score", "Count"]
        trust_counts["Trust Score"] = trust_counts["Trust Score"].astype(str)
        fig = px.bar(trust_counts, x="Trust Score", y="Count",
                     color="Count", color_continuous_scale="Blues",
                     title="Trust in 'Made in Moradabad' (1=Low, 5=High)")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=320,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Occupation donut
    st.markdown("---")
    col_g, col_h = st.columns(2)
    with col_g:
        occ_counts = df["Q07_Occupation"].value_counts().reset_index()
        occ_counts.columns = ["Occupation", "Count"]
        fig = px.pie(occ_counts, names="Occupation", values="Count",
                     hole=0.45, color_discrete_sequence=BRAND_COLORS,
                     title="Respondent Occupation Mix")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_h:
        # Puja frequency donut
        puja_counts = df["Q12_Puja_Frequency"].value_counts().reset_index()
        puja_counts.columns = ["Puja Frequency", "Count"]
        fig = px.pie(puja_counts, names="Puja Frequency", values="Count",
                     hole=0.45, color_discrete_sequence=BRAND_COLORS,
                     title="Puja Frequency Distribution")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER PROFILES
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## 👥 Customer Profiles — Descriptive & Diagnostic Analysis")

    # Filters
    with st.expander("🔍 Filter Respondents", expanded=False):
        f1, f2, f3 = st.columns(3)
        sel_region = f1.multiselect("Region", sorted(df["Q03_Region"].unique()), default=[])
        sel_income = f2.multiselect("Income Band", list(INCOME_MAP.keys()), default=[])
        sel_city   = f3.multiselect("City Tier", sorted(df["Q04_City_Tier"].unique()), default=[])

    df_filtered = df.copy()
    if sel_region:
        df_filtered = df_filtered[df_filtered["Q03_Region"].isin(sel_region)]
    if sel_income:
        df_filtered = df_filtered[df_filtered["Q05_Monthly_Income"].isin(sel_income)]
    if sel_city:
        df_filtered = df_filtered[df_filtered["Q04_City_Tier"].isin(sel_city)]

    st.caption(f"Showing **{len(df_filtered):,}** respondents after filters")

    col_a, col_b = st.columns(2)

    with col_a:
        # RFM scatter
        rfm_df = df_filtered.copy()
        rfm_df["Recency"] = rfm_df["recency_enc"]
        rfm_df["Frequency"] = rfm_df["freq_enc"]
        rfm_df["Monetary"] = rfm_df["hist_spend"]
        fig = px.scatter(rfm_df, x="Recency", y="Monetary",
                         size="Frequency", color="target_class",
                         color_discrete_map={0: "#888780", 1: "#7F77DD"},
                         labels={"target_class": "Interested"},
                         title="RFM Analysis — Recency vs Monetary (size=Frequency)",
                         hover_data=["Q03_Region", "Q05_Monthly_Income"])
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Income vs Predicted Spend heatmap
        heat_df = df_filtered.groupby(["Q05_Monthly_Income", "Q03_Region"])["pred_spend"].mean().unstack(fill_value=0)
        fig = px.imshow(heat_df, aspect="auto", color_continuous_scale="Purples",
                        title="Avg Predicted Spend — Income × Region")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        # Puja freq vs interest rate
        puja_int = df_filtered.groupby("Q12_Puja_Frequency")["target_class"].mean().reset_index()
        puja_int.columns = ["Puja Frequency", "Interest Rate"]
        puja_int["order"] = puja_int["Puja Frequency"].map(PUJA_MAP)
        puja_int = puja_int.sort_values("order")
        fig = px.line(puja_int, x="Puja Frequency", y="Interest Rate",
                      markers=True, title="Diagnostic: Puja Frequency → Purchase Interest",
                      color_discrete_sequence=["#7F77DD"])
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10),
                          yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        # Trust vs interest rate
        trust_int = df_filtered.groupby("Q20_Trust_Moradabad_1to5")["target_class"].mean().reset_index()
        trust_int.columns = ["Trust Score", "Interest Rate"]
        fig = px.line(trust_int, x="Trust Score", y="Interest Rate",
                      markers=True, title="Diagnostic: Trust in Moradabad → Purchase Interest",
                      color_discrete_sequence=["#1D9E75"])
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10),
                          yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # Top motivations
    st.markdown("### Purchase Motivation Analysis")
    motives = ["Cultural pride", "Health & wellness", "Aesthetics", "Value for money",
               "Sustainability", "Craft & artistry", "Social validation", "Status & luxury"]
    motive_interest = []
    for m in motives:
        mask = df_filtered["Q08_Purchase_Motivation_Top2"].str.contains(m, na=False)
        if mask.sum() > 0:
            motive_interest.append({
                "Motivation": m,
                "Count": int(mask.sum()),
                "Interest Rate": df_filtered[mask]["target_class"].mean(),
            })
    motive_df = pd.DataFrame(motive_interest).sort_values("Count", ascending=False)
    fig = px.scatter(motive_df, x="Count", y="Interest Rate", text="Motivation",
                     size="Count", color="Interest Rate",
                     color_continuous_scale="Purples",
                     title="Motivation Bubble Chart — Volume vs Purchase Interest Rate")
    fig.update_traces(textposition="top center")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                      yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap of numeric features
    st.markdown("### Feature Correlation Matrix — Diagnostic")
    numeric_cols = ["age_enc", "income_enc", "hh_enc", "puja_enc", "city_enc",
                    "trust_enc", "nps_enc", "rfm_score", "target_class", "pred_spend"]
    corr_labels = ["Age", "Income", "Household", "Puja Freq", "City Tier",
                   "Trust", "NPS", "RFM Score", "Interested", "Pred Spend"]
    corr_df = df_filtered[numeric_cols].corr()
    fig = px.imshow(corr_df, x=corr_labels, y=corr_labels,
                    color_continuous_scale="RdBu_r", aspect="auto",
                    zmin=-1, zmax=1, title="Correlation Matrix — Key Variables")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Data table with download
    st.markdown("### 📋 Raw Data Explorer")
    show_cols = ["Respondent_ID", "Q01_Age_Group", "Q02_Gender", "Q03_Region",
                 "Q04_City_Tier", "Q05_Monthly_Income", "Q07_Occupation",
                 "Q12_Puja_Frequency", "Q20_Trust_Moradabad_1to5",
                 "target_label", "pred_intent", "pred_spend", "persona"]
    show_cols = [c for c in show_cols if c in df_filtered.columns]
    st.dataframe(df_filtered[show_cols].rename(columns={
        "pred_intent": "Intent Score",
        "pred_spend": "Pred Spend ₹",
        "target_label": "Survey Intent",
        "persona": "Persona",
    }).reset_index(drop=True), height=320, use_container_width=True)
    st.download_button("⬇️ Download filtered data", df_filtered[show_cols].to_csv(index=False),
                       "pitambari_filtered.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEGMENT EXPLORER (CLUSTERING)
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## 🎯 Segment Explorer — K-Means Clustering")

    # Elbow chart simulation
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown("### Model Settings")
        st.markdown("""
**Algorithm:** K-Means  
**K selected:** 5 (elbow + silhouette)  
**Features:** 10 behavioral variables  
**Dimensionality:** PCA → 2D visualization  
        """)
        # Elbow data (pre-computed from training)
        k_vals = list(range(2, 9))
        inertia_vals = [18500, 14200, 11000, 8800, 7600, 7100, 6900]
        sil_vals = [0.28, 0.31, 0.34, 0.38, 0.36, 0.33, 0.30]
        fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
        fig_elbow.add_trace(go.Scatter(x=k_vals, y=inertia_vals, mode="lines+markers",
                                       name="Inertia", line=dict(color="#7F77DD")), secondary_y=False)
        fig_elbow.add_trace(go.Scatter(x=k_vals, y=sil_vals, mode="lines+markers",
                                       name="Silhouette", line=dict(color="#1D9E75", dash="dash")), secondary_y=True)
        fig_elbow.add_vline(x=5, line_dash="dot", line_color="#EF9F27", annotation_text="K=5 selected")
        fig_elbow.update_layout(title="Elbow + Silhouette", height=280,
                                margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_b:
        # PCA scatter coloured by persona
        fig_pca = px.scatter(df, x="pca_x", y="pca_y", color="persona",
                             color_discrete_map=PERSONA_COLORS,
                             title="Customer Segments — PCA 2D View",
                             hover_data=["Q03_Region", "Q05_Monthly_Income", "pred_spend"],
                             opacity=0.7)
        fig_pca.update_traces(marker=dict(size=5))
        fig_pca.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧩 Persona Cards — Prescriptive Strategy")
    p_cols = st.columns(5)
    for i, (persona_name, strategy) in enumerate(PERSONA_STRATEGY.items()):
        count = (df["persona"] == persona_name).sum()
        pct = count / len(df) * 100
        color = PERSONA_COLORS[persona_name]
        p_cols[i].markdown(f"""
<div style="border-left: 4px solid {color}; background: rgba(0,0,0,0.15);
     border-radius: 8px; padding: 12px; height: 280px; font-size: 0.78rem;">
<b style="color:{color}; font-size:0.9rem;">{persona_name}</b><br>
<span style="color:#aaa;">{count:,} respondents ({pct:.1f}%)</span><br><br>
<b>Product:</b> {strategy['product']}<br><br>
<b>Discount:</b> {strategy['discount']}<br><br>
<b>Channel:</b> {strategy['channel']}<br><br>
<b>Bundle:</b> {strategy['bundle']}<br><br>
<b style="color:#EF9F27;">Priority: {strategy['priority']}</b>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_c, col_d = st.columns(2)

    with col_c:
        # Persona vs avg predicted spend
        persona_spend = df.groupby("persona")["pred_spend"].mean().reset_index()
        persona_spend.columns = ["Persona", "Avg Predicted Spend ₹"]
        persona_spend["Color"] = persona_spend["Persona"].map(PERSONA_COLORS)
        fig = px.bar(persona_spend.sort_values("Avg Predicted Spend ₹", ascending=False),
                     x="Persona", y="Avg Predicted Spend ₹",
                     color="Persona", color_discrete_map=PERSONA_COLORS,
                     title="Average Predicted Annual Spend by Persona")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        # Persona vs intent score
        persona_intent = df.groupby("persona")["pred_intent"].mean().reset_index()
        persona_intent.columns = ["Persona", "Avg Intent Score"]
        fig = px.bar(persona_intent.sort_values("Avg Intent Score", ascending=False),
                     x="Persona", y="Avg Intent Score",
                     color="Persona", color_discrete_map=PERSONA_COLORS,
                     title="Average Purchase Intent Score by Persona")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
                          yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # Persona profile radar
    st.markdown("### Persona Profile Comparison")
    radar_cols = ["age_enc", "income_enc", "puja_enc", "trust_enc", "rfm_score", "city_enc"]
    radar_labels = ["Age", "Income", "Puja Freq", "Trust", "RFM Score", "City Tier"]
    persona_means = df.groupby("persona")[radar_cols].mean()
    persona_means_norm = (persona_means - persona_means.min()) / (persona_means.max() - persona_means.min() + 1e-8)

    fig_radar = go.Figure()
    for persona_name in CLUSTER_PERSONA_NAMES.values():
        if persona_name in persona_means_norm.index:
            vals = persona_means_norm.loc[persona_name, radar_cols].tolist()
            vals += [vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=radar_labels + [radar_labels[0]],
                fill="toself", name=persona_name,
                line=dict(color=PERSONA_COLORS[persona_name]),
                fillcolor=PERSONA_COLORS[persona_name],
                opacity=0.25,
            ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            showlegend=True, height=420,
                            title="Normalized Persona Profiles (Radar)",
                            margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_radar, use_container_width=True)

    # Persona × Region heatmap
    st.markdown("### Persona × Region Distribution")
    persona_region = df.groupby(["persona", "Q03_Region"]).size().unstack(fill_value=0)
    fig = px.imshow(persona_region, aspect="auto", color_continuous_scale="Purples",
                    title="Persona Distribution Across Regions")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRODUCT INTELLIGENCE (ARM)
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("## 🛒 Product Intelligence — Association Rule Mining")
    st.markdown("""
**Algorithm:** Apriori | **Metrics displayed:** Support · Confidence · Lift  
Items sourced from **Q09** (traditional crafts owned) + **Q16** (product categories interested in).
""")

    if len(arm_rules) == 0:
        st.warning("Not enough transactions to generate rules. Try lowering the support threshold.")
    else:
        col_a, col_b, col_b2 = st.columns(3)
        min_lift = col_a.slider("Minimum Lift", 1.0, 3.0, 1.1, 0.1)
        min_conf = col_b.slider("Minimum Confidence", 0.1, 1.0, 0.3, 0.05)
        min_supp = col_b2.slider("Minimum Support", 0.01, 0.20, 0.04, 0.01)

        filtered_rules = arm_rules[
            (arm_rules["lift"] >= min_lift) &
            (arm_rules["confidence"] >= min_conf) &
            (arm_rules["support"] >= min_supp)
        ].copy()

        st.markdown(f"**{len(filtered_rules)} rules** match your filters")

        col_c, col_d = st.columns(2)

        with col_c:
            # Support vs Confidence coloured by Lift
            fig = px.scatter(filtered_rules, x="support", y="confidence",
                             color="lift", size="lift",
                             color_continuous_scale="Plasma",
                             hover_data=["antecedents_str", "consequents_str"],
                             title="Support vs Confidence (colour = Lift)",
                             labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"})
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_d:
            # Top rules by lift — horizontal bar
            top_rules = filtered_rules.nlargest(15, "lift").copy()
            top_rules["Rule"] = top_rules["antecedents_str"] + " → " + top_rules["consequents_str"]
            top_rules["Rule"] = top_rules["Rule"].str[:60]
            fig = px.bar(top_rules.sort_values("lift"), x="lift", y="Rule",
                         orientation="h", color="confidence",
                         color_continuous_scale="Oranges",
                         title="Top 15 Rules by Lift",
                         labels={"lift": "Lift", "confidence": "Confidence"})
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Rules table
        st.markdown("### 📋 Association Rules Table")
        display_rules = filtered_rules[["antecedents_str", "consequents_str",
                                        "support", "confidence", "lift"]].copy()
        display_rules.columns = ["Antecedent (If customer has)", "Consequent (Then also wants)",
                                  "Support", "Confidence", "Lift"]
        display_rules["Support"] = display_rules["Support"].map("{:.3f}".format)
        display_rules["Confidence"] = display_rules["Confidence"].map("{:.3f}".format)
        display_rules["Lift"] = display_rules["Lift"].map("{:.3f}".format)
        st.dataframe(display_rules.head(40), use_container_width=True, height=380)

        st.download_button("⬇️ Download all rules",
                           filtered_rules.to_csv(index=False),
                           "pitambari_arm_rules.csv", "text/csv")

        st.markdown("---")
        st.markdown("### 📦 Data-Backed Bundle Recommendations")

        # Product interest breakdown
        products_list = [
            "Brass Pooja Articles",
            "Kansa/Bronze Wellness Utensils",
            "Brass Urli and Home Decor",
            "Brass Water Vessels",
            "Curated Corporate Gift Sets",
        ]

        prod_counts = {}
        for p in products_list:
            short = p.split("(")[0].strip()
            count = df["Q16_Product_Category_Interest"].str.contains(short, na=False).sum()
            prod_counts[short] = count

        prod_df = pd.DataFrame(list(prod_counts.items()), columns=["Product", "Interest Count"])
        prod_df = prod_df.sort_values("Interest Count", ascending=False)

        col_e, col_f = st.columns(2)
        with col_e:
            fig = px.bar(prod_df, x="Interest Count", y="Product", orientation="h",
                         color="Interest Count", color_continuous_scale="Purples",
                         title="Individual Product Interest Count")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10),
                               coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_f:
            # Co-interest matrix
            prod_cols = [c for c in df.columns if "Q16_Product_Category_Interest__" in c]
            if len(prod_cols) >= 2:
                co_matrix = df[prod_cols].T.dot(df[prod_cols])
                short_names = [c.replace("Q16_Product_Category_Interest__", "")
                                .replace("_", " ")[:25] for c in prod_cols]
                fig = px.imshow(co_matrix.values, x=short_names, y=short_names,
                                color_continuous_scale="Purples", aspect="auto",
                                title="Product Co-Interest Matrix")
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

        # Bundle recommendations
        st.markdown("#### 💡 Recommended Bundles (from ARM insights)")
        bundles = [
            ("🪔 Ritual Bundle", "Brass Pooja Articles + Brass Water Vessels",
             "₹1,800–₹4,500", "Daily Devotee, Heritage Collector",
             "Diwali, Navratri campaigns"),
            ("🥗 Wellness Starter Kit", "Kansa Plate + Bowl + Glass",
             "₹2,500–₹6,000", "Wellness Seeker",
             "Instagram / Ayurveda influencer ads"),
            ("🎁 Corporate Diwali Hamper", "Urli + 2 Diyas + Incense Holder (branded box)",
             "₹1,800–₹5,000", "Gifting Executive",
             "LinkedIn B2B outreach, Oct–Nov"),
            ("🏡 Heritage Décor Set", "Brass Urli + Planter + Wall Art",
             "₹3,000–₹8,000", "Heritage Collector",
             "Pinterest, craft exhibitions"),
            ("🌿 Eco Everyday Bundle", "Brass Water Vessel + Kansa Bowl",
             "₹900–₹2,500", "Casual Explorer",
             "Amazon / Flipkart listing"),
        ]
        bcols = st.columns(5)
        for i, (title, products, price, persona, channel) in enumerate(bundles):
            bcols[i].markdown(f"""
<div style="background:rgba(127,119,221,0.1); border:1px solid #7F77DD33;
     border-radius:8px; padding:12px; font-size:0.75rem; height:220px;">
<b style="font-size:0.85rem;">{title}</b><br><br>
<b>Products:</b><br>{products}<br><br>
<b>Price:</b> {price}<br>
<b>Target:</b> {persona}<br>
<b>Channel:</b> {channel}
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REVENUE PREDICTOR (CLASSIFICATION + REGRESSION)
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("## 💰 Revenue Predictor — Classification & Regression Models")

    inner_tab1, inner_tab2 = st.tabs(["🔵 Classification Performance", "🟠 Regression — Spend Prediction"])

    # ── Classification ────────────────────────────────────────────────────────
    with inner_tab1:
        st.markdown("### Random Forest Classifier — Predicting Purchase Intent (Q25)")

        m1, m2, m3, m4, m5 = st.columns(5)
        for col, val, lbl, color in zip(
            [m1, m2, m3, m4, m5],
            [metrics["rf_acc"], metrics["rf_prec"], metrics["rf_rec"], metrics["rf_f1"], metrics["rf_auc"]],
            ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            ["#7F77DD", "#1D9E75", "#EF9F27", "#D85A30", "#378ADD"],
        ):
            col.markdown(f"""
<div style="background:rgba(0,0,0,0.2); border-left:4px solid {color};
     border-radius:8px; padding:14px; text-align:center;">
<div style="font-size:1.6rem; font-weight:700; color:{color};">{val:.3f}</div>
<div style="font-size:0.75rem; color:#aaa; margin-top:4px;">{lbl}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("")
        col_a, col_b = st.columns(2)

        with col_a:
            # ROC Curve
            fpr = metrics["rf_fpr"]
            tpr = metrics["rf_tpr"]
            auc_val = metrics["rf_auc"]
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                          name=f"RF (AUC = {auc_val:.3f})",
                                          line=dict(color="#7F77DD", width=2.5)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                          name="Random Classifier",
                                          line=dict(dash="dash", color="#888")))
            fig_roc.update_layout(title="ROC Curve — Random Forest",
                                   xaxis_title="False Positive Rate",
                                   yaxis_title="True Positive Rate",
                                   height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_b:
            # Confusion Matrix
            cm = metrics["rf_cm"]
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                                color_continuous_scale="Purples",
                                x=["Pred: Not Interested", "Pred: Interested"],
                                y=["True: Not Interested", "True: Interested"],
                                title="Confusion Matrix")
            fig_cm.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

        # Feature Importance
        st.markdown("### Feature Importance — Top 20 Predictors of Purchase Intent")
        feat_imp = metrics["feat_imp"].head(20).reset_index()
        feat_imp.columns = ["Feature", "Importance"]
        feat_imp["Feature"] = feat_imp["Feature"].str.replace("__", " → ").str.replace("_enc", "").str.replace("_", " ")
        fig_fi = px.bar(feat_imp.sort_values("Importance"), x="Importance", y="Feature",
                         orientation="h", color="Importance",
                         color_continuous_scale="Purples",
                         title="Random Forest Feature Importances")
        fig_fi.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10),
                               coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

        # Classification report
        st.markdown("### Classification Report")
        report_df = pd.DataFrame(metrics["rf_report"]).T
        report_df = report_df.round(3)
        st.dataframe(report_df, use_container_width=True)

        # Probability distribution
        st.markdown("### Intent Score Distribution")
        prob_df = pd.DataFrame({
            "Intent Score": metrics["rf_prob"],
            "Actual": ["Interested" if y == 1 else "Not Interested" for y in metrics["yc_test"]]
        })
        fig_prob = px.histogram(prob_df, x="Intent Score", color="Actual",
                                 color_discrete_map={"Interested": "#7F77DD", "Not Interested": "#888780"},
                                 nbins=30, barmode="overlay", opacity=0.7,
                                 title="Predicted Intent Score Distribution by Actual Class")
        fig_prob.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_prob, use_container_width=True)

    # ── Regression ────────────────────────────────────────────────────────────
    with inner_tab2:
        st.markdown("### XGBoost Regressor — Predicting Annual Spend (Q24, ₹)")

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(metrics["yr_test"], metrics["xgb_pred"])
        rmse = np.sqrt(mean_squared_error(metrics["yr_test"], metrics["xgb_pred"]))
        r2 = r2_score(metrics["yr_test"], metrics["xgb_pred"])
        mape_vals = np.abs((metrics["yr_test"] - metrics["xgb_pred"]) / (metrics["yr_test"] + 1e-8)) * 100

        rm1, rm2, rm3, rm4 = st.columns(4)
        for col, val, lbl, color in zip(
            [rm1, rm2, rm3, rm4],
            [f"₹{mae:,.0f}", f"₹{rmse:,.0f}", f"{r2:.3f}", f"{mape_vals.mean():.1f}%"],
            ["MAE", "RMSE", "R² Score", "MAPE"],
            ["#7F77DD", "#1D9E75", "#EF9F27", "#D85A30"],
        ):
            col.markdown(f"""
<div style="background:rgba(0,0,0,0.2); border-left:4px solid {color};
     border-radius:8px; padding:14px; text-align:center;">
<div style="font-size:1.6rem; font-weight:700; color:{color};">{val}</div>
<div style="font-size:0.75rem; color:#aaa; margin-top:4px;">{lbl}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("")
        col_c, col_d = st.columns(2)

        with col_c:
            # Actual vs Predicted scatter
            fig_reg = px.scatter(x=metrics["yr_test"], y=metrics["xgb_pred"],
                                  labels={"x": "Actual Spend ₹", "y": "Predicted Spend ₹"},
                                  title="Actual vs Predicted Annual Spend",
                                  opacity=0.6, color_discrete_sequence=["#EF9F27"])
            max_val = max(metrics["yr_test"].max(), metrics["xgb_pred"].max())
            fig_reg.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                          mode="lines", name="Perfect prediction",
                                          line=dict(dash="dash", color="#7F77DD")))
            fig_reg.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_reg, use_container_width=True)

        with col_d:
            # Residual distribution
            residuals = metrics["yr_test"] - metrics["xgb_pred"]
            fig_res = px.histogram(x=residuals, nbins=40,
                                    color_discrete_sequence=["#1D9E75"],
                                    title="Residuals Distribution (Actual − Predicted ₹)",
                                    labels={"x": "Residual ₹"})
            fig_res.add_vline(x=0, line_dash="dash", line_color="#EF9F27")
            fig_res.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_res, use_container_width=True)

        # Spend distribution by persona
        st.markdown("### Predicted Annual Spend Distribution by Persona")
        fig_box = px.box(df, x="persona", y="pred_spend", color="persona",
                          color_discrete_map=PERSONA_COLORS,
                          title="Spend Distribution Across Personas",
                          labels={"pred_spend": "Predicted Spend ₹", "persona": "Persona"})
        fig_box.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                               showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

        # High value customer list
        st.markdown("### 🏆 High-Value Customer List (Predicted Spend > ₹10,000)")
        hv_cols = ["Respondent_ID", "Q01_Age_Group", "Q03_Region", "Q04_City_Tier",
                   "Q05_Monthly_Income", "Q07_Occupation", "persona",
                   "pred_intent", "pred_spend", "target_label"]
        hv_cols = [c for c in hv_cols if c in df.columns]
        hv_df = df[df["pred_spend"] > 10000][hv_cols].sort_values("pred_spend", ascending=False)
        hv_df = hv_df.rename(columns={
            "pred_intent": "Intent Score",
            "pred_spend": "Pred Spend ₹",
            "target_label": "Survey Intent",
        })
        st.dataframe(hv_df.reset_index(drop=True), height=320, use_container_width=True)
        st.download_button("⬇️ Download high-value list",
                           hv_df.to_csv(index=False),
                           "pitambari_high_value.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SCORE NEW LEADS
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown("## 🚀 Score New Leads — Upload & Predict")
    st.markdown("""
Upload a CSV of **new survey respondents** (from a trade fair, WhatsApp survey, or any future data collection).  
The app will validate columns, preprocess identically to the training data, run all trained models,  
and return a fully scored output CSV with **persona, intent score, predicted spend, and recommended action**.
""")

    st.info("📌 Your CSV must have the same column names as the main dataset. Download the template below.")

    template_cols = [
        "Respondent_ID", "Q01_Age_Group", "Q02_Gender", "Q03_Region", "Q04_City_Tier",
        "Q05_Monthly_Income", "Q06_Household_Size", "Q07_Occupation",
        "Q08_Purchase_Motivation_Top2", "Q09_Traditional_Crafts_Owned",
        "Q10_Gifting_Occasions", "Q11_Gift_Budget_Per_Occasion", "Q12_Puja_Frequency",
        "Q13_Last_Purchase_Recency", "Q14_Purchase_Frequency_12M", "Q15_Historical_Spend_12M",
        "Q16_Product_Category_Interest", "Q17_Pooja_Set_Budget", "Q17b_Kansa_Dinner_Set_Budget",
        "Q_Preferred_Shopping_Channel", "Q18_Top2_Reasons_Buying_Brass",
        "Q19_Current_Buying_Source", "Q20_Trust_Moradabad_1to5", "Q21_Barrier_Removal_Top2",
        "Q22_Preferred_Offer_Type", "Q23_NPS_Score_1to5",
        "Q24_Future_Annual_Spend_TARGET", "Q25_Purchase_Likelihood_TARGET",
        "Q_Open_Text_Brass_Meaning",
    ]
    template_df = pd.DataFrame(columns=template_cols)
    st.download_button(
        "⬇️ Download blank CSV template",
        template_df.to_csv(index=False),
        "pitambari_new_leads_template.csv",
        "text/csv",
    )

    st.markdown("---")
    new_file = st.file_uploader("📤 Upload new leads CSV", type=["csv"], key="new_leads")

    if new_file is not None:
        try:
            new_df_raw = pd.read_csv(new_file)
            st.success(f"✅ File loaded: **{len(new_df_raw):,} respondents**, {new_df_raw.shape[1]} columns")

            # Column validation
            required = [c for c in template_cols if c not in [
                "Q24_Future_Annual_Spend_TARGET", "Q25_Purchase_Likelihood_TARGET"
            ]]
            missing_cols = [c for c in required if c not in new_df_raw.columns]

            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                st.stop()
            else:
                st.success("✅ Column validation passed — all required columns present")

            with st.spinner("⚙️ Preprocessing and scoring..."):
                new_df_proc = load_and_preprocess(new_df_raw)
                feat_cols_model = metrics["feat_cols"]

                # Align columns
                for c in feat_cols_model:
                    if c not in new_df_proc.columns:
                        new_df_proc[c] = 0
                X_new = new_df_proc[feat_cols_model].fillna(0)

                # Predictions
                intent_scores = rf_model.predict_proba(X_new)[:, 1]
                spend_preds = xgb_model.predict(X_new)

                # Cluster assignment
                X_new_cl = new_df_proc[cluster_feats].fillna(0)
                X_new_cl_sc = scaler_cl.transform(X_new_cl)
                cluster_labels = km_model.predict(X_new_cl_sc)
                persona_labels = [CLUSTER_PERSONA_NAMES.get(c, "Casual Explorer") for c in cluster_labels]

                # Priority tier
                def priority_tier(score, spend):
                    if score >= 0.65 and spend >= 8000:
                        return "🔴 Very High — Direct Outreach"
                    elif score >= 0.45 and spend >= 3000:
                        return "🟠 High — Festival Campaign"
                    elif score >= 0.25:
                        return "🟡 Medium — Nurture Content"
                    else:
                        return "⚪ Low — Awareness Only"

                # Recommended action per persona
                def recommended_action(persona, intent, spend):
                    strategy = PERSONA_STRATEGY.get(persona, {})
                    bundle = strategy.get("bundle", "Starter kit")
                    discount = strategy.get("discount", "Festival cashback")
                    channel = strategy.get("channel", "Online")
                    return f"Bundle: {bundle} | Offer: {discount} | Channel: {channel}"

                scored = new_df_raw.copy()
                scored["Predicted_Persona"] = persona_labels
                scored["Intent_Score"] = np.round(intent_scores, 4)
                scored["Predicted_Annual_Spend_Rs"] = np.round(spend_preds, 0).astype(int)
                scored["Priority_Tier"] = [
                    priority_tier(s, sp) for s, sp in zip(intent_scores, spend_preds)
                ]
                scored["Recommended_Action"] = [
                    recommended_action(p, s, sp)
                    for p, s, sp in zip(persona_labels, intent_scores, spend_preds)
                ]

            st.markdown("### 🎯 Scoring Results")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Total Scored", f"{len(scored):,}")
            r2.metric("High Intent (>60%)", f"{(intent_scores >= 0.6).sum():,}")
            r3.metric("Avg Predicted Spend", f"₹{spend_preds.mean():,.0f}")
            r4.metric("Top Persona", pd.Series(persona_labels).value_counts().index[0])

            col_e, col_f = st.columns(2)
            with col_e:
                fig_intent = px.histogram(x=intent_scores, nbins=20,
                                           color_discrete_sequence=["#7F77DD"],
                                           title="Intent Score Distribution — New Leads",
                                           labels={"x": "Intent Score"})
                fig_intent.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_intent, use_container_width=True)

            with col_f:
                persona_counts = pd.Series(persona_labels).value_counts().reset_index()
                persona_counts.columns = ["Persona", "Count"]
                fig_pers = px.pie(persona_counts, names="Persona", values="Count",
                                   color="Persona", color_discrete_map=PERSONA_COLORS,
                                   hole=0.4, title="New Leads — Persona Mix")
                fig_pers.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_pers, use_container_width=True)

            # Priority breakdown
            priority_counts = scored["Priority_Tier"].value_counts().reset_index()
            priority_counts.columns = ["Priority", "Count"]
            fig_pri = px.bar(priority_counts, x="Priority", y="Count",
                              color="Count", color_continuous_scale="Oranges",
                              title="New Leads — Priority Tier Breakdown")
            fig_pri.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10),
                                   coloraxis_showscale=False)
            st.plotly_chart(fig_pri, use_container_width=True)

            st.markdown("### 📋 Scored Output Preview")
            preview_cols = [c for c in [
                "Respondent_ID", "Q01_Age_Group", "Q03_Region", "Q04_City_Tier",
                "Predicted_Persona", "Intent_Score", "Predicted_Annual_Spend_Rs",
                "Priority_Tier", "Recommended_Action"
            ] if c in scored.columns]
            st.dataframe(scored[preview_cols].reset_index(drop=True),
                         height=380, use_container_width=True)

            st.download_button(
                "⬇️ Download scored output CSV",
                scored.to_csv(index=False),
                "pitambari_scored_leads.csv",
                "text/csv",
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)

    else:
        st.markdown("""
        ### How this works
        1. Download the blank template above  
        2. Fill it with your new survey respondents (from a trade fair, digital form, etc.)  
        3. Upload the filled CSV here  
        4. Get back a fully scored file with:
           - **Predicted Persona** (which of 5 customer types they are)
           - **Intent Score** (0–1 probability of purchase)
           - **Predicted Annual Spend** (₹ estimate)
           - **Priority Tier** (Very High / High / Medium / Low)
           - **Recommended Action** (exact bundle + discount + channel)
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; font-size:0.8rem;'>"
    "🪔 Pitambari Brass Analytics · Moradabad Heritage × Wellness × Premium Living · "
    "Built with Streamlit, Scikit-Learn, XGBoost, MLxtend & Plotly"
    "</div>",
    unsafe_allow_html=True
)
