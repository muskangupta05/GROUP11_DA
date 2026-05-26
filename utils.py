import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ── Ordinal mappings ──────────────────────────────────────────────────────────

AGE_MAP = {
    "Under 18 years": 0,
    "18-24 years": 1,
    "25-34 years": 2,
    "35-44 years": 3,
    "45-59 years": 4,
    "60 years and above": 5,
}

INCOME_MAP = {
    "Below Rs 20000": 1,
    "Rs 20000-40000": 2,
    "Rs 40001-75000": 3,
    "Rs 75001-150000": 4,
    "Rs 150001-300000": 5,
    "Above Rs 300000": 6,
    "Prefer not to disclose": 3,
}

INCOME_MIDPOINT = {
    "Below Rs 20000": 15000,
    "Rs 20000-40000": 30000,
    "Rs 40001-75000": 57500,
    "Rs 75001-150000": 112500,
    "Rs 150001-300000": 225000,
    "Above Rs 300000": 400000,
    "Prefer not to disclose": 57500,
}

HH_MAP = {
    "1-2 people (nuclear)": 1,
    "3-4 people (small nuclear)": 2,
    "5-6 people (medium family)": 3,
    "7+ people (large joint family)": 4,
}

PUJA_MAP = {
    "Every single day": 6,
    "Most days (5-6 days a week)": 5,
    "A few times a week": 4,
    "Only on weekends": 3,
    "Only on festivals": 2,
    "Rarely or never": 1,
}

RECENCY_MAP = {
    "Within last 1 month": 6,
    "1-3 months ago": 5,
    "3-6 months ago": 4,
    "6-12 months ago": 3,
    "More than 1 year ago": 2,
    "Never bought": 1,
}

FREQ_MAP = {
    "0 times": 0,
    "1-2 times": 1,
    "3-5 times": 3,
    "6-10 times": 8,
    "More than 10 times": 12,
}

HIST_SPEND_MIDPOINT = {
    "Rs 0 (did not buy)": 0,
    "Rs 1-1000": 500,
    "Rs 1001-3000": 2000,
    "Rs 3001-8000": 5500,
    "Rs 8001-20000": 14000,
    "More than Rs 20000": 30000,
}

FUTURE_SPEND_MIDPOINT = {
    "Under Rs 1000": 500,
    "Rs 1000-3000": 2000,
    "Rs 3001-8000": 5500,
    "Rs 8001-20000": 14000,
    "Rs 20001-50000": 35000,
    "Above Rs 50000": 75000,
}

GIFT_BUDGET_MAP = {
    "Under Rs 500": 1,
    "Rs 500-1500": 2,
    "Rs 1500-3500": 3,
    "Rs 3500-7000": 4,
    "Rs 7000-15000": 5,
    "Above Rs 15000": 6,
    float("nan"): 3,
}

POOJA_BUDGET_MAP = {
    "Under Rs 500": 1,
    "Rs 500-1500": 2,
    "Rs 1500-4000": 3,
    "Rs 4000-8000": 4,
    "Rs 8000-15000": 5,
    "Above Rs 15000": 6,
}

DINNER_BUDGET_MAP = {
    "Under Rs 1000": 1,
    "Rs 1000-3000": 2,
    "Rs 3000-7000": 3,
    "Rs 7000-15000": 4,
    "Rs 15000-30000": 5,
    "Above Rs 30000": 6,
    float("nan"): 3,
}

CITY_MAP = {
    "Metro city": 5,
    "Tier-2 city": 4,
    "Tier-3 town": 3,
    "Semi-rural area": 2,
    "Village or rural area": 1,
}

LIKELIHOOD_BINARY = {
    "Definitely yes - buy within first month": 1,
    "Probably yes - explore and very likely purchase": 1,
    "Maybe - need to see quality and pricing first": 0,
    "Probably not - have trusted alternatives": 0,
    "Definitely not - category not relevant": 0,
}

LIKELIHOOD_LABEL = {
    "Definitely yes - buy within first month": "Highly Interested",
    "Probably yes - explore and very likely purchase": "Interested",
    "Maybe - need to see quality and pricing first": "Neutral",
    "Probably not - have trusted alternatives": "Not Interested",
    "Definitely not - category not relevant": "Not Interested",
}

CLUSTER_PERSONA_NAMES = {
    0: "Daily Devotee",
    1: "Wellness Seeker",
    2: "Gifting Executive",
    3: "Heritage Collector",
    4: "Casual Explorer",
}

PERSONA_COLORS = {
    "Daily Devotee": "#7F77DD",
    "Wellness Seeker": "#1D9E75",
    "Gifting Executive": "#EF9F27",
    "Heritage Collector": "#D85A30",
    "Casual Explorer": "#888780",
}

PERSONA_STRATEGY = {
    "Daily Devotee": {
        "product": "Brass Pooja Articles — full ritual sets",
        "discount": "Festival cashback (Navratri, Diwali, Akshaya Tritiya)",
        "channel": "WhatsApp, regional language YouTube",
        "bundle": "Diya set + Kalash + Thali bundle",
        "priority": "High",
    },
    "Wellness Seeker": {
        "product": "Kansa/Bronze Wellness Utensils",
        "discount": "Buy more save more (3 items 20% off)",
        "channel": "Instagram reels, Ayurveda influencers",
        "bundle": "Kansa plate + bowl + glass starter kit",
        "priority": "High",
    },
    "Gifting Executive": {
        "product": "Curated Corporate Gift Sets",
        "discount": "Custom engraving + bulk pricing",
        "channel": "LinkedIn, direct B2B outreach",
        "bundle": "Urli + 2 diyas + incense holder in premium box",
        "priority": "Very High",
    },
    "Heritage Collector": {
        "product": "Brass Urli & Home Décor",
        "discount": "Early member access — limited-edition drops",
        "channel": "Instagram, Pinterest, craft exhibitions",
        "bundle": "Urli + brass planter + wall art combo",
        "priority": "High",
    },
    "Casual Explorer": {
        "product": "Brass Water Vessels — entry-level",
        "discount": "Low-cost trial kit / sample set",
        "channel": "Amazon, Flipkart",
        "bundle": "Starter lota + care guide",
        "priority": "Medium",
    },
}

# ── Multi-select expander ─────────────────────────────────────────────────────

def expand_multiselect(df, col, sep="|"):
    """One-hot encode a pipe-separated multi-select column."""
    all_vals = set()
    for row in df[col].dropna():
        for v in str(row).split(sep):
            all_vals.add(v.strip())
    for val in sorted(all_vals):
        safe = val.replace(" ", "_").replace("/", "_").replace(",", "").replace("(", "").replace(")", "").replace("-", "_")[:40]
        colname = f"{col}__{safe}"
        df[colname] = df[col].apply(
            lambda x: 1 if pd.notna(x) and val in str(x) else 0
        )
    return df

# ── Main preprocessing ────────────────────────────────────────────────────────

def load_and_preprocess(df_raw):
    df = df_raw.copy()

    # -- Ordinal encodings
    df["age_enc"] = df["Q01_Age_Group"].map(AGE_MAP).fillna(2)
    df["income_enc"] = df["Q05_Monthly_Income"].map(INCOME_MAP).fillna(3)
    df["income_mid"] = df["Q05_Monthly_Income"].map(INCOME_MIDPOINT).fillna(57500)
    df["hh_enc"] = df["Q06_Household_Size"].map(HH_MAP).fillna(2)
    df["puja_enc"] = df["Q12_Puja_Frequency"].map(PUJA_MAP).fillna(3)
    df["recency_enc"] = df["Q13_Last_Purchase_Recency"].map(RECENCY_MAP).fillna(3)
    df["freq_enc"] = df["Q14_Purchase_Frequency_12M"].map(FREQ_MAP).fillna(1)
    df["hist_spend"] = df["Q15_Historical_Spend_12M"].map(HIST_SPEND_MIDPOINT).fillna(0)
    df["gift_budget_enc"] = df["Q11_Gift_Budget_Per_Occasion"].map(GIFT_BUDGET_MAP).fillna(3)
    df["pooja_budget_enc"] = df["Q17_Pooja_Set_Budget"].map(POOJA_BUDGET_MAP).fillna(3)
    df["dinner_budget_enc"] = df["Q17b_Kansa_Dinner_Set_Budget"].map(DINNER_BUDGET_MAP).fillna(3)
    df["city_enc"] = df["Q04_City_Tier"].map(CITY_MAP).fillna(3)
    df["trust_enc"] = df["Q20_Trust_Moradabad_1to5"].fillna(3).astype(float)
    df["nps_enc"] = df["Q23_NPS_Score_1to5"].fillna(3).astype(float)

    # -- Nominal label encode
    le = LabelEncoder()
    for col in ["Q02_Gender", "Q03_Region", "Q07_Occupation", "Q19_Current_Buying_Source", "Q_Preferred_Shopping_Channel"]:
        df[col + "_enc"] = le.fit_transform(df[col].fillna("Unknown"))

    # -- RFM composite score
    df["rfm_score"] = (
        df["recency_enc"] * 0.3
        + (df["freq_enc"] / df["freq_enc"].max()) * 10 * 0.3
        + (df["hist_spend"] / df["hist_spend"].max()) * 10 * 0.4
    )

    # -- Multi-select one-hot
    df = expand_multiselect(df, "Q08_Purchase_Motivation_Top2")
    df = expand_multiselect(df, "Q09_Traditional_Crafts_Owned")
    df = expand_multiselect(df, "Q10_Gifting_Occasions")
    df = expand_multiselect(df, "Q16_Product_Category_Interest")
    df = expand_multiselect(df, "Q18_Top2_Reasons_Buying_Brass")
    df = expand_multiselect(df, "Q21_Barrier_Removal_Top2")
    df = expand_multiselect(df, "Q22_Preferred_Offer_Type")

    # -- Targets
    df["target_class"] = df["Q25_Purchase_Likelihood_TARGET"].map(LIKELIHOOD_BINARY).fillna(0).astype(int)
    df["target_label"] = df["Q25_Purchase_Likelihood_TARGET"].map(LIKELIHOOD_LABEL).fillna("Neutral")
    df["target_spend"] = df["Q24_Future_Annual_Spend_TARGET"].map(FUTURE_SPEND_MIDPOINT).fillna(2000)

    return df


def get_feature_cols(df):
    base = [
        "age_enc", "income_enc", "income_mid", "hh_enc", "puja_enc",
        "recency_enc", "freq_enc", "hist_spend", "gift_budget_enc",
        "pooja_budget_enc", "dinner_budget_enc", "city_enc",
        "trust_enc", "nps_enc", "rfm_score",
        "Q02_Gender_enc", "Q03_Region_enc", "Q07_Occupation_enc",
        "Q19_Current_Buying_Source_enc", "Q_Preferred_Shopping_Channel_enc",
    ]
    multi = [c for c in df.columns if any(
        c.startswith(p) for p in [
            "Q08_Purchase_Motivation_Top2__",
            "Q09_Traditional_Crafts_Owned__",
            "Q10_Gifting_Occasions__",
            "Q16_Product_Category_Interest__",
            "Q18_Top2_Reasons_Buying_Brass__",
            "Q21_Barrier_Removal_Top2__",
            "Q22_Preferred_Offer_Type__",
        ]
    )]
    return [c for c in base + multi if c in df.columns]


def prepare_splits(df):
    feat_cols = get_feature_cols(df)
    X = df[feat_cols].fillna(0)
    y_cls = df["target_class"]
    y_reg = df["target_spend"]
    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
    )
    return X_train, X_test, yc_train, yc_test, yr_train, yr_test, feat_cols


def get_arm_transactions(df):
    """Build transaction list for Apriori from Q09 + Q16."""
    transactions = []
    for _, row in df.iterrows():
        items = set()
        for col in ["Q09_Traditional_Crafts_Owned", "Q16_Product_Category_Interest"]:
            val = row.get(col, "")
            if pd.notna(val) and str(val) != "None":
                for item in str(val).split("|"):
                    item = item.strip()
                    if item and item.lower() not in ["none", "none of these interest me", "i do not own any traditional craft or heritage items at present"]:
                        short = item.split("(")[0].strip()[:45]
                        items.add(short)
        if len(items) >= 2:
            transactions.append(list(items))
    return transactions
