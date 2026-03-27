import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer

MULTILABEL_COLS = [
    "products_interested", "preferred_fabric", "preferred_colors",
    "preferred_prints", "preferred_channel", "purchase_occasion",
    "discovery_channel"
]

ORDINAL_MAPS = {
    "age_group":             {"Under18": 0, "18-24": 1, "25-34": 2, "35-44": 3, "45-55": 4, "55+": 5},
    "city_tier":             {"Rural": 0, "Tier3": 1, "Tier2": 2, "Metro": 3},
    "monthly_hh_income":     {"<15k": 0, "15k-30k": 1, "30k-60k": 2, "60k-1L": 3, ">1L": 4},
    "education":             {"Below10th": 0, "12th": 1, "Graduate": 2, "PostGrad": 3},
    "purchase_frequency":    {"OccasionOnly": 0, "Quarterly": 1, "Monthly": 2, "Bimonthly": 3, "Weekly": 4},
    "last_purchase_recency": {">1yr": 0, "6-12mo": 1, "3-6mo": 2, "1-3mo": 3, "<1mo": 4},
    "avg_items_per_order":   {"1": 0, "2-3": 1, "4-5": 2, "6+": 3},
    "price_range_apparel":   {"<500": 0, "500-999": 1, "1000-1999": 2, "2000-3999": 3, "4000+": 4},
    "price_range_bedding":   {"<800": 0, "800-1499": 1, "1500-2999": 2, "3000-5000": 3, "5000+": 4},
    "clothing_size":         {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4, "XXL": 5, "3XL+": 6},
    "has_returned_product":  {"NeverReturned": 0, "Unhappy_NoReturn": 1, "OnceOrTwice": 2, "MultipleReturns": 3},
    "gifting_behaviour":     {"SelfOnly": 0, "Rarely": 1, "Occasionally": 2, "Frequently": 3},
}

NOMINAL_COLS = [
    "region", "occupation", "marital_status", "style_identity",
    "brand_loyalty", "discount_preference", "main_pain_point"
]

CONTINUOUS_COLS = [
    "monthly_spend_inr", "madein_india_importance",
    "referral_propensity", "current_satisfaction"
]

TARGET_COL = "zaria_interest_label"


def load_data(path="zaria_fashion_clean_data.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in MULTILABEL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def encode_features(df, fit_encoders=None):
    df = df.copy()
    encoders = fit_encoders if fit_encoders else {}

    # Ordinal encoding
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # Nominal encoding
    for col in NOMINAL_COLS:
        if col not in df.columns:
            continue
        if fit_encoders is None:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col])

    # Multi-label binarization (top items only to keep dimensions manageable)
    for col in MULTILABEL_COLS:
        if col not in df.columns:
            continue
        split = df[col].str.split("|")
        if fit_encoders is None:
            mlb = MultiLabelBinarizer()
            binarized = mlb.fit_transform(split)
            encoders[col] = mlb
        else:
            mlb = encoders.get(col)
            if mlb is None:
                continue
            binarized = mlb.transform(split)
        cols_out = [f"{col}__{c}" for c in mlb.classes_]
        bdf = pd.DataFrame(binarized, columns=cols_out, index=df.index)
        df = pd.concat([df.drop(columns=[col]), bdf], axis=1)

    # Continuous — ensure numeric
    for col in CONTINUOUS_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if df[col].dtype != object else 0)

    return df, encoders


def get_feature_matrix(df, encoders=None, drop_target=True):
    df_enc, enc = encode_features(df, fit_encoders=encoders)
    drop_cols = ["respondent_id"]
    if drop_target and TARGET_COL in df_enc.columns:
        drop_cols.append(TARGET_COL)
    feature_cols = [c for c in df_enc.columns if c not in drop_cols]
    X = df_enc[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, enc, feature_cols


def get_cluster_features(df):
    cluster_raw = [
        "monthly_spend_inr", "madein_india_importance",
        "referral_propensity", "current_satisfaction",
        "monthly_hh_income", "purchase_frequency",
        "last_purchase_recency", "city_tier", "avg_items_per_order",
        "price_range_apparel", "price_range_bedding"
    ]
    df_c = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in df_c.columns:
            df_c[col] = df_c[col].map(mapping).fillna(0)
    for col in CONTINUOUS_COLS:
        if col in df_c.columns:
            df_c[col] = pd.to_numeric(df_c[col], errors="coerce").fillna(0)
    cols = [c for c in cluster_raw if c in df_c.columns]
    X = df_c[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, cols


def prepare_arm_transactions(df):
    transactions = []
    for _, row in df.iterrows():
        items = set()
        for col in ["products_interested", "preferred_fabric",
                    "preferred_colors", "preferred_prints", "purchase_occasion"]:
            if col in df.columns and str(row.get(col, "")).strip():
                for item in str(row[col]).split("|"):
                    item = item.strip()
                    if item:
                        items.add(f"{col.split('_')[0].upper()[:4]}:{item}")
        if items:
            transactions.append(list(items))
    return transactions
