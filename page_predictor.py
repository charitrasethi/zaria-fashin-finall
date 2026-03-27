import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.utils import resample as sk_resample
from preprocessing import (
    get_feature_matrix, get_cluster_features,
    TARGET_COL, ORDINAL_MAPS, CONTINUOUS_COLS
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


MARKETING_ACTIONS = {
    (2, 0): "HOT LEAD — Immediate WhatsApp outreach + 20% first-order discount",
    (2, 1): "HOT LEAD — Instagram retarget + Festival bundle offer",
    (2, 2): "HOT LEAD — YouTube influencer campaign + free shipping offer",
    (2, 3): "HOT LEAD — WhatsApp broadcast with traditional combo pack",
    (2, 4): "HOT LEAD — Premium D2C campaign + early-access loyalty invite",
    (1, 0): "WARM LEAD — Festival sale reminder + Buy2Get1 offer",
    (1, 1): "WARM LEAD — Instagram story ads + new arrival showcase",
    (1, 2): "WARM LEAD — Influencer reel targeting fusion-style audience",
    (1, 3): "WARM LEAD — Referral programme + community discount",
    (1, 4): "WARM LEAD — Email/WhatsApp with curated home-textile offer",
    (0, 0): "LOW PRIORITY — Add to retargeting pool; revisit in 6 months",
    (0, 1): "LOW PRIORITY — Monitor; re-engage after brand establishes in city",
    (0, 2): "LOW PRIORITY — Low conversion probability; pause outreach",
    (0, 3): "LOW PRIORITY — Very traditional; wait for regional collection launch",
    (0, 4): "LOW PRIORITY — Brand-loyal competitor customer; long-term nurture only",
}

CLUSTER_NAMES = {
    0: "Budget Festival Shoppers",
    1: "Urban Trend Seekers",
    2: "Young Experimenters",
    3: "Traditional Value Buyers",
    4: "Premium Home & Ethnic",
}


def _balanced_resample(X_train, y_train):
    if HAS_SMOTE:
        sm = SMOTE(random_state=42, k_neighbors=3)
        return sm.fit_resample(X_train, y_train)
    classes    = y_train.unique()
    max_count  = y_train.value_counts().max()
    X_parts, y_parts = [], []
    for c in classes:
        mask = y_train == c
        Xc, yc = X_train[mask], y_train[mask]
        Xc_r, yc_r = sk_resample(Xc, yc, replace=True,
                                  n_samples=max_count, random_state=42)
        X_parts.append(Xc_r)
        y_parts.append(yc_r)
    return pd.concat(X_parts), pd.concat(y_parts)


def _prep_reg_features(df):
    df_r = df.copy()
    reg_cols = [
        "monthly_hh_income", "occupation", "city_tier", "age_group",
        "purchase_frequency", "last_purchase_recency", "avg_items_per_order",
        "gifting_behaviour", "has_returned_product", "price_range_apparel",
        "price_range_bedding", "madein_india_importance",
        "referral_propensity", "current_satisfaction", "brand_loyalty",
        "style_identity", "education", "marital_status", "clothing_size",
    ]
    for col, mapping in ORDINAL_MAPS.items():
        if col in df_r.columns:
            df_r[col] = df_r[col].map(mapping).fillna(0).astype(int)
    for col in ["occupation", "brand_loyalty", "style_identity",
                "marital_status", "region", "discount_preference"]:
        if col in df_r.columns:
            le = LabelEncoder()
            df_r[col] = le.fit_transform(df_r[col].astype(str))
    for col in CONTINUOUS_COLS:
        if col in df_r.columns:
            df_r[col] = pd.to_numeric(df_r[col], errors="coerce").fillna(0)
    avail = [c for c in reg_cols if c in df_r.columns]
    X = df_r[avail].apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, avail


@st.cache_resource
def train_all_models(_df_id, df):
    # Classifier
    X_clf, encoders, feat_cols = get_feature_matrix(df, drop_target=True)
    y = df[TARGET_COL].astype(int)
    Xtr, _, ytr, _ = train_test_split(
        X_clf, y, test_size=0.2, random_state=42, stratify=y
    )
    Xtr_bal, ytr_bal = _balanced_resample(Xtr, ytr)
    rf_clf = RandomForestClassifier(
        n_estimators=150, random_state=42,
        class_weight="balanced", max_depth=12, n_jobs=-1
    )
    rf_clf.fit(Xtr_bal, ytr_bal)

    # Regressor
    if "monthly_spend_inr" in df.columns:
        Xr, avail = _prep_reg_features(df)
        yr = pd.to_numeric(df["monthly_spend_inr"], errors="coerce").fillna(0)
        rf_reg = RandomForestRegressor(
            n_estimators=150, random_state=42, max_depth=10, n_jobs=-1
        )
        rf_reg.fit(Xr, yr)
    else:
        rf_reg, avail = None, []

    # Clusterer
    X_cl, scaler_cl, cl_cols = get_cluster_features(df)
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    km.fit(X_cl)

    return rf_clf, encoders, feat_cols, rf_reg, avail, km, scaler_cl, cl_cols


def _encode_new(df_new, encoders, feat_cols):
    from preprocessing import encode_features
    df_enc, _ = encode_features(df_new, fit_encoders=encoders)
    drop_cols  = ["respondent_id", TARGET_COL]
    avail      = [c for c in feat_cols if c in df_enc.columns]
    X = df_enc[avail].apply(pd.to_numeric, errors="coerce").fillna(0)
    for mc in [c for c in feat_cols if c not in X.columns]:
        X[mc] = 0
    return X[feat_cols]


def render(df: pd.DataFrame):
    st.markdown("## New Customer Predictor")
    st.markdown(
        "*Upload a new survey CSV → instant predictions: interest label, "
        "spend capacity, cluster assignment & personalised marketing action.*"
    )

    with st.spinner("Training models on base dataset..."):
        (rf_clf, encoders, feat_cols,
         rf_reg, reg_avail,
         km, scaler_cl, cl_cols) = train_all_models(id(df), df)

    st.success("✅ Models trained on base dataset and ready for prediction.")

    # ── Column guide ──────────────────────────────────────────────────────────
    req_cols = [
        "respondent_id", "age_group", "region", "city_tier", "occupation",
        "monthly_hh_income", "education", "marital_status",
        "purchase_frequency", "last_purchase_recency", "avg_items_per_order",
        "preferred_channel", "purchase_occasion", "has_returned_product",
        "gifting_behaviour", "products_interested", "preferred_fabric",
        "preferred_colors", "preferred_prints", "price_range_apparel",
        "price_range_bedding", "clothing_size", "style_identity",
        "brand_loyalty", "madein_india_importance", "discovery_channel",
        "discount_preference", "referral_propensity",
        "current_satisfaction", "main_pain_point",
    ]
    st.markdown(
        f"**{len(req_cols)} columns required** — same as base survey, "
        "excluding `monthly_spend_inr` and `zaria_interest_label`"
    )
    with st.expander("Show required column list"):
        st.write(req_cols)

    # Template download
    sample_df = pd.DataFrame([{col: "SAMPLE_VALUE" for col in req_cols}])
    st.download_button(
        "⬇️  Download upload template (CSV)",
        sample_df.to_csv(index=False),
        "zaria_upload_template.csv", "text/csv"
    )

    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("### Upload new respondent data")
    uploaded = st.file_uploader(
        "Drop your new survey CSV here",
        type=["csv"],
        help="Columns must match the required list above"
    )

    st.markdown("#### Or generate a quick demo (10 synthetic respondents)")
    use_sample = st.button("Generate & predict on 10 sample customers")

    new_df = None

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            st.success(f"Uploaded: **{len(new_df)} rows × {len(new_df.columns)} columns**")
        except Exception as e:
            st.error(f"File read error: {e}")
            return

    elif use_sample:
        import random
        random.seed(77)
        opts = dict(
            age_group    = ["18-24", "25-34", "35-44", "45-55"],
            region       = ["North", "South", "West", "East"],
            city_tier    = ["Metro", "Tier2", "Tier3", "Rural"],
            occupation   = ["Student", "Homemaker", "Salaried_Pvt", "Business"],
            income       = ["<15k", "15k-30k", "30k-60k", "60k-1L", ">1L"],
            edu          = ["Graduate", "PostGrad", "12th"],
            marital      = ["Single", "Married", "Married_with_kids"],
            freq         = ["Monthly", "Quarterly", "Bimonthly", "Weekly"],
            recency      = ["<1mo", "1-3mo", "3-6mo", "6-12mo"],
            items        = ["1", "2-3", "4-5", "6+"],
            channel      = ["Meesho|Amazon", "Myntra|D2C_Brand", "LocalStore|WhatsApp_Seller"],
            occasion     = ["Diwali|Wedding", "NoOccasion", "Birthday|Holi"],
            returned     = ["NeverReturned", "OnceOrTwice", "MultipleReturns"],
            gifting      = ["Occasionally", "Rarely", "Frequently", "SelfOnly"],
            products     = ["SalwarSuit|Kurti", "IndoWestern|NightSuit", "BeddingSet|Kurti|CoordSet"],
            fabric       = ["Cotton|Rayon", "Silk|Linen", "CottonBlend|Georgette"],
            colors       = ["Red|Pink", "Blue|White|Beige", "Green|Yellow|Printed"],
            prints       = ["Floral|Solid", "BlockPrint|Embroidery", "DigitalPrint|Geometric"],
            price_app    = ["<500", "500-999", "1000-1999", "2000-3999"],
            price_bed    = ["<800", "800-1499", "1500-2999", "3000-5000"],
            size         = ["S", "M", "L", "XL", "XXL"],
            style        = ["EthnicComfort", "FusionForward", "TraditionallyRooted", "HomeFirst"],
            loyalty      = ["SwitchByPrice", "ExperimentsFreely", "NoBrandPreference", "StickTo1-2"],
            discovery    = ["Instagram|YouTube", "WhatsApp|MeeshoApp", "WordOfMouth|PhysicalStore"],
            discount     = ["FlatDiscount", "BundleCombo", "Buy2Get1", "FestivalSale"],
            pain         = ["HighPrice", "FabricQuality", "SizingIssue", "LimitedDesign", "SlowDelivery"],
        )
        rows = []
        for i in range(10):
            rows.append({
                "respondent_id":           f"NEW{i+1:03d}",
                "age_group":               random.choice(opts["age_group"]),
                "region":                  random.choice(opts["region"]),
                "city_tier":               random.choice(opts["city_tier"]),
                "occupation":              random.choice(opts["occupation"]),
                "monthly_hh_income":       random.choice(opts["income"]),
                "education":               random.choice(opts["edu"]),
                "marital_status":          random.choice(opts["marital"]),
                "purchase_frequency":      random.choice(opts["freq"]),
                "last_purchase_recency":   random.choice(opts["recency"]),
                "avg_items_per_order":     random.choice(opts["items"]),
                "preferred_channel":       random.choice(opts["channel"]),
                "purchase_occasion":       random.choice(opts["occasion"]),
                "has_returned_product":    random.choice(opts["returned"]),
                "gifting_behaviour":       random.choice(opts["gifting"]),
                "products_interested":     random.choice(opts["products"]),
                "preferred_fabric":        random.choice(opts["fabric"]),
                "preferred_colors":        random.choice(opts["colors"]),
                "preferred_prints":        random.choice(opts["prints"]),
                "price_range_apparel":     random.choice(opts["price_app"]),
                "price_range_bedding":     random.choice(opts["price_bed"]),
                "clothing_size":           random.choice(opts["size"]),
                "style_identity":          random.choice(opts["style"]),
                "brand_loyalty":           random.choice(opts["loyalty"]),
                "madein_india_importance": random.randint(4, 10),
                "discovery_channel":       random.choice(opts["discovery"]),
                "discount_preference":     random.choice(opts["discount"]),
                "referral_propensity":     random.randint(4, 10),
                "current_satisfaction":    random.randint(3, 8),
                "main_pain_point":         random.choice(opts["pain"]),
            })
        new_df = pd.DataFrame(rows)
        st.success("Generated **10 synthetic new customers** for prediction demo.")

    if new_df is None:
        return

    st.divider()
    st.markdown("### Running prediction pipeline...")

    try:
        # 1. Classification
        X_clf_new = _encode_new(new_df, encoders, feat_cols)
        pred_labels = rf_clf.predict(X_clf_new)
        pred_probs  = rf_clf.predict_proba(X_clf_new)

        # 2. Regression
        if rf_reg is not None:
            df_tmp = new_df.copy()
            df_tmp["monthly_spend_inr"] = 0
            Xr_new, _ = _prep_reg_features(df_tmp)
            for mc in [c for c in reg_avail if c not in Xr_new.columns]:
                Xr_new[mc] = 0
            Xr_new = Xr_new[reg_avail]
            pred_spend = np.round(rf_reg.predict(Xr_new), 0).astype(int)
        else:
            pred_spend = np.array([1500] * len(new_df))

        # 3. Clustering
        df_cl = new_df.copy()
        df_cl["monthly_spend_inr"] = pred_spend
        X_cl_new, _, _ = get_cluster_features(df_cl)
        pred_clusters = km.predict(X_cl_new)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.exception(e)
        return

    # ── Results ───────────────────────────────────────────────────────────────
    label_map = {0: "Not interested", 1: "Interested", 2: "Definitely buying"}

    res = pd.DataFrame()
    res["respondent_id"]              = new_df["respondent_id"] if "respondent_id" in new_df.columns else range(len(new_df))
    res["predicted_label"]            = pred_labels
    res["interest"]                   = [label_map[l] for l in pred_labels]
    res["prob_not_interested"]        = np.round(pred_probs[:, 0], 3)
    res["prob_interested"]            = np.round(pred_probs[:, 1], 3)
    res["prob_definitely_buying"]     = np.round(pred_probs[:, 2], 3)
    res["predicted_spend_inr"]        = pred_spend
    res["cluster"]                    = pred_clusters
    res["segment"]                    = [CLUSTER_NAMES.get(c, f"Cluster {c}") for c in pred_clusters]
    res["marketing_action"]           = [
        MARKETING_ACTIONS.get((int(l), int(c)), "Review manually")
        for l, c in zip(pred_labels, pred_clusters)
    ]

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Respondents scored",  len(res))
    k2.metric("Hot leads (label 2)", int((res["predicted_label"] == 2).sum()))
    k3.metric("Warm leads (label 1)", int((res["predicted_label"] == 1).sum()))
    k4.metric("Avg predicted spend", f"₹{res['predicted_spend_inr'].mean():,.0f}")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        ld = res["interest"].value_counts().reset_index()
        ld.columns = ["Label", "Count"]
        fig_ld = px.pie(
            ld, values="Count", names="Label", hole=0.45,
            color_discrete_sequence=["#E24B4A", "#378ADD", "#1D9E75"],
            title="Predicted interest distribution"
        )
        fig_ld.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=280)
        st.plotly_chart(fig_ld, use_container_width=True)

    with col_r2:
        fig_sp = px.bar(
            res.sort_values("predicted_spend_inr", ascending=True),
            x="predicted_spend_inr", y="respondent_id", orientation="h",
            color="predicted_spend_inr", color_continuous_scale="Greens",
            title="Predicted monthly spend per customer",
            labels={"predicted_spend_inr": "Spend (₹)", "respondent_id": "Customer ID"}
        )
        fig_sp.update_layout(
            margin=dict(t=40, b=10, l=10, r=10),
            coloraxis_showscale=False, height=280
        )
        st.plotly_chart(fig_sp, use_container_width=True)

    # Full table
    st.markdown("### Full prediction table")
    st.dataframe(
        res[["respondent_id", "interest", "prob_definitely_buying",
             "predicted_spend_inr", "segment", "marketing_action"]]
        .reset_index(drop=True),
        use_container_width=True
    )

    # Marketing action breakdown
    st.markdown("### Marketing actions breakdown")
    act_df = res["marketing_action"].value_counts().reset_index()
    act_df.columns = ["Action", "Count"]
    fig_act = px.bar(
        act_df, x="Count", y="Action", orientation="h",
        color="Count", color_continuous_scale="Blues",
        text="Count", title="Recommended marketing actions"
    )
    fig_act.update_traces(textposition="outside")
    fig_act.update_layout(
        margin=dict(t=40, b=20, l=20, r=20),
        coloraxis_showscale=False, height=320
    )
    st.plotly_chart(fig_act, use_container_width=True)

    # Download
    st.download_button(
        "⬇️  Download enriched predictions CSV",
        res.to_csv(index=False),
        "zaria_new_customer_predictions.csv", "text/csv"
    )
    st.success("Pipeline complete. Use the **marketing_action** column to prioritise your outreach.")
