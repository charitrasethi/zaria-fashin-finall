import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from preprocessing import ORDINAL_MAPS, NOMINAL_COLS, CONTINUOUS_COLS
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


def prepare_regression_features(df):
    df_r = df.copy()
    reg_cols = [
        "monthly_hh_income", "occupation", "city_tier", "age_group",
        "purchase_frequency", "last_purchase_recency", "avg_items_per_order",
        "gifting_behaviour", "has_returned_product", "price_range_apparel",
        "price_range_bedding", "madein_india_importance",
        "referral_propensity", "current_satisfaction", "brand_loyalty",
        "style_identity", "education", "marital_status", "clothing_size"
    ]

    for col, mapping in ORDINAL_MAPS.items():
        if col in df_r.columns:
            df_r[col] = df_r[col].map(mapping).fillna(0).astype(int)

    nominal_r = ["occupation", "brand_loyalty", "style_identity",
                 "marital_status", "region", "discount_preference"]
    for col in nominal_r:
        if col in df_r.columns:
            le = LabelEncoder()
            df_r[col] = le.fit_transform(df_r[col].astype(str))

    for col in CONTINUOUS_COLS:
        if col in df_r.columns:
            df_r[col] = pd.to_numeric(df_r[col], errors="coerce").fillna(0)

    available = [c for c in reg_cols if c in df_r.columns]
    X = df_r[available].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df_r["monthly_spend_inr"], errors="coerce").fillna(0)
    return X, y, available


def render(df: pd.DataFrame):
    st.markdown("## Spend Regression — Predicting Customer Spending Capacity")
    st.markdown("*Predict how much a customer will spend monthly so Zaria can price, discount, and bundle appropriately.*")

    X, y, feature_cols = prepare_regression_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    with st.spinner("Training regression models..."):
        # Random Forest Regressor
        rf_r = RandomForestRegressor(n_estimators=150, random_state=42,
                                     max_depth=10, n_jobs=-1)
        rf_r.fit(X_train, y_train)
        y_pred_rf = rf_r.predict(X_test)

        # Ridge Regression baseline
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_sc, y_train)
        y_pred_ridge = ridge.predict(X_test_sc)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("### Model performance comparison")

    def metrics_dict(y_true, y_pred, name):
        return {
            "Model": name,
            "MAE (₹)": mean_absolute_error(y_true, y_pred),
            "RMSE (₹)": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R² Score": r2_score(y_true, y_pred),
        }

    m_rf    = metrics_dict(y_test, y_pred_rf,    "Random Forest Regressor")
    m_ridge = metrics_dict(y_test, y_pred_ridge, "Ridge Regression")

    col1, col2 = st.columns(2)
    for col, m in [(col1, m_rf), (col2, m_ridge)]:
        with col:
            st.markdown(f"**{m['Model']}**")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MAE",  f"₹{m['MAE (₹)']:,.0f}")
            mc2.metric("RMSE", f"₹{m['RMSE (₹)']:,.0f}")
            mc3.metric("R²",   f"{m['R² Score']:.3f}")

    # Comparison bar
    metrics_bar = pd.DataFrame([
        {"Model": "Random Forest", "Metric": "R²", "Value": m_rf["R² Score"]},
        {"Model": "Ridge Regression", "Metric": "R²", "Value": m_ridge["R² Score"]},
    ])
    fig_comp = px.bar(metrics_bar, x="Model", y="Value", color="Model",
                      color_discrete_sequence=["#378ADD", "#1D9E75"],
                      text=metrics_bar["Value"].apply(lambda x: f"{x:.3f}"),
                      title="R² Score comparison", range_y=[0, 1.1])
    fig_comp.update_traces(textposition="outside")
    fig_comp.update_layout(margin=dict(t=40, b=20, l=20, r=20),
                           showlegend=False, height=300)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    # ── Actual vs Predicted ────────────────────────────────────────────────────
    st.markdown("### Actual vs predicted spend (Random Forest)")
    avp_df = pd.DataFrame({
        "Actual Spend (₹)": y_test.values,
        "Predicted Spend (₹)": y_pred_rf
    })
    fig_avp = px.scatter(
        avp_df, x="Actual Spend (₹)", y="Predicted Spend (₹)",
        opacity=0.55, color_discrete_sequence=["#378ADD"],
        title="Actual vs Predicted monthly spend"
    )
    max_val = max(avp_df["Actual Spend (₹)"].max(), avp_df["Predicted Spend (₹)"].max())
    fig_avp.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                  mode="lines", line=dict(color="#E24B4A", dash="dash"),
                                  name="Perfect prediction"))
    fig_avp.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=420)
    st.plotly_chart(fig_avp, use_container_width=True)

    st.divider()

    # ── Residuals ─────────────────────────────────────────────────────────────
    st.markdown("### Residual analysis")
    residuals = y_test.values - y_pred_rf
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        fig_res = px.histogram(residuals, nbins=40,
                               color_discrete_sequence=["#534AB7"],
                               title="Residual distribution",
                               labels={"value": "Residual (₹)", "count": "Frequency"})
        fig_res.add_vline(x=0, line_dash="dash", line_color="#E24B4A")
        fig_res.update_layout(margin=dict(t=40, b=20, l=20, r=20),
                              height=300, showlegend=False)
        st.plotly_chart(fig_res, use_container_width=True)

    with res_col2:
        fig_res2 = px.scatter(x=y_pred_rf, y=residuals,
                              opacity=0.5, color_discrete_sequence=["#D85A30"],
                              title="Residuals vs predicted",
                              labels={"x": "Predicted Spend (₹)", "y": "Residual (₹)"})
        fig_res2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_res2.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig_res2, use_container_width=True)

    st.divider()

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown("### Feature importance (Random Forest Regressor)")
    fi = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf_r.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Oranges",
                    text=fi["Importance"].apply(lambda x: f"{x:.3f}"),
                    title="Features driving monthly spend prediction")
    fig_fi.update_traces(textposition="outside")
    fig_fi.update_layout(margin=dict(t=40, b=20, l=20, r=20),
                         coloraxis_showscale=False, height=500)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    # ── Spend tier segmentation ───────────────────────────────────────────────
    st.markdown("### Spend tier segmentation")
    df_tier = df.copy()
    df_tier["monthly_spend_inr"] = pd.to_numeric(df_tier["monthly_spend_inr"], errors="coerce")

    def tier(s):
        if s < 800:   return "Budget (< ₹800)"
        elif s < 2500: return "Mid (₹800–₹2,500)"
        elif s < 6000: return "Premium (₹2,500–₹6,000)"
        else:          return "Luxury (> ₹6,000)"

    df_tier["Spend Tier"] = df_tier["monthly_spend_inr"].apply(tier)
    tier_order = ["Budget (< ₹800)", "Mid (₹800–₹2,500)", "Premium (₹2,500–₹6,000)", "Luxury (> ₹6,000)"]
    tier_counts = df_tier["Spend Tier"].value_counts().reindex(tier_order, fill_value=0).reset_index()
    tier_counts.columns = ["Spend Tier", "Count"]

    tc1, tc2 = st.columns(2)
    with tc1:
        fig_tier = px.bar(tier_counts, x="Spend Tier", y="Count",
                          color="Spend Tier",
                          color_discrete_sequence=["#D85A30", "#BA7517", "#1D9E75", "#534AB7"],
                          text="Count", title="Customer count by spend tier")
        fig_tier.update_traces(textposition="outside")
        fig_tier.update_layout(margin=dict(t=40, b=20, l=20, r=20),
                               showlegend=False, height=320)
        st.plotly_chart(fig_tier, use_container_width=True)

    with tc2:
        fig_tier2 = px.box(df_tier, x="Spend Tier", y="monthly_spend_inr",
                           color="Spend Tier",
                           color_discrete_sequence=["#D85A30", "#BA7517", "#1D9E75", "#534AB7"],
                           category_orders={"Spend Tier": tier_order},
                           title="Spend distribution within tiers",
                           labels={"monthly_spend_inr": "Monthly Spend (₹)"})
        fig_tier2.update_layout(margin=dict(t=40, b=20, l=20, r=20),
                                showlegend=False, height=320)
        st.plotly_chart(fig_tier2, use_container_width=True)

    st.markdown("### Cross-validation scores (Random Forest Regressor, 5-fold)")
    with st.spinner("Running cross-validation..."):
        cv_scores = cross_val_score(rf_r, X, y, cv=5, scoring="r2", n_jobs=-1)
    cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(5)], "R²": cv_scores})
    fig_cv = px.bar(cv_df, x="Fold", y="R²", color="R²",
                    color_continuous_scale="Greens",
                    text=cv_df["R²"].apply(lambda x: f"{x:.3f}"),
                    title=f"5-Fold CV R² scores — Mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    fig_cv.update_traces(textposition="outside")
    fig_cv.update_layout(margin=dict(t=40, b=20, l=20, r=20),
                         coloraxis_showscale=False, height=300,
                         yaxis_range=[0, 1.1])
    st.plotly_chart(fig_cv, use_container_width=True)

    # Store regressor
    st.session_state["rf_regressor"] = rf_r
    st.session_state["reg_feature_cols"] = feature_cols
    st.session_state["reg_scaler"] = scaler
