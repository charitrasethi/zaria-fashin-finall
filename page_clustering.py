import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from preprocessing import get_cluster_features, ORDINAL_MAPS


CLUSTER_PRESCRIPTIONS = {
    0: {
        "name": "Budget Festival Shoppers",
        "emoji": "🛍️",
        "discount": "Buy 2 Get 1 Free + Festival flash sale",
        "bundle": "Salwar Suit + matching Dupatta combo",
        "channel": "WhatsApp broadcast + Meesho listing",
        "message": "Traditional value collections under ₹999",
        "color": "#1D9E75"
    },
    1: {
        "name": "Urban Trend Seekers",
        "emoji": "💼",
        "discount": "Loyalty points + 15% first-order discount",
        "bundle": "Kurti + Co-ord Set + NightSuit gift set",
        "channel": "Instagram reels + Myntra partnership",
        "message": "New arrivals — fusion ethnic for the modern woman",
        "color": "#378ADD"
    },
    2: {
        "name": "Young Experimenters",
        "emoji": "✨",
        "discount": "Flat 20% off + free shipping above ₹799",
        "bundle": "IndoWestern Top + Palazzo Pants",
        "channel": "Instagram stories + YouTube influencer",
        "message": "Express your style — IndoWestern drops every Friday",
        "color": "#534AB7"
    },
    3: {
        "name": "Traditional Value Buyers",
        "emoji": "🌸",
        "discount": "Festival bundle deal + referral cashback",
        "bundle": "Cotton Saree + Salwar Suit combo pack",
        "channel": "WhatsApp groups + local store tie-up",
        "message": "Pure cotton, pure tradition — proudly Made in India",
        "color": "#D85A30"
    },
    4: {
        "name": "Premium Home & Ethnic",
        "emoji": "🏡",
        "discount": "Loyalty membership + exclusive early access",
        "bundle": "Bedding Set + NightSuit premium gift box",
        "channel": "D2C website + Instagram premium ads",
        "message": "Curated collections for the discerning Indian home",
        "color": "#993556"
    },
}


def render(df: pd.DataFrame):
    st.markdown("## Customer Clustering — Persona Discovery")
    st.markdown("*K-Means clustering to identify natural customer segments for targeted marketing.*")

    X_scaled, scaler, feature_cols = get_cluster_features(df)

    # ── Elbow + Silhouette ────────────────────────────────────────────────────
    st.markdown("### Step 1 — Optimal number of clusters")
    k_range = range(2, 10)
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    col1, col2 = st.columns(2)
    with col1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertias, mode="lines+markers",
                                       line=dict(color="#378ADD", width=2),
                                       marker=dict(size=8)))
        fig_elbow.add_vline(x=5, line_dash="dash", line_color="#E24B4A",
                            annotation_text="k=5 selected", annotation_position="top right")
        fig_elbow.update_layout(title="Elbow curve (Inertia)",
                                xaxis_title="Number of clusters (k)",
                                yaxis_title="Inertia",
                                margin=dict(t=40, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(x=list(k_range), y=sil_scores, mode="lines+markers",
                                     line=dict(color="#1D9E75", width=2),
                                     marker=dict(size=8)))
        fig_sil.add_vline(x=5, line_dash="dash", line_color="#E24B4A",
                          annotation_text="k=5 selected", annotation_position="top right")
        fig_sil.update_layout(title="Silhouette scores",
                              xaxis_title="Number of clusters (k)",
                              yaxis_title="Silhouette Score",
                              margin=dict(t=40, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig_sil, use_container_width=True)

    st.info(f"**Silhouette score at k=5: {sil_scores[3]:.3f}** — values closer to 1.0 indicate well-separated clusters.")

    # ── Fit final model ───────────────────────────────────────────────────────
    n_clusters = st.slider("Select number of clusters", 2, 8, 5)
    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X_scaled)
    df_c = df.copy()
    df_c["Cluster"] = cluster_labels

    st.divider()
    st.markdown("### Step 2 — Cluster visualisation (PCA 2D projection)")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Cluster"] = cluster_labels.astype(str)
    pca_df["respondent_id"] = df["respondent_id"].values

    fig_pca = px.scatter(
        pca_df, x="PC1", y="PC2", color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.7, hover_data=["respondent_id"],
        title=f"Customer clusters in 2D PCA space (k={n_clusters})"
    )
    fig_pca.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=420)
    st.plotly_chart(fig_pca, use_container_width=True)

    st.divider()
    st.markdown("### Step 3 — Cluster profiles & marketing prescriptions")

    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()

    for cid in range(n_clusters):
        presc = CLUSTER_PRESCRIPTIONS.get(cid, {
            "name": f"Segment {cid}", "emoji": "👤",
            "discount": "Flat discount", "bundle": "Mixed combo",
            "channel": "Multi-channel", "message": "Explore Zaria Fashion",
            "color": "#888780"
        })
        sub = df_c[df_c["Cluster"] == cid]
        size = len(sub)
        pct = size / len(df_c) * 100

        with st.expander(f"{presc['emoji']} Cluster {cid} — {presc['name']}  |  {size} customers ({pct:.1f}%)", expanded=(cid == 0)):
            pc1, pc2, pc3 = st.columns(3)

            spend_m = pd.to_numeric(sub["monthly_spend_inr"], errors="coerce").mean()
            sat_m = pd.to_numeric(sub["current_satisfaction"], errors="coerce").mean()
            ref_m = pd.to_numeric(sub["referral_propensity"], errors="coerce").mean()
            label2_pct = (sub["zaria_interest_label"] == 2).sum() / max(len(sub), 1) * 100

            pc1.metric("Avg Monthly Spend", f"₹{spend_m:,.0f}")
            pc2.metric("Avg Satisfaction", f"{sat_m:.1f}/10")
            pc3.metric("Avg Referral Score", f"{ref_m:.1f}/10")

            mc1, mc2 = st.columns(2)
            with mc1:
                # Top city tier
                fig_tier = px.pie(sub, names="city_tier", hole=0.45,
                                  color_discrete_sequence=px.colors.qualitative.Pastel,
                                  title="City tier mix")
                fig_tier.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=260)
                st.plotly_chart(fig_tier, use_container_width=True)
            with mc2:
                # Style identity
                fig_style = px.pie(sub, names="style_identity", hole=0.45,
                                   color_discrete_sequence=px.colors.qualitative.Bold,
                                   title="Style identity mix")
                fig_style.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=260)
                st.plotly_chart(fig_style, use_container_width=True)

            # Marketing prescription card
            st.markdown(f"""
<div style='background:#f8f9fa;border-left:4px solid {presc["color"]};padding:14px 18px;border-radius:0 8px 8px 0;margin-top:8px'>
<b style='color:{presc["color"]}'>Marketing Prescription for Cluster {cid}</b><br><br>
<b>Recommended discount:</b> {presc["discount"]}<br>
<b>Product bundle:</b> {presc["bundle"]}<br>
<b>Best channel:</b> {presc["channel"]}<br>
<b>Campaign message:</b> <i>{presc["message"]}</i><br>
<b>Definitely buying %:</b> {label2_pct:.1f}%
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Step 4 — Cluster feature heatmap")
    cluster_means = []
    for col in feature_cols:
        df_c[col] = pd.to_numeric(df_c.get(col, 0), errors="coerce").fillna(0)

    heat_cols = [c for c in feature_cols if c in df_c.columns][:12]
    heat_df = df_c.groupby("Cluster")[heat_cols].mean()
    fig_heat = px.imshow(
        heat_df.T, text_auto=".1f",
        color_continuous_scale="RdYlGn",
        title="Average feature values per cluster (higher = darker green)",
        labels=dict(x="Cluster", y="Feature", color="Mean value")
    )
    fig_heat.update_layout(margin=dict(t=50, b=20, l=20, r=20), height=420)
    st.plotly_chart(fig_heat, use_container_width=True)

    csv_out = df_c.to_csv(index=False)
    st.download_button("Download data with cluster labels", csv_out,
                       "zaria_clustered.csv", "text/csv")
