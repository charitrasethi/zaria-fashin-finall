import streamlit as st
import pandas as pd
import plotly.express as px


def render(df: pd.DataFrame):
    st.markdown("## Customer Explorer")
    st.markdown("*Filter and drill into any segment to understand who your customers are and what drives their interest.*")

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("### Filters — Customer Explorer")

    regions = ["All"] + sorted(df["region"].dropna().unique().tolist())
    sel_region = st.sidebar.selectbox("Region", regions)

    tiers = ["All"] + sorted(df["city_tier"].dropna().unique().tolist())
    sel_tier = st.sidebar.selectbox("City Tier", tiers)

    age_groups = ["All"] + ["Under18", "18-24", "25-34", "35-44", "45-55", "55+"]
    sel_age = st.sidebar.selectbox("Age Group", age_groups)

    incomes = ["All"] + ["<15k", "15k-30k", "30k-60k", "60k-1L", ">1L"]
    sel_income = st.sidebar.selectbox("Income Bracket", incomes)

    occupations = ["All"] + sorted(df["occupation"].dropna().unique().tolist())
    sel_occ = st.sidebar.selectbox("Occupation", occupations)

    labels = ["All", "0 — Not interested", "1 — Interested", "2 — Definitely buying"]
    sel_label = st.sidebar.selectbox("Interest Label", labels)

    # ── Apply filters ─────────────────────────────────────────────────────────
    fdf = df.copy()
    if sel_region != "All":
        fdf = fdf[fdf["region"] == sel_region]
    if sel_tier != "All":
        fdf = fdf[fdf["city_tier"] == sel_tier]
    if sel_age != "All":
        fdf = fdf[fdf["age_group"] == sel_age]
    if sel_income != "All":
        fdf = fdf[fdf["monthly_hh_income"] == sel_income]
    if sel_occ != "All":
        fdf = fdf[fdf["occupation"] == sel_occ]
    if sel_label != "All":
        lval = int(sel_label[0])
        fdf = fdf[fdf["zaria_interest_label"] == lval]

    st.info(f"**{len(fdf):,} respondents** match the current filters (out of {len(df):,} total)")

    if fdf.empty:
        st.warning("No data matches the selected filters. Please adjust.")
        return

    # ── KPIs for filtered segment ─────────────────────────────────────────────
    spend = pd.to_numeric(fdf["monthly_spend_inr"], errors="coerce")
    sat   = pd.to_numeric(fdf["current_satisfaction"], errors="coerce")
    ref   = pd.to_numeric(fdf["referral_propensity"], errors="coerce")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Monthly Spend", f"₹{spend.mean():,.0f}")
    c2.metric("Avg Satisfaction", f"{sat.mean():.1f}/10")
    c3.metric("Avg Referral Score", f"{ref.mean():.1f}/10")
    c4.metric("Definitely Buying %",
              f"{(fdf['zaria_interest_label']==2).sum()/len(fdf)*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    # Occupation breakdown
    with col1:
        st.markdown("#### Occupation breakdown")
        occ_df = fdf["occupation"].value_counts().reset_index()
        occ_df.columns = ["Occupation", "Count"]
        fig = px.bar(occ_df, x="Count", y="Occupation", orientation="h",
                     color="Count", color_continuous_scale="Teal", text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10),
                          coloraxis_showscale=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Brand loyalty
    with col2:
        st.markdown("#### Brand loyalty distribution")
        bl_df = fdf["brand_loyalty"].value_counts().reset_index()
        bl_df.columns = ["Loyalty Type", "Count"]
        fig2 = px.pie(bl_df, values="Count", names="Loyalty Type",
                      color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.4)
        fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    # Discount preference
    with col3:
        st.markdown("#### Preferred discount type")
        disc_df = fdf["discount_preference"].value_counts().reset_index()
        disc_df.columns = ["Discount", "Count"]
        fig3 = px.bar(disc_df, x="Discount", y="Count",
                      color="Count", color_continuous_scale="Purples", text="Count")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(margin=dict(t=10, b=10, l=10, r=10),
                           coloraxis_showscale=False, height=300,
                           xaxis_tickangle=-25)
        st.plotly_chart(fig3, use_container_width=True)

    # Spend by city tier
    with col4:
        st.markdown("#### Spend distribution by city tier")
        tier_order = ["Metro", "Tier2", "Tier3", "Rural"]
        fdf2 = fdf.copy()
        fdf2["monthly_spend_inr"] = pd.to_numeric(fdf2["monthly_spend_inr"], errors="coerce")
        fdf2["city_tier"] = pd.Categorical(fdf2["city_tier"], categories=tier_order, ordered=True)
        fig4 = px.box(fdf2, x="city_tier", y="monthly_spend_inr",
                      color="city_tier",
                      color_discrete_sequence=["#534AB7", "#1D9E75", "#BA7517", "#D85A30"],
                      labels={"city_tier": "City Tier", "monthly_spend_inr": "Monthly Spend (₹)"})
        fig4.update_layout(margin=dict(t=10, b=10, l=10, r=10),
                           showlegend=False, height=300)
        st.plotly_chart(fig4, use_container_width=True)

    # Cross tab: interest label vs style identity
    st.markdown("#### Interest label vs style identity (cross-tab)")
    ct = pd.crosstab(fdf["style_identity"], fdf["zaria_interest_label"])
    ct.columns = [f"Label {c}" for c in ct.columns]
    ct = ct.reset_index()
    fig5 = px.bar(ct, x="style_identity", y=ct.columns[1:].tolist(),
                  barmode="stack",
                  color_discrete_sequence=["#E24B4A", "#378ADD", "#1D9E75"],
                  labels={"style_identity": "Style Identity", "value": "Count"})
    fig5.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
    st.plotly_chart(fig5, use_container_width=True)

    # Satisfaction vs spend scatter
    st.markdown("#### Satisfaction vs monthly spend")
    scatter_df = fdf.copy()
    scatter_df["monthly_spend_inr"] = pd.to_numeric(scatter_df["monthly_spend_inr"], errors="coerce")
    scatter_df["current_satisfaction"] = pd.to_numeric(scatter_df["current_satisfaction"], errors="coerce")
    scatter_df["Interest"] = scatter_df["zaria_interest_label"].map(
        {0: "Not interested", 1: "Interested", 2: "Definitely buying"})
    fig6 = px.scatter(
        scatter_df.dropna(subset=["monthly_spend_inr", "current_satisfaction"]),
        x="current_satisfaction", y="monthly_spend_inr",
        color="Interest",
        color_discrete_map={"Not interested": "#E24B4A", "Interested": "#378ADD", "Definitely buying": "#1D9E75"},
        opacity=0.6,
        labels={"current_satisfaction": "Satisfaction (1–10)", "monthly_spend_inr": "Monthly Spend (₹)"},
        hover_data=["age_group", "city_tier", "occupation"]
    )
    fig6.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=380)
    st.plotly_chart(fig6, use_container_width=True)

    # Discovery channel analysis
    st.markdown("#### Discovery channel frequency (how customers find new brands)")
    all_channels = []
    for val in fdf["discovery_channel"].dropna():
        all_channels.extend([c.strip() for c in str(val).split("|") if c.strip()])
    ch_df = pd.Series(all_channels).value_counts().reset_index()
    ch_df.columns = ["Channel", "Count"]
    fig7 = px.bar(ch_df, x="Channel", y="Count",
                  color="Count", color_continuous_scale="Blues", text="Count")
    fig7.update_traces(textposition="outside")
    fig7.update_layout(margin=dict(t=10, b=10, l=10, r=10),
                       coloraxis_showscale=False, height=320)
    st.plotly_chart(fig7, use_container_width=True)

    # Raw data table
    with st.expander("View filtered raw data"):
        st.dataframe(fdf.reset_index(drop=True), use_container_width=True)
        csv = fdf.to_csv(index=False)
        st.download_button("Download filtered segment as CSV", csv,
                           "zaria_filtered_segment.csv", "text/csv")
