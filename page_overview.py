import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render(df: pd.DataFrame):
    st.markdown("## Executive Overview")
    st.markdown("*Zaria Fashion · Customer Insight Dashboard · Data-Driven Decision Layer*")

    # ── KPI row ──────────────────────────────────────────────────────────────
    total = len(df)
    label_counts = df["zaria_interest_label"].value_counts()
    hot = int(label_counts.get(2, 0))
    warm = int(label_counts.get(1, 0))
    cold = int(label_counts.get(0, 0))

    spend_col = pd.to_numeric(df["monthly_spend_inr"], errors="coerce")
    avg_spend = spend_col.mean()
    avg_sat = pd.to_numeric(df["current_satisfaction"], errors="coerce").mean()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Respondents", f"{total:,}")
    k2.metric("Definitely Buying (2)", f"{hot}", f"{hot/total*100:.1f}%")
    k3.metric("Interested (1)", f"{warm}", f"{warm/total*100:.1f}%")
    k4.metric("Not Interested (0)", f"{cold}", f"{cold/total*100:.1f}%")
    k5.metric("Avg Monthly Spend", f"₹{avg_spend:,.0f}")
    k6.metric("Avg Satisfaction", f"{avg_sat:.1f} / 10")

    st.divider()

    col1, col2 = st.columns(2)

    # Interest label donut
    with col1:
        st.markdown("#### Purchase interest distribution")
        fig = px.pie(
            values=[hot, warm, cold],
            names=["Definitely buying (2)", "Interested (1)", "Not interested (0)"],
            hole=0.55,
            color_discrete_sequence=["#1D9E75", "#378ADD", "#E24B4A"]
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Region distribution
    with col2:
        st.markdown("#### Respondents by region")
        region_counts = df["region"].value_counts().reset_index()
        region_counts.columns = ["Region", "Count"]
        fig2 = px.bar(
            region_counts, x="Region", y="Count",
            color="Count", color_continuous_scale="Teal",
            text="Count"
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(margin=dict(t=20, b=20, l=20, r=20),
                           coloraxis_showscale=False, height=320)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    # City tier breakdown
    with col3:
        st.markdown("#### City tier breakdown")
        tier_order = ["Metro", "Tier2", "Tier3", "Rural"]
        tier_df = df["city_tier"].value_counts().reindex(tier_order, fill_value=0).reset_index()
        tier_df.columns = ["Tier", "Count"]
        fig3 = px.bar(
            tier_df, x="Tier", y="Count",
            color="Tier",
            color_discrete_sequence=["#534AB7", "#1D9E75", "#BA7517", "#D85A30"],
            text="Count"
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(margin=dict(t=20, b=20, l=20, r=20),
                           showlegend=False, height=300)
        st.plotly_chart(fig3, use_container_width=True)

    # Age group distribution
    with col4:
        st.markdown("#### Age group distribution")
        age_order = ["Under18", "18-24", "25-34", "35-44", "45-55", "55+"]
        age_df = df["age_group"].value_counts().reindex(age_order, fill_value=0).reset_index()
        age_df.columns = ["Age Group", "Count"]
        fig4 = px.bar(
            age_df, x="Age Group", y="Count",
            color="Age Group",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text="Count"
        )
        fig4.update_traces(textposition="outside")
        fig4.update_layout(margin=dict(t=20, b=20, l=20, r=20),
                           showlegend=False, height=300)
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    col5, col6 = st.columns(2)

    # Spend distribution
    with col5:
        st.markdown("#### Monthly spend distribution (₹)")
        spend_data = pd.to_numeric(df["monthly_spend_inr"], errors="coerce").dropna()
        fig5 = px.histogram(
            spend_data, nbins=40,
            color_discrete_sequence=["#378ADD"],
            labels={"value": "Monthly Spend (₹)", "count": "Respondents"}
        )
        fig5.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300, showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    # Style identity
    with col6:
        st.markdown("#### Style identity breakdown")
        style_df = df["style_identity"].value_counts().reset_index()
        style_df.columns = ["Style", "Count"]
        fig6 = px.bar(
            style_df, x="Count", y="Style",
            orientation="h",
            color="Count", color_continuous_scale="Purples",
            text="Count"
        )
        fig6.update_traces(textposition="outside")
        fig6.update_layout(margin=dict(t=20, b=20, l=20, r=20),
                           coloraxis_showscale=False, height=300)
        st.plotly_chart(fig6, use_container_width=True)

    # Interest label by income
    st.markdown("#### Purchase interest by income bracket")
    income_order = ["<15k", "15k-30k", "30k-60k", "60k-1L", ">1L"]
    label_map = {0: "Not interested", 1: "Interested", 2: "Definitely buying"}
    df_plot = df.copy()
    df_plot["Interest"] = df_plot["zaria_interest_label"].map(label_map)
    df_plot["monthly_hh_income"] = pd.Categorical(df_plot["monthly_hh_income"], categories=income_order, ordered=True)
    grp = df_plot.groupby(["monthly_hh_income", "Interest"], observed=True).size().reset_index(name="Count")
    fig7 = px.bar(
        grp, x="monthly_hh_income", y="Count", color="Interest",
        barmode="group",
        color_discrete_map={
            "Not interested": "#E24B4A",
            "Interested": "#378ADD",
            "Definitely buying": "#1D9E75"
        },
        labels={"monthly_hh_income": "Income Bracket"}
    )
    fig7.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
    st.plotly_chart(fig7, use_container_width=True)

    # Top pain points
    st.markdown("#### Top pain points reported")
    pain_df = df["main_pain_point"].value_counts().reset_index()
    pain_df.columns = ["Pain Point", "Count"]
    fig8 = px.bar(
        pain_df, x="Count", y="Pain Point",
        orientation="h",
        color="Count", color_continuous_scale="Reds",
        text="Count"
    )
    fig8.update_traces(textposition="outside")
    fig8.update_layout(margin=dict(t=20, b=20, l=20, r=20),
                       coloraxis_showscale=False, height=320)
    st.plotly_chart(fig8, use_container_width=True)
