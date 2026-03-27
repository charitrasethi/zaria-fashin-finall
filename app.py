import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Zaria Fashion — Analytics Dashboard",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label { color: #b0b0b0 !important; }
    .metric-container { background: #f8f9fa; border-radius: 8px; padding: 12px; }
    h2 { color: #1a1a2e; }
    h3 { color: #2d2d5e; }
    .stAlert { border-radius: 8px; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("zaria_fashion_clean_data.csv")
    df.columns = df.columns.str.strip()
    for col in ["products_interested", "preferred_fabric", "preferred_colors",
                "preferred_prints", "preferred_channel", "purchase_occasion",
                "discovery_channel"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    df["monthly_spend_inr"] = pd.to_numeric(df["monthly_spend_inr"], errors="coerce").fillna(1200)
    df["madein_india_importance"] = pd.to_numeric(df["madein_india_importance"], errors="coerce").fillna(7)
    df["referral_propensity"] = pd.to_numeric(df["referral_propensity"], errors="coerce").fillna(6)
    df["current_satisfaction"] = pd.to_numeric(df["current_satisfaction"], errors="coerce").fillna(5)
    df["zaria_interest_label"] = pd.to_numeric(df["zaria_interest_label"], errors="coerce").fillna(1).astype(int)
    return df


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👗 Zaria Fashion")
    st.markdown("### Analytics Dashboard")
    st.markdown("---")

    pages = {
        "Executive Overview":       "📊",
        "Customer Explorer":        "🔍",
        "Clustering — Personas":    "🎯",
        "Association Rules":        "🔗",
        "Classification":           "🤖",
        "Spend Regression":         "📈",
        "New Customer Predictor":   "🆕",
    }

    page = st.radio(
        "Navigate to",
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]}  {x}"
    )

    st.markdown("---")
    st.markdown("**Analysis layers**")
    st.markdown("📊 Descriptive · 🔬 Diagnostic")
    st.markdown("🤖 Predictive · 🎯 Prescriptive")
    st.markdown("---")
    st.markdown("*Zaria Fashion · v1.0*")
    st.markdown("*Data-driven decision engine*")


# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()

# ── Route to page ─────────────────────────────────────────────────────────────
if page == "Executive Overview":
    import page_overview
    page_overview.render(df)

elif page == "Customer Explorer":
    import page_explorer
    page_explorer.render(df)

elif page == "Clustering — Personas":
    import page_clustering
    page_clustering.render(df)

elif page == "Association Rules":
    import page_arm
    page_arm.render(df)

elif page == "Classification":
    import page_classification
    page_classification.render(df)

elif page == "Spend Regression":
    import page_regression
    page_regression.render(df)

elif page == "New Customer Predictor":
    import page_predictor
    page_predictor.render(df)
