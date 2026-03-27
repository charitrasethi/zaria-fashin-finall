import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample as sk_resample
from preprocessing import get_feature_matrix, TARGET_COL
import warnings
warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


def balanced_resample(X_train, y_train):
    if HAS_SMOTE:
        sm = SMOTE(random_state=42, k_neighbors=3)
        return sm.fit_resample(X_train, y_train)
    classes = y_train.unique()
    max_count = y_train.value_counts().max()
    X_parts, y_parts = [], []
    for c in classes:
        mask = y_train == c
        Xc, yc = X_train[mask], y_train[mask]
        Xc_r, yc_r = sk_resample(Xc, yc, replace=True,
                                  n_samples=max_count, random_state=42)
        X_parts.append(Xc_r)
        y_parts.append(yc_r)
    return pd.concat(X_parts), pd.concat(y_parts)


@st.cache_resource
def run_classification(_df_id, df):
    X, encoders, feature_cols = get_feature_matrix(df, drop_target=True)
    y = df[TARGET_COL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_bal, y_bal = balanced_resample(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=150, random_state=42,
        class_weight="balanced", max_depth=12, n_jobs=-1
    )
    rf.fit(X_bal, y_bal)

    lr = LogisticRegression(
        max_iter=1000, random_state=42,
        class_weight="balanced", multi_class="ovr"
    )
    lr.fit(X_bal, y_bal)

    return (rf, lr,
            X_test, y_test,
            rf.predict(X_test), lr.predict(X_test),
            rf.predict_proba(X_test), lr.predict_proba(X_test),
            feature_cols, encoders)


def render(df: pd.DataFrame):
    st.markdown("## Classification — Predicting Customer Interest")
    st.markdown(
        "*Random Forest vs Logistic Regression · Oversampling for class balance · "
        "Accuracy · Precision · Recall · F1-Score · ROC-AUC · Feature Importance*"
    )
    st.markdown(
        "**Label mapping:** `0` = Not Interested &nbsp;|&nbsp; "
        "`1` = Interested &nbsp;|&nbsp; `2` = Definitely Buying"
    )

    if not HAS_SMOTE:
        st.info("ℹ️ Using sklearn resample oversampling (SMOTE fallback).")

    with st.spinner("Training classifiers — Random Forest + Logistic Regression..."):
        (rf, lr, X_test, y_test,
         y_pred_rf, y_pred_lr,
         y_prob_rf, y_prob_lr,
         feature_cols, encoders) = run_classification(id(df), df)

    classes     = [0, 1, 2]
    class_names = ["Not Interested (0)", "Interested (1)", "Definitely Buying (2)"]

    # ── Metrics comparison ────────────────────────────────────────────────────
    st.markdown("### Model performance — Accuracy · Precision · Recall · F1")
    rows = []
    for name, yp in [("Random Forest", y_pred_rf), ("Logistic Regression", y_pred_lr)]:
        rows.append({
            "Model":                name,
            "Accuracy":             accuracy_score(y_test, yp),
            "Precision (weighted)": precision_score(y_test, yp, average="weighted", zero_division=0),
            "Recall (weighted)":    recall_score(y_test, yp, average="weighted", zero_division=0),
            "F1 Score (weighted)":  f1_score(y_test, yp, average="weighted", zero_division=0),
        })
    metrics_df = pd.DataFrame(rows).set_index("Model")

    col_m1, col_m2 = st.columns(2)
    for i, (mname, row) in enumerate(metrics_df.iterrows()):
        col = col_m1 if i == 0 else col_m2
        with col:
            st.markdown(f"**{mname}**")
            a, b = st.columns(2)
            a.metric("Accuracy",  f"{row['Accuracy']:.3f}")
            b.metric("F1 Score",  f"{row['F1 Score (weighted)']:.3f}")
            c, d = st.columns(2)
            c.metric("Precision", f"{row['Precision (weighted)']:.3f}")
            d.metric("Recall",    f"{row['Recall (weighted)']:.3f}")

    melt = metrics_df.reset_index().melt(
        id_vars="Model", var_name="Metric", value_name="Score"
    )
    fig_cmp = px.bar(
        melt, x="Metric", y="Score", color="Model", barmode="group",
        text=melt["Score"].apply(lambda x: f"{x:.3f}"),
        color_discrete_sequence=["#378ADD", "#1D9E75"],
        title="RF vs Logistic Regression — all metrics"
    )
    fig_cmp.update_traces(textposition="outside")
    fig_cmp.update_layout(
        margin=dict(t=40, b=20, l=20, r=20),
        height=360, yaxis_range=[0, 1.15]
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.divider()

    # ── Per-class report ──────────────────────────────────────────────────────
    st.markdown("### Per-class Precision · Recall · F1 (Random Forest)")
    report = classification_report(
        y_test, y_pred_rf,
        target_names=class_names,
        output_dict=True, zero_division=0
    )
    rep_df = (
        pd.DataFrame(report).T
        .loc[class_names, ["precision", "recall", "f1-score", "support"]]
        .round(3)
    )
    rep_df["support"] = rep_df["support"].astype(int)
    st.dataframe(rep_df, use_container_width=True)

    # Per-class bar chart
    rep_melt = rep_df[["precision", "recall", "f1-score"]].reset_index()
    rep_melt = rep_melt.rename(columns={"index": "Class"})
    rep_melt = rep_melt.melt(id_vars="Class", var_name="Metric", value_name="Score")
    fig_rep = px.bar(
        rep_melt, x="Class", y="Score", color="Metric", barmode="group",
        text=rep_melt["Score"].apply(lambda x: f"{x:.2f}"),
        color_discrete_sequence=["#534AB7", "#1D9E75", "#BA7517"],
        title="Per-class Precision · Recall · F1 (Random Forest)"
    )
    fig_rep.update_traces(textposition="outside")
    fig_rep.update_layout(
        margin=dict(t=40, b=20, l=20, r=20),
        height=340, yaxis_range=[0, 1.15]
    )
    st.plotly_chart(fig_rep, use_container_width=True)

    st.divider()

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.markdown("### Confusion Matrices")
    cm1, cm2 = st.columns(2)
    for col, name, yp in [
        (cm1, "Random Forest",       y_pred_rf),
        (cm2, "Logistic Regression", y_pred_lr),
    ]:
        with col:
            cm = confusion_matrix(y_test, yp, labels=classes)
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=class_names, y=class_names,
                color_continuous_scale="Blues",
                title=f"Confusion Matrix — {name}",
                labels=dict(x="Predicted", y="Actual")
            )
            fig_cm.update_layout(
                margin=dict(t=50, b=20, l=20, r=20), height=340
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()

    # ── ROC Curves ───────────────────────────────────────────────────────────
    st.markdown("### ROC Curves — One-vs-Rest (per class)")
    y_test_bin  = label_binarize(y_test, classes=classes)
    roc_colors  = ["#E24B4A", "#378ADD", "#1D9E75"]

    rc1, rc2 = st.columns(2)
    for col, name, ypr in [
        (rc1, "Random Forest",       y_prob_rf),
        (rc2, "Logistic Regression", y_prob_lr),
    ]:
        with col:
            fig_roc = go.Figure()
            for i, (cname, clr) in enumerate(zip(class_names, roc_colors)):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], ypr[:, i])
                roc_auc     = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{cname} (AUC={roc_auc:.3f})",
                    line=dict(color=clr, width=2)
                ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Random baseline"
            ))
            fig_roc.update_layout(
                title=f"ROC — {name}",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                margin=dict(t=50, b=20, l=20, r=20),
                height=380
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    # AUC summary table
    st.markdown("#### AUC summary table")
    auc_rows = []
    for name, ypr in [("Random Forest", y_prob_rf), ("Logistic Regression", y_prob_lr)]:
        for i, cname in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], ypr[:, i])
            auc_rows.append({"Model": name, "Class": cname, "AUC": round(auc(fpr, tpr), 3)})
    auc_df = pd.DataFrame(auc_rows)
    fig_auc = px.bar(
        auc_df, x="Class", y="AUC", color="Model", barmode="group",
        text="AUC",
        color_discrete_sequence=["#378ADD", "#1D9E75"],
        title="AUC per class — both models",
        range_y=[0, 1.1]
    )
    fig_auc.update_traces(textposition="outside")
    fig_auc.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=320)
    st.plotly_chart(fig_auc, use_container_width=True)

    st.divider()

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("### Feature Importance (Random Forest)")
    fi_df = (
        pd.DataFrame({
            "Feature":    feature_cols,
            "Importance": rf.feature_importances_
        })
        .sort_values("Importance", ascending=True)
        .tail(25)
    )
    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Teal",
        text=fi_df["Importance"].apply(lambda x: f"{x:.3f}"),
        title="Top 25 features driving purchase-interest prediction"
    )
    fig_fi.update_traces(textposition="outside")
    fig_fi.update_layout(
        margin=dict(t=40, b=20, l=20, r=20),
        coloraxis_showscale=False, height=660
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    # ── Founder insight ───────────────────────────────────────────────────────
    st.markdown("### Founder's Interpretation")
    top3 = fi_df.tail(3)["Feature"].tolist()[::-1]
    st.success(
        f"**Top 3 drivers of Zaria purchase interest:**\n\n"
        f"1. `{top3[0]}` — highest predictive power\n"
        f"2. `{top3[1]}`\n"
        f"3. `{top3[2]}`\n\n"
        "Focus your survey qualification and ad targeting on these variables "
        "to maximise conversion probability per marketing rupee spent."
    )

    st.session_state["rf_model"]     = rf
    st.session_state["encoders"]     = encoders
    st.session_state["feature_cols"] = feature_cols
