import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from itertools import combinations
from preprocessing import prepare_arm_transactions
import warnings
warnings.filterwarnings("ignore")

try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False


# ── Inline Apriori (fallback when mlxtend absent) ─────────────────────────────
def _get_freq_itemsets(transactions, min_support):
    n = len(transactions)
    t_sets = [set(t) for t in transactions]

    # 1-itemsets
    counts1 = defaultdict(int)
    for t in t_sets:
        for item in t:
            counts1[frozenset([item])] += 1
    freq = {k: v / n for k, v in counts1.items() if v / n >= min_support}

    all_freq = dict(freq)
    prev = list(freq.keys())
    k = 2

    while prev:
        candidates = set()
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                union = prev[i] | prev[j]
                if len(union) == k:
                    candidates.add(union)

        counts_k = defaultdict(int)
        for t in t_sets:
            for cand in candidates:
                if cand.issubset(t):
                    counts_k[cand] += 1

        freq_k = {k2: v / n for k2, v in counts_k.items() if v / n >= min_support}
        all_freq.update(freq_k)
        prev = list(freq_k.keys())
        k += 1
        if k > 4:          # cap depth for performance
            break

    return all_freq, n


def _mine_rules(all_freq, n, min_confidence, min_lift):
    rows = []
    for itemset, sup in all_freq.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for size in range(1, len(items)):
            for ant_tuple in combinations(items, size):
                ant = frozenset(ant_tuple)
                con = itemset - ant
                if not con:
                    continue
                ant_sup = all_freq.get(ant, 0)
                con_sup = all_freq.get(con, 0)
                if ant_sup == 0 or con_sup == 0:
                    continue
                conf = sup / ant_sup
                lift = conf / con_sup
                if conf >= min_confidence and lift >= min_lift:
                    rows.append({
                        "antecedents":  " + ".join(sorted(ant)),
                        "consequents":  " + ".join(sorted(con)),
                        "support":      round(sup, 4),
                        "confidence":   round(conf, 4),
                        "lift":         round(lift, 4),
                    })
    return pd.DataFrame(rows).sort_values("lift", ascending=False) if rows else pd.DataFrame()


def run_arm_mlxtend(transactions, min_support, min_confidence, min_lift):
    te = TransactionEncoder()
    te_arr = te.fit_transform(transactions)
    te_df = pd.DataFrame(te_arr, columns=te.columns_)
    freq_items = apriori(te_df, min_support=min_support, use_colnames=True)
    if freq_items.empty:
        return pd.DataFrame(), pd.DataFrame(), 0
    rules = association_rules(
        freq_items, metric="confidence",
        min_threshold=min_confidence,
        num_itemsets=len(freq_items)
    )
    rules = rules[rules["lift"] >= min_lift].sort_values("lift", ascending=False)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: " + ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: " + ".join(sorted(x)))
    rules["rule"] = rules["antecedents"] + " → " + rules["consequents"]

    # Item frequency from 1-itemsets
    one_items = freq_items[freq_items["itemsets"].apply(len) == 1].copy()
    one_items["item"] = one_items["itemsets"].apply(lambda x: list(x)[0])
    return freq_items, rules, one_items


def run_arm_fallback(transactions, min_support, min_confidence, min_lift):
    all_freq, n = _get_freq_itemsets(transactions, min_support)
    rules = _mine_rules(all_freq, n, min_confidence, min_lift)
    # Build one-item freq df
    one_items = pd.DataFrame([
        {"item": list(k)[0], "support": v}
        for k, v in all_freq.items() if len(k) == 1
    ])
    return all_freq, rules, one_items


def render(df: pd.DataFrame):
    st.markdown("## Association Rule Mining")
    st.markdown(
        "*Apriori algorithm — discover which products, colours, fabrics, prints "
        "and occasions travel together. Drive cross-sell, bundling and campaign design.*"
    )

    if not HAS_MLXTEND:
        st.info("ℹ️ Running with inline Apriori (mlxtend not available in this environment). "
                "Results are equivalent; mlxtend will be used when deployed on Streamlit Cloud.")

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        min_support = st.slider(
            "Min support", 0.01, 0.30, 0.05, 0.01,
            help="Fraction of respondents containing the itemset"
        )
    with c2:
        min_confidence = st.slider(
            "Min confidence", 0.10, 1.00, 0.35, 0.05,
            help="How often the rule is correct when antecedent is present"
        )
    with c3:
        min_lift = st.slider(
            "Min lift", 1.00, 5.00, 1.20, 0.10,
            help="Lift > 1 means items appear together more than by chance"
        )

    transactions = prepare_arm_transactions(df)
    st.info(f"**{len(transactions):,} transactions** built from product, colour, fabric, print & occasion columns.")

    with st.spinner("Running Apriori algorithm..."):
        try:
            if HAS_MLXTEND:
                freq_items, rules, one_items = run_arm_mlxtend(
                    transactions, min_support, min_confidence, min_lift
                )
                n_freq = len(freq_items)
            else:
                all_freq, rules, one_items = run_arm_fallback(
                    transactions, min_support, min_confidence, min_lift
                )
                n_freq = len(all_freq)
        except Exception as e:
            st.error(f"ARM error: {e}. Try lowering support threshold.")
            return

    if rules is None or (isinstance(rules, pd.DataFrame) and rules.empty):
        st.warning("No rules found with current thresholds. Try lowering confidence or lift.")
        return

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Frequent itemsets", n_freq)
    k2.metric("Rules generated",   len(rules))
    k3.metric("Max lift",          f"{rules['lift'].max():.2f}")
    k4.metric("Max confidence",    f"{rules['confidence'].max():.2f}")

    st.divider()

    # ── Most frequent items ───────────────────────────────────────────────────
    st.markdown("### Most frequent items (by support)")
    if not one_items.empty:
        one_plot = one_items.sort_values("support", ascending=True).tail(22)
        fig_items = px.bar(
            one_plot, x="support", y="item", orientation="h",
            color="support", color_continuous_scale="Teal",
            text=one_plot["support"].apply(lambda x: f"{x:.3f}"),
            title="Top frequent items — support ≥ min threshold"
        )
        fig_items.update_traces(textposition="outside")
        fig_items.update_layout(
            margin=dict(t=40, b=20, l=20, r=20),
            coloraxis_showscale=False, height=540
        )
        st.plotly_chart(fig_items, use_container_width=True)

    st.divider()

    # ── Rules table ───────────────────────────────────────────────────────────
    st.markdown("### Association rules — Support · Confidence · Lift")
    display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    show_rules = rules[display_cols].head(60).copy()
    show_rules.columns = ["If customer wants…", "They also want…",
                          "Support", "Confidence", "Lift"]
    show_rules["Support"]    = show_rules["Support"].apply(lambda x: f"{float(x):.3f}")
    show_rules["Confidence"] = show_rules["Confidence"].apply(lambda x: f"{float(x):.3f}")
    show_rules["Lift"]       = show_rules["Lift"].apply(lambda x: f"{float(x):.2f}")
    st.dataframe(show_rules.reset_index(drop=True), use_container_width=True, height=380)

    st.divider()

    # ── Scatter: Confidence vs Lift ───────────────────────────────────────────
    st.markdown("### Confidence vs Lift map (bubble = support)")
    rules_plot = rules.copy()
    rules_plot["rule"] = rules_plot["antecedents"] + " → " + rules_plot["consequents"]
    fig_scat = px.scatter(
        rules_plot.head(100),
        x="confidence", y="lift",
        size="support", color="lift",
        color_continuous_scale="RdYlGn",
        hover_data={"rule": True, "support": ":.3f",
                    "confidence": ":.3f", "lift": ":.2f"},
        labels={"confidence": "Confidence", "lift": "Lift"},
        title="Rule quality map — strongest rules appear top-right"
    )
    fig_scat.add_hline(y=1.0, line_dash="dash", line_color="gray",
                       annotation_text="Lift=1 (no association)")
    fig_scat.update_layout(
        margin=dict(t=40, b=20, l=20, r=20), height=420
    )
    st.plotly_chart(fig_scat, use_container_width=True)

    st.divider()

    # ── Top 15 by lift ────────────────────────────────────────────────────────
    st.markdown("### Top 15 rules by lift")
    top15 = rules_plot.nlargest(15, "lift").sort_values("lift")
    fig_lift = px.bar(
        top15, x="lift", y="rule", orientation="h",
        color="confidence", color_continuous_scale="Blues",
        text=top15["lift"].apply(lambda x: f"{x:.2f}"),
        title="Highest lift rules — strongest non-random associations",
        labels={"lift": "Lift", "rule": "Rule", "confidence": "Confidence"}
    )
    fig_lift.update_traces(textposition="outside")
    fig_lift.update_layout(
        margin=dict(t=40, b=20, l=20, r=20),
        height=520, coloraxis_showscale=True
    )
    st.plotly_chart(fig_lift, use_container_width=True)

    st.divider()

    # ── Distributions ─────────────────────────────────────────────────────────
    st.markdown("### Confidence & Lift distributions")
    dc1, dc2 = st.columns(2)
    with dc1:
        fig_conf = px.histogram(
            rules, x="confidence", nbins=30,
            color_discrete_sequence=["#378ADD"],
            title="Confidence distribution",
            labels={"confidence": "Confidence"}
        )
        fig_conf.update_layout(
            margin=dict(t=40, b=20, l=20, r=20), height=280
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    with dc2:
        fig_lift2 = px.histogram(
            rules, x="lift", nbins=30,
            color_discrete_sequence=["#1D9E75"],
            title="Lift distribution",
            labels={"lift": "Lift"}
        )
        fig_lift2.update_layout(
            margin=dict(t=40, b=20, l=20, r=20), height=280
        )
        st.plotly_chart(fig_lift2, use_container_width=True)

    st.divider()

    # ── Top 5 actionable rules ────────────────────────────────────────────────
    st.markdown("### Top 5 actionable rules — founder's read")
    top5 = rules_plot.nlargest(5, "lift").reset_index(drop=True)
    for i, row in top5.iterrows():
        st.markdown(
            f"**Rule {i+1}:** `{row['antecedents']}` → `{row['consequents']}`\n\n"
            f"- Support: **{float(row['support']):.3f}** &nbsp;|&nbsp; "
            f"Confidence: **{float(row['confidence']):.3f}** &nbsp;|&nbsp; "
            f"Lift: **{float(row['lift']):.2f}**\n\n"
            f"- *Action: When a customer shows interest in **{row['antecedents']}**, "
            f"proactively recommend **{row['consequents']}** in cart or WhatsApp follow-up.*"
        )

    # Download
    csv_rules = show_rules.to_csv(index=False)
    st.download_button(
        "Download all rules as CSV",
        csv_rules, "zaria_association_rules.csv", "text/csv"
    )
