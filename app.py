import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

st.set_page_config(
    page_title="Group 4 Retention Copilot",
    layout="wide"
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ===== Custom Style =====
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 0;
}
.sub-title {
    font-size: 18px;
    color: #5f6368;
    margin-top: 0;
    margin-bottom: 20px;
}
.section-card {
    padding: 18px;
    border-radius: 14px;
    background-color: #f7f9fc;
    margin-bottom: 18px;
    border: 1px solid #e6eaf0;
}
.small-note {
    color: #6b7280;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ===== Title =====
st.markdown('<p class="main-title">📡 Group 4 Customer Retention Copilot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">AI-Driven Churn Prediction, Customer Segmentation, and Retention Strategy Support</p>',
    unsafe_allow_html=True
)
st.caption("Built on Group 4 Customer Analytics Pipeline")

st.markdown("---")

# ===== Load Data =====
@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn_risk_scores.csv")
    df.columns = [c.strip() for c in df.columns]
    if "customerID" in df.columns:
        df["customerID"] = df["customerID"].astype(str).str.strip()
    return df

df = load_data()

# ===== Sidebar =====
st.sidebar.header("🔍 Customer Lookup")
customer_id = st.sidebar.text_input("Enter Customer ID")
st.sidebar.markdown("---")
st.sidebar.markdown("### About this app")
st.sidebar.write(
    "This internal decision-support tool uses our churn model outputs to identify high-risk customers and recommend cost-effective retention actions."
)

# ===== Helper Functions =====
def generate_ai_explanation(row, churn_prob, predicted, priority):
    prompt = f"""
You are a customer retention analyst working for our telecom company.

Your task is to explain this customer's churn risk in a business-oriented way
and recommend a practical action aligned with our company objectives:
1. reduce churn
2. protect high-value customers
3. avoid unnecessary retention cost

Customer information:
- Customer ID: {row.get('customerID', 'N/A')}
- Churn probability: {churn_prob:.2%}
- Predicted churn: {"Yes" if predicted == 1 else "No"}
- Retention priority: {priority}
- Contract: {row.get('Contract', 'N/A')}
- Tenure: {row.get('tenure', 'N/A')}
- Monthly Charges: {row.get('MonthlyCharges', 'N/A')}
- Risk-adjusted CLV: {row.get('risk_adjusted_clv', 'N/A')}
- Cluster: {row.get('rfm_cluster', 'N/A')}

Please produce:
1. Risk explanation
2. Business implication
3. Recommended retention action

Requirements:
- Keep it under 140 words
- Use concise business language
- Do not invent unavailable variables
- Make the advice specific to our company, not generic to the telecom industry
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )
    return response.output_text
    
def get_priority(churn_prob: float) -> str:
    if churn_prob > 0.7:
        return "High"
    elif churn_prob > 0.4:
        return "Medium"
    return "Low"

def get_risk_label(churn_prob: float) -> str:
    if churn_prob > 0.7:
        return "🔴 High Risk Customer"
    elif churn_prob > 0.4:
        return "🟡 Medium Risk Customer"
    return "🟢 Low Risk Customer"

def build_reasons(churn_prob: float, predicted: int, row: pd.Series):
    reasons = []

    if churn_prob > 0.7:
        reasons.append("very high predicted churn probability")
    elif churn_prob > 0.4:
        reasons.append("moderately elevated churn probability")

    if predicted == 1:
        reasons.append("the predictive model classifies this customer as churn risk")

    if "Contract" in row.index:
        contract_val = str(row["Contract"]).strip().lower()
        if contract_val == "month-to-month":
            reasons.append("month-to-month contract indicates lower relationship stability")

    if "tenure" in row.index:
        try:
            if float(row["tenure"]) < 12:
                reasons.append("short tenure suggests a weaker customer relationship")
        except:
            pass

    if "MonthlyCharges" in row.index:
        try:
            if float(row["MonthlyCharges"]) > 80:
                reasons.append("higher monthly charges may increase churn sensitivity")
        except:
            pass

    return reasons

def get_recommendation(churn_prob: float, row: pd.Series):
    value_hint = None

    if "risk_adjusted_clv" in row.index:
        try:
            if float(row["risk_adjusted_clv"]) > df["risk_adjusted_clv"].median():
                value_hint = "high"
            else:
                value_hint = "low"
        except:
            value_hint = None

    if churn_prob > 0.7 and value_hint == "high":
        return (
            "🔥 **Priority retention**: proactive outreach, targeted incentive, "
            "contract upgrade recommendation, and service recovery follow-up."
        )
    elif churn_prob > 0.7:
        return (
            "⚠️ **High-risk retention**: launch a focused retention campaign with "
            "low-cost incentives and customer support intervention."
        )
    elif churn_prob > 0.4:
        return (
            "🟡 **Medium-priority action**: targeted engagement campaign, personalized "
            "messaging, and usage-based offer nudges."
        )
    else:
        return (
            "💎 **Low-risk maintenance**: maintain engagement, explore upsell or bundle "
            "expansion opportunities."
        )

# ===== Tabs =====
tab1, tab2 = st.tabs(["Customer View", "Segment Overview"])

with tab1:
    st.markdown('<div class="section-card">Use the sidebar to enter a customer ID and view churn risk, prediction results, and retention recommendations.</div>', unsafe_allow_html=True)

    if customer_id:
        row = df[df["customerID"] == customer_id]

        if not row.empty:
            row = row.iloc[0]

            churn_prob = float(row["churn_probability"]) if "churn_probability" in row.index else 0.0
            predicted = int(row["predicted_churn"]) if "predicted_churn" in row.index else 0
            priority = get_priority(churn_prob)

            cluster = row["rfm_cluster"] if "rfm_cluster" in row.index else "N/A"
            risk_adjusted_clv = row["risk_adjusted_clv"] if "risk_adjusted_clv" in row.index else "N/A"

            st.subheader("Customer Risk Summary")

            col1, col2, col3 = st.columns(3)
            col1.metric("Churn Probability", f"{churn_prob:.2%}")
            col2.metric("Predicted Churn", "Yes" if predicted == 1 else "No")
            col3.metric("Retention Priority", priority)

            st.markdown(get_risk_label(churn_prob))

            st.markdown("### 📊 Segment / Value Information")
            info_col1, info_col2 = st.columns(2)
            info_col1.write(f"**Cluster:** {cluster}")
            info_col2.write(f"**Risk-Adjusted CLV:** {risk_adjusted_clv}")

            st.markdown("### 🤖 AI Risk Explanation")

            # 先保留规则解释，保证即使 API 失败也能展示
            reasons = build_reasons(churn_prob, predicted, row)
            if reasons:
                st.info("Rule-based summary: " + "; ".join(reasons) + ".")
            else:
                st.info("Rule-based summary: this customer currently shows relatively low churn risk.")
            
            # 再加 ChatGPT 按钮，避免每次刷新都调用 API
            if st.button("Generate Management Recommendation"):
                with st.spinner("Generating explanation..."):
                    try:
                        ai_text = generate_ai_explanation(row, churn_prob, predicted, priority)
                        st.success(ai_text)
                    except Exception as e:
                        st.error(f"OpenAI API call failed: {e}")

            st.markdown("### 🎯 Recommended Retention Action")
            recommendation = get_recommendation(churn_prob, row)
            if churn_prob > 0.7:
                st.error(recommendation)
            elif churn_prob > 0.4:
                st.warning(recommendation)
            else:
                st.success(recommendation)

            st.markdown("### 🧾 Raw Customer Output")
            display_cols = [c for c in [
                "customerID", "churn_probability", "predicted_churn",
                "rfm_cluster", "risk_adjusted_clv", "tenure",
                "MonthlyCharges", "Contract"
            ] if c in df.columns]

            st.dataframe(pd.DataFrame([row[display_cols]]), use_container_width=True)

        else:
            st.error("Customer ID not found.")
    else:
        st.markdown(
            '<p class="small-note">Enter a valid customer ID in the sidebar to see customer-level results.</p>',
            unsafe_allow_html=True
        )

with tab2:
    st.subheader("Segment Overview")

    if "rfm_cluster" in df.columns:
        group_cols = {"customerID": "count", "churn_probability": "mean"}
        if "predicted_churn" in df.columns:
            group_cols["predicted_churn"] = "mean"
        if "risk_adjusted_clv" in df.columns:
            group_cols["risk_adjusted_clv"] = "mean"

        segment_summary = (
            df.groupby("rfm_cluster", dropna=False)
              .agg(group_cols)
              .reset_index()
              .rename(columns={
                  "customerID": "customers",
                  "churn_probability": "avg_churn_probability",
                  "predicted_churn": "predicted_churn_rate",
                  "risk_adjusted_clv": "avg_risk_adjusted_clv"
              })
        )

        st.dataframe(segment_summary, use_container_width=True)

        st.markdown("### 📈 Churn Probability by Cluster")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(segment_summary["rfm_cluster"].astype(str), segment_summary["avg_churn_probability"])
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Average Churn Probability")
        ax.set_title("Average Churn Probability Across Clusters")
        st.pyplot(fig)

    else:
        st.warning("No cluster column found in the uploaded result file.")

# ===== Footer =====
st.markdown("---")
st.markdown(
    '<p class="small-note">This interface is designed for project presentation and decision-support demonstration using precomputed model outputs.</p>',
    unsafe_allow_html=True
)
