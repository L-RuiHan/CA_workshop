import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

st.set_page_config(
    page_title="Group 4 Retention Copilot",
    layout="wide"
)

# ===== OpenAI client =====
client = None
if "OPENAI_API_KEY" in st.secrets:
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception:
        client = None

# ===== Cluster business rules =====
cluster_rules = {
    0: "High-value risk customer: high spending, highest churn risk, moderate tenure. Priority retention is required.",
    1: "General stable customer: low spending but very stable. Suitable for upsell and cross-sell.",
    2: "High-value loyal customer: highest value, longest tenure, strong loyalty. Focus on VIP relationship maintenance.",
    3: "Churn-risk customer: short-tenure, low-activity, low-value customer. Focus on activation and conversion."
}

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
    df = pd.read_csv("final_app_dataset.csv")
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
    "This internal decision-support tool uses churn model outputs, customer segmentation, and behavioral profiles to recommend cost-effective retention actions."
)

# ===== Helper Functions =====
def active_service(val):
    text = str(val).strip().lower()
    return text not in ["no", "no internet service", "no phone service", "nan", "none", ""]

def get_persona(row):
    entertainment_score = sum([
        active_service(row.get("StreamingTV", "No")),
        active_service(row.get("StreamingMovies", "No"))
    ])

    security_score = sum([
        active_service(row.get("OnlineSecurity", "No")),
        active_service(row.get("OnlineBackup", "No")),
        active_service(row.get("TechSupport", "No"))
    ])

    if entertainment_score >= 1 and entertainment_score >= security_score:
        return "Entertainment Seeker"
    elif security_score >= 1:
        return "Security Conscious"
    return "General User"

def get_lda_risk_flags(row):
    flags = []

    if str(row.get("InternetService", "")).strip().lower() == "fiber optic":
        flags.append("Fiber optic risk")
    if str(row.get("Contract", "")).strip().lower() == "month-to-month":
        flags.append("Month-to-month vulnerability")
    if str(row.get("PaymentMethod", "")).strip().lower() == "electronic check":
        flags.append("Electronic check friction")

    try:
        if float(row.get("tenure", 999)) < 12:
            flags.append("Short-tenure risk")
    except Exception:
        pass

    if active_service(row.get("StreamingMovies", "No")) or active_service(row.get("StreamingTV", "No")):
        flags.append("Entertainment bundle opportunity")

    if active_service(row.get("OnlineSecurity", "No")) or active_service(row.get("TechSupport", "No")):
        flags.append("Security-value opportunity")

    return flags

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

    if str(row.get("Contract", "")).strip().lower() == "month-to-month":
        reasons.append("month-to-month contract indicates lower relationship stability")

    try:
        if float(row.get("tenure", 999)) < 12:
            reasons.append("short tenure suggests a weaker customer relationship")
    except Exception:
        pass

    try:
        if float(row.get("MonthlyCharges", 0)) > 80:
            reasons.append("higher monthly charges may increase churn sensitivity")
    except Exception:
        pass

    if str(row.get("InternetService", "")).strip().lower() == "fiber optic":
        reasons.append("fiber optic service has been identified as a churn risk driver")

    return reasons

def get_recommendation(churn_prob: float, row: pd.Series):
    value_hint = None

    if "risk_adjusted_clv" in row.index and "risk_adjusted_clv" in df.columns:
        try:
            if float(row["risk_adjusted_clv"]) > df["risk_adjusted_clv"].median():
                value_hint = "high"
            else:
                value_hint = "low"
        except Exception:
            value_hint = None

    persona = get_persona(row)
    contract = str(row.get("Contract", "")).strip().lower()

    if churn_prob > 0.7 and value_hint == "high":
        if persona == "Entertainment Seeker":
            return "🔥 **Priority retention**: recommend contract conversion with a value-added entertainment bundle instead of a generic discount."
        elif persona == "Security Conscious":
            return "🔥 **Priority retention**: offer upgraded protection / support services tied to a longer-term plan."
        return "🔥 **Priority retention**: proactive outreach, value-added service bundle, and contract upgrade recommendation."

    elif churn_prob > 0.7:
        if contract == "month-to-month":
            return "⚠️ **High-risk retention**: encourage migration to a longer-term contract through a low-cost value-added offer."
        return "⚠️ **High-risk retention**: launch a focused retention campaign with low-cost incentives and service support."

    elif churn_prob > 0.4:
        return "🟡 **Medium-priority action**: targeted engagement campaign, personalized messaging, and usage-based offer nudges."

    else:
        if value_hint == "high":
            return "💎 **Low-risk growth strategy**: maintain engagement and explore upsell or bundle expansion opportunities."
        return "💎 **Low-risk maintenance**: keep the customer engaged with light-touch lifecycle communication."

def generate_fallback_explanation(row, churn_prob, predicted, priority):
    cluster_desc = cluster_rules.get(row.get("rfm_cluster", -1), "Unknown segment")
    persona = get_persona(row)
    lda_flags = get_lda_risk_flags(row)

    text = f"""Risk explanation:
This customer is classified as {priority.lower()} priority with churn probability at {churn_prob:.2%}. The profile aligns with {cluster_desc.lower()}.

Business implication:
The customer belongs to the {persona} persona. Losing this customer would have {'high' if float(row.get('risk_adjusted_clv', 0)) > df['risk_adjusted_clv'].median() else 'moderate'} business impact based on value and risk.

Recommended retention action:
{get_recommendation(churn_prob, row)}

Strategic reasoning:
The recommendation is based on the customer's segment, persona, and current risk signals: {", ".join(lda_flags) if lda_flags else "no major additional risk flag"}.
"""
    return text

def generate_ai_explanation(row, churn_prob, predicted, priority):
    cluster_desc = cluster_rules.get(row.get("rfm_cluster", -1), "Unknown segment")
    persona = get_persona(row)
    lda_flags = get_lda_risk_flags(row)

    prompt = f"""
You are a senior telecom retention strategist working inside our company.

Your goal is to generate a targeted retention strategy based on analytics results.

Customer Profile:
- Churn Probability: {churn_prob:.2%}
- Predicted Churn: {"Yes" if predicted else "No"}
- Priority: {priority}
- Cluster: {cluster_desc}
- Persona: {persona}
- Contract: {row.get("Contract", "N/A")}
- Tenure: {row.get("tenure", "N/A")}
- Monthly Charges: {row.get("MonthlyCharges", "N/A")}
- Risk-adjusted CLV: {row.get("risk_adjusted_clv", "N/A")}
- Current risk flags: {", ".join(lda_flags) if lda_flags else "No major LDA risk flag triggered"}

Known churn drivers from our analysis:
- Fiber optic users are high-risk
- Month-to-month contracts are unstable
- Low tenure customers are more likely to churn
- Electronic check may increase churn friction

Company rules:
- Do NOT give generic discounts
- Prefer value-added offers (bundles, upgrades)
- Must consider cost efficiency
- Align recommendations with the customer's segment and persona

Return in exactly this format:

Risk explanation:
...

Business implication:
...

Recommended retention action:
...

Strategic reasoning:
...
"""

    if client is None:
        return generate_fallback_explanation(row, churn_prob, predicted, priority)

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )
        return response.output_text
    except Exception:
        return generate_fallback_explanation(row, churn_prob, predicted, priority)

# ===== Tabs =====
tab1, tab2 = st.tabs(["Customer View", "Segment Overview"])

with tab1:
    st.markdown(
        '<div class="section-card">Use the sidebar to enter a customer ID and view churn risk, prediction results, customer persona, and retention recommendations.</div>',
        unsafe_allow_html=True
    )

    if customer_id:
        row = df[df["customerID"] == customer_id]

        if not row.empty:
            row = row.iloc[0]

            churn_prob = float(row["churn_probability"]) if "churn_probability" in row.index else 0.0
            predicted = int(row["predicted_churn"]) if "predicted_churn" in row.index else 0
            priority = get_priority(churn_prob)

            cluster = row["rfm_cluster"] if "rfm_cluster" in row.index else "N/A"
            cluster_name = row["cluster_name"] if "cluster_name" in row.index else "N/A"
            risk_adjusted_clv = row["risk_adjusted_clv"] if "risk_adjusted_clv" in row.index else "N/A"
            persona = get_persona(row)
            lda_flags = get_lda_risk_flags(row)

            st.subheader("Customer Risk Summary")

            col1, col2, col3 = st.columns(3)
            col1.metric("Churn Probability", f"{churn_prob:.2%}")
            col2.metric("Predicted Churn", "Yes" if predicted == 1 else "No")
            col3.metric("Retention Priority", priority)

            st.markdown(get_risk_label(churn_prob))

            st.markdown("### 📊 Segment / Value Information")
            info_col1, info_col2, info_col3 = st.columns(3)
            info_col1.write(f"**Cluster:** {cluster}")
            info_col1.write(f"**Cluster Name:** {cluster_name}")
            info_col2.write(f"**Risk-Adjusted CLV:** {risk_adjusted_clv}")
            info_col2.write(f"**Persona:** {persona}")
            info_col3.write(f"**LDA Risk Flags:** {', '.join(lda_flags) if lda_flags else 'None'}")

            st.markdown("### 🤖 AI Risk Explanation")

            reasons = build_reasons(churn_prob, predicted, row)
            if reasons:
                st.info("Rule-based summary: " + "; ".join(reasons) + ".")
            else:
                st.info("Rule-based summary: this customer currently shows relatively low churn risk.")

            if st.button("Generate Management Recommendation"):
                with st.spinner("Generating explanation..."):
                    ai_text = generate_ai_explanation(row, churn_prob, predicted, priority)
                    st.success(ai_text)

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
                "rfm_cluster", "cluster_name", "risk_adjusted_clv", "tenure",
                "MonthlyCharges", "Contract", "InternetService", "PaymentMethod",
                "StreamingTV", "StreamingMovies", "OnlineSecurity", "OnlineBackup", "TechSupport"
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
            df.groupby(["rfm_cluster", "cluster_name"], dropna=False)
              .agg(group_cols)
              .reset_index()
              .rename(columns={
                  "customerID": "customers",
                  "churn_probability": "avg_churn_probability",
                  "predicted_churn": "predicted_churn_rate",
                  "risk_adjusted_clv": "avg_risk_adjusted_clv"
              })
              .sort_values("rfm_cluster")
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
    '<p class="small-note">This interface is designed for project presentation and decision-support demonstration using model outputs, customer segmentation, persona interpretation, and churn-driver insights.</p>',
    unsafe_allow_html=True
)
