import streamlit as st
import pandas as pd

st.set_page_config(page_title="Retention Copilot", layout="wide")

st.title("📡 Group 4 Customer Retention Copilot")
st.markdown("""
This app operationalizes our churn prediction pipeline by integrating:
- Predictive churn model outputs
- Customer segmentation
- Value-based prioritisation
- AI-assisted retention recommendations
""")

# ===== 读取你的模型结果 =====
df = pd.read_csv("customer_churn_risk_scores.csv")

# ===== 输入 customerID =====
customer_id = st.text_input("Enter Customer ID")

if customer_id:

    row = df[df["customerID"] == customer_id]

    if not row.empty:

        # ===== 提取真实模型结果 =====
        churn_prob = float(row["churn_probability"].values[0])
        predicted = int(row["predicted_churn"].values[0])

        # 如果你有这些字段就用（没有可以删）
        cluster = row["rfm_cluster"].values[0] if "rfm_cluster" in df.columns else "N/A"

        # ===== 优先级逻辑（结合你项目）=====
        if churn_prob > 0.7:
            priority = "High"
        elif churn_prob > 0.4:
            priority = "Medium"
        else:
            priority = "Low"

        # ===== 页面展示 =====
        col1, col2, col3 = st.columns(3)

        col1.metric("Churn Probability", f"{churn_prob:.2%}")
        col2.metric("Predicted Churn", "Yes" if predicted == 1 else "No")
        col3.metric("Retention Priority", priority)

        st.subheader("📊 Customer Segment Info")
        st.write(f"Cluster: {cluster}")

        # ===== AI解释（关键加分点）=====
        reasons = []

        if churn_prob > 0.7:
            reasons.append("High predicted churn probability")
        if predicted == 1:
            reasons.append("Model classifies customer as churn risk")

        st.subheader("🤖 Why this customer may churn")
        if reasons:
            st.write("This customer is at risk because: " + ", ".join(reasons))
        else:
            st.write("Low churn risk based on model prediction")

        # ===== 推荐策略 =====
        st.subheader("🎯 Recommended Retention Action")

        if churn_prob > 0.7:
            st.error("🔥 Priority retention: proactive outreach, discount, contract upgrade")
        elif churn_prob > 0.4:
            st.warning("⚠️ Medium priority: targeted engagement campaign")
        else:
            st.success("💎 Low risk: maintain engagement and upsell opportunities")

    else:
        st.error("Customer ID not found ❌")
