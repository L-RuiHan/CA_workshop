import streamlit as st
import pandas as pd

st.title("📊 Customer Churn Retention Dashboard")

st.write("Input customer information to predict churn risk and get retention recommendations")

# ===== 用户输入 =====
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0, 150, 70)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# ===== 简单规则模拟模型 =====
def predict_churn(tenure, monthly_charges, contract):
    score = 0
    
    if contract == "Month-to-month":
        score += 0.4
    if tenure < 12:
        score += 0.3
    if monthly_charges > 80:
        score += 0.3
        
    return min(score, 1)

churn_prob = predict_churn(tenure, monthly_charges, contract)

# ===== 简单分群 =====
if monthly_charges > 80 and tenure > 24:
    segment = "High-Value"
elif tenure < 12:
    segment = "New Customer"
else:
    segment = "Mid-Value"

# ===== 推荐策略 =====
def recommendation(prob, segment):
    if prob > 0.7 and segment == "High-Value":
        return "🔥 Priority retention: offer discount and contract upgrade"
    elif prob > 0.7:
        return "⚠️ Risk control: low-cost retention campaign"
    elif segment == "High-Value":
        return "💎 Loyalty strategy: upsell or bundle services"
    else:
        return "🙂 Maintain engagement"

rec = recommendation(churn_prob, segment)

# ===== 输出 =====
st.subheader("🔍 Prediction Result")
st.write(f"Churn Probability: {churn_prob:.2f}")
st.write(f"Customer Segment: {segment}")

st.subheader("📌 Recommended Action")
st.write(rec)
