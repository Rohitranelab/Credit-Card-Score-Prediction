import streamlit as st
import pickle

st.set_page_config(
    page_title="Credit Score Prediction",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}

.stButton > button {
    width: 100%;
    height: 50px;
    font-size:18px;
    font-weight:bold;
    border-radius:10px;
    background-color:#2563eb;
    color:white;
}

.stButton > button:hover {
    background-color:#1d4ed8;
    color:white;
}

.block-container{
    padding-top:2rem;
}

div[data-testid="stVerticalBlock"]{
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

with open('artifact/encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('artifact/credit_score.pkl', 'rb') as file:
    model = pickle.load(file)

st.markdown("<h2 style='text-align: center;'>💳 Credit Score Prediction</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Enter customer details to predict the credit score.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2, vertical_alignment="top")

with col1:

    st.subheader("👤 Customer Information")

    age = st.text_input("Age")

    annual_income = st.text_input("Annual Income ($)")

    employment_years = st.text_input("Employment Years")

    credit_utilization = st.slider(
        "Credit Utilization (%)",
        0.0,
        100.0,
        30.0
    )

    payment_history = st.selectbox(
        "Payment History",
        ("Excellent", "Good", "Average", "Poor")
    )

with col2:

    st.subheader("💰 Financial Details")

    num_credit_cards = st.text_input("Number of Credit Cards")

    loan_balance = st.text_input("Loan Balance ($)")

    debt_to_income = st.slider(
        "Debt-to-Income Ratio (%)",
        0.0,
        100.0,
        20.0
    )

    credit_inquiries = st.text_input("Credit Inquiries")

payment_history = encoder.transform([payment_history])[0]

st.divider()

if st.button("🚀 Predict Credit Score"):

    credit_data = [[
        age,
        annual_income,
        employment_years,
        credit_utilization,
        payment_history,
        num_credit_cards,
        loan_balance,
        debt_to_income,
        credit_inquiries
    ]]

    prediction = model.predict(credit_data)

    st.success(f"### Predicted Credit Score: **{prediction[0]:.2f}**")