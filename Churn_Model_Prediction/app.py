import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load('best_rf_churn_model.pkl')
    return model

# Main function
def main():
    # Title
    st.title("Bank Customer Churn Prediction 🏦")
    st.write("""
    ### Use this application to predict if a customer is likely to churn
    Fill in the customer information below and click 'Predict' to get the churn probability.
    """)
    
    # Create input columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Profile")
        tenure = st.number_input("Tenure (years)", min_value=0.0, max_value=50.0, value=5.0)
        total_balance = st.number_input("Total Balance ($)", min_value=0.0, max_value=1000000.0, value=10000.0)
        num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=1)
        
    with col2:
        st.subheader("Transaction Information")
        recency = st.number_input("Days Since Last Transaction", min_value=0, max_value=365, value=30)
        frequency = st.number_input("Transaction Frequency (monthly)", min_value=0, max_value=100, value=10)
        monetary = st.number_input("Average Transaction Amount ($)", min_value=0.0, max_value=10000.0, value=100.0)

    # Additional features
    st.subheader("Additional Information")
    col3, col4 = st.columns(2)
    
    with col3:
        has_loan = st.selectbox("Has Active Loan?", ["Yes", "No"])
        num_cards = st.number_input("Number of Cards", min_value=0, max_value=10, value=1)
    
    with col4:
        support_calls = st.number_input("Number of Support Calls", min_value=0, max_value=50, value=0)
        resolution_rate = st.slider("Support Resolution Rate", min_value=0.0, max_value=1.0, value=1.0)

    # Create feature dictionary
    features = {
        'tenure': tenure,
        'total_balance': total_balance,
        'number_of_accounts': num_products,
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary,
        'has_active_loan': 1 if has_loan == "Yes" else 0,
        'number_of_cards': num_cards,
        'support_call_frequency': support_calls,
        'resolution_rate': resolution_rate
    }

    # Create prediction button
    if st.button("Predict Churn Probability"):
        # Load model
        model = load_model()
        
        # Convert features to DataFrame
        input_df = pd.DataFrame([features])
        
        # Make prediction
        prediction_prob = model.predict_proba(input_df)[0][1]
        
        # Show prediction
        st.subheader("Prediction Results")
        
        # Create a color-coded probability gauge
        if prediction_prob < 0.3:
            color = "green"
            risk_level = "Low Risk"
        elif prediction_prob < 0.7:
            color = "orange"
            risk_level = "Medium Risk"
        else:
            color = "red"
            risk_level = "High Risk"
            
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {color}25;">
            <h3 style="color: {color};">{risk_level}</h3>
            <h2 style="color: {color};">{prediction_prob:.1%}</h2>
            <p>Probability of Customer Churn</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional insights based on risk level
        st.subheader("Risk Analysis")
        if prediction_prob < 0.3:
            st.write("📈 This customer shows strong loyalty indicators.")
            st.write("Recommendations:")
            st.write("- Consider offering premium services or products")
            st.write("- Enroll in loyalty rewards program")
        elif prediction_prob < 0.7:
            st.write("⚠️ This customer shows moderate churn risk.")
            st.write("Recommendations:")
            st.write("- Proactive engagement through targeted offers")
            st.write("- Schedule customer satisfaction survey")
            st.write("- Review product usage patterns")
        else:
            st.write("🚨 High risk of customer churn!")
            st.write("Recommendations:")
            st.write("- Immediate customer outreach")
            st.write("- Develop retention strategy")
            st.write("- Consider special retention offers")

# Run the app
if __name__ == '__main__':
    main()
