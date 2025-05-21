import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    return model

def main():
    # Title and description
    st.title("Bank Loan Default Prediction 💰")
    st.write("""
    ### Predict the likelihood of loan default
    Enter the loan application details below to assess the risk of default.
    """)
    
    # Create input sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Loan Details")
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000.0, max_value=500000.0, value=50000.0)
        interest_rate = st.slider("Interest Rate (%)", min_value=2.0, max_value=15.0, value=7.0, step=0.1)
        loan_type = st.selectbox("Loan Type", ["Car", "Home", "Personal"])
        loan_term = st.slider("Loan Term (years)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)
        
    with col2:
        st.subheader("Customer Information")
        tenure = st.number_input("Customer Tenure (years)", min_value=0.0, max_value=20.0, value=2.0)
        total_balance = st.number_input("Total Balance ($)", min_value=0.0, max_value=500000.0, value=10000.0)
        total_payments = st.number_input("Total Previous Payments ($)", min_value=0.0, max_value=100000.0, value=5000.0)
        support_issues = st.number_input("Number of Support Issues", min_value=0, max_value=10, value=0)

    # Create feature dictionary
    features = {
        'LoanAmount': loan_amount,
        'InterestRate': interest_rate,
        'LoanTerm': loan_term,
        'Tenure': tenure,
        'TotalBalance': total_balance,
        'TotalPayments': total_payments,
        'SupportIssueFrequency': support_issues
    }
    
    # Add one-hot encoding for loan type
    loan_types = ['Car', 'Home', 'Personal']
    for lt in loan_types:
        features[f'LoanType_{lt}'] = 1 if loan_type == lt else 0

    # Predict button
    if st.button("Predict Default Risk"):
        # Load model
        model = load_model()
        
        # Convert features to DataFrame
        input_df = pd.DataFrame([features])
        
        # Make prediction
        try:
            prediction_prob = model.predict_proba(input_df)[0][1]
            
            # Show prediction
            st.subheader("Risk Assessment Results")
            
            # Color-coded risk level
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
                <p>Probability of Loan Default</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk analysis and recommendations
            st.subheader("Risk Analysis")
            if prediction_prob < 0.3:
                st.write("✅ This application shows positive indicators")
                st.write("Recommendations:")
                st.write("- Proceed with standard loan processing")
                st.write("- Consider offering preferential rates")
                st.write("- Potential for upselling other financial products")
            elif prediction_prob < 0.7:
                st.write("⚠️ This application requires additional review")
                st.write("Recommendations:")
                st.write("- Request additional documentation")
                st.write("- Consider adjusting loan terms")
                st.write("- Implement stricter monitoring if approved")
            else:
                st.write("🚫 High risk application detected")
                st.write("Recommendations:")
                st.write("- Detailed credit review required")
                st.write("- Consider requesting collateral")
                st.write("- Evaluate alternative loan structures")
            
            # Feature importance visualization
            st.subheader("Key Risk Factors")
            importance_dict = {
                'Loan Amount': loan_amount/500000,
                'Interest Rate': interest_rate/15,
                'Loan Term': loan_term/30,
                'Customer Tenure': 1 - (tenure/20),  # Inverse relationship
                'Total Balance': total_balance/500000,
                'Payment History': 1 - (total_payments/100000)  # Inverse relationship
            }
            
            # Sort factors by importance
            sorted_factors = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            for factor, value in sorted_factors.items():
                st.progress(value, text=f"{factor}: {value:.0%} impact")

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("Please check if all input values are within expected ranges.")

# Run the app
if __name__ == '__main__':
    main()
