# app/streamlit_app.py

import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #E5E7EB;
    }
    .high-risk {
        background-color: #FEE2E2;
        border-color: #DC2626;
    }
    .medium-risk {
        background-color: #FEF3C7;
        border-color: #D97706;
    }
    .low-risk {
        background-color: #D1FAE5;
        border-color: #059669;
    }
    .stButton button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📱 Telco Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
This tool predicts whether a telecom customer is likely to churn (cancel service). 
The model analyzes customer demographics, account information, and service usage patterns.
""")

# Load models with caching
@st.cache_resource
def load_models():
    """Load trained models and scaler"""
    try:
        # Load models
        lr_model = joblib.load('models/logistic_regression_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        
        # Load selected features
        with open('models/selected_features.pkl', 'rb') as f:
            selected_features = pickle.load(f)
        
        st.sidebar.success("✅ Models loaded successfully!")
        return lr_model, scaler, selected_features
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {e}")
        return None, None, None

# Load models
lr_model, scaler, selected_features = load_models()

# Only show input form if models loaded successfully
if lr_model is not None and scaler is not None and selected_features is not None:
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header("Customer Details")
        
        # Model selection
        st.subheader("Model Settings")
        model_choice = st.radio(
            "Selected Model for Prediction:",
            ["Logistic Regression"],
            help="Random Forest generally performs better for this dataset"
        )
        
        # Create tabs for organized inputs
        tab1, tab2, tab3 = st.tabs(["👤 Demographics", "💳 Account Info", "🔧 Services"])
        
        with tab1:
            gender = st.radio("Gender", ["Female", "Male"])
            senior_citizen = st.radio("Senior Citizen", ["No", "Yes"])
            partner = st.radio("Partner", ["No", "Yes"])
        
        with tab2:
            tenure = st.slider("Tenure (months)", 0, 72, 12, 
                              help="How long the customer has been with the company")
            
            # Auto-calculate TenureGroup based on tenure
            if tenure < 12:
                tenure_group = '0-1yr'
            elif tenure < 24:
                tenure_group = '1-2yr'
            elif tenure < 36:
                tenure_group = '2-3yr'
            elif tenure < 48:
                tenure_group = '3-4yr'
            else:
                tenure_group = '4-5yr'
            
            st.info(f"Tenure Group: {tenure_group}")
            
            contract = st.selectbox("Contract Type", 
                                   ["Month-to-month", "One year", "Two year"],
                                   help="Contract duration")
            
            paperless_billing = st.radio("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", 
                                         ["Electronic check", "Mailed check", 
                                          "Bank transfer (automatic)", 
                                          "Credit card (automatic)"])
            
            monthly_charges = st.number_input("Monthly Charges ($)", 
                                             min_value=18.0, max_value=120.0, 
                                             value=65.0, step=1.0)
            total_charges = st.number_input("Total Charges ($)", 
                                           min_value=0.0, max_value=9000.0, 
                                           value=2000.0, step=100.0)
            
            # Calculate MonthlyToTotalRatio (handle division by zero)
            if total_charges > 0:
                monthly_to_total = monthly_charges / total_charges
            else:
                monthly_to_total = 0
            
            st.metric("Monthly/Total Ratio", f"{monthly_to_total:.3f}")
        
        with tab3:
            phone_service = st.radio("Phone Service", ["No", "Yes"])
            
            if phone_service == "Yes":
                multiple_lines = st.radio("Multiple Lines", ["No", "Yes"])
            else:
                multiple_lines = "No phone service"
            
            internet_service = st.radio("Internet Service", 
                                       ["DSL", "Fiber optic", "No"])
            
            if internet_service != "No":
                col1, col2 = st.columns(2)
                with col1:
                    online_security = st.radio("Online Security", ["No", "Yes"])
                    online_backup = st.radio("Online Backup", ["No", "Yes"])
                    device_protection = st.radio("Device Protection", ["No", "Yes"])
                with col2:
                    tech_support = st.radio("Tech Support", ["No", "Yes"])
                    streaming_tv = st.radio("Streaming TV", ["No", "Yes"])
                    streaming_movies = st.radio("Streaming Movies", ["No", "Yes"])
            else:
                online_security = online_backup = device_protection = "No internet service"
                tech_support = streaming_tv = streaming_movies = "No internet service"
        
        # Service Count Calculation
        service_count = sum([
            1 if online_security == "Yes" else 0,
            1 if online_backup == "Yes" else 0,
            1 if device_protection == "Yes" else 0,
            1 if tech_support == "Yes" else 0,
            1 if streaming_tv == "Yes" else 0,
            1 if streaming_movies == "Yes" else 0
        ])
        st.sidebar.metric("Additional Services Used", service_count)
        
        # Predict button
        predict_button = st.button("🎯 Predict Churn Risk", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_button:
            
            tenure_mapping = {'0-1yr': 0, '1-2yr': 1, '2-3yr': 2, 
                     '3-4yr': 3, '4-5yr': 4, '5-6yr': 5}
            
            input_data = {
                'gender': 1 if gender == "Male" else 0,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': 1 if partner == "Yes" else 0,
                'tenure': float(tenure),  # Ensure float
                'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
                'MonthlyCharges': float(monthly_charges),
                'TotalCharges': float(total_charges),
                'TenureGroup': float(tenure_mapping[tenure_group]),  # Convert to number
                'MonthlyToTotalRatio': float(monthly_to_total),
                'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
                'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
                'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
                'OnlineBackup_Yes': 1 if online_backup == "Yes" else 0,
                'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
                'Contract_One year': 1 if contract == "One year" else 0,
                'Contract_Two year': 1 if contract == "Two year" else 0,
                'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
            }
            
            # Add other one-hot encoded features with 0 values
            for feature in selected_features:
                if feature not in input_data:
                    # Extract base feature and category
                    if '_' in feature:
                        base_feature = feature.split('_')[0]
                        category = '_'.join(feature.split('_')[1:])
                        
                        # Check if this category matches user input
                        if base_feature == "MultipleLines" and multiple_lines == category:
                            input_data[feature] = 1
                        elif base_feature == "InternetService" and internet_service == category:
                            input_data[feature] = 1
                        elif base_feature == "PaymentMethod" and payment_method.replace(" (automatic)", "") == category:
                            input_data[feature] = 1
                        else:
                            input_data[feature] = 0
                    else:
                        input_data[feature] = 0
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure columns are in correct order
            input_df = input_df[selected_features]
            
            input_df = input_df.apply(pd.to_numeric, errors='coerce')
            
            # Scale features for Logistic Regression
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            churn_prob = lr_model.predict_proba(input_scaled)[0][1]
            
            # Calculate churn percentage
            churn_percentage = churn_prob * 100
            
            # Display prediction result
            st.subheader("🎯 Prediction Result")
            
            # Determine risk level and styling
            if churn_prob > 0.7:
                risk_level = "HIGH"
                risk_class = "high-risk"
                risk_color = "#DC2626"
            elif churn_prob > 0.4:
                risk_level = "MEDIUM"
                risk_class = "medium-risk"
                risk_color = "#D97706"
            else:
                risk_level = "LOW"
                risk_class = "low-risk"
                risk_color = "#059669"
            
            # Create prediction box
            st.markdown(f'<div class="prediction-box {risk_class}">', unsafe_allow_html=True)
            
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.metric("Churn Probability", f"{churn_percentage:.1f}%")
                st.progress(churn_prob)
            
            with col_b:
                st.metric("Risk Level", risk_level)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Visual gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_percentage,
                title={'text': "Churn Risk Meter"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': churn_percentage
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance visualization (only for Random Forest)
            if model_choice == "Random Forest" and feature_importances is not None:
                st.subheader("🔍 Key Factors Influencing Prediction")
                
                top_features = feature_importances.head(10)
                fig2 = px.bar(top_features, 
                             x='Importance', 
                             y='Feature',
                             orientation='h',
                             color='Importance',
                             color_continuous_scale='Blues')
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Retention Recommendations")
            recommendations = []
            
            if churn_prob > 0.3:
                if contract == "Month-to-month":
                    recommendations.append("📝 Offer a discounted annual contract to increase commitment")
                
                if tenure < 12:
                    recommendations.append("🎁 Provide a 'welcome loyalty' bonus after 12 months")
                
                if internet_service == "Fiber optic" and monthly_charges > 80:
                    recommendations.append("💰 Consider promotional pricing or bundle discounts")
                
                if payment_method == "Electronic check":
                    recommendations.append("💳 Suggest automatic payment setup with a $5 monthly credit")
                
                if service_count < 2:
                    recommendations.append("🔧 Offer a free trial of additional services (Security, Backup, etc.)")
                
                if senior_citizen == 1:
                    recommendations.append("👵 Consider senior discount or simplified billing")
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")
            else:
                st.info("Customer appears satisfied. Maintain current service quality and check in periodically.")
    
    with col2:
        # Customer summary
        st.subheader("👤 Customer Summary")
        
        summary_data = {
            "Category": ["Tenure", "Contract", "Monthly Charges", "Services Used"],
            "Value": [f"{tenure} months", contract, f"${monthly_charges:.2f}", f"{service_count} add-ons"]
        }
        st.table(pd.DataFrame(summary_data))
        
        # Industry comparison
        st.subheader("📊 Industry Comparison")
        col_c, col_d = st.columns(2)
        with col_c:
            st.metric("Avg Industry Churn", "25%")
        with col_d:
            diff = churn_percentage - 25 if predict_button else 0
            st.metric("Customer vs Avg", f"{churn_percentage:.1f}%" if predict_button else "N/A", 
                     f"{diff:+.1f}%" if predict_button else "")
        
        # Quick insights
        st.subheader("💡 Quick Insights")
        
        insights = []
        if tenure < 6:
            insights.append("⚠️ New customer (higher risk)")
        if contract == "Month-to-month":
            insights.append("⚠️ Month-to-month contract")
        if payment_method == "Electronic check":
            insights.append("⚠️ Manual payment method")
        if internet_service == "Fiber optic":
            insights.append("✅ Premium internet service")
        if service_count >= 3:
            insights.append("✅ Multiple services (more engaged)")
        
        if insights:
            for insight in insights:
                if insight.startswith("⚠️"):
                    st.warning(insight[2:])
                else:
                    st.success(insight[2:])
        
        # Model info
        with st.expander("ℹ️ Model Information"):
            st.write(f"**Selected Model:** {model_choice}")
            st.write(f"**Features Used:** {len(selected_features)}")
            st.write(f"**Selected Features:**")
            for feature in selected_features:
                st.write(f"- {feature}")
            
            if model_choice == "Logistic Regression":
                st.write("**Note:** Logistic Regression requires feature scaling")
            else:
                st.write("**Note:** Random Forest handles features without scaling")
        
        # Download prediction
        if predict_button:
            st.download_button(
                label="📥 Download Prediction Report",
                data=f"""
Customer Churn Prediction Report
================================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Used: {model_choice}
Churn Probability: {churn_percentage:.1f}%
Risk Level: {risk_level}

Customer Details:
- Tenure: {tenure} months
- Contract: {contract}
- Monthly Charges: ${monthly_charges:.2f}
- Total Services: {service_count}

Top Risk Factors:
1. {recommendations[0] if recommendations else 'No specific risk factors identified'}
2. {recommendations[1] if len(recommendations) > 1 else ''}
3. {recommendations[2] if len(recommendations) > 2 else ''}
                """,
                file_name=f"churn_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
else:
    # Show error message if models didn't load
    st.error("""
    ⚠️ Models could not be loaded. Please ensure:
    1. The model files exist in the `models/` directory
    2. Files are named correctly:
       - `logistic_regression_model.joblib`
       - `random_forest_model.joblib`
       - `scaler.joblib`
       - `selected_features.pkl`
    3. You have the required permissions to read the files
    """)
    
    # Debug information
    with st.expander("Debug Information"):
        st.write("Current directory:", Path.cwd())
        st.write("Models directory:", Path('models').resolve())
        try:
            files = list(Path('models').glob('*'))
            st.write("Files in models directory:", [f.name for f in files])
        except:
            st.write("Could not list files in models directory")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📊 <em>This tool uses machine learning to predict customer churn based on historical data.</em></p>
    <p><small>For educational purposes. Predictions are estimates, not guarantees.</small></p>
</div>
""", unsafe_allow_html=True)