import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import logging
import datetime
import os
from joblib import load
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page configuration FIRST, before any other Streamlit commands
st.set_page_config(
    page_title="Agri Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return load('best_model.joblib')
    except Exception as e:
        return None

model = load_model()
if model is None:
    st.error("Error loading model. Please ensure best_model.joblib exists in the current directory.")

# Load logo
current_dir = os.path.dirname(__file__)
logo_path = os.path.join(current_dir, "Agriculture.jpeg")
try:
    st.sidebar.image(logo_path, width=400)
except Exception as e:
    logging.error(f"Logo loading error: {e}")


# App title and description
st.title("🌾 AI-ML Based Commodity Price Prediction System")
st.markdown("""
    This application uses Random Forest machine learning to predict agricultural commodity prices.
    Built for businesses and government bodies to make data-driven decisions.
""")

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 5px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar instructions
st.sidebar.header("📋 Instructions")
st.sidebar.markdown("""
1. Select location and commodity details
2. Choose prediction date(s)
3. View predictions and analytics
4. Export results as needed
""")

# Load and cache data
@st.cache_data
def load_data(file_path="Price_Agriculture_commodities_Week.csv"):
    try:
        data = pd.read_csv(file_path)
        data['Arrival_Date'] = pd.to_datetime(data['Arrival_Date'], format="%d-%m-%Y")
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
commodity_data = load_data()

if commodity_data.empty:
    st.error("No data available. Please check the dataset.")
    st.stop()

# Feature engineering function
def create_features(df, date):
    """Create features for the model from date"""
    return pd.DataFrame({
        'year': [date.year],
        'month': [date.month],
        'day': [date.day],
        'dayofweek': [date.dayofweek],
        'quarter': [date.quarter]
    })

# Prediction function
def predict_price(date, commodity, market, district, state):
    """Predict price using the random forest model"""
    if model is None:
        return None, None
    
    try:
        # Create features
        features = create_features(commodity_data, date)
        
        # Add categorical features
        features['Commodity'] = commodity
        features['Market'] = market
        features['District'] = district
        features['State'] = state
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Calculate confidence using model's built-in feature
        predictions = []
        for estimator in model.estimators_:
            predictions.append(estimator.predict(features)[0])
        confidence = np.std(predictions) * 1.96  # 95% confidence interval
        
        return prediction, confidence
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None, None

# UI Components
st.subheader("🔍 Select Details")
col1, col2 = st.columns(2)

with col1:
    # Location selectors
    states = sorted(commodity_data['State'].unique())
    selected_state = st.selectbox("Select State", states)
    
    filtered_districts = sorted(commodity_data[commodity_data['State'] == selected_state]['District'].unique())
    selected_district = st.selectbox("Select District", filtered_districts)
    
    filtered_markets = sorted(commodity_data[
        (commodity_data['State'] == selected_state) &
        (commodity_data['District'] == selected_district)
    ]['Market'].unique())
    selected_market = st.selectbox("Select Market", filtered_markets)

with col2:
    # Commodity and date selectors
    filtered_commodities = sorted(commodity_data[
        (commodity_data['State'] == selected_state) &
        (commodity_data['District'] == selected_district) &
        (commodity_data['Market'] == selected_market)
    ]['Commodity'].unique())
    selected_commodity = st.selectbox("Select Commodity", filtered_commodities)
    
    selected_date = st.date_input("Select Prediction Date")

# Make prediction
if st.button("Predict Price"):
    price, confidence = predict_price(
        pd.to_datetime(selected_date),
        selected_commodity,
        selected_market,
        selected_district,
        selected_state
    )
    
    if price is not None and confidence is not None:
        st.success(f"Predicted Price: ₹{price:.2f} ± ₹{confidence:.2f}")
        
        # Show historical context
        historical_data = commodity_data[
            (commodity_data['State'] == selected_state) &
            (commodity_data['District'] == selected_district) &
            (commodity_data['Market'] == selected_market) &
            (commodity_data['Commodity'] == selected_commodity)
        ].copy()
        
        if not historical_data.empty:
            st.subheader("📊 Historical Context")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_price = historical_data['Modal Price'].mean()
                st.metric("Average Historical Price", f"₹{avg_price:.2f}")
            
            with col2:
                max_price = historical_data['Modal Price'].max()
                st.metric("Maximum Historical Price", f"₹{max_price:.2f}")
            
            with col3:
                volatility = historical_data['Modal Price'].std()
                st.metric("Price Volatility", f"₹{volatility:.2f}")

# Date range prediction
st.subheader("📅 Date Range Prediction")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", key="start_date")
with col2:
    end_date = st.date_input("End Date", key="end_date")

if st.button("Predict Range"):
    if start_date <= end_date:
        dates = pd.date_range(start_date, end_date)
        predictions = []
        
        for date in dates:
            price, confidence = predict_price(
                date,
                selected_commodity,
                selected_market,
                selected_district,
                selected_state
            )
            if price is not None and confidence is not None:
                predictions.append({
                    'Date': date,
                    'Predicted_Price': price,
                    'Confidence': confidence
                })
        
        if predictions:
            df_predictions = pd.DataFrame(predictions)
            
            # Plot predictions
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_predictions['Date'], df_predictions['Predicted_Price'], 'b-')
            ax.fill_between(
                df_predictions['Date'],
                df_predictions['Predicted_Price'] - df_predictions['Confidence'],
                df_predictions['Predicted_Price'] + df_predictions['Confidence'],
                alpha=0.2
            )
            plt.xticks(rotation=45)
            plt.title(f"Price Predictions: {selected_commodity}")
            plt.xlabel("Date")
            plt.ylabel("Predicted Price (₹)")
            st.pyplot(fig)
            
            # Show predictions table
            st.dataframe(df_predictions.round(2))
            
            # Export option
            csv = df_predictions.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name='price_predictions.csv',
                mime='text/csv'
            )
        else:
            st.error("Could not generate predictions for the selected date range")
    else:
        st.error("End date must be after start date")

# Add model performance metrics if available
if model is not None:
    st.subheader("🎯 Model Performance")
    
    try:
        # Get R² score if available
        if hasattr(model, 'score'):
            r2 = model.score
            st.metric("R² Score", f"{r2:.2f}")
        
        # Show feature importance if available
        if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
            feature_imp = pd.DataFrame({
                'Feature': model.feature_names_in_,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_imp.head(10), x='Importance', y='Feature')
            plt.title("Top 10 Important Features")
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        logging.error(f"Error displaying model metrics: {e}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit and Random Forest</p>
    </div>
""", unsafe_allow_html=True)