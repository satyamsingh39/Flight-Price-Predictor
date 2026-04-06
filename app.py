import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Flight Price Predictor Pro", page_icon="✈️", layout="wide")

# --- MONTH MAPPING ---
MONTH_OPTIONS = ["January", "February", "March", "April", "May", "June", 
                 "July", "August", "September", "October", "November", "December"]
MONTH_TO_NUM = {name: i+1 for i, name in enumerate(MONTH_OPTIONS)}
NUM_TO_MONTH = {i+1: name for i, name in enumerate(MONTH_OPTIONS)}

# --- PREMIUM THEME-AWARE CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Market Insights Visibility */
    .market-insights-header {
        color: #f8fafc !important;
        font-weight: 800;
        margin-top: 40px;
        margin-bottom: 20px;
        font-size: 2.2rem;
    }

    /* Container Styling (Avoiding the 'white box' bug) */
    /* We style the built-in Streamlit containers instead of using partial HTML divs */
    [data-testid="stVerticalBlock"] > div:has(div.stSubheader) {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 30px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    /* Heading Colors for Visibility */
    h1, h2, h3 {
        color: #f8fafc !important;
    }
    
    .app-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 40px;
    }

    /* Predict Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
        font-weight: 700;
        border-radius: 12px;
        height: 60px;
        border: none;
        transition: all 0.3s ease;
        margin-top: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
    }

    /* Digital Result Card */
    .results-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 14px;
        padding: 35px;
        text-align: center;
        margin-top: 30px;
        box-shadow: 0 15px 30px rgba(16, 185, 129, 0.3);
    }
    
    /* Plotly Chart Container Fixes */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- ENCODING MAPPINGS ---
airline_map = {
    "Air Asia": 0, "Air India": 1, "GoAir": 2, "IndiGo": 3, "Jet Airways": 4,
    "Jet Airways Business": 5, "Multiple carriers": 6,
    "Multiple carriers Premium economy": 7, "SpiceJet": 8, "Trujet": 9, 
    "Vistara": 10, "Vistara Premium economy": 11
}

source_map = {
    "Banglore": 0, "Chennai": 1, "Delhi": 2, "Kolkata": 3, "Mumbai": 4
}

destination_map = {
    "Banglore": 0, "Cochin": 1, "Delhi": 2, "Hyderabad": 3, "Kolkata": 4, "New Delhi": 5
}

# --- HEADER SECTION ---
st.markdown('<h1 style="font-size:3.2rem; margin-bottom:0;">✈️ Flight Price Predictor Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Intelligent airline pricing insights & real-time estimates</p>', unsafe_allow_html=True)

# --- PREDICTION FORM SECTION ---
with st.container():
    st.subheader("🛠️ Travel Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        airline = st.selectbox("Airline Carrier", list(airline_map.keys()))
        source = st.selectbox("Source City", list(source_map.keys()))
        destination = st.selectbox("Destination City", list(destination_map.keys()))
    
    with col2:
        dep_hour = st.slider("Departure Hour", 0, 23, 12)
        arrival_hour = st.slider("Arrival Hour", 0, 23, 18)
        journey_month_name = st.selectbox("Travel Month", MONTH_OPTIONS, index=5)
    
    with col3:
        journey_day = st.slider("Day of Month", 1, 31, 15)
        total_stops = st.selectbox("Number of Stops", [0, 1, 2, 3, 4])
        duration = st.number_input("Duration (minutes)", min_value=30, max_value=1000, value=120)

    predict_btn = st.button("Calculate Predicted Price")

    if predict_btn:
        with st.spinner("Analyzing market data..."):
            try:
                features = np.array([[
                    airline_map[airline], source_map[source], destination_map[destination],
                    total_stops, journey_day, MONTH_TO_NUM[journey_month_name],
                    duration, dep_hour, arrival_hour
                ]])
                
                model = joblib.load("flight_price_model.pkl")
                log_pred = model.predict(features)
                prediction = np.exp(log_pred)[0]
                
                st.markdown(f"""
                <div class="results-card">
                    <p style="font-size: 1rem; text-transform: uppercase; letter-spacing: 2px; opacity: 0.9; margin-bottom: 15px;">Estimated Market Fare</p>
                    <p style="font-size: 3rem; font-weight: 800; margin: 0;">₹ {prediction:,.2f}</p>
                    <p style="font-size: 0.8rem; margin-top: 15px; opacity: 0.8;">Quantum Index Applied | Fuel Surcharge Included</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")

# --- VISUALIZATION SECTION ---
st.markdown('<h2 class="market-insights-header">📊 Market Insights</h2>', unsafe_allow_html=True)

try:
    # Load data
    df = pd.read_csv("cleaned_flight_price_data.csv")
    inv_airline_map = {v: k for k, v in airline_map.items()}
    plot_df = df.copy()
    plot_df['Airline_Name'] = plot_df['Airline'].map(inv_airline_map)
    plot_df['Month_Name'] = plot_df['Journey_Month'].map(NUM_TO_MONTH)
    
    vcol1, vcol2 = st.columns(2)
    
    with vcol1:
        airline_order = plot_df.groupby('Airline_Name')['Price'].median().sort_values().index
        fig1 = px.box(plot_df, x='Airline_Name', y='Price', 
                      title="Ticket Price Distribution by Airline",
                      category_orders={"Airline_Name": list(airline_order)},
                      color='Airline_Name', template="plotly_dark")
        fig1.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)
        
        fig3 = px.scatter(plot_df, x='Duration_mins', y='Price', color='Total_Stops',
                          title="Price vs. Duration Correlation",
                          labels={'Duration_mins': 'Duration (min)', 'Price': 'Price (₹)'},
                          template="plotly_dark", opacity=0.4)
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

    with vcol2:
        fig2 = px.box(plot_df, x='Total_Stops', y='Price', 
                      title="Impact of Layovers on Pricing",
                      color='Total_Stops', template="plotly_dark")
        fig2.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)
        
        avg_price_month = plot_df.groupby(['Journey_Month', 'Month_Name'])['Price'].mean().reset_index().sort_values('Journey_Month')
        fig4 = px.line(avg_price_month, x='Month_Name', y='Price', 
                       title="Seasonal Average Price (Monthly)",
                       markers=True, template="plotly_dark")
        fig4.update_traces(line_color='#3b82f6', line_width=4)
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width=True)

except Exception as e:
    st.error(f"Visualization Error: {e}")

st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 60px; padding-bottom: 40px;">
    Flight Predictor Pro | Aviation Intelligence Engine | 2026 Index
</div>
""", unsafe_allow_html=True)
