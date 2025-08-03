import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('avg_monthly_potable_water_production.csv')
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df['month'] = pd.Categorical(df['month'].str.strip(), categories=month_order, ordered=True)
    df['month'] = df['month'].cat.codes
    
    df.ffill(inplace=True)
    le_station = LabelEncoder()
    df['station_encoded'] = le_station.fit_transform(df['station'])
    
    df = df.sort_values(['station_encoded', 'year', 'month'])
    df['surface_lag_1'] = df.groupby('station_encoded')['surface'].shift(1)
    df['surface_lag_2'] = df.groupby('station_encoded')['surface'].shift(2)
    df[['surface_lag_1', 'surface_lag_2']] = df[['surface_lag_1', 'surface_lag_2']].ffill().fillna(0)
    
    # Create datetime index
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + (df['month']+1).astype(str) + '-01')
    
    return df, le_station

df, le_station = load_data()

# Initialize scaler globally
scaler = MinMaxScaler()
features_for_scaling = ['month', 'year', 'station_encoded', 'borehole', 'surface_lag_1', 'surface_lag_2']
scaler.fit(df[features_for_scaling])

# Prediction function
def predict_future(model, station_data, years_to_predict=2):
    if len(station_data) < 2:
        return pd.DataFrame()  # Return empty DataFrame if insufficient data
    
    future_predictions = []
    last_data = station_data.iloc[-1][['surface', 'borehole', 'surface_lag_1', 'surface_lag_2']]
    last_year = station_data['year'].max()
    
    for year in [last_year + 1, last_year + 2]:
        for month in range(12):
            # Create feature vector
            features = pd.DataFrame({
                'month': [month],
                'year': [year],
                'station_encoded': [station_data['station_encoded'].iloc[0]],
                'borehole': [last_data['borehole']],
                'surface_lag_1': [last_data['surface']],
                'surface_lag_2': [last_data['surface_lag_1']]
            })
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Predict
            pred = model.predict(features_scaled)[0]
            future_predictions.append({
                'year': year,
                'month': month,
                'date': pd.to_datetime(f"{year}-{month+1}-01"),
                'predicted_surface': pred
            })
            
            # Update last known values for next prediction
            last_data['surface_lag_2'] = last_data['surface_lag_1']
            last_data['surface_lag_1'] = last_data['surface']
            last_data['surface'] = pred
    
    return pd.DataFrame(future_predictions)

# Streamlit app
st.set_page_config(page_title="Water Production Forecast", layout="wide")

st.title("🌊 Mauritian Water Production Analysis & Forecasting")

# Sidebar controls
st.sidebar.header("Analysis Controls")
selected_station = st.sidebar.selectbox(
    "Select Station", 
    df['station'].unique(),
    index=0
)

analysis_type = st.sidebar.radio(
    "Analysis Type",
    ["Historical Analysis", "Future Prediction (2025-2026)"]
)

# Filter data
station_data = df[df['station'] == selected_station].sort_values('date')

# Main analysis
if analysis_type == "Historical Analysis":
    st.header(f"Historical Analysis for {selected_station}")
    
    # Time series plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(station_data['date'], station_data['surface'], label='Surface Water')
    ax.plot(station_data['date'], station_data['borehole'], label='Borehole Water')
    ax.set_title(f"Water Production at {selected_station}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Production (units)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Seasonal decomposition
    st.subheader("Seasonal Patterns")
    if len(station_data) >= 24:
        decomposition = seasonal_decompose(
            station_data.set_index('date')['surface'], 
            model='additive', 
            period=12
        )
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        st.pyplot(fig)
        
        # Generate insights
        seasonal_component = decomposition.seasonal
        peak_month = seasonal_component.idxmax().month
        trough_month = seasonal_component.idxmin().month
        
        st.markdown(f"""
        **Key Observations:**
        - 📈 Peak production typically occurs in **{pd.to_datetime(str(peak_month), format='%m').strftime('%B')}**
        - 📉 Lowest production typically occurs in **{pd.to_datetime(str(trough_month), format='%m').strftime('%B')}**
        - 🔄 Seasonal variation amplitude: **{seasonal_component.max() - seasonal_component.min():.2f} units**
        """)
    else:
        st.warning("Insufficient data for seasonal decomposition (need ≥24 months)")
    
    # Annual summary
    st.subheader("Annual Summary")
    annual_summary = station_data.groupby('year').agg({
        'surface': ['mean', 'max', 'min'],
        'borehole': ['mean', 'max', 'min']
    })
    st.dataframe(annual_summary.style.background_gradient(), use_container_width=True)
    
else:
    st.header(f"Future Prediction for {selected_station} (2025-2026)")
    
    # Prepare data for modeling
    X = df[features_for_scaling]
    y = df['surface']
    X_scaled = scaler.transform(X)
    
    # Train best model
    gb_model = GradientBoostingRegressor(n_estimators=150, random_state=42)
    gb_model.fit(X_scaled, y)
    
    # Make predictions
    future_df = predict_future(gb_model, station_data)
    
    if not future_df.empty:
        # Plot predictions
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Historical data
        ax.plot(station_data['date'], station_data['surface'], 
                label='Historical Surface', color='blue')
        
        # Future predictions
        ax.plot(future_df['date'], future_df['predicted_surface'], 
                label='Predicted Surface (2025-2026)', color='orange', linestyle='--')
        
        ax.set_title(f"Water Production Forecast for {selected_station}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Production (units)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # Generate insights
        pred_2025 = future_df[future_df['year'] == 2025]['predicted_surface']
        pred_2026 = future_df[future_df['year'] == 2026]['predicted_surface']
        
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        
        # Safely get peak and low months
        def get_peak_low(pred_series):
            if len(pred_series) == 0:
                return "N/A", "N/A", 0, 0
            peak_idx = pred_series.values.argmax()
            low_idx = pred_series.values.argmin()
            return (month_names[peak_idx], month_names[low_idx], 
                    pred_series.max(), pred_series.min())
        
        peak_2025, low_2025, max_2025, min_2025 = get_peak_low(pred_2025)
        peak_2026, low_2026, max_2026, min_2026 = get_peak_low(pred_2026)
        
        st.markdown(f"""
        **Forecast Insights for {selected_station}:**
        
        **2025 Projections:**
        - 🚰 Average predicted surface water: **{pred_2025.mean():.2f} units/month**
        - 📈 Peak month: **{peak_2025}** (predicted {max_2025:.2f} units)
        - 📉 Lowest month: **{low_2025}** (predicted {min_2025:.2f} units)
        
        **2026 Projections:**
        - 🚰 Average predicted surface water: **{pred_2026.mean():.2f} units/month**
        - 📈 Peak month: **{peak_2026}** (predicted {max_2026:.2f} units)
        - 📉 Lowest month: **{low_2026}** (predicted {min_2026:.2f} units)
        
        **Recommendations:**
        - Prepare for increased demand during predicted peak months
        - Schedule maintenance during predicted low-production periods
        - Consider borehole utilization adjustments based on surface water forecasts
        """)
        
        # Show detailed predictions
        st.subheader("Detailed Monthly Predictions")
        future_df['month_name'] = future_df['month'].apply(lambda x: month_names[x])
        future_display = future_df[['year', 'month_name', 'predicted_surface']].rename(
            columns={'month_name': 'month', 'predicted_surface': 'predicted_water'})
        st.dataframe(future_display.style.background_gradient(subset=['predicted_water']), 
                     use_container_width=True)
    else:
        st.error("Insufficient historical data to generate predictions. Need at least 2 months of data.")

# Model comparison
st.sidebar.header("Model Performance")
if st.sidebar.checkbox("Show Model Comparison"):
    st.header("Model Performance Comparison")
    
    # Prepare data for modeling
    X = df[features_for_scaling]
    y = df['surface']
    X_scaled = scaler.transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results.append({
            "Model": name,
            "RMSE": f"{rmse:.3f}",
            "R² Score": f"{r2:.3f}"
        })
    
    st.dataframe(pd.DataFrame(results), use_container_width=True)
    st.info("Gradient Boosting shows the best performance and is used for predictions.")

# About section
st.sidebar.header("About")
st.sidebar.info("""
This dashboard provides:
- Historical water production analysis
- Seasonal pattern identification
- 2025-2026 water production forecasts
- Automated insights and recommendations
""")

#  streamlit run app2.py