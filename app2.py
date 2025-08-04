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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Mauritian Water Production Forecasting",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #2d3436;
        border-left: 5px solid #fdcb6e;
    }
    
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #2d3436;
        border-left: 5px solid #00b894;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div > select {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
    }
    
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #2c3e50;
    }
    
    .recommendation-card strong {
        color: #1a252f;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    try:
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
    except FileNotFoundError:
        st.error("⚠️ Data file not found. Please ensure 'avg_monthly_potable_water_production.csv' is in the correct directory.")
        return None, None

# Header
st.markdown('<h1 class="main-header">🌊 Mauritian Water Production Forecasting System</h1>', unsafe_allow_html=True)

# Load data with error handling
data_load_state = st.text('🔄 Loading data...')
df, le_station = load_data()

if df is None:
    st.stop()

data_load_state.text('✅ Data loaded successfully!')

# Initialize scaler globally
scaler = MinMaxScaler()
features_for_scaling = ['month', 'year', 'station_encoded', 'borehole', 'surface_lag_1', 'surface_lag_2']
scaler.fit(df[features_for_scaling])

# Sidebar with enhanced styling
st.sidebar.markdown("## 🎛️ Dashboard Controls")
st.sidebar.markdown("---")

# Station selection with search functionality
station_list = sorted(df['station'].unique())
selected_station = st.sidebar.selectbox(
    "🏭 Select Water Treatment Station", 
    station_list,
    index=0,
    help="Choose a water treatment station to analyze"
)

st.sidebar.markdown("---")

# Analysis type selection
analysis_type = st.sidebar.radio(
    "📊 Analysis Mode",
    ["🔍 Historical Analysis", "🔮 Future Prediction (2025-2026)", "📈 Advanced Analytics"],
    help="Select the type of analysis you want to perform"
)

st.sidebar.markdown("---")

# Quick stats in sidebar
station_data = df[df['station'] == selected_station].sort_values('date')
if not station_data.empty:
    st.sidebar.markdown("### 📊 Quick Statistics")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("📅 Years", f"{station_data['year'].min()}-{station_data['year'].max()}")
        st.metric("🚰 Avg Surface", f"{station_data['surface'].mean():.1f}")
    
    with col2:
        st.metric("📊 Records", len(station_data))
        st.metric("🕳️ Avg Borehole", f"{station_data['borehole'].mean():.1f}")

# Prediction function
def predict_future(model, station_data, years_to_predict=2):
    if len(station_data) < 2:
        return pd.DataFrame()
    
    future_predictions = []
    last_data = station_data.iloc[-1][['surface', 'borehole', 'surface_lag_1', 'surface_lag_2']]
    last_year = station_data['year'].max()
    
    for year in [last_year + 1, last_year + 2]:
        for month in range(12):
            features = pd.DataFrame({
                'month': [month],
                'year': [year],
                'station_encoded': [station_data['station_encoded'].iloc[0]],
                'borehole': [last_data['borehole']],
                'surface_lag_1': [last_data['surface']],
                'surface_lag_2': [last_data['surface_lag_1']]
            })
            
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]
            future_predictions.append({
                'year': year,
                'month': month,
                'date': pd.to_datetime(f"{year}-{month+1}-01"),
                'predicted_surface': pred
            })
            
            last_data['surface_lag_2'] = last_data['surface_lag_1']
            last_data['surface_lag_1'] = last_data['surface']
            last_data['surface'] = pred
    
    return pd.DataFrame(future_predictions)

# Main content area
if analysis_type == "🔍 Historical Analysis":
    st.markdown("## 📈 Historical Water Production Analysis")
    st.markdown(f"### Analysis for **{selected_station}**")
    
    if station_data.empty:
        st.error("❌ No data available for the selected station.")
    else:
        # Create tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Production Trends", "🌀 Seasonal Patterns", "📋 Summary Statistics", "🎯 Key Insights"])
        
        with tab1:
            # Interactive time series plot using Plotly
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Surface Water Production', 'Borehole Water Production'),
                vertical_spacing=0.12
            )
            
            # Surface water
            fig.add_trace(
                go.Scatter(
                    x=station_data['date'],
                    y=station_data['surface'],
                    mode='lines+markers',
                    name='Surface Water',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=6),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Surface Water:</b> %{y:.2f} units<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Borehole water
            fig.add_trace(
                go.Scatter(
                    x=station_data['date'],
                    y=station_data['borehole'],
                    mode='lines+markers',
                    name='Borehole Water',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=6),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Borehole Water:</b> %{y:.2f} units<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=700,
                showlegend=True,
                title={
                    'text': f"Water Production Timeline - {selected_station}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=60, r=60, t=120, b=60),
                template="plotly_white"
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Production (units)", row=1, col=1)
            fig.update_yaxes(title_text="Production (units)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Production metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                surface_avg = station_data['surface'].mean()
                st.markdown(f'<div class="metric-container"><h3>{surface_avg:.1f}</h3><p>Avg Surface Production</p></div>', unsafe_allow_html=True)
            
            with col2:
                borehole_avg = station_data['borehole'].mean()
                st.markdown(f'<div class="metric-container"><h3>{borehole_avg:.1f}</h3><p>Avg Borehole Production</p></div>', unsafe_allow_html=True)
            
            with col3:
                total_records = len(station_data)
                st.markdown(f'<div class="metric-container"><h3>{total_records}</h3><p>Total Records</p></div>', unsafe_allow_html=True)
            
            with col4:
                years_span = station_data['year'].max() - station_data['year'].min() + 1
                st.markdown(f'<div class="metric-container"><h3>{years_span}</h3><p>Years of Data</p></div>', unsafe_allow_html=True)
        
        with tab2:
            # Seasonal decomposition
            if len(station_data) >= 24:
                decomposition = seasonal_decompose(
                    station_data.set_index('date')['surface'], 
                    model='additive', 
                    period=12
                )
                
                # Create subplots for decomposition
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
                    vertical_spacing=0.08
                )
                
                components = [
                    (decomposition.observed, 'Original', '#1f77b4'),
                    (decomposition.trend, 'Trend', '#ff7f0e'),
                    (decomposition.seasonal, 'Seasonal', '#2ca02c'),
                    (decomposition.resid, 'Residual', '#d62728')
                ]
                
                for i, (component, name, color) in enumerate(components, 1):
                    fig.add_trace(
                        go.Scatter(
                            x=component.index,
                            y=component.values,
                            mode='lines',
                            name=name,
                            line=dict(color=color, width=2),
                            showlegend=False
                        ),
                        row=i, col=1
                    )
                
                fig.update_layout(
                    height=900, 
                    title={
                        'text': "Seasonal Decomposition Analysis",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 18}
                    },
                    margin=dict(l=60, r=60, t=120, b=60),
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Seasonal insights
                seasonal_component = decomposition.seasonal
                peak_month = seasonal_component.idxmax().month
                trough_month = seasonal_component.idxmin().month
                
                month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December']
                
                st.markdown(f"""
                <div class="insight-box">
                    <h3>🔍 Seasonal Pattern Insights</h3>
                    <ul>
                        <li><b>Peak Production Month:</b> {month_names[peak_month-1]} 📈</li>
                        <li><b>Lowest Production Month:</b> {month_names[trough_month-1]} 📉</li>
                        <li><b>Seasonal Variation:</b> {seasonal_component.max() - seasonal_component.min():.2f} units</li>
                        <li><b>Trend Direction:</b> {'Increasing' if decomposition.trend.iloc[-12:].mean() > decomposition.trend.iloc[:12].mean() else 'Decreasing'} 📊</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠️ Insufficient Data</h4>
                    <p>Seasonal decomposition requires at least 24 months of data. Current dataset has {len(station_data)} records.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            # Annual summary with enhanced visualization
            annual_summary = station_data.groupby('year').agg({
                'surface': ['mean', 'max', 'min', 'std'],
                'borehole': ['mean', 'max', 'min', 'std']
            }).round(2)
            
            st.markdown("### 📊 Annual Production Summary")
            st.dataframe(
                annual_summary.style.background_gradient(cmap='Blues'),
                use_container_width=True
            )
            
            # Yearly comparison chart
            yearly_avg = station_data.groupby('year')[['surface', 'borehole']].mean().reset_index()
            
            fig = px.bar(
                yearly_avg.melt(id_vars='year', var_name='Source', value_name='Production'),
                x='year', y='Production', color='Source',
                title="Annual Average Production Comparison",
                color_discrete_map={'surface': '#1f77b4', 'borehole': '#ff7f0e'}
            )
            fig.update_layout(
                height=450,
                title={
                    'x': 0.5,
                    'xanchor': 'center'
                },
                margin=dict(l=50, r=50, t=80, b=50),
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Key insights and recommendations
            surface_trend = np.polyfit(range(len(station_data)), station_data['surface'], 1)[0]
            borehole_trend = np.polyfit(range(len(station_data)), station_data['borehole'], 1)[0]
            
            st.markdown("### 🎯 Key Performance Insights")
            
            insights = []
            
            if surface_trend > 0:
                insights.append("📈 Surface water production shows an increasing trend over time")
            else:
                insights.append("📉 Surface water production shows a decreasing trend over time")
            
            if borehole_trend > 0:
                insights.append("📈 Borehole water production shows an increasing trend over time")
            else:
                insights.append("📉 Borehole water production shows a decreasing trend over time")
            
            cv_surface = station_data['surface'].std() / station_data['surface'].mean()
            if cv_surface > 0.3:
                insights.append("⚠️ High variability in surface water production detected")
            else:
                insights.append("✅ Surface water production shows stable patterns")
            
            for insight in insights:
                st.markdown(f"""
                <div class="recommendation-card">
                    {insight}
                </div>
                """, unsafe_allow_html=True)

elif analysis_type == "🔮 Future Prediction (2025-2026)":
    st.markdown("## 🔮 Future Water Production Forecasting")
    st.markdown(f"### Predictions for **{selected_station}** (2025-2026)")
    
    if station_data.empty:
        st.error("❌ No data available for the selected station.")
    else:
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text('🔄 Training prediction model...')
        progress_bar.progress(25)
        
        # Prepare data for modeling
        X = df[features_for_scaling]
        y = df['surface']
        X_scaled = scaler.transform(X)
        
        # Train best model
        gb_model = GradientBoostingRegressor(n_estimators=150, random_state=42)
        gb_model.fit(X_scaled, y)
        
        progress_bar.progress(50)
        status_text.text('🎯 Generating predictions...')
        
        # Make predictions
        future_df = predict_future(gb_model, station_data)
        
        progress_bar.progress(75)
        
        if not future_df.empty:
            progress_bar.progress(100)
            status_text.text('✅ Predictions generated successfully!')
            
            # Create prediction visualization
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=station_data['date'],
                y=station_data['surface'],
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=6),
                hovertemplate='<b>Date:</b> %{x}<br><b>Production:</b> %{y:.2f} units<extra></extra>'
            ))
            
            # Future predictions
            fig.add_trace(go.Scatter(
                x=future_df['date'],
                y=future_df['predicted_surface'],
                mode='lines+markers',
                name='Predictions (2025-2026)',
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Predicted:</b> %{y:.2f} units<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': f"Water Production Forecast - {selected_station}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                xaxis_title="Date",
                yaxis_title="Production (units)",
                height=600,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=80, b=50),
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            
            # Prediction insights
            pred_2025 = future_df[future_df['year'] == 2025]['predicted_surface']
            pred_2026 = future_df[future_df['year'] == 2026]['predicted_surface']
            
            month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            
            def get_peak_low(pred_series):
                if len(pred_series) == 0:
                    return "N/A", "N/A", 0, 0
                peak_idx = pred_series.values.argmax()
                low_idx = pred_series.values.argmin()
                return (month_names[peak_idx], month_names[low_idx], 
                        pred_series.max(), pred_series.min())
            
            peak_2025, low_2025, max_2025, min_2025 = get_peak_low(pred_2025)
            peak_2026, low_2026, max_2026, min_2026 = get_peak_low(pred_2026)
            
            # Display predictions in cards
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>📅 2025 Forecast</h3>
                    <p><b>🚰 Average:</b> {pred_2025.mean():.2f} units/month</p>
                    <p><b>📈 Peak:</b> {peak_2025} ({max_2025:.2f} units)</p>
                    <p><b>📉 Low:</b> {low_2025} ({min_2025:.2f} units)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>📅 2026 Forecast</h3>
                    <p><b>🚰 Average:</b> {pred_2026.mean():.2f} units/month</p>
                    <p><b>📈 Peak:</b> {peak_2026} ({max_2026:.2f} units)</p>
                    <p><b>📉 Low:</b> {low_2026} ({min_2026:.2f} units)</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### 💡 Strategic Recommendations")
            
            # Calculate additional metrics for better recommendations
            historical_avg = station_data['surface'].mean()
            pred_2025_avg = pred_2025.mean() if len(pred_2025) > 0 else 0
            pred_2026_avg = pred_2026.mean() if len(pred_2026) > 0 else 0
            
            # Calculate percentage changes
            change_2025 = ((pred_2025_avg - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0
            change_2026 = ((pred_2026_avg - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0
            
            # Calculate seasonal variation
            if len(pred_2025) > 0 and len(pred_2026) > 0:
                seasonal_variation_2025 = (max_2025 - min_2025) / pred_2025_avg * 100 if pred_2025_avg > 0 else 0
                seasonal_variation_2026 = (max_2026 - min_2026) / pred_2026_avg * 100 if pred_2026_avg > 0 else 0
                
                recommendations = [
                    f"🔧 **Maintenance Scheduling**: Plan major maintenance during {low_2025} 2025 and {low_2026} 2026 (predicted low production periods with {min_2025:.1f} and {min_2026:.1f} units respectively)",
                    
                    f"⚡ **Capacity Planning**: Prepare for peak demand in {peak_2025} 2025 ({max_2025:.1f} units) and {peak_2026} 2026 ({max_2026:.1f} units) - approximately {((max_2025 + max_2026)/2 - historical_avg)/historical_avg*100:.1f}% above historical average",
                    
                    f"📈 **Production Trend**: {'Expected increase' if change_2025 > 0 else 'Expected decrease'} of {abs(change_2025):.1f}% in 2025 and {abs(change_2026):.1f}% in 2026 compared to historical average ({historical_avg:.1f} units)",
                    
                    f"🔄 **Seasonal Management**: High seasonal variation expected ({seasonal_variation_2025:.1f}% in 2025, {seasonal_variation_2026:.1f}% in 2026) - consider flexible borehole utilization strategies",
                    
                    f"📊 **Resource Allocation**: During predicted low months ({low_2025}, {low_2026}), increase borehole water production to maintain overall supply stability",
                    
                    f"⚠️ **Risk Management**: Monitor actual vs predicted values closely, especially during peak months when demand is highest",
                    
                    f"🎯 **Infrastructure Planning**: Consider infrastructure upgrades if predicted peak production ({max((max_2025 + max_2026)/2, historical_avg):.1f} units) exceeds current capacity limits"
                ]
            else:
                recommendations = [
                    "🔧 **Data Collection**: Insufficient prediction data available. Focus on improving data collection processes",
                    "📊 **Monitoring**: Implement comprehensive monitoring systems to gather more historical data",
                    "⚡ **Baseline Establishment**: Work on establishing production baselines for better future predictions",
                    "🔄 **System Optimization**: Focus on optimizing current production processes while building historical dataset"
                ]
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div class="recommendation-card">
                    <strong>Recommendation {i}:</strong><br>
                    {rec}
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed predictions table
            st.markdown("### 📊 Detailed Monthly Predictions")
            future_df['month_name'] = future_df['month'].apply(lambda x: month_names[x])
            future_display = future_df[['year', 'month_name', 'predicted_surface']].rename(
                columns={'month_name': 'Month', 'year': 'Year', 'predicted_surface': 'Predicted Production'})
            
            st.dataframe(
                future_display.style.background_gradient(subset=['Predicted Production'], cmap='Blues'),
                use_container_width=True
            )
            
        else:
            st.error("❌ Insufficient historical data to generate predictions. Need at least 2 months of data.")

elif analysis_type == "📈 Advanced Analytics":
    st.markdown("## 📈 Advanced Analytics Dashboard")
    
    # Model comparison section
    st.markdown("### 🤖 Model Performance Comparison")
    
    with st.spinner('Training and comparing models...'):
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
                "RMSE": rmse,
                "R² Score": r2,
                "Performance": "🥇" if r2 == max([r['R² Score'] for r in results] + [r2]) else "🥈" if len(results) > 1 else "🥉"
            })
        
        results_df = pd.DataFrame(results)
        
        # Display results with styling
        st.dataframe(
            results_df.style.background_gradient(subset=['RMSE', 'R² Score'], cmap='RdYlGn_r'),
            use_container_width=True
        )
        
        # Model performance visualization
        fig = px.bar(
            results_df,
            x='Model',
            y='R² Score',
            title='Model Performance Comparison (R² Score)',
            color='R² Score',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            height=400,
            title={
                'x': 0.5,
                'xanchor': 'center'
            },
            margin=dict(l=50, r=50, t=60, b=50),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (for tree-based models)
    st.markdown("### 🎯 Feature Importance Analysis")
    
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_scaled, y)
    
    feature_importance = pd.DataFrame({
        'Feature': features_for_scaling,
        'Importance': gb_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance in Water Production Prediction',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        height=400,
        title={
            'x': 0.5,
            'xanchor': 'center'
        },
        margin=dict(l=100, r=50, t=60, b=50),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### 🔗 Feature Correlation Analysis")
    
    correlation_matrix = df[['surface', 'borehole', 'month', 'year']].corr()
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Heatmap",
        color_continuous_scale='RdBu'
    )
    fig.update_layout(
        title={
            'x': 0.5,
            'xanchor': 'center'
        },
        margin=dict(l=50, r=50, t=60, b=50),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 15px; color: white; margin-top: 2rem;">
    <h3>🌊 Mauritian Water Production Forecasting System</h3>
    <p>Advanced AI-powered analytics for sustainable water resource management</p>
    <p><i>Developed with ❤️ for efficient water production planning</i></p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📋 Features
- 🔍 **Historical Analysis**: Comprehensive time series analysis
- 🔮 **AI Forecasting**: Machine learning predictions for 2025-2026
- 📊 **Advanced Analytics**: Model comparison and feature analysis
- 💡 **Smart Insights**: Automated recommendations and alerts
- 📱 **Interactive Charts**: Modern, responsive visualizations

### 📊 Data Quality
- ✅ Automated data preprocessing
- 🔧 Missing value handling
- 📈 Trend analysis
- 🌀 Seasonal decomposition
""")

st.sidebar.success("Dashboard is running optimally! ✨")


#  streamlit run claude.py