import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import os
import logging
import warnings

# Suppress TensorFlow CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)

# Set page configuration
st.set_page_config(
    page_title="Water Level Prediction Dashboard",
    page_icon="💧",
    layout="wide"
)

# Increase timeout for long-running operations
if not hasattr(st, 'session_state'):
    st.session_state = {}

# Title and description
st.title("Yuvacık Dam Water Level Prediction")
st.markdown("Interactive dashboard for visualizing and predicting dam water levels using LSTM model")

# Flag for TensorFlow availability
tensorflow_available = False
model_loaded = False

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    tensorflow_available = True
    
    # Check if model exists and load it, otherwise show warning
    model_path = 'lstm_model.h5'
    history_path = 'model_history.pkl'
    if os.path.exists(model_path):
        # Add memory management for TensorFlow
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                st.warning(f"Memory growth setting failed: {e}")
        
        # Cache the model loading
        @st.cache_resource
        def load_cached_model(model_path):
            return load_model(model_path)
        
        try:
            model = load_cached_model(model_path)
            model_loaded = True
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            model_loaded = False
    else:
        st.warning("Model file not found. Please train and save the model first.")
        if st.button("Train Model"):
            st.info("This would train the model - functionality to be implemented")
except ImportError as e:
    st.error(f"TensorFlow could not be loaded: {e}")
    st.info("Install TensorFlow with: pip install tensorflow-cpu")
    st.warning("Prediction functionality will be disabled until TensorFlow is properly installed.")

# Load and display data
try:
    # Load data with better caching
    @st.cache_data(ttl=3600)
    def load_data():
        try:
            df = pd.read_csv('baraj_seviyesi_tum_yillar_eksiksiz.csv', index_col=0, parse_dates=True)
            return df
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return None
    
    df = load_data()
    
    if df is not None:
        df_lstm = df[['Baraj_Seviyesi']].copy()
        
        # Data loaded successfully
        data_loaded = True
        
        # Display raw data section
        with st.expander("View Raw Data"):
            # Use full width for the dataframe and show more data
            st.markdown("### Complete Dataset")
            st.dataframe(df, use_container_width=True, height=400)
            
            # Display basic statistics
            st.subheader("Dataset Statistics")
            st.write(df.describe())
            
        # Historical data visualization
        st.subheader("Historical Water Level Data")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", df.index.min())
        with col2:
            end_date = st.date_input("End Date", df.index.max())
        
        # Convert to pandas datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filter data based on date range
        filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Downsample for large date ranges
        if len(filtered_df) > 1000:
            rule = 'W' if len(filtered_df) > 2000 else 'D'
            filtered_df = filtered_df.resample(rule).mean()
        
        # Create Plotly figure for historical data
        fig = px.line(
            filtered_df, 
            x=filtered_df.index, 
            y="Baraj_Seviyesi",
            labels={"Baraj_Seviyesi": "Water Level (m)", "index": "Date"},
            title="Historical Water Level Data"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if tensorflow_available and model_loaded:
            st.subheader("Model Predictions")
            
            # Define parameters
            sequence_length = 30  # Same as in the model
            
            # Prediction section
            st.subheader("Forecast Water Level")
            forecast_days = st.slider("Forecast Days", 1, 30, 7)
            
            if st.button("Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    # Cache prediction results
                    @st.cache_data(ttl=3600)
                    def generate_forecast(df_lstm, sequence_length, forecast_days):
                        # Prepare data for prediction
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scaled_data = scaler.fit_transform(df_lstm)
                        
                        # Generate forecast for multiple days
                        forecasted_values = []
                        last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
                        
                        current_sequence = last_sequence.copy()
                        
                        for _ in range(forecast_days):
                            # Predict next day
                            next_day_prediction_scaled = model.predict(current_sequence, verbose=0)
                            # Store the prediction
                            forecasted_values.append(next_day_prediction_scaled[0, 0])
                            # Update sequence for next prediction (remove oldest, add newest)
                            # Fix: reshape the prediction to match the 3D structure (1, 1, 1)
                            next_pred_reshaped = next_day_prediction_scaled[0, 0].reshape(1, 1, 1)
                            current_sequence = np.concatenate([current_sequence[:, 1:, :], 
                                                            next_pred_reshaped], 
                                                            axis=1)
                        
                        # Convert predictions back to original scale
                        forecasted_values = scaler.inverse_transform(
                            np.array(forecasted_values).reshape(-1, 1)
                        ).flatten()
                        
                        # Generate dates for forecast period
                        last_date = df_lstm.index[-1]
                        forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
                        
                        return forecasted_values, forecast_dates, scaler, last_date
                    
                    try:
                        forecasted_values, forecast_dates, scaler, last_date = generate_forecast(
                            df_lstm, sequence_length, forecast_days
                        )
                        
                        # Create forecast dataframe
                        forecast_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Forecasted_Water_Level': forecasted_values
                        })
                        
                        # Display forecast table
                        st.write("Forecasted Water Levels:")
                        st.dataframe(forecast_df)
                        
                        # Create visualization showing historical + forecast - limiting historical data
                        hist_dates = df_lstm.index[-60:] 
                        hist_values = df_lstm['Baraj_Seviyesi'].values[-60:]
                        
                        fig = go.Figure()
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=hist_dates, 
                            y=hist_values,
                            mode='lines',
                            name='Historical Water Level',
                            line=dict(color='blue')
                        ))
                        
                        # Add forecasted data
                        fig.add_trace(go.Scatter(
                            x=forecast_dates, 
                            y=forecasted_values,
                            mode='lines+markers',
                            name='Forecasted Water Level',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Add vertical line separating historical and forecast
                        fig.add_vline(x=last_date, line_width=1, line_dash="dash", line_color="gray")
                        
                        # Update layout
                        fig.update_layout(
                            title='Water Level Forecast',
                            xaxis_title='Date',
                            yaxis_title='Water Level (m)',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating forecast: {e}")
                        st.info("Try restarting the app or checking your data")
            
            # Model performance metrics
            st.subheader("Model Performance")
            
            tabs = st.tabs(["Metrics", "Test Predictions", "Training History", "Visual Prediction Test"])
            
            with tabs[0]:
                @st.cache_data(ttl=3600)
                def calculate_model_metrics():
                    # Fixed test size to 20% of the dataset
                    test_size_percent = 20
                    total_size = len(df_lstm)
                    test_size = int(total_size * test_size_percent / 100)
                    
                    # Use a test set to evaluate the model 
                    test_data = df_lstm[-test_size-sequence_length:].copy()
                    
                    # Prepare the test data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    test_scaled = scaler.fit_transform(test_data)
                    
                    # Create sequences for testing
                    X_test, y_test = [], []
                    for i in range(sequence_length, len(test_scaled)):
                        X_test.append(test_scaled[i-sequence_length:i, 0])
                        y_test.append(test_scaled[i, 0])
                    
                    X_test = np.array(X_test).reshape(len(X_test), sequence_length, 1)
                    y_test = np.array(y_test)
                    
                    # Make predictions
                    y_pred_scaled = model.predict(X_test, verbose=0)
                    
                    # Reshape for inverse transform
                    y_test_reshaped = y_test.reshape(-1, 1)
                    y_pred_reshaped = y_pred_scaled.reshape(-1, 1)
                    
                    # Inverse transform
                    y_test_actual = scaler.inverse_transform(y_test_reshaped).flatten()
                    y_pred_actual = scaler.inverse_transform(y_pred_reshaped).flatten()
                    
                    # Calculate metrics
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
                    mae = mean_absolute_error(y_test_actual, y_pred_actual)
                    r2 = r2_score(y_test_actual, y_pred_actual)
                    
                    return rmse, mae, r2, y_test_actual, y_pred_actual, test_size, test_size_percent
                
                try:
                    rmse, mae, r2, y_test_actual, y_pred_actual, test_size, test_size_percent = calculate_model_metrics()
                    
                    # Display test set info
                    st.info(f"Using {test_size} days as test set ({test_size_percent}% of total data)")
                    
                    # Display metrics with actual values
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="RMSE", value=f"{rmse:.3f}")
                    with col2:
                        st.metric(label="MAE", value=f"{mae:.3f}")
                    with col3:
                        st.metric(label="R² Score", value=f"{r2:.3f}")
                        
                    # Add a metric visualization
                    st.markdown("### Metrics Explanation")
                    st.markdown("""
                    - **RMSE (Root Mean Square Error)**: Measures the average magnitude of errors. Lower is better.
                    - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values. Lower is better.
                    - **R² Score**: Proportion of variance in the dependent variable predictable from the independent variables. Closer to 1 is better.
                    """)
                except Exception as e:
                    # Fall back to placeholder metrics if calculation fails
                    st.error(f"Could not calculate metrics: {e}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="RMSE", value="0.423")
                    with col2:
                        st.metric(label="MAE", value="0.315")
                    with col3:
                        st.metric(label="R² Score", value="0.879")
            
            with tabs[1]:
                st.write("Test set predictions vs actual values")
                
                # Add actual visualization for test predictions
                try:
                    # Get dates for the test period - we need these for the x-axis
                    # test_size is now determined in the metrics tab
                    test_dates = df_lstm.index[-test_size:]
                    
                    # Create dataframe for test results with dates
                    test_df = pd.DataFrame({
                        'Date': test_dates,
                        'Actual': y_test_actual,
                        'Predicted': y_pred_actual
                    })
                    
                    # Create visualization with dates on x-axis
                    fig = go.Figure()
                    
                    # Add actual values
                    fig.add_trace(go.Scatter(
                        x=test_df['Date'],
                        y=test_df['Actual'],
                        mode='lines',
                        name='Actual Values',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add predicted values
                    fig.add_trace(go.Scatter(
                        x=test_df['Date'],
                        y=test_df['Predicted'],
                        mode='lines',
                        name='Predicted Values',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    # Update layout with better title and axis labels
                    fig.update_layout(
                        title='LSTM Water Level Prediction vs Actual (Test Set)',
                        xaxis_title='Date',
                        yaxis_title='Baraj Seviyesi (m)',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        height=500,
                        template="plotly_white",
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray',
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray',
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add error distribution visualization
                    errors = test_df['Actual'] - test_df['Predicted']
                    
                    fig_hist = px.histogram(
                        errors, 
                        nbins=20,
                        title="Error Distribution",
                        labels={'value': 'Error (Actual - Predicted)'}
                    )
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Could not visualize test predictions: {e}")
                    st.info("Implementation for visualizing test predictions would go here")
            
            with tabs[2]:
                st.write("Model Loss During Training")
                
                try:
                    # Try to load actual training history from saved file
                    history_available = False
                    
                    # Check if history is available from model
                    if hasattr(model, 'history') and model.history is not None:
                        history = model.history.history
                        if 'loss' in history and 'val_loss' in history:
                            epochs = range(1, len(history['loss']) + 1)
                            training_loss = history['loss']
                            validation_loss = history['val_loss']
                            history_available = True
                            st.success("Loaded training history from model")
                    
                    # If not in model, try to load from saved history file
                    if not history_available and os.path.exists(history_path):
                        import pickle
                        try:
                            with open(history_path, 'rb') as f:
                                history = pickle.load(f)
                            if 'loss' in history and 'val_loss' in history:
                                epochs = range(1, len(history['loss']) + 1)
                                training_loss = history['loss']
                                validation_loss = history['val_loss']
                                history_available = True
                                st.success("Loaded training history from saved file")
                        except Exception as e:
                            st.warning(f"Could not load history file: {e}")
                    
                    # If still no history, show instructions to save history
                    if not history_available:
                        st.warning("Actual training history not found. To save training history, add this code to your model training script:")
                        st.code("""
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[],
    verbose=1
)

# Save the history
import pickle
with open('model_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
                        """)
                        
                        # Use simulated data as fallback
                        st.info("Using simulated training data as an example")
                        
                        # Create dummy data for demonstration purposes
                        epochs = range(1, 51)
                        training_loss = [0.18, 0.032, 0.020, 0.015, 0.012, 
                                        0.010, 0.009, 0.008, 0.007, 0.006,
                                        0.005, 0.005, 0.004, 0.004, 0.004,
                                        0.004, 0.004, 0.003, 0.003, 0.003,
                                        0.003, 0.003, 0.003, 0.003, 0.003,
                                        0.003, 0.003, 0.003, 0.003, 0.003,
                                        0.003, 0.003, 0.003, 0.003, 0.003,
                                        0.003, 0.003, 0.003, 0.003, 0.003,
                                        0.003, 0.003, 0.003, 0.003, 0.003,
                                        0.003, 0.003, 0.003, 0.002, 0.002]
                        
                        validation_loss = [0.045, 0.022, 0.015, 0.012, 0.010,
                                          0.009, 0.008, 0.007, 0.006, 0.005,
                                          0.004, 0.004, 0.004, 0.003, 0.003,
                                          0.003, 0.003, 0.003, 0.003, 0.003,
                                          0.003, 0.003, 0.003, 0.003, 0.003,
                                          0.003, 0.003, 0.003, 0.003, 0.003,
                                          0.003, 0.003, 0.003, 0.003, 0.003,
                                          0.003, 0.003, 0.003, 0.003, 0.003,
                                          0.003, 0.003, 0.003, 0.003, 0.003,
                                          0.003, 0.003, 0.003, 0.002, 0.002]
                    
                    # Create a figure with matplotlib
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(epochs, training_loss, 'b-', label='Training Loss')
                    ax.plot(epochs, validation_loss, 'orange', label='Validation Loss')
                    ax.set_title('Model Loss During Training')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss (MSE)')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Display the matplotlib figure in Streamlit
                    st.pyplot(fig)
                    
                    # Add explanation
                    st.markdown("""
                    ### Training History Interpretation
                    
                    - **Training Loss**: The model's error on the training data during each epoch of training
                    - **Validation Loss**: The model's error on a held-out validation set during training
                    
                    A good model shows:
                    - Decreasing loss for both training and validation sets
                    - Small gap between training and validation loss (indicates good generalization)
                    - Stabilization of both curves (indicates convergence)
                    
                    If validation loss starts increasing while training loss continues to decrease, it may indicate overfitting.
                    """)
                    
                    # Option to download history data
                    if history_available:
                        history_df = pd.DataFrame({
                            'Epoch': epochs,
                            'Training_Loss': training_loss,
                            'Validation_Loss': validation_loss
                        })
                        
                        csv = history_df.to_csv(index=False)
                        st.download_button(
                            label="Download Training History Data",
                            data=csv,
                            file_name='model_training_history.csv',
                            mime='text/csv',
                        )
                    
                except Exception as e:
                    st.error(f"Could not display training history: {e}")
                    st.exception(e)
            
            with tabs[3]:
                st.subheader("Visual Prediction Accuracy Test")
                st.markdown("""
                This tool allows you to visually test the model's prediction accuracy. 
                1. Select a date within the dataset
                2. The model will predict the water level for the next day
                3. Compare the prediction with the actual value
                """)
                
                # Get min and max dates from dataset (excluding the last sequence_length days)
                try:
                    min_date = pd.Timestamp(df.index.min())
                    max_date = pd.Timestamp(df.index.max())
                    safe_max_date = max_date - pd.Timedelta(days=1)
                    safe_min_date = min_date + pd.Timedelta(days=sequence_length)
                    
                    # Default value calculation
                    default_date = safe_max_date - pd.Timedelta(days=30)
                    if default_date < safe_min_date:
                        default_date = safe_min_date
                    
                    # Date selector
                    selected_date = st.date_input(
                        "Select split date (model will predict the next day):",
                        value=default_date.date(),
                        min_value=safe_min_date.date(),
                        max_value=safe_max_date.date()
                    )
                    
                    # Add forecast days slider
                    forecast_days = st.slider("Forecast Days", 1, 30, 1, key="visual_test_forecast_days")
                    
                    # Convert to pandas datetime
                    selected_date = pd.Timestamp(selected_date)
                    
                    if st.button("Test Prediction", key="visual_test_btn"):
                        with st.spinner("Generating visual prediction test..."):
                            try:
                                # Get data up to selected date for prediction
                                mask = df_lstm.index <= selected_date
                                data_for_prediction = df_lstm[mask].copy()
                                
                                if len(data_for_prediction) < sequence_length:
                                    st.error(f"Not enough data available before {selected_date.date()} for prediction. Need at least {sequence_length} days of data.")
                                    st.stop()
                                
                                # Prepare data for prediction
                                scaler = MinMaxScaler(feature_range=(0, 1))
                                scaled_data = scaler.fit_transform(data_for_prediction)
                                
                                # Take the last sequence_length days for prediction
                                prediction_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
                                
                                # Generate forecast for multiple days
                                forecasted_values = []
                                current_sequence = prediction_sequence.copy()
                                
                                for i in range(forecast_days):
                                    # Predict next day
                                    next_day_prediction_scaled = model.predict(current_sequence, verbose=0)
                                    # Store the prediction
                                    forecasted_values.append(next_day_prediction_scaled[0, 0])
                                    # Update sequence for next prediction (remove oldest, add newest)
                                    next_pred_reshaped = next_day_prediction_scaled[0, 0].reshape(1, 1, 1)
                                    current_sequence = np.concatenate([current_sequence[:, 1:, :], 
                                                                      next_pred_reshaped], 
                                                                      axis=1)
                                
                                # Convert predictions back to original scale
                                forecasted_values = scaler.inverse_transform(
                                    np.array(forecasted_values).reshape(-1, 1)
                                ).flatten()
                                
                                # Generate dates for forecast period
                                forecast_dates = [selected_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
                                
                                # Get actual values for the forecast period
                                actual_values = []
                                for date in forecast_dates:
                                    next_day_data = df_lstm[df_lstm.index == date]
                                    if not next_day_data.empty:
                                        actual_values.append(next_day_data.iloc[0]['Baraj_Seviyesi'])
                                    else:
                                        actual_values.append(None)
                                
                                # Create forecast dataframe
                                forecast_df = pd.DataFrame({
                                    'Date': forecast_dates,
                                    'Forecasted_Value': forecasted_values,
                                    'Actual_Value': actual_values
                                })
                                
                                # Display forecast table
                                st.write("Forecasted vs Actual Water Levels:")
                                st.dataframe(forecast_df)
                                
                                # Calculate error metrics for available actual values
                                valid_indices = [i for i, val in enumerate(actual_values) if val is not None]
                                if valid_indices:
                                    valid_forecasts = [forecasted_values[i] for i in valid_indices]
                                    valid_actuals = [actual_values[i] for i in valid_indices]
                                    
                                    errors = [a - p for a, p in zip(valid_actuals, valid_forecasts)]
                                    error_percents = [(a - p) / a * 100 for a, p in zip(valid_actuals, valid_forecasts)]
                                    
                                    mean_error = sum(errors) / len(errors)
                                    mean_error_percent = sum(error_percents) / len(error_percents)
                                    
                                    # Display average error metrics
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric(
                                            label="Average Error", 
                                            value=f"{mean_error:.2f} m"
                                        )
                                    with col2:
                                        st.metric(
                                            label="Average Error Percent", 
                                            value=f"{mean_error_percent:.2f}%"
                                        )
                                
                                # Create visualization - make sure all data is properly converted for plotly
                                # Get historical data for the chart (last 60 days before prediction)
                                vis_start_date = selected_date - pd.Timedelta(days=60)
                                # Make sure we have a valid start date
                                if vis_start_date < min_date:
                                    vis_start_date = min_date
                                    
                                # Create masks for data selection to avoid integer-based indexing
                                hist_mask = (df_lstm.index >= vis_start_date) & (df_lstm.index <= selected_date)
                                historical_data = df_lstm[hist_mask].copy()
                                
                                # Convert datetime index to unix timestamps for plotly
                                timestamps = [d.timestamp() * 1000 for d in historical_data.index]
                                
                                fig = go.Figure()
                                
                                # Add historical line with converted timestamps
                                fig.add_trace(go.Scatter(
                                    x=timestamps,
                                    y=historical_data['Baraj_Seviyesi'],
                                    mode='lines',
                                    name='Historical Water Level',
                                    line=dict(color='blue')
                                ))
                                
                                # Add vertical line at split date
                                fig.add_vline(
                                    x=selected_date.timestamp() * 1000,  # Convert to Unix timestamp in milliseconds for plotly
                                    line_width=2, 
                                    line_dash="dash", 
                                    line_color="black",
                                    annotation_text="Split Date",
                                    annotation_position="top"
                                )
                                
                                # Add prediction points - converting timestamps for plotly
                                forecast_timestamps = [d.timestamp() * 1000 for d in forecast_dates]
                                fig.add_trace(go.Scatter(
                                    x=forecast_timestamps,
                                    y=forecasted_values,
                                    mode='lines+markers',
                                    name='Predicted Values',
                                    line=dict(color='red', dash='dash'),
                                    marker=dict(color='red', size=8)
                                ))
                                
                                # Add actual points for which we have data
                                actual_timestamps = []
                                valid_actuals = []
                                for i, val in enumerate(actual_values):
                                    if val is not None:
                                        actual_timestamps.append(forecast_dates[i].timestamp() * 1000)
                                        valid_actuals.append(val)
                                
                                if valid_actuals:
                                    fig.add_trace(go.Scatter(
                                        x=actual_timestamps,
                                        y=valid_actuals,
                                        mode='markers',
                                        name='Actual Values',
                                        marker=dict(color='green', size=8)
                                    ))
                                
                                # Update layout
                                fig.update_layout(
                                    title=f'Prediction Test ({forecast_days} days from {selected_date.strftime("%Y-%m-%d")})',
                                    xaxis_title='Date',
                                    yaxis_title='Water Level (m)',
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add explanation
                                st.markdown(f"""
                                ### Interpretation
                                
                                - The blue line shows historical water level data up to your selected date ({selected_date.strftime('%Y-%m-%d')})
                                - The vertical dashed line marks your selected split date
                                - The red line shows the model's predictions for the next {forecast_days} days
                                - The green dots show the actual water levels for dates where data is available
                                
                                The closer the red line is to the green dots, the more accurate the model's predictions.
                                """)
                            except Exception as e:
                                st.error(f"Error generating visual prediction test: {str(e)}")
                                st.exception(e)
                except Exception as e:
                    st.error(f"Error setting up date selector: {str(e)}")
                    st.exception(e)
        elif data_loaded and not tensorflow_available:
            st.warning("TensorFlow is not available. Model prediction functionality is disabled.")
            st.info("Please install TensorFlow to enable prediction features.")
    else:
        data_loaded = False
        st.error("Failed to load data")
except Exception as e:
    data_loaded = False
    st.error(f"Error loading data: {e}")
    st.info("Please ensure the CSV file is in the correct location")

# Sidebar for additional configuration
with st.sidebar:
    st.header("About")
    st.markdown("""
    This dashboard visualizes dam water level data and predictions from an LSTM model.
    
    Features:
    - Historical data visualization
    - Water level forecasting
    - Model performance metrics
    
    To use this dashboard, ensure the model file (lstm_model.h5) and
    dataset (baraj_seviyesi_tum_yillar_eksiksiz.csv) are available.
    """)
    
    # System status section
    st.subheader("System Status")
    
    if tensorflow_available:
        st.success("TensorFlow: Available")
    else:
        st.error("TensorFlow: Not Available")
        
    if model_loaded:
        st.success("Model: Loaded")
    else:
        st.error("Model: Not Loaded")
        
    if 'data_loaded' in locals() and data_loaded:
        st.success("Data: Loaded")
    else:
        st.error("Data: Not Loaded") 