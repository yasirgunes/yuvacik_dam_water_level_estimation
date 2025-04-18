# Dam Water Level Prediction Dashboard

A Streamlit web application for visualizing and forecasting dam water levels using an LSTM model.

## Features

- Interactive historical data visualization
- Adjustable time series forecasting
- Model performance metrics display
- Raw data exploration

## Setup Instructions

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the LSTM model to train and save it:
   ```
   python lstm_model.py
   ```

3. Launch the Streamlit app:
   ```
   streamlit run app.py
   ```

4. Access the dashboard in your web browser at `http://localhost:8501`

## Required Files

- `lstm_model.py` - LSTM model training script
- `app.py` - Streamlit dashboard application
- `baraj_seviyesi_tum_yillar_eksiksiz.csv` - Dataset with dam water level data
- `lstm_model.h5` - Saved model file (created after running the LSTM script)

## Dashboard Sections

1. **Historical Data Visualization**: View and filter historical water level data with an interactive chart
2. **Forecast**: Generate and visualize future water level predictions
3. **Model Performance**: View metrics on model accuracy and test predictions
4. **Raw Data Exploration**: Examine the dataset and its statistics 