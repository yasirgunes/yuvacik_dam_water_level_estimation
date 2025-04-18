# Yuvacık Dam Water Level Prediction Dashboard

An interactive Streamlit dashboard and accompanying scripts for visualizing historical dam water level data and forecasting future levels using machine learning models (LSTM and Random Forest).

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
  - [LSTM Model](#lstm-model)
  - [Random Forest Model](#random-forest-model)
- [Notebooks](#notebooks)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- Interactive visualization of historical water level data with Plotly and Matplotlib
- Adjustable date range filters and automated downsampling for large time windows
- Configurable multi-day time-series forecasting using a cached LSTM model
- Baseline Random Forest model for comparison and performance metrics
- Raw data exploration with summary statistics
- Pre-generated plots available in `su_seviyesi_grafikleri/`

## Project Structure

```bash
├── app.py                  # Streamlit dashboard application
├── data_acquisition.py     # Data acquisition and preprocessing script
├── lstm_model.py           # LSTM model training script
├── model.py                # Random Forest training and evaluation script
├── lstm_model.h5           # Saved LSTM model weights
├── model_history.pkl       # Pickled training history for LSTM
├── baraj_seviyesi_tum_yillar_eksiksiz.csv  # Cleaned dataset used in app
├── baraj_seviyesi_tum_yillar.csv           # Raw dataset
├── su_seviyesi_grafikleri/ # Generated visualization images
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
3. Upgrade `pip` and install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Acquisition & Preprocessing

If you wish to fetch or update data, run:
```bash
python data_acquisition.py
```

### 2. Model Training

- **LSTM Model**: Train and save the model weights
  ```bash
  python lstm_model.py
  ```
- **Random Forest Model**: Train and evaluate baseline model
  ```bash
  python model.py
  ```

### 3. Launch Dashboard

Start the Streamlit application:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to interact with the dashboard.

## Data

- **baraj_seviyesi_tum_yillar.csv**: Raw collected dam water level data.
- **baraj_seviyesi_tum_yillar_eksiksiz.csv**: Cleaned and imputed dataset used for modeling and visualization.

## Models

### LSTM Model

- Implemented in `lstm_model.py` and `lstm_model.ipynb`.
- Architecture: Single LSTM layer (50 units), dropout, dense output.
- Input window: 30 days of past water levels.
- Forecast: Next-day predictions extendable to a multi-day horizon via iterative forecasting.
- Metrics and training history stored in `model_history.pkl`.

### Random Forest Model

- Implemented in `model.py` and `random_forest_model.ipynb`.
- Uses lag features and date-based features for regression.
- Provides baseline metrics (MSE, MAE) for comparison.

## Notebooks

- **lstm_model.ipynb**: Exploratory data analysis and LSTM development walkthrough.
- **random_forest_model.ipynb**: Random Forest modeling and results visualization.

## Visualizations

Pre-generated images (monthly and yearly water level trends) are in the `su_seviyesi_grafikleri/` directory.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m "Add feature description"`)
4. Push to your branch (`git push origin feature/YourFeature`)
5. Open a Pull Request for review

## License

This project is released under the MIT License. See `LICENSE` for details.

## Contact

- Maintainer: [Your Name]
- Email: your.email@example.com 