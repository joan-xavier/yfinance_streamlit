# yfinance_streamlit
#  Real-World Stock Forecasting Dashboard with Streamlit

An interactive data science web app for stock price analysis and prediction using real-time data from [Yahoo Finance](https://finance.yahoo.com/). Built with `Python`, `Streamlit`, and popular machine learning and deep learning libraries, this project helps users analyze market trends, explore financial statements, and forecast future prices using models like ARIMA, SARIMA, and LSTM.

## Why This Project Matters

While many stock prediction projects focus on just one model, this project uniquely combines multiple forecasting models, classification algorithms, EDA, feature engineering, time series models and deep learning in a single interactive app.

 
 1. Live Data Ingestion & Preprocessing
 - Pulled real-time historical stock data from Yahoo Finance API using yfinance.
 -	Enabled user-configurable date ranges, trading intervals, and stock tickers.
 -	Applied rolling window functions for moving average & volatility indicators.

 2. Time Series Forecasting with ARIMA, SARIMA, and ARIMAX
 -	Trained and evaluated classical models (ARIMA, SARIMA, ARIMAX) to forecast closing prices.
 -	Visualized model outputs against test data for performance comparison.
 -	Integrated exogenous variables like volume for enhanced ARIMAX prediction.

 3. Deep Learning with LSTM for Stock Price Prediction
 -	Built an LSTM-based Recurrent Neural Network using TensorFlow/Keras for multi-day sequence prediction.
 -	Normalized data using MinMaxScaler and structured temporal input for training.
 -	Reported and visualized RMSE for both training and test sets to assess performance.

4. Stock Movement Classification with ML Models
 -	Engineered target labels for price direction prediction (Up/Down) over customizable horizons.
 -	Implemented Logistic Regression, Support Vector Machine, and XGBoost classifiers.
 -	Evaluated classifiers using ROC-AUC curves and highlighted the best-performing model.

 5. Interactive EDA & Comparative Insights Across Tickers
 -	Compared multiple stocks (AAPL, GOOGL, TSLA, BTC, etc.) on:
   -	Cumulative Returns
   -	Volatility vs Return
   -	Bar Charts of Risk/Reward
   -	Correlation Heatmaps
   -	Confusion Matrices of directional consistency.
-	Enabled full visualization using Plotly, Seaborn, and Matplotlib with responsive layouts in Streamlit.


## Features

- **Stock Picker**: Choose from popular tickers like AAPL, GOOGL, META, TSLA, MSFT, and BTC-USD.
- **Custom Date Range**: Analyze trends over a selected historical time frame.
- **Technical Indicators**: View moving averages, Bollinger Bands, and price volatility.
-  **Forecasting Models**:
  - **ARIMA, ARIMAX, SARIMA** for classical statistical time series modeling.
  - **LSTM (Deep Learning)** for sequence learning and price forecasting.
- **ROC-AUC Based Classification**: Predict stock movement using ML classifiers.
- **Financial Reports**: Retrieve balance sheets and financials directly from Yahoo Finance.
-  **Automated EDA**: Generate detailed data profiling reports and full visualization with one click.

##  Tech Stack

- **Frontend**: Streamlit
- **Backend/Modeling**:
  - `yfinance`, `pandas`, `numpy`
  - `scikit-learn`, `XGBoost`, `statsmodels`
  - `TensorFlow Keras` for LSTM
  - `matplotlib`, `plotly`, `cufflinks`
- **EDA**: `pandas, `numpy`,`matplotlib`, `plotly`, `streamlit-pandas-profiling`

## Getting Started

### 1. Clone the Repo

### 2. Install dependancies
This project uses python 3.11, and the common data science libraries

pip install -r requirements.txt

### 3. Run the app
streamlit run filename.py
