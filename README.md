# yfinance_streamlit
#  Real-World Stock Forecasting Dashboard with Streamlit

An interactive data science web app for stock price analysis and prediction using real-time data from [Yahoo Finance](https://finance.yahoo.com/). Built with `Python`, `Streamlit`, and popular machine learning and deep learning libraries, this project helps users analyze market trends, explore financial statements, and forecast future prices using models like ARIMA, SARIMA, and LSTM.

## Why This Project Matters

While many stock prediction projects focus on just one model, this project uniquely combines multiple forecasting models, classification algorithms, EDA, feature engineering, time series models and deep learning in a single interactive app.

- **Real-time Data Handling**: Integration with live financial data from Yahoo Finance using `yfinance`.
-  **Time Series Forecasting**: Predicting future stock prices using:
  - **ARIMA / ARIMAX**, **SARIMA**,**LSTM (Deep Learning)**
-  **Machine Learning for Classification**: Predict whether a stock will rise using:
  - **Logistic Regression**,**SVM**, **XGBoost**
-  **Interactive Data Visualizations**: Created using `Plotly`, `Matplotlib`, and `Cufflinks`.
-  **EDA Automation**: Using `pandas-profiling` (`ydata_profiling`) for quick and deep insights into the dataset.
- **Skill Development**: Showcases model evaluation using ROC-AUC and Mean Squared Error (MSE), and applies data preprocessing techniques like scaling and sequence generation for LSTM.

## Features

- **Stock Picker**: Choose from popular tickers like AAPL, GOOGL, META, TSLA, MSFT, and BTC-USD.
- **Custom Date Range**: Analyze trends over a selected historical time frame.
- **Technical Indicators**: View moving averages, Bollinger Bands, and price volatility.
-  **Forecasting Models**:
  - **ARIMA, ARIMAX, SARIMA** for classical statistical time series modeling.
  - **LSTM (Deep Learning)** for sequence learning and price forecasting.
- **ROC-AUC Based Classification**: Predict stock movement using ML classifiers.
- **Financial Reports**: Retrieve balance sheets and financials directly from Yahoo Finance.
-  **Automated EDA**: Generate detailed data profiling reports with one click.

##  Tech Stack

- **Frontend**: Streamlit
- **Backend/Modeling**:
  - `yfinance`, `pandas`, `numpy`
  - `scikit-learn`, `XGBoost`, `statsmodels`
  - `TensorFlow Keras` for LSTM
  - `matplotlib`, `plotly`, `cufflinks`
- **EDA**: `ydata-profiling`, `streamlit-pandas-profiling`

## Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/yfinance_streamlit.git
cd yfinance_streamlit

### 2. Install dependancies

pip install -r requirements.txt


### 3. Run the app
streamlit run filename.py
