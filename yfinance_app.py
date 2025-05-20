import streamlit as st
import yfinance as yf
import numpy as np
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import cufflinks as cf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


# st.title("Real world Yahoo stocks Dashboard -Interactive App")
# App title
st.markdown('''
## Real world Stocks Price and Forecasting Dashboard -Interactive App
### All the stock price data are taken from [**Live Yahoo financing!**](https://finance.yahoo.com/quote/GOOGL/)

**Credits**
- App built by [Joanofarc Xavier](https://joan-xavier.github.io/portfolio/) 
- Built in `Python` using `streamlit`,`yfinance`, `plotly`,`matplotlib`, `pandas`,`scikit learn`, `Tensorflow`and `datetime`
- Trends and Patterns of popular stocks - META, GOOGL, AAPL(Apple), MSFT(Microsoft), TSLA (Tesla) BTC-USD(Bitcoin)
- Time series Analysis -Stock Prediction using ARIMA, LSTM Models, Logistic Regression, SVM and XGBoost
''')
st.write('---')

# sidebar layout(start date, end date, ticker symbol, ma_window)
st.sidebar.title('Stock Settings')
# ticker_symbol = st.sidebar.selectbox('Enter Ticker Symbol', ('AAPL','GOOGL','META','MSFT','TSLA','BTC-USD'))
ticker_symbol = st.sidebar.selectbox("Ticker Symbol", ['AAPL', 'GOOGL', 'META', 'MSFT', 'TSLA', 'BTC-USD'])
st.sidebar.write(" ### Trading Period")


# Set default to 1 year ago from today
default_start = datetime.date.today() - datetime.timedelta(days=365)
default_end = datetime.date.today()
start_date = st.sidebar.date_input('Start Date', value=default_start)
end_date =st.sidebar.date_input('End Date', value=default_end)


trade_interval = st.sidebar.selectbox('Trading interval',("3mo", "1mo", "1wk", "5d", "1d", "1h", "90m", "30m", "15m"))
st.sidebar.write(" #### Note: for recurring intervals less than 1 day, only the last 60 days of data are available")
st.sidebar.write('#### m - minute, d -day, wk - week, mo - month')
#get yfinance data on this ticker
ticker = yf.Ticker(ticker_symbol)

# ------------------- Cache Data Download --------------------
@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    df = yf.download(ticker_symbol, start=start, end=end)
    df.reset_index(inplace=True)
    return df

if start_date and end_date:
    stockData = load_data(ticker_symbol, start_date, end_date)
    if stockData.empty:
        st.error("No data found for selected range.")
        st.stop()

    # Fetching data
    st.write(f"Fetching data for **{ticker_symbol}** from {start_date} to {end_date}")
    st.subheader(f'{ticker_symbol} Stock Overview')
    
    # stockData = yf.download(ticker_symbol, start =start_date, end = end_date)
    # the data frame stockData has multiple indices
    if isinstance(stockData.columns, pd.MultiIndex):
        stockData.columns = stockData.columns.get_level_values(0)  # flatten columns
    
    # Initialize Tabs
    raw_data_tab,prediction_tab, chart_tab, finance_tab, EDA_tab = st.tabs(["Price Summary","Prediction Analysis","Charts & Trends","Financial Statements","Data Analysis"])
    
    # Tab 1: Raw Data

    with raw_data_tab:
        st.subheader(f"Stock Price Summary for {ticker_symbol} stocks for the selected period ")
        st.write(stockData.tail())# data frame
        st.download_button("Download Raw Data as CSV", stockData.to_csv(), file_name=f"{ticker_symbol}_data.csv")

     
    # Tab 2: Stock Data for Prediction
    # Tab 2, part 1 : Rolling window
    with prediction_tab:
        # Tab 2, part 1 : Rolling window
        st.subheader("Stock Prediction using Time series - ARIMA/SARIMA/ARIMAX/LSTM Models")
        st.write('''##### Models are trained to forecast stock prices and their accuracy is evaluated using past data''')
        ma_window = st.slider("Rolling Window Size (in Days)", min_value=5, max_value=50, value=10)
        # Calculate Rolling Mean and Standard Deviation
        
        stockData['roll_mean'] = stockData['Close'].rolling(window=ma_window).mean()
        stockData['roll_std'] = stockData['Close'].rolling(window=ma_window).std()


        # Safely access the last value of the rolling mean
        if not stockData['roll_mean'].dropna().empty:
            mean_value = stockData['roll_mean'].dropna().iloc[-1]
            std_value = stockData['roll_std'].iloc[-1]
            st.markdown(f"**Latest Moving Average (Window Size {ma_window}):** {mean_value:.2f}")
            st.markdown(f"**Latest Standard Deviation (Window Size {ma_window}):** {std_value:.2f}")
        else:
            st.warning("Insufficient data to calculate rolling mean. Please choose a smaller window size or a larger data range.")


        stockData.reset_index(inplace=True)  # Reset index to use Date as a column
        # Plotting using Plotly Express
        fig = px.line(stockData, x='Date', y=['Close', 'roll_mean', 'roll_std'],
                     labels={
                         "value": "Price (USD)",
                         "Date": "Date",
                         "variable": "Legend"
                     },
                     title=f"{ticker_symbol} Stock Price with {ma_window}-Day Moving Average and Std Dev")

        # Customize traces for better visualization
        fig.for_each_trace(lambda trace: trace.update(mode='lines+markers'))
        fig.update_traces(line=dict(width=1))

        # Increase figure size
        fig.update_layout(
            width=1000,
            height=600
        )

        # Update legend names
        fig.for_each_trace(lambda trace: trace.update(name={
            "Close": "Closing Price",
            "roll_mean": f"{ma_window}-Day Moving Average",
            "roll_std": f"{ma_window}-Day Rolling Std Dev"
        }[trace.name]))

        # Display the chart
        st.plotly_chart(fig)

        
        # Tab 2 , part 2: ARIMA
   
        # Ensure DateTimeIndex for ARIMA
        if 'Date' in stockData.columns:
            stockData.set_index('Date', inplace=True)

        # Split data into train and test
        train_size = int(len(stockData) * 0.7)
        train, test = stockData.iloc[:train_size], stockData.iloc[train_size:]

        # ARIMA model
        arima_model = ARIMA(train["Close"], order=(1,1,1))  
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=len(test))

        # ARIMAX model (adding exogenous variable if available, e.g., 'Volume')
        if 'Volume' in stockData.columns:
            arimax_model = SARIMAX(train["Close"], exog=train["Volume"], order=(1,1,1))
            arimax_fit = arimax_model.fit()
            arimax_forecast = arimax_fit.forecast(steps=len(test), exog=test["Volume"])
        else:
            print("No exogenous variable available for ARIMAX; using ARIMA instead.")
            arimax_forecast = arima_forecast

        # SARIMA model
        sarima_model = SARIMAX(train["Close"], order=(1,1,1), seasonal_order=(1,1,1,12))
        sarima_fit = sarima_model.fit()
        sarima_forecast = sarima_fit.forecast(steps=len(test))
        # Plot the results
        plt.figure(figsize=(14,7))

        # Actual data
        plt.plot(train.index, train["Close"], label='Train', color='#203147')
        plt.plot(test.index, test["Close"], label='Test', color='#01ef63')

        # Forecasts
        plt.plot(test.index, arima_forecast, label='ARIMA Forecast', color='orange', linestyle='--')
        plt.plot(test.index, arimax_forecast, label='ARIMAX Forecast', color='blue', linestyle='-.')
        plt.plot(test.index, sarima_forecast, label='SARIMA Forecast', color='red', linestyle=':')

        # Title and labels
        plt.title('Comparison of ARIMA, ARIMAX, and SARIMA Models')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()
        st.pyplot(plt)
        #  Reset index AFTER forecasting
        stockData.reset_index(inplace=True)
        

        # Tab 2, part 3: LSTM

        # %%%%%%%%%%%%%%%%%%
        # Streamlit App Title
        st.subheader("Stock Price Prediction using RNN based LSTM Model ")
        seq_length = st.selectbox("Forecasting sequence length in days", (1, 7, 14, 30, 45, 60))

        # Data Fetching
                  
        if stockData.empty:
            st.error("No data found for the selected ticker symbol and date range. Please check your inputs.")
            st.stop()

        st.write("Data fetched successfully.")
        st.write(stockData.tail())

        # Data Preprocessing
        scaler = MinMaxScaler(feature_range=(0, 1))

        if 'Close' not in stockData.columns or stockData['Close'].isnull().all():
            st.error("No valid 'Close' price data available. Please select another ticker symbol or date range.")
            st.stop()

        clean_data = stockData['Close'].dropna()
        if clean_data.empty:
            st.error("No valid data available after removing NaN values.")
            st.stop()

        # Scale the data
        scaled_data = scaler.fit_transform(clean_data.values.reshape(-1, 1))
        st.success("Data successfully scaled.")

        # Train-Test Split
        train_size = int(len(scaled_data) * 0.7)
        if train_size == 0:
            st.error("Insufficient data for training. Please select a larger date range.")
            st.stop()

        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

        # Function to create LSTM sequences
        def create_sequences(data, seq_length):
            x, y = [], []
            for i in range(len(data) - seq_length):
                x.append(data[i:(i + seq_length)])
                y.append(data[i + seq_length])
            return np.array(x), np.array(y)

        # Generate sequences
        if len(train_data) <= seq_length or len(test_data) <= seq_length:
            st.error("Not enough data points to create sequences. Try a smaller sequence length or larger date range.")
            st.stop()

        x_train, y_train = create_sequences(train_data, seq_length)
        x_test, y_test = create_sequences(test_data, seq_length)

        # Reshape to fit LSTM input
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        # Model Building
        model = Sequential([
            LSTM(50, return_sequences=False, input_shape=(seq_length, 1)),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Model Training
        with st.spinner("Training LSTM model..."):
            model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), verbose=0)

        # Prediction
        train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)

        # Reshape predictions
        train_predict = train_predict.reshape(-1, 1)
        test_predict = test_predict.reshape(-1, 1)

        # Inverse scaling
        train_predict = scaler.inverse_transform(train_predict)
        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        test_predict = scaler.inverse_transform(test_predict)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Mean Squared Error
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))

        st.success(f"Training RMSE: {train_rmse:.4f}")
        st.success(f"Testing RMSE: {test_rmse:.4f}")

        # Plotting the results
        plt.figure(figsize=(14, 7))

        # Get the date indices from the stockData DataFrame
        if 'Date' in stockData.columns:
            dates = stockData['Date'].values
        else:
            dates = stockData.index.values

           
        # Plot training data (actual values)
        plt.plot(dates[:train_size], scaler.inverse_transform(train_data), label='Training Data', color='#203147')

        # Plot testing data (actual values)
        plt.plot(dates[train_size:], scaler.inverse_transform(test_data), label='Testing Data', color='#01ef63')

        # Align the predictions with the correct dates
        train_pred_dates = dates[seq_length:train_size]  # Train prediction dates
        test_pred_dates = dates[train_size + seq_length:]  # Test prediction dates

        # Plot LSTM predictions with correct dates
        plt.plot(train_pred_dates, train_predict, label='LSTM Train Prediction', color='orange', linestyle='--')
        plt.plot(test_pred_dates, test_predict, label='LSTM Test Prediction', color='red', linestyle='--')


        plt.title(f'LSTM Prediction vs Actual for {ticker_symbol}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        st.pyplot(plt)

    
        # Tab 2, part 4 : ROC-AUC Evaluation using SVM, XGBoost and Logistic Regression
        st.markdown("""
###  ROC-AUC Evaluation for Stock Movement (Up/Down) Prediction

- In this classification task, prediction is made on whether a stock's **closing price will go up the next day (1)** or **not (0)**.
- Instead of just predicting **hard labels** like 0 or 1, these ML models generate **soft probabilities** — values between 0 and 1 — indicating the **confidence** that the stock will rise.
- The **ROC curve** plots the **True Positive Rate (correctly predicting upward movement)** vs. the **False Positive Rate (incorrectly predicting upward movement)** at various threshold levels.
- The **AUC (Area Under the Curve)** gives a single performance score:
  - **AUC = 1.0** - perfect model to distinguish between rising and non-rising stock outcomes.
  - **AUC = 0.5** - no better than random guessing
""")

        n_days_ahead = st.selectbox("Predict how many days ahead?", [1, 2, 3, 5, 7, 10], index=0)
        # Create target variable: user-selected N-day shift if price increases next day, else 0
        stockData['Target'] = (stockData['Close'].shift(-n_days_ahead) > stockData['Close']).astype(int)
        stockData.dropna(inplace=True)  # drop last row with NaN in Target

        # Feature selection (example features, can expand)
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = stockData[features]
        y = stockData['Target']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define models
        models = {
            "Logistic Regression": LogisticRegression(),
            "Support Vector Machine": SVC(probability=True),
            "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

        roc_fig = go.Figure()
        auc_scores = {}

        # Train and evaluate each model
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            auc_scores[name] = roc_auc
            
            # Plot ROC curve
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC = {roc_auc:.2f})'))

        # Plot formatting
        roc_fig.update_layout(
            title="ROC Curve Comparison of ML Models",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=900,
            height=600
        )
        st.plotly_chart(roc_fig)

        # Display best model
        best_model = max(auc_scores, key=auc_scores.get)
        st.success(f"Best Model: **{best_model}** with ROC-AUC = {auc_scores[best_model]:.4f}")

    
    # Tab 3: charts tab
    with chart_tab:
        
        st.write(f" ### {ticker_symbol} Charts with High, low, close prices and Volume (total no of shares traded) for the specified interval ")
        # 4a.

        st.subheader("High Price, Low Price in (USD) and Volume")

        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        if 'Date' in stockData.columns:
            stockData.set_index('Date', inplace=True)

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price (USD)", color='black')
        ax1.plot(stockData.index, stockData['High'], label="High Price", color='blue',linewidth=2)
        ax1.plot(stockData.index, stockData['Low'], label="Low Price", color='red', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Volume', color='tab:cyan')
        ax2.bar(stockData.index, stockData['Volume'], alpha=0.3, label='Volume', color='cyan')
        ax2.tick_params(axis='y', labelcolor='tab:cyan')

        plt.title(f'High {ticker_symbol} Stock Price and Volume')
        fig.tight_layout()
        fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
        plt.grid(True)
        plt.show()
        st.pyplot(fig)

        ### 4c. 
        # Bollinger bands
        st.subheader(f' Bollinger Bands for {ticker_symbol}')
        st.markdown(''' #### To visualize price volatility and potential overbought/oversold conditions on stock charts [Click for more details](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-bands)''')
        tickerDf = ticker.history(start=start_date, end = end_date, interval = trade_interval)
        qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')
        qf.add_bollinger_bands()
        fig = qf.iplot(asFigure=True)
        st.plotly_chart(fig)
        

        ### 4d.
        st.subheader(f"Closing Price in (USD) with {ma_window}-Days Moving Average")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(stockData.index, stockData['Close'], label="Closing Price", color='blue')
        ax.plot(stockData.index, stockData['roll_mean'], label=f"{ma_window}-Day Moving Average", color='orange')
        ax.set_title(f"Closing Price with {ma_window}-Days Moving Average")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
    

    # Tab 4: Finance detail tab   
    
    with finance_tab:
        st.subheader(f" Financial Overview of {ticker_symbol}")
        st.markdown("Extracting **Balance Sheet**, **Income Statement**, and **Cash Flow** data with trend analysis and key ratios.")

        # Load Statements
        balance_sheet = ticker.balance_sheet
        financials = ticker.financials
        cashflow = ticker.cashflow

        # Convert columns to string dates for better display
        balance_sheet.columns = balance_sheet.columns.strftime('%Y-%m')
        financials.columns = financials.columns.strftime('%Y-%m')
        cashflow.columns = cashflow.columns.strftime('%Y-%m')

        # Show Expandable Sections
        with st.expander(" Balance Sheet"):
            st.dataframe(balance_sheet)

        with st.expander(" Income Statement"):
            st.dataframe(financials)

        with st.expander("Cash Flow Statement"):
            st.dataframe(cashflow)

        # -- Financial Ratios (most recent column)
        st.markdown("###  Key Financial Ratios (Latest Available)")
        try:
            latest = balance_sheet.columns[0]

            total_assets = balance_sheet.loc["Total Assets", latest]
            total_liabilities = balance_sheet.loc["Total Liabilities Net Minority Interest", latest]
            total_equity = balance_sheet.loc["Ordinary Shares Number", latest]  # You may choose another equity measure
            current_assets = balance_sheet.loc["Current Assets", latest]
            current_liabilities = balance_sheet.loc["Current Liabilities", latest]

            revenue = financials.loc["Total Revenue", latest]
            net_income = financials.loc["Net Income", latest]

            current_ratio = current_assets / current_liabilities if current_liabilities != 0 else np.nan
            debt_to_equity = total_liabilities / total_equity if total_equity != 0 else np.nan
            net_profit_margin = net_income / revenue if revenue != 0 else np.nan

            st.write(f"**Current Ratio**: {current_ratio:.2f}")
            st.write(f"**Debt-to-Equity Ratio**: {debt_to_equity:.2f}")
            st.write(f"**Net Profit Margin**: {net_profit_margin:.2%}")

            # Text Insight
            if current_ratio > 1.5:
                st.success(f"{ticker_symbol} has a strong liquidity position.")
            if debt_to_equity > 2:
                st.warning(f" {ticker_symbol} carries a relatively high level of debt compared to its equity.")
            if net_profit_margin < 0:
                st.error(f"🔻{ticker_symbol} is currently operating at a net loss.")

        except Exception as e:
            st.warning(f"Could not compute some ratios: {e}")

        # -- Trend Plot for Revenue and Net Income
        st.markdown("###  Trend Analysis of Revenue & Net Income")
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=financials.columns, y=financials.loc["Total Revenue"], mode='lines+markers', name="Revenue"))
            fig.add_trace(go.Scatter(x=financials.columns, y=financials.loc["Net Income"], mode='lines+markers', name="Net Income"))

            fig.update_layout(
                title="Revenue and Net Income Over Time",
                xaxis_title="Period",
                yaxis_title="USD",
                height=400
            )
            st.plotly_chart(fig)
        except Exception as e:
            st.warning(f"Trend plot failed: {e}")


    
     # Tab 5: stock history tab   
    with EDA_tab:
        

        st.markdown("<h4>Comparative EDA for Popular Stocks</h4>", unsafe_allow_html=True)
        tickers = ['AAPL','GOOGL','META','MSFT','TSLA','BTC-USD']

        st.info("Fetching and comparing data for: AAPL, GOOGL, META, MSFT, TSLA, BTC-USD")

        price_data = yf.download(tickers, start=start_date, end=end_date)['Close']
        returns = price_data.pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        risk = returns.std()
        mean_return = returns.mean()
        correlation = returns.corr()

        # -- 1. Cumulative Returns Line Chart
                # Custom color mapping for tickers
        tickers = cumulative_returns.columns
        ticker_colors = px.colors.qualitative.Plotly  # Use any other palette if preferred
        color_map = {ticker: ticker_colors[i % len(ticker_colors)] for i, ticker in enumerate(tickers)}

        # -- 1. Cumulative Returns Line Chart
        st.subheader("Cumulative Returns")
        fig1 = px.line(
            cumulative_returns,
            title="Cumulative Returns Over Time",
            color_discrete_map=color_map
        )
        st.plotly_chart(fig1)

        # -- 2. Risk vs Return Scatter Plot
        st.subheader("Risk vs Return (Volatility vs Mean Return)")
        risk_return_df = pd.DataFrame({
            "Ticker": tickers,
            "Risk": risk.values,
            "Mean Return": mean_return.values
        })

        fig2 = px.scatter(
            risk_return_df,
            x="Risk",
            y="Mean Return",
            text="Ticker",
            color="Ticker",
            title="Risk vs Return",
            labels={"Risk": "Risk (Std Dev)", "Mean Return": "Mean Return"},
            color_discrete_map=color_map
        )
        fig2.update_traces(marker=dict(size=12), textposition='top center')
        st.plotly_chart(fig2)

        # -- 3. Bar Chart of Risk and Return
        st.subheader("Bar Chart of Risk and Mean Return")
        st.dataframe(risk_return_df)

        fig3 = go.Figure()

        # Plot each ticker with matching colors
        for ticker in tickers:
            fig3.add_trace(go.Bar(
                name=f'{ticker} - Risk',
                x=[ticker],
                y=[risk[ticker]],
                marker_color=color_map[ticker]
            ))
            fig3.add_trace(go.Bar(
                name=f'{ticker} - Mean Return',
                x=[ticker],
                y=[mean_return[ticker]],
                marker_color=color_map[ticker],
                opacity=0.5  # faded to distinguish from risk
            ))

        fig3.update_layout(
            barmode='group',
            title="Risk and Mean Return for Each Ticker",
            xaxis_title="Ticker",
            yaxis_title="Value",
            width=900,
            height=500
        )
        st.plotly_chart(fig3)

        # -- 4. Correlation Heatmap
        
        st.markdown("<h4 style='color:#336699;'>Correlation Between Assets</h4>", unsafe_allow_html=True)

        fig4 = px.imshow(
            correlation,
            text_auto=".2f",
            title="Correlation Matrix of Daily Returns",
            color_continuous_scale='Viridis',
            width=700,
            height=600
        )
        fig4.update_layout(font=dict(size=12), title_font_size=18)
        st.plotly_chart(fig4)

        # -- 5. Up vs Down Days Confusion Matrix
        st.markdown("<h4 style='color:#336699;'>Confusion Matrix: Movement Consistency</h4>", unsafe_allow_html=True)

        for ticker in tickers:
            direction = np.where(returns[ticker] > 0, 1, 0)
            y_true = direction[:-1]
            y_pred = direction[1:]
            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(2, 2), dpi=150)  # Higher resolution

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=False,
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'],
                annot_kws={"size": 8},
                ax=ax
            )

            ax.set_title(f'{ticker}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=9)
            ax.set_ylabel('Actual', fontsize=9)
            ax.tick_params(axis='both', labelsize=8)

            st.pyplot(fig)


        


