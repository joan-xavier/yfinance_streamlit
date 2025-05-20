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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


 ### Details of YAhoo finance:
#  The following ticker symbols   AAPL -apple stocks;  GOOGL - google;  MSFT- Microsoft



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
st.sidebar.title(' Fill the stock details')
ticker_symbol = st.sidebar.selectbox('Enter Ticker Symbol', ('AAPL','GOOGL','META','MSFT','TSLA','BTC-USD'))

st.sidebar.write(" ### Trading Period")
start_date = st.sidebar.date_input('Start Date', value=None)
end_date =st.sidebar.date_input('End Date', value=None)


trade_interval = st.sidebar.selectbox('Trading interval',("3mo", "1mo", "1wk", "5d", "1d", "1h", "90m", "30m", "15m"))
st.sidebar.write(" #### Note: for recurring intervals less than 1 day, only the last 60 days of data are available")
st.sidebar.write('#### m - minute, d -day, wk - week, mo - month')
#get yfinance data on this ticker
ticker = yf.Ticker(ticker_symbol)

if start_date is not None and end_date is not None:
    # Fetching data
    st.write(f"Fetching data for **{ticker_symbol}** from {start_date} to {end_date}")
    st.subheader(f'{ticker_symbol} Stock Overview')
    stockData = yf.download(ticker_symbol, start =start_date, end = end_date)
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
    
    with prediction_tab:
        #$$$$$$$$$$$$$$$$$$$$$4
        st.subheader("Stock Prediction using Time series - ARIMA/SARIMA/ARIMAX Models")
        # Tab 2, part 1 : Rolling window
     
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
     ##############################################################
        # Tab 2, part 2 : ARIMA
       
        # Ensure DateTimeIndex for ARIMA
        # if 'Date' in stockData.columns:
        #     stockData.set_index('Date', inplace=True)

        # Reset index before modeling to ensure numeric RangeIndex
        # stockData.reset_index(drop=True, inplace=True)   

        # # Split data into train and test
        # train_size = int(len(stockData) * 0.7)
        # train, test = stockData.iloc[:train_size], stockData.iloc[train_size:]

        # # ARIMA model
        # arima_model = ARIMA(train["Close"], order=(1,1,1))  
        # arima_fit = arima_model.fit()
        # # arima_forecast = arima_fit.forecast(steps=len(test))
        # arima_forecast = arima_fit.get_forecast(steps=len(test)).predicted_mean


        # # ARIMAX model (adding exogenous variable if available, e.g., 'Volume')
        # if 'Volume' in stockData.columns:
        #     arimax_model = SARIMAX(train["Close"], exog=train["Volume"], order=(1,1,1))
        #     arimax_fit = arimax_model.fit()
        #     # arimax_forecast = arimax_fit.forecast(steps=len(test), exog=test["Volume"])
        #     arimax_forecast = arimax_fit.get_forecast(steps=len(test), exog=test["Volume"]).predicted_mean


        # else:
        #     print("No exogenous variable available for ARIMAX; using ARIMA instead.")
        #     arimax_forecast = arima_forecast

        # # SARIMA model
        # sarima_model = SARIMAX(train["Close"], order=(1,1,1), seasonal_order=(1,1,1,12))
        # sarima_fit = sarima_model.fit()
        # # sarima_forecast = sarima_fit.forecast(steps=len(test))
        # sarima_forecast = sarima_fit.get_forecast(steps=len(test)).predicted_mean

        # # Plot the results
        # plt.figure(figsize=(14,7))

        # # Actual data
        # plt.plot(train.index, train["Close"], label='Train', color='#203147')
        # plt.plot(test.index, test["Close"], label='Test', color='#01ef63')

        # # Forecasts
        # plt.plot(test.index, arima_forecast, label='ARIMA Forecast', color='orange', linestyle='--')
        # plt.plot(test.index, arimax_forecast, label='ARIMAX Forecast', color='blue', linestyle='-.')
        # plt.plot(test.index, sarima_forecast, label='SARIMA Forecast', color='red', linestyle=':')

        # # Title and labels
        # plt.title('Comparison of ARIMA, ARIMAX, and SARIMA Models')
        # plt.xlabel('Date')
        # plt.ylabel('Close Price')
        # plt.legend()
        # plt.show()
        # st.pyplot(plt)
        # #  Reset index AFTER forecasting
        stockData.reset_index(inplace=True)
        #$$$$$$$$$$$$$$$$$$$
    
        # Tab 2, part 3: LSTM

        # %%%%%%%%%%%%%%%%%%
        # Streamlit App Title
        st.subheader("LSTM Stock Price Prediction Dashboard")
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
        train_mse = mean_squared_error(y_train, train_predict)
        test_mse = mean_squared_error(y_test, test_predict)

        st.success(f"Training MSE: {train_mse:.4f}")
        st.success(f"Testing MSE: {test_mse:.4f}")

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

    # %%%%%%%%%%%%%%%%%%##################3
        # Tab 2, part 4 : ROC-AUC Evaluation using SVM, XGBoost and Logistic Regression
        st.markdown("""
###  ROC-AUC Evaluation for Stock Movement (Up/Down) Prediction

- In this classification task, prediction is maade on whether a stock's **closing price will go up the next day (1)** or **not (0)**.
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

    # ################################3
    # Tab 3: charts tab
    with chart_tab:
        
        st.write(f" ### {ticker_symbol} Charts with High, low, close prices and Volume (total no of shares traded) for the specified interval ")
        ########### 4a.

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
        st.write(f"Retrieving Financial Statements of {ticker_symbol} stocks")
        
        st.write("Balance Sheet:")
        st.write(ticker.balance_sheet.head())
        st.write("Financial Statements:")
        st.write(ticker.financials.head())

    
     # Tab 5: stock history tab   
    with EDA_tab:
        ###

        st.header(" Comparative EDA for Popular Stocks")
        tickers = ['AAPL','GOOGL','META','MSFT','TSLA','BTC-USD']

        st.info("Fetching and comparing data for: AAPL, GOOGL, META, MSFT, TSLA, BTC-USD")

        price_data = yf.download(tickers, start=start_date, end=end_date)['Close']
        returns = price_data.pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        risk = returns.std()
        mean_return = returns.mean()
        correlation = returns.corr()

        # -- 1. Cumulative Returns Line Chart
        st.subheader(" Cumulative Returns")
        fig1 = px.line(cumulative_returns, title="Cumulative Returns Over Time")
        st.plotly_chart(fig1)

        # -- 2. Risk vs Return Scatter
        st.subheader("Risk vs Return (Volatility vs Mean Return)")
        fig2 = px.scatter(x=risk, y=mean_return, text=risk.index,
                        labels={"x": "Risk (Std Dev)", "y": "Mean Return"},
                        title="Risk vs Return")
        fig2.update_traces(marker=dict(size=12), textposition='top center')
        st.plotly_chart(fig2)

        # -- 3. Bar Chart of Risk and Return
        st.subheader("Bar Chart of Risk and Mean Return")
        risk_return_df = pd.DataFrame({'Risk': risk, 'Mean Return': mean_return})
        st.dataframe(risk_return_df)

        fig3 = go.Figure(data=[
            go.Bar(name='Risk', x=risk.index, y=risk.values),
            go.Bar(name='Mean Return', x=mean_return.index, y=mean_return.values)
        ])
        fig3.update_layout(barmode='group', title="Risk and Mean Return for Each Ticker")
        st.plotly_chart(fig3)

        # -- 4. Correlation Heatmap
        st.subheader("Correlation Between Assets")
        fig4 = px.imshow(correlation, text_auto=True, title="Correlation Matrix of Daily Returns")
        st.plotly_chart(fig4)

        # -- 5. Up vs Down Days Confusion Matrix
        st.subheader("Confusion Matrix: Movement Consistency")

        for ticker in tickers:
            direction = np.where(returns[ticker] > 0, 1, 0)
            y_true = direction[:-1]
            y_pred = direction[1:]
            cm = confusion_matrix(y_true, y_pred)

            # Smaller figure size
            fig, ax = plt.subplots(figsize=(1, 1))  # width, height in inches

            # Heatmap with small annotation text
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=False,
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'],
                annot_kws={"size": 3},  # smaller numbers
                ax=ax
            )

            # Axis label and title font sizes
            ax.set_title(f'{ticker}', fontsize=5)
            ax.set_xlabel('Predicted', fontsize=5)
            ax.set_ylabel('Actual', fontsize=5)
            ax.tick_params(axis='both', labelsize=3)  # smaller tick labels

            # Render in Streamlit
            st.pyplot(fig)

        ####
else:
    st.error("No data found. Please choose the 'ticker symbol' and 'date range' of your choice from the Side bar.")
    st.stop()

