    # Tab 2: Stock Data for Prediction
    # Tab 2, part 1 : Rolling window
with prediction_tab:
        # Tab 2, part 1 : Rolling window
    st.subheader("Stock Prediction using Time series - ARIMA/SARIMA/ARIMAX Models")
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

    #$$$$$$$$$$$$$$$$$$$$$4
    # Tab 2 , part 2: ARIMA
    # Split data into train and test
    train_size = int(len(stockData) * 0.7)
    train, test = stockData.iloc[:train_size], stockData.iloc[train_size:]

    # # ARIMA model
    # arima_model = ARIMA(train["Close"], order=(1,1,1))  
    # arima_fit = arima_model.fit()
    # arima_forecast = arima_fit.forecast(steps=len(test))

    # # ARIMAX model (adding exogenous variable if available, e.g., 'Volume')
    # if 'Volume' in stockData.columns:
    #     arimax_model = SARIMAX(train["Close"], exog=train["Volume"], order=(1,1,1))
    #     arimax_fit = arimax_model.fit()
    #     arimax_forecast = arimax_fit.forecast(steps=len(test), exog=test["Volume"])
    # else:
    #     print("No exogenous variable available for ARIMAX; using ARIMA instead.")
    #     arimax_forecast = arima_forecast

    # # SARIMA model
    # sarima_model = SARIMAX(train["Close"], order=(1,1,1), seasonal_order=(1,1,1,12))
    # sarima_fit = sarima_model.fit()
    # sarima_forecast = sarima_fit.forecast(steps=len(test))
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

