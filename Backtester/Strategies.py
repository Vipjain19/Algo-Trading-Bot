# -*- coding: utf-8 -*-
"""

@author: vipul
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import warnings

import ta
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ta.trend import EMAIndicator
#from ta.trend import SuperTrend
from ta.momentum import RSIIndicator
from sklearn.neighbors import KNeighborsClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical


warnings.filterwarnings("ignore")
plt.style.use("seaborn")

#1.

def mean_reversion_strategy(data, lookback_period=5, z_score_threshold=1.0, stop_loss=0.02, take_profit=0.05):
    # Calculate the rolling mean and standard deviation
    rolling_mean = data['Close'].rolling(window=lookback_period).mean()
    rolling_std = data['Close'].rolling(window=lookback_period).std()

    # Calculate the z-score
    z_score = (data['Close'] - rolling_mean) / rolling_std

    # Generate signals
    signals = pd.Series(index=data.index)
    signals[z_score > z_score_threshold] = 1.0  # Buy signal
    signals[z_score < -z_score_threshold] = -1.0  # Sell signal
    signals = signals.ffill().fillna(0.0)  # Fill NaN values with 0 (no signal)

    # Apply stop loss and take profit levels
    long_stop_loss = data['Close'] * (1 - stop_loss)
    short_stop_loss = data['Close'] * (1 + stop_loss)
    long_take_profit = data['Close'] * (1 + take_profit)
    short_take_profit = data['Close'] * (1 - take_profit)

    long_exit = (signals.shift(-1) == 1) & (data['Close'] <= long_stop_loss) | (data['Close'] >= long_take_profit)
    short_exit = (signals.shift(-1) == -1) & (data['Close'] >= short_stop_loss) | (data['Close'] <= short_take_profit)

    signals[long_exit] = 0.0
    signals[short_exit] = 0.0

    return signals


# Example usage:
# historical_data = ...  # Replace this with your DataFrame containing OHLC data
# backtest = run_mean_reversion_backtest(historical_data)
# result = bt.run(backtest)
# print(result)


#2.
def heiken_ashi_ichimoku_strategy(data):
    TenkanSenPeriods = 9
    KijunSenPeriods = 24
    SenkouSpanBPeriods = 51
    displacement = 24

    def donchian(series, window):
        return pd.concat([series.rolling(window=window).min(), series.rolling(window=window).max()], axis=1).mean(axis=1)

    data['hahigh'] = data['High'].rolling(2).max()
    data['halow'] = data['Low'].rolling(2).min()
    data['TenkanSen'] = donchian(data['hahigh'], TenkanSenPeriods)
    data['KijunSen'] = donchian(data['hahigh'], KijunSenPeriods)
    data['SenkouSpanA'] = (data['TenkanSen'] + data['KijunSen']) / 2
    data['SenkouSpanB'] = donchian(data['hahigh'], SenkouSpanBPeriods)
    data['SenkouSpanH'] = data[['SenkouSpanA', 'SenkouSpanB']].max(axis=1).shift(displacement)
    data['SenkouSpanL'] = data[['SenkouSpanA', 'SenkouSpanB']].min(axis=1).shift(displacement)
    data['ChikouSpan'] = data['Close'].shift(-displacement)

    data['longCondition'] = (
        (data['hahigh'] > data['hahigh'].shift(1)) &
        (data['hahigh'] > data['hahigh'].shift(2)) &
        (data['Close'] > data['ChikouSpan']) &
        (data['Close'] > data['SenkouSpanH']) &
        ((data['TenkanSen'] >= data['KijunSen']) | (data['Close'] > data['KijunSen']))
    )

    data['shortCondition'] = (
        (data['halow'] < data['halow'].shift(1)) &
        (data['halow'] < data['halow'].shift(2)) &
        (data['Close'] < data['ChikouSpan']) &
        (data['Close'] < data['SenkouSpanL']) &
        ((data['TenkanSen'] <= data['KijunSen']) | (data['Close'] < data['KijunSen']))
    )
    
    # Position management
    data['position'] = 0
    in_long_position = False
    in_short_position = False

    for index, row in data.iterrows():
        if row['longCondition']:
            if not in_long_position:
                data.loc[index, 'position'] = 1
                in_long_position = True
                in_short_position = False
        elif row['shortCondition']:
            if not in_short_position:
                data.loc[index, 'position'] = -1
                in_short_position = True
                in_long_position = False
        elif in_long_position:
            data.loc[index, 'position'] = 1
        elif in_short_position:
            data.loc[index, 'position'] = -1

    return data


#3.
def hull_ma_pack_strategy(data):
    def HMA(src, length):
        return pd.Series(2 * pd.Series.ewm(src, span=length // 2, min_periods=1).mean() - pd.Series.ewm(src, span=length, min_periods=1).mean(), name="HMA")

    def EHMA(src, length):
        return pd.Series(2 * pd.Series.ewm(src, span=length // 2, min_periods=1).mean() - pd.Series.ewm(src, span=length, min_periods=1).mean(), name="EHMA")

    def THMA(src, length):
        return pd.Series(pd.Series.rolling(src, window=length // 3).mean() * 3 - pd.Series.rolling(src, window=length // 2).mean() - pd.Series.rolling(src, window=length).mean(), name="THMA")

    def Mode(modeSwitch, src, length):
        if modeSwitch == "Hma":
            return HMA(src, length)
        elif modeSwitch == "Ehma":
            return EHMA(src, length)
        elif modeSwitch == "Thma":
            return THMA(src, length // 2)
        else:
            return pd.Series(np.nan)

    modeSwitch = "Hma"  # You can change this to "Thma" or "Ehma" if needed
    length = 55  # You can change this to your desired length

    data['HULL'] = Mode(modeSwitch, data['Close'], length)
    data['MHULL'] = data['HULL'].shift(1)
    data['SHULL'] = data['HULL'].shift(2)

    data['position'] = np.where(data['HULL'] > data['SHULL'], 1, np.where(data['HULL'] < data['SHULL'], -1, 0))

    return data


#4.Preprocessing for ML strategies.

def preprocess_data(data):
    # Calculate RSI
    # data['rsi'] = RSI(data['close'])

    # Calculate MACD
    macd = MACD(data['Close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()

    # Calculate Bollinger Bands
    bb = BollingerBands(data['Close'])
    data['bb_upper'] = bb.bollinger_hband()
    data['bb_lower'] = bb.bollinger_lband()

    return data

def generate_positions(data):
    # Create the target variable "position" based on the current and next day's closing price
    data['next_close'] = data['Close'].shift(-1)
    data['position'] = np.where(data['next_close'] > data['Close'], 1, np.where(data['next_close'] < data['Close'], -1, 0))
    data.drop(columns=['next_close'], inplace=True)
    
    return data


#5.

def intraday_random_forest_strategy(data):
    # Drop NaN values resulting from technical indicator calculations
    data.dropna(inplace=True)

    # Create the feature matrix X and the target vector y
    X = data[['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']].values
    y = data['position'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Train the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = clf.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Model Accuracy:', accuracy)

    # Make predictions on the entire dataset and create the "position" column
    data['position'] = clf.predict(X)
    
    return data


#6.

def intraday_svm_strategy(data):
    # Drop NaN values resulting from technical indicator calculations
    data.dropna(inplace=True)

    # Create the feature matrix X and the target vector y
    X = data[['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']].values
    y = data['position'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Support Vector Machine classifier
    clf = SVC(kernel='rbf', random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = clf.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Model Accuracy:', accuracy)

    # Make predictions on the entire dataset and create the "position" column
    data['position'] = clf.predict(X)

    return data


#7.

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Multi-class classification, so output layer has num_classes neurons with softmax activation

    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def intraday_dl_strategy(data):
    # Drop NaN values resulting from technical indicator calculations
    data.dropna(inplace=True)

    # Create the feature matrix X and the target vector y
    X = data[['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']].values
    y = data['position'].values

    # Encode the target variable to categorical one-hot vectors
    label_to_index = {label: index for index, label in enumerate(np.unique(y))}
    y_encoded = np.array([label_to_index[label] for label in y])
    y_categorical = to_categorical(y_encoded)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # Create the Deep Learning model
    num_classes = len(np.unique(y))
    model = create_model(input_shape=(X_train.shape[1],), num_classes=num_classes)

    # Train the model on the training dataset
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Evaluate the model's accuracy on the testing dataset
    _, accuracy = model.evaluate(X_test, y_test)
    print('Model Accuracy:', accuracy)

    # Make predictions on the entire dataset and create the "position" column
    predictions = model.predict(X)
    data['position'] = [np.argmax(pred) for pred in predictions]
    data['position'] = data['position'].map({index: label for label, index in label_to_index.items()})

    return data

#8.

def generate_knn_signals(data, window):
    # Generate features: RSI and price change
    #data['rsi'] = calculate_rsi(data['Close'])
    data['price_change'] = data['Close'].pct_change()

    # Create target variable: 1 for buy, -1 for sell
    data['target'] = np.where(data['price_change'] > 0, 1, -1)

    # Drop rows with NaN in RSI and target columns
    data.dropna(subset=['rsi', 'target'], inplace=True)

    # Scale features
    scaler = StandardScaler()
    data[['rsi', 'price_change']] = scaler.fit_transform(data[['rsi', 'price_change']])

    # Create the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=window)

    # Split the data into features (X) and target (y)
    X = data[['rsi', 'price_change']].values
    y = data['target'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the KNN classifier
    knn.fit(X_train, y_train)

    # Generate predictions using the KNN classifier
    data['knn_signal'] = knn.predict(X)

    return data

def apply_ema_ribbon(data, short_window, long_window):
    # Calculate short-term and long-term EMAs
    data['ema_short'] = EMAIndicator(data['Close'], window=short_window).ema_indicator()
    data['ema_long'] = EMAIndicator(data['Close'], window=long_window).ema_indicator()

    # Create EMA ribbon signals: 1 for buy (short EMA above long EMA), -1 for sell (short EMA below long EMA)
    data['ema_ribbon_signal'] = np.where(data['ema_short'] > data['ema_long'], 1, -1)

    # Filter out KNN signals where EMA ribbon signal is opposite
    data['filtered_signal'] = data['knn_signal'] * data['ema_ribbon_signal']

    return data

def apply_rsi_confirmation(data, rsi_threshold):
    # Calculate RSI
    # data['rsi'] = calculate_rsi(data['Close'])

    # Create RSI confirmation signals: 1 for strong buy (RSI above threshold), -1 for strong sell (RSI below threshold)
    data['rsi_confirmation'] = np.where(data['rsi'] > rsi_threshold, 1, -1)

    # Final position is the product of the filtered signal and RSI confirmation
    data['position'] = data['filtered_signal'] * data['rsi_confirmation']

    return data

#9.More features.

def generate_features(data):
    #Calculate SuperTrend
    data["atr"] = ta.volatility.AverageTrueRange(data["High"], data["Low"], data["Close"], window=14).average_true_range()
    data["upper_band"] = data["High"] + 3.0 * data["atr"]
    data["lower_band"] = data["Low"] - 3.0 * data["atr"]
    data["in_uptrend"] = data["Close"] > data["lower_band"].shift()
    data["super_trend"] = np.where(data["in_uptrend"], data["lower_band"], data["upper_band"])
    
    # Calculate EMA Ribbon
    data["ema_20"] = EMAIndicator(data["Close"], window=20).ema_indicator()
    data["ema_40"] = EMAIndicator(data["Close"], window=40).ema_indicator()
    data["ema_60"] = EMAIndicator(data["Close"], window=60).ema_indicator()
    data["ema_80"] = EMAIndicator(data["Close"], window=80).ema_indicator()
    data["ema_100"] = EMAIndicator(data["Close"], window=100).ema_indicator()
    
    ema_conditions1 = (
        (data["Close"] > data["ema_20"]) &
        (data["Close"] > data["ema_40"]) &
        (data["Close"] > data["ema_60"]) 
    )
    ema_conditions2 = (
        (data["Close"] < data["ema_20"]) &
        (data["Close"] < data["ema_40"]) &
        (data["Close"] < data["ema_60"]) 
    )

    data["ema_ribbon"] =  0
    data.loc[ema_conditions1,'ema_ribbon'] = -1
    data.loc[ema_conditions2,'ema_ribbon'] = 1
    
    
    

    # Calculate RSI
    data["rsi"] = RSIIndicator(data["Close"], window=14).rsi()

    return data

