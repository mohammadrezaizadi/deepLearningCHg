import os
from datetime import timedelta, datetime, date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.models import load_model
from keras.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tradeHelper as th

# region params

SymbolName = "EURUSD_i"
timeframe = 'm1'
FetchDaysOfSymbol = 25
TrainingNumber: int = 20000
NumberOfFuturesData = 30
ModelSizeForFutures = 10000
ModelSize = 500
ModelUnits = 50
ModelEpochs = 2
ModelBatchSize = 1
ModelVerbs = 2
DenseUnit = 25
IsTrainTestData = True

# endregion


# region fetch Forex data
t = th.Getsymbols(SymbolName)
today = datetime.combine(date.today(), datetime.min.time())
th.login()
st = today + timedelta(days=-FetchDaysOfSymbol)
c = th.GetCandels(t[0].name, st, today, timeframe)
th.shotdown()
df = pd.DataFrame(c)  # تمام داده ها گرفته شده
df = df.rename(columns={'close': 'Close'})

df["Date"] = pd.to_datetime(df.time, unit='s')
df.index = df['Date']
Mytest = df[-NumberOfFuturesData:]
df = df[:-NumberOfFuturesData]

# endregion

# region cleaning Data


data = df.sort_index(ascending=True, axis=0)
new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_dataset["Date"].iloc[i] = data['Date'].iloc[i]
    new_dataset["Close"].iloc[i] = data["Close"].iloc[i]

new_dataset.index = new_dataset.Date
new_dataset.drop("Date", axis=1, inplace=True)

final_dataset = new_dataset.values

# endregion

# region split train and test data

train_data = final_dataset[0:TrainingNumber, :]
valid_data = final_dataset[TrainingNumber:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_dataset)
x_train_data, y_train_data = [], []

for i in range(ModelSize, len(train_data)):
    x_train_data.append(scaled_data[i - ModelSize:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

# endregion

# region train model

lstm_model = Sequential()
lstm_model.add(LSTM(units=ModelUnits, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
lstm_model.add(LSTM(units=ModelUnits, return_sequences=True))
lstm_model.add(LSTM(units=ModelUnits, return_sequences=True))
lstm_model.add(LSTM(units=ModelUnits, return_sequences=True))
lstm_model.add(LSTM(units=ModelUnits, return_sequences=True))
lstm_model.add(LSTM(units=ModelUnits, return_sequences=True))
lstm_model.add(LSTM(units=ModelUnits))
lstm_model.add(Dense(1))
# lstm_model.add(LSTM(ModelUnits, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
# lstm_model.add(LSTM(ModelUnits, return_sequences=True))
# # lstm_model.add(LSTM(ModelUnits, return_sequences=True))
# lstm_model.add(LSTM(ModelUnits, return_sequences=False))
# lstm_model.add(Dense(DenseUnit))
# lstm_model.add(Dense(1))

path = 'saved_lstm_model.h5'
if os.path.isfile(path):
    lstm_model = load_model(path)
else:
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(x_train_data, y_train_data, epochs=ModelEpochs, batch_size=ModelBatchSize, verbose=ModelVerbs)
    lstm_model.save(path)

# endregion

# region train test data

if IsTrainTestData:
    inputs_data = final_dataset[len(final_dataset) - len(valid_data) - ModelSize:]
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)

    X_test = []
    for i in range(ModelSize, inputs_data.shape[0]):
        X_test.append(inputs_data[i - ModelSize:i, 0])

    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    closing_price = lstm_model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

# endregion

# region check model
# محاسبه معیارهای ارزیابی (در اینجا میانگین مربعات خطا)
if IsTrainTestData:
    mse = 0
    try:
        mse = mean_squared_error(new_dataset[TrainingNumber:]['Close'], closing_price)
        print(f'Mean Squared Error: {mse}')
    finally:
        mse = 0


# endregion

# region futures

def predict_future(model, data, n_steps, future_steps):
    inputs_data = data[-n_steps:].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)
    last_sequence = inputs_data
    predictions = []
    for _ in range(future_steps):
        last_sequence = np.reshape(last_sequence, (1, n_steps, 1))
        prediction = model.predict(last_sequence)
        p = np.reshape(prediction, (1, 1, 1))
        last_sequence = np.roll(last_sequence, shift=-1, axis=1)  # جابه‌جایی داده‌های ورودی
        # last_sequence = np.concatenate((last_sequence[:, 1:, :], p), axis=1)
        predictions.append(prediction)

    predictions = np.array(predictions)
    predictions = np.reshape(predictions, (-1, NumberOfFuturesData))
    predictions = np.reshape(predictions, (NumberOfFuturesData, -1))
    predictions = scaler.inverse_transform(predictions)

    return predictions


future_predictions = predict_future(lstm_model, new_dataset, ModelSizeForFutures, NumberOfFuturesData)
mse2 = mean_squared_error(new_dataset[-NumberOfFuturesData:]['Close'], future_predictions)
print(f'Mean Squared Error 2 : {mse2}')


# endregion

# region   plot
train_data = new_dataset[:TrainingNumber]
valid_data = new_dataset[TrainingNumber:]
if IsTrainTestData:
    valid_data['Predictions'] = closing_price
    valid_data['Predictions'] =  valid_data['Predictions'].shift(-1)
Mytest['Predictions'] = future_predictions
# Mytest['Predictions'] = Mytest['Predictions'].shift(1)

# plt.figure(figsize=(8, 4))
plt.plot(valid_data[['Close']], c='g', label='train')
if IsTrainTestData:
    plt.plot(valid_data[["Predictions"]], c='r', label='predict')
plt.plot(Mytest[['Close']], c='b', label='real')
plt.plot(Mytest[['Predictions']], c='c', label='future')
# rcParams['figure.figsize'] = 20, 10
plt.legend()
plt.show()
# endregion
