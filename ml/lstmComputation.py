import numpy as np
import pandas as pd
from keras_tuner.src.backend.io import tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import keras_tuner as kt


class LSTMVolatilityModel:
    def __init__(self, lookback=60, grid_search=False):
        self.lookback = lookback
        self.grid_search = grid_search
        self.scaler = MinMaxScaler()

    def train(self, returns):
        scaled_data = self.scaler.fit_transform(returns)
        X, y = self._create_sequences(scaled_data)
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        if self.grid_search:
            tuner = kt.GridSearch(
                self._build_model,
                objective='val_loss',
                max_trials=10,
                directory='tuner',
                project_name='lstm_volatility'
            )
            tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val),
                         callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])
            self.model = tuner.get_best_models(num_models=1)[0]
            self.best_params = tuner.get_best_hyperparameters()[0].values
        else:
            self.model = self._build_model()
            self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                           callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])
            self.best_params = {'units': 50, 'layers': 1, 'dropout': 0.2, 'learning_rate': 0.001}

        return self.best_params

    def predict_vol_multi_period(self, last_sequence, periods):
        forecasts = []
        current_sequence = last_sequence.copy()

        for _ in range(max(periods)):
            scaled_sequence = self.scaler.transform(current_sequence)
            X = np.array([scaled_sequence[-self.lookback:]])
            pred = self.model.predict(X, verbose=0)
            forecasts.append(np.sqrt(np.mean(np.square(pred[0]))))
            current_sequence = np.vstack((current_sequence, pred))

        return [forecasts[p - 1] for p in periods]

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def _build_model(self, hp=None):
        model = Sequential()
        if hp:
            units = hp.Int('units', min_value=50, max_value=150, step=50)
            layers = hp.Int('layers', min_value=1, max_value=2)
            dropout = hp.Float('dropout', min_value=0.1, max_value=0.3, step=0.1)
            learning_rate = hp.Choice('learning_rate', values=[0.001, 0.01])
        else:
            units, layers, dropout, learning_rate = 50, 1, 0.2, 0.001

        for i in range(layers):
            if i == 0:
                model.add(LSTM(units=units, return_sequences=(layers > 1), input_shape=(self.lookback, 1)))
            else:
                model.add(LSTM(units=units))
            model.add(Dropout(dropout))

        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
        return model