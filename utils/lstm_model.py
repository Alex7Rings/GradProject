import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

class LSTMVolatilityModel:
    def __init__(self, model_path: Path, input_window: int = 60):
        self.model_path = Path(model_path)
        self.input_window = input_window
        self.model = None

    def build(self):
        m = Sequential()
        m.add(LSTM(64, input_shape=(self.input_window, 1)))
        m.add(Dense(1, activation="linear"))
        m.compile(optimizer="adam", loss="mse")
        self.model = m
        return m

    def train(self, X, y, epochs: int = 50, batch_size: int = 32):
        if self.model is None:
            self.build()
        es = EarlyStopping(patience=5, restore_best_weights=True)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[es])
        self.model.save(self.model_path)

    def predict(self, X):
        if self.model is None:
            if self.model_path.exists():
                self.model = load_model(self.model_path)
            else:
                self.build()
        return self.model.predict(X).ravel()

    @staticmethod
    def make_sequences(series, window):
        x, y = [], []
        arr = np.asarray(series).astype(float)
        for i in range(len(arr) - window):
            x.append(arr[i:i+window])
            y.append(arr[i+window])
        x = np.expand_dims(np.array(x), -1)
        y = np.array(y)
        return x, y
