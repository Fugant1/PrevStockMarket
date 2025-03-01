import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import dados
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def function(success):
    # Carregar o dataset
    df = success

    # Separar features (X) e rótulos (y)
    X = df.drop("Close", axis=1).values  # Remove a coluna 'Close' para obter as features
    y = df["Close"].values  # Usa 'Close' como rótulo

    # Divisão em treino (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar as features
    scaler = MinMaxScaler()  # Normalizar entre 0 e 1
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Construir o modelo
    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),  # Camada de entrada
        Dense(64, activation="relu"),  # Camada oculta
        Dense(32, activation="relu"),  # Camada oculta
        Dense(1)  # Camada de saída para regressão
    ])

    # Compilar o modelo
    model.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss="mean_squared_error", metrics=["mae"])

    # Treinar o modelo
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    # Avaliação no conjunto de teste
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Loss: {loss}, MAE: {mae}")

    # Fazer previsões
    predictions = model.predict(X_test)
    print("Previsões:", predictions.flatten())

if __name__ == '__main__':
    while True:
        ticker_symbol = input("Escreva o Ticker da ação: ")
        flag = False
        success = dados.plot_monthly_mean_closing_prices(ticker_symbol, flag)
        if success is not None:
            function(success)
            break
        else:
            choice = input("Nada foi encontrado, gostaria de tentar novamente? (y/n): ")
            if choice.lower() != 'y':
                break