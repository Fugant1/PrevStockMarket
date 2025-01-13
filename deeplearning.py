import pandas as pd
import numpy as np
import keras
import dados

def function(success):
# Carregar o dataset
    df = success
    # Exemplo: Supondo que a coluna 'target' seja o rótulo e o resto sejam as features
    X = df.drop("YearMonth", axis=1).values  # Entradas (features) como NumPy array
    y = df["Close"].values

    from sklearn.model_selection import train_test_split

    # Divisão em treino (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()  # Normalizar entre 0 e 1
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    from keras.utils import to_categorical

    num_classes = int(max(y_train)+ 1)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    from keras.models import Sequential
    from keras.layers import Dense

    # Construir um modelo simples
    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),  # Camada de entrada
        Dense(64, activation="relu"),     
        Dense(32, activation="relu"),                                   # Camada oculta
        Dense(num_classes, activation="softmax")                        # Camada de saída 
    ])

    # Compilar o modelo
    model.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

    # Treinar o modelo
    history = model.fit(X_train, y_train, epochs=1000, batch_size=256, validation_data=(X_test, y_test)) #~93% de accuracy

    # Avaliação no conjunto de teste
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    # Fazer previsões
    predictions = model.predict(X_test)

if __name__ == '__main__':
    while True:
        ticker_symbol = input("Escreva o Ticker da ação: ")
        flag = False
        success = dados.plot_monthly_mean_closing_prices(ticker_symbol, flag)
        function(success)
        if flag:
            choice = input("Nada foi encontrado, gostaria de tentar novamente? (y/n): ")
            if choice.lower() != 'y':
                break
        else:
            break