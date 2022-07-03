import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

def lstm_Model(units,trainX, trainY, epochs, batch_size, verbose, values):
    model = Sequential()
    model.add(LSTM(units=units[0], return_sequences=True, input_shape=values.shape[1:]))
    model.add(Dropout(0.2))

    model.add(LSTM(units=units[1], return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=units[2]))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))  # Prediction of the next closing value

    model.compile(optimizer="adam", loss="mean_squared_error")
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
    plt.Figure(figsize=(8,5))
    plt.plot(history.history["loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Model accuracy")
    return model