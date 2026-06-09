from tensorflow import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_flat = x_train.reshape(x_train.shape[0], -1).astype("float32") / 255.0
y_train_ohe = keras.utils.to_categorical(y_train, 10)

model = keras.Sequential([
    keras.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("Starting fit...")
model.fit(x_train_flat, y_train_ohe, epochs=1, batch_size=64, validation_split=0.1, verbose=2)
print("Finished fit.")

import sklearn.metrics
print("Imported sklearn")
