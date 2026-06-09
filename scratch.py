import numpy as np
from tensorflow import keras

model = keras.models.load_model("saved_model/mnist_ann.keras")

# Test empty black image
empty_black = np.zeros((1, 784))
pred = model.predict(empty_black, verbose=0)
print("Empty black prediction:", np.argmax(pred[0]), "Probs:", pred[0])

# Test completely white image
empty_white = np.ones((1, 784))
pred = model.predict(empty_white, verbose=0)
print("Empty white prediction:", np.argmax(pred[0]), "Probs:", pred[0])
