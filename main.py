# основной модуль
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical, plot_model


folder_path = "crop/"
img_size = (29, 22)
epoch_num = 5

# X - images, Y - labels
X = np.zeros((69000, 29 * 22))
Y = np.zeros((69000,))

loaded = np.load("XY.npz")
X = loaded["x"]
Y = loaded["y"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=123
)

# one-hot encoding
Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)

# параметры сети
batch_size = 128

X_train = X_train.reshape(X_train.shape[0], 29, 22, 1)
X_test = X_test.reshape(X_test.shape[0], 29, 22, 1)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(29, 22, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(Y_train.shape[1], activation="softmax"))
model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=[
        "accuracy",
        "mean_absolute_percentage_error",
        "mean_absolute_error",
        "mean_squared_error",
    ],
)

train_acc = []
train_mape = []
train_mae = []
train_mse = []

history = model.fit(X_train, Y_train, epochs=epoch_num, batch_size=batch_size)

train_acc.extend(history.history["accuracy"])
train_mape.extend(history.history["mean_absolute_percentage_error"])
train_mae.extend(history.history["mean_absolute_error"])
train_mse.extend(history.history["mean_squared_error"])
loss = history.history["loss"]

evaluation = model.evaluate(X_test, Y_test, batch_size=batch_size)
print(f"Test accuracy: {100 * evaluation[1]}")

epochs = range(1, epoch_num + 1)

# График точности
plt.plot(epochs, train_acc, "b")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.show()

# График MAPE
plt.plot(epochs, train_mape, "r")
plt.xlabel("Epochs")
plt.ylabel("MAPE")
plt.title("Training MAPE")
plt.show()

# График MAE
plt.plot(epochs, train_mae, "g")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.title("Training MAE")
plt.show()

# График MSE
plt.plot(epochs, train_mse, "m")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Training MSE")
plt.show()

# График loss
plt.plot(epochs, loss, "b")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
