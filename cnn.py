import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical

# Initialize parameter.
classes = ["car", "motorcycle"]
num_classes = len(classes)
image_size = 150

image_files_data = np.load("./imagefiles.npz")
X_train, X_test, Y_train, Y_test = (
    image_files_data["X_train"],
    image_files_data["X_test"],
    image_files_data["Y_train"],
    image_files_data["Y_test"],
)

Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

model = Sequential()
model.add(
    Conv2D(32, (3, 3), activation="relu", input_shape=(image_size, image_size, 3))
)
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

opt = Adam()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

model.fit(X_train, Y_train, batch_size=32, epochs=20)

score = model.evaluate(X_test, Y_test, batch_size=32)
