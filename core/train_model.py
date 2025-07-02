import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


def train_emotion_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(7, activation="softmax"),  # 7 emotion classes
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    print(
        "Model defined. Please download and load FER2013 data manually if training from scratch."
    )
    model.save("core/emotion_model/model.h5")


# Uncomment to train
# train_emotion_model()
