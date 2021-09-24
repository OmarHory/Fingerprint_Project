import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
    concatenate,
)
from tensorflow.keras import Model
from config import training_config


class multipleInputAlexNet(object):
    def __init__(self, height, width, channels):
        self.height = height
        self.width = width
        self.channels = channels

    def multiple_input_alex_net(self):
        input_A = Input(shape=(self.height, self.width, self.channels))
        model_A = self.create_convolutional_layers(input_A)
        input_B = Input(shape=(self.height, self.width, self.channels))
        model_B = self.create_convolutional_layers(input_B)
        combined_vector = concatenate([model_A, model_B])

        X = Flatten()(combined_vector)
        X = Dense(4096, activation="relu")(X)
        X = Dropout(0.5)(X)
        X = Dense(4096, activation="relu")(X)
        X = Dropout(0.5)(X)
        output = Dense(1, activation="softmax")(X)

        model = Model(inputs=[input_A, input_B], outputs=[output])
        print(model.summary())

        return model

    def create_convolutional_layers(self, input_img):
        X = Conv2D(
            filters=96,
            kernel_size=(11, 11),
            strides=(4, 4),
            activation="relu",
            input_shape=(self.height, self.width, self.channels),
        )(input_img)

        X = BatchNormalization()(X)
        X = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(X)
        X = Conv2D(
            filters=256,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(X)
        X = BatchNormalization()(X)
        X = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(X)
        X = Conv2D(
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(X)
        X = BatchNormalization()(X)
        X = Conv2D(
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(X)
        X = BatchNormalization()(X)
        X = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(X)
        X = BatchNormalization()(X)
        X = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(X)

        return X

    def original_alex_net(self):
        return tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=96,
                    kernel_size=(11, 11),
                    strides=(4, 4),
                    activation="relu",
                    input_shape=(self.height, self.width, self.channels),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=256,
                    kernel_size=(5, 5),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=384,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=384,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=256,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
