from config import image_config, training_config
from process_images import ProcessImages
from model import multipleInputAlexNet
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

mlflow.autolog()

# TODO:check mlflow how to track


class TrainAndEvaluate(multipleInputAlexNet, ProcessImages):
    def __init__(self):

        self.process_obj = ProcessImages()
        self.network_obj = multipleInputAlexNet(
            image_config["height"],
            image_config["width"],
            image_config["channels"],
        )
        exp_path = os.path.join(
            training_config["experiment_path"], training_config["experiment_name"]
        )
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)

    def train(self):
        model = self.network_obj.multiple_input_alex_net()
        model.compile(
            loss=training_config["loss"],
            optimizer=tf.optimizers.Adam(lr=training_config["lr"]),
            metrics=["accuracy"],
        )

        print(model.summary())
        print(self.process_obj.train_ds)
        with mlflow.start_run(run_name=training_config["run_name"]) as run:
            history = model.fit(
                self.process_obj.train_ds,
                epochs=training_config["epochs"],
                validation_data=self.process_obj.validation_ds,
                batch_size=training_config["batch_size"],
                verbose=1,
                callbacks=[self.callbacks()],
            )
            self.plot_loss(history)
            accuracy_custom = model.evaluate(self.process_obj.test_ds)[1]
            mlflow.log_metric("accuracy_custom", accuracy_custom)
            print(model.evaluate(self.process_obj.test_ds))

    @staticmethod
    def plot_loss(history):
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def callbacks():
        # location = "experimentssaved-model-{epoch:02d}-{val_loss:.2f}.hdf5"
        exp_dir = os.path.join(
            training_config["experiment_path"], training_config["experiment_name"]
        )
        temp = "saved-model-{epoch:02d}-{val_loss:.2f}.hdf5"
        location = os.path.join(exp_dir, temp)
        rlr = ReduceLROnPlateau(
            monitor="val_loss", patience=training_config["patience"], min_lr=0.0005
        )
        checkpoint = ModelCheckpoint(
            temp,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="min",
        )
        early_stop = EarlyStopping(
            monitor="val_loss", patience=training_config["patience"], mode="auto"
        )
        return [rlr, checkpoint, early_stop]
