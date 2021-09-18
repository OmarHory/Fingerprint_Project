from config import image_config, training_config
from process_images import ProcessImages
from model import multipleInputAlexNet
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf

mlflow.autolog()
# TODO:check mlflow how to track
# TODO: Save model after 10 epochs.


class TrainAndEvaluate(multipleInputAlexNet, ProcessImages):
    def __init__(self):
        self.process_obj = ProcessImages()
        self.network_obj = multipleInputAlexNet(
            image_config["height"],
            image_config["width"],
            image_config["channels"],
        )

    def train(self):
        model = self.network_obj.multiple_input_alex_net()
        model.compile(
            loss=training_config["loss"],
            optimizer=tf.optimizers.Adam(lr=training_config["lr"]),
            metrics=["accuracy"],
        )

        print(model.summary())

        with mlflow.start_run(run_name=training_config["run_name"]) as run:
            history = model.fit(
                self.process_obj.train_ds,
                epochs=training_config["epochs"],
                validation_data=self.process_obj.validation_ds,
                validation_freq=1,
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
