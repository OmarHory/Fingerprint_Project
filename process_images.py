import tensorflow as tf
import matplotlib.pyplot as plt
from config import image_config, training_config

# TODO: Add a function to make for contrastive learning.


class ProcessImages(object):
    def __init__(self):
        (train_images, train_labels), (
            test_images,
            test_labels,
        ) = tf.keras.datasets.cifar10.load_data()

        validation_images, validation_labels = (
            train_images[: training_config["val_size"]],
            train_labels[: training_config["val_size"]],
        )
        train_images, train_labels = (
            train_images[
                training_config["val_size"] : training_config["val_size"] + 5000
            ],
            train_labels[
                training_config["val_size"] : training_config["val_size"] + 5000
            ],
        )

        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

        validation_ds = tf.data.Dataset.from_tensor_slices(
            (validation_images, validation_labels)
        )

        train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
        test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
        validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
        print("Training data size:", train_ds_size)
        print("Test data size:", test_ds_size)
        print("Validation data size:", validation_ds_size)

        self.train_ds = (
            train_ds.map(self.process_images)
            .shuffle(buffer_size=train_ds_size)
            .batch(batch_size=training_config["batch_size"], drop_remainder=True)
        )

        self.test_ds = (
            test_ds.map(self.process_images)
            .shuffle(buffer_size=train_ds_size)
            .batch(batch_size=training_config["batch_size"], drop_remainder=True)
        )
        self.validation_ds = (
            validation_ds.map(self.process_images)
            .shuffle(buffer_size=train_ds_size)
            .batch(batch_size=training_config["batch_size"], drop_remainder=True)
        )

        # self.draw_num_samples(num=5)

    def draw_num_samples(self, num=5):
        plt.figure(figsize=(20, 20))
        for i, (image, label) in enumerate(self.train_ds.take(num)):
            ax = plt.subplot(num, num, i + 1)
            plt.imshow(image)
            plt.title(image_config["class_names"][label.numpy()[0]])
            plt.axis("off")
        plt.show()

    def process_images(self, image, label):
        # Normalize images to have a mean of 0 and standard deviation of 1
        image = tf.image.per_image_standardization(image)
        # Resize images from 32x32 to 277x277
        image = tf.image.resize(image, (image_config["height"], image_config["width"]))
        return image, label
