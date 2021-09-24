import tensorflow as tf
import matplotlib.pyplot as plt
from config import image_config, training_config, dataset_config
import pickle
import random
import numpy as np


class ProcessImages(object):
    def __init__(self):

        dataset = self.__load_fingerprint()
        (
            (train_images, train_labels),
            (validation_images, validation_labels),
            (test_images, test_labels),
        ) = self.__train_val_test_split(dataset)

        train_ds = tf.data.Dataset.from_tensor_slices(
            (np.array(train_images), np.array(train_labels))
        )
        validation_ds = tf.data.Dataset.from_tensor_slices(
            (np.array(validation_images), np.array(validation_labels))
        )
        test_ds = tf.data.Dataset.from_tensor_slices(
            (np.array(test_images), np.array(test_labels))
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

    @staticmethod
    def __load_fingerprint():
        with open(dataset_config["save_dataset_path"], "rb") as f:
            dataset = pickle.load(f)
        for _ in range(0, dataset_config["number_of_shuffles"]):
            random.shuffle(dataset)
        return dataset

    @staticmethod
    def __train_val_test_split(dataset):
        # [(p1,p2), labels]

        images = []
        labels = []

        for example in dataset:
            images.append(example[0])
            labels.append(example[1])

        training_images = images[: training_config["train_size"]]
        training_labels = labels[: training_config["train_size"]]

        val_images = images[
            training_config["train_size"] : training_config["train_size"]
            + training_config["val_size"]
        ]
        val_labels = labels[
            training_config["train_size"] : training_config["train_size"]
            + training_config["val_size"]
        ]

        test_images = images[
            training_config["train_size"]
            + training_config["val_size"] : training_config["train_size"]
            + training_config["val_size"]
            + training_config["test_size"]
        ]
        test_labels = labels[
            training_config["train_size"]
            + training_config["val_size"] : training_config["train_size"]
            + training_config["val_size"]
            + training_config["test_size"]
        ]

        del images
        del labels

        return (
            (training_images, training_labels),
            (val_images, val_labels),
            (test_images, test_labels),
        )

    def draw_num_samples(self, num=5):
        plt.figure(figsize=(20, 20))
        for i, (image, label) in enumerate(self.train_ds.take(num)):
            ax = plt.subplot(num, num, i + 1)
            plt.imshow(image)
            plt.title(image_config["class_names"][label.numpy()[0]])
            plt.axis("off")
        plt.show()

    def process_images(self, image, label):
        image1 = image[0]
        image2 = image[1]
        image1 = tf.reshape(image1, [image1.shape[0], image1.shape[1], 1])
        image2 = tf.reshape(image2, [image2.shape[0], image2.shape[1], 1])

        # Normalize images to have a mean of 0 and standard deviation of 1
        image1 = tf.image.per_image_standardization(image1)
        image2 = tf.image.per_image_standardization(image2)

        # Resize images from 32x32 to 277x277
        image1 = tf.image.resize(
            image1, (image_config["height"], image_config["width"])
        )
        image2 = tf.image.resize(
            image2, (image_config["height"], image_config["width"])
        )

        return (image1, image2), label
