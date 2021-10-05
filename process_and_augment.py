import os
import pdb
import cv2
from skimage import transform
import random
import numpy as np
import pickle
from config import dataset_config, image_config, augmentation_config, training_config
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gc

# TODO: Make it a class?
# TODO: Add another augmentation technique
# TODO: OPTIMIIIZE CODE


def process_dataset():
    images_names = []
    dataset = {}
    dataset_images = []
    dataset_labels = []

    for image_name in os.listdir(dataset_config["path"]):
        if image_name.split("_")[1].split(".")[0] == "1":
            images_names.append(image_name)

    for image in images_names:
        dataset[image] = [
            augment_image(os.path.join(dataset_config["path"], image))
            for _ in range(0, augmentation_config["num_augmentations"])
        ]

    for key in dataset:
        all_pairs = [
            (dataset[key][i], dataset[key][j])
            for i in range(len(dataset[key]))
            for j in range(i + 1, len(dataset[key]))
        ]

        dataset_images.append(all_pairs)
        dataset_labels.append(image_config["class_names"][0])

    matrix = []

    for key in dataset:
        matrix.append(dataset[key])

    for i in range(0, len(matrix) - 1):
        all_pairs = [
            (matrix[i][i], matrix[i + 1][j])
            for i in range(len(matrix[i]))
            for j in range(len(matrix[i + 1]))
        ]

        dataset_images.append(all_pairs)
        dataset_labels.append(image_config["class_names"][1])

    whole_dataset = []
    for image, label in zip(dataset_images, dataset_labels):
        for pair in image:
            pair = (
                cv2.resize(pair[0], ((image_config["height"], image_config["width"]))),
                cv2.resize(pair[1], ((image_config["height"], image_config["width"]))),
            )
            pair = (
                cv2.normalize(
                    pair[0],
                    None,
                    alpha=0,
                    beta=1,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_32F,
                ),
                cv2.normalize(
                    pair[1],
                    None,
                    alpha=0,
                    beta=1,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_32F,
                ),
            )
            whole_dataset.append([pair, label])
    print("Finished resizing and normalizing.")

    gc.collect()



    del dataset
    del matrix
    del images_names
    del dataset_images
    del dataset_labels

    return whole_dataset


def augment_image(img_path):
    random_degree = random.randint(
        augmentation_config["min_rotation_degree"],
        augmentation_config["max_rotation_degree"],
    )

    rotated = transform.rotate(
        cv2.imread(img_path, -1), angle=-1 * random_degree, preserve_range=True
    ).astype(np.uint8)

    return rotated


def save_dataset(dataset):
    with open(dataset_config["save_dataset_path"], "wb") as f:
        pickle.dump(dataset, f)


def load_fingerprint():
    with open(dataset_config["save_dataset_path"], "rb") as f:
        return pickle.load(f)


def train_val_test_split(dataset):
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


if __name__ == "__main__":
    dataset = process_dataset()
    print('Finished processing')
    gc.collect()

    save_dataset(
        dataset
    )
    print('Saved')
