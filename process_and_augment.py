import os
import pdb
import cv2
from skimage import transform
import random
import numpy as np
import pickle
from config import dataset_config, image_config, augmentation_config, training_config
import matplotlib.pyplot as plt

# TODO: Make it a class?
# TODO: Add another augmentation technique

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

    final_dataset = []
    for image, label in zip(dataset_images, dataset_labels):
        for pair in image:
            final_dataset.append([pair, label])

    del dataset
    del matrix
    del images_names
    del dataset_images
    del dataset_labels

    return final_dataset


def augment_image(img_path):
    random_degree = random.randint(
        augmentation_config["min_rotation_degree"],
        augmentation_config["max_rotation_degree"],
    )

    rotated = transform.rotate(
        cv2.imread(img_path, -1), angle=-1 * random_degree, preserve_range=True
    ).astype(np.uint8)

    return rotated


def save_dataset(dataset: list):
    with open(
        dataset_config["save_dataset_path"],
        "wb",
    ) as f:
        pickle.dump(dataset, f)


def load_fingerprint():
    with open(dataset_config["save_dataset_path"], "rb") as f:
        dataset = pickle.load(f)
    for _ in range(0, dataset_config["number_of_shuffles"]):
        random.shuffle(dataset)

    print('Augmented and loaded sucessfully\nPassed Health Check.')
    
    return dataset



if __name__ == "__main__":
    fingerprint_dataset = process_dataset()
    save_dataset(fingerprint_dataset)
    dataset = load_fingerprint()
    print('Augmented sucessfully')

