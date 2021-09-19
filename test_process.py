
import os
import cv2
from skimage import transform

# from skimage.util import crop
import random
import numpy as np

path = "/home/omar/Desktop/fingerprint_dataset/second_session/"
NUM_AUGMENTATIONS = 12
MAX_DEGREE = 359
MIN_DEGREE = 0
def process_dataset():
    images_names = []
    dataset = {}
    train_images = []
    train_labels = []

    for image_name in os.listdir(path):
        if image_name.split("_")[1].split(".")[0] == "1":
            images_names.append(image_name)

    for image in images_names:
        dataset[image] = [
            augment_image(os.path.join(path, image))
            for _ in range(0, NUM_AUGMENTATIONS)
        ]

    for key in dataset:
        all_pairs = [
            (dataset[key][i], dataset[key][j])
            for i in range(len(dataset[key]))
            for j in range(i + 1, len(dataset[key]))
        ]

        train_images.append(all_pairs)
        train_labels.append("same")

    matrix = []

    for key in dataset:
        matrix.append(dataset[key])

    for i in range(0, len(matrix) - 1):
        all_pairs = [
            (matrix[i], matrix[i + 1])
            for i in range(len(matrix[i]))
            for j in range(len(matrix[i + 1]))
        ]

        train_images.append(all_pairs)
        train_labels.append("different")

    final_dataset = []
    for image, label in zip(train_images, train_labels):
        for pair in image:
            final_dataset.append([pair, label])

    del dataset
    del matrix
    del images_names

    return final_dataset


def augment_image(img_path):
    random_degree = random.randint(MIN_DEGREE, MAX_DEGREE)

    rotated = transform.rotate(
        cv2.imread(img_path, -1), angle=-1 * random_degree, preserve_range=True
    ).astype(np.uint8)

    return rotated


fingerprint_dataset = process_dataset()
