path = "/home/omar/Desktop/fingerprint_dataset/second_session/"

import os
import cv2
from skimage import transform

# from skimage.util import crop
import random
import numpy as np

# TODO: make sure to validate the pairs of same and different.


def get_single_image():
    images_names = []
    train_images = []
    train_labels = []
    for image_name in os.listdir(path):
        if image_name.split("_")[1].split(".")[0] == "1":
            images_names.append(image_name)

    all_pairs = [
        (images_names[i], images_names[j])
        for i in range(len(images_names))
        for j in range(i + 1, len(images_names))
    ]

    for p1, p2 in all_pairs:
        train_images.append(
            (
                cv2.imread(os.path.join(path, p1), -1),
                augment_image(os.path.join(path, p2)),
            )
        )
        train_labels.append("different")

    for image in images_names:
        try:
            train_images.append(
                (
                    cv2.imread(os.path.join(path, image), -1),
                    augment_image(os.path.join(path, image)),
                )
            )
            train_labels.append("same")
        except:
            pass


def augment_image(img_path):
    random_degree = random.randint(0, 360)

    file_name = os.path.join(
        "/home/omar/Desktop/temp_images",
        img_path.split("/")[-1].split(".")[0] + "_rotated.jpg",
    )
    rotated = transform.rotate(
        cv2.imread(img_path, -1), angle=-1 * random_degree, preserve_range=True
    ).astype(np.uint8)

    cv2.imwrite(
        file_name,
        rotated,
    )

    return rotated


get_single_image()
