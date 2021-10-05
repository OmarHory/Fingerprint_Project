dataset_config = dict(
    path="/home/omar/Desktop/fingerprint_dataset/second_session/",
    save_dataset_path="/home/omar/Desktop/fingerprint_dataset/fingerprint_second_session.pkl",
    number_of_shuffles=4,
)

augmentation_config = dict(
    num_augmentations=6,
    max_rotation_degree=359,
    min_rotation_degree=0,
    if_augment=False,
)

image_config = dict(
    height=277,
    width=277,
    channels=1,
    class_names=[0, 1],
)

training_config = dict(
    train_size=1000,
    val_size=1000,
    test_size=1000,
    epochs=3,
    batch_size=128,
    run_name="AlexNet_Multiple_Inputs",
    loss="binary_crossentropy",
    lr=0.001,
    patience=5,
    experiment_name="trial1",
    experiment_path="./experiments",
)
