dataset_config = dict(
    path="/home/omar/Desktop/fingerprint_dataset/second_session/",
    save_dataset_path="/home/omar/Desktop/fingerprint_dataset/fingerprint_second_session.pkl",
    number_of_shuffles=4,
)

augmentation_config = dict(
    num_augmentations=12,
    max_rotation_degree=359,
    min_rotation_degree=0,
)

image_config = dict(
    height=277,
    width=277,
    channels=1,
    class_names=["same", "different"],
)

training_config = dict(
    train_size=100,
    val_size=100,
    test_size=100,
    epochs=50,
    batch_size=1,
    run_name="AlexNet_Multiple_Inputs",
    loss="binary_crossentropy",
    lr=0.001,
)
