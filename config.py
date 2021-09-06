image_config = dict(
    height=277,
    width=277,
    channels=3,
    class_names=[
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
)
training_config = dict(
    epochs=50,
    val_size=5000,
    batch_size=32,
    run_name="AlexNet_Multiple_Inputs",
    loss="sparse_categorical_crossentropy",
    lr=0.001,
)

# binary_crossentropy
