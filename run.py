from train_eval import TrainAndEvaluate
from config import augmentation_config
from process_and_augment import process_dataset, save_dataset, load_fingerprint

# TODO: let the process classes not to take config files as global variables.


def run(augment):
    if augment:
        fingerprint_dataset = process_dataset()
        save_dataset(fingerprint_dataset)
        load_fingerprint()

    train_object = TrainAndEvaluate()
    train_object.train()


if __name__ == "__main__":
    run(augmentation_config['if_augment'])
