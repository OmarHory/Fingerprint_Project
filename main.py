from train_eval import TrainAndEvaluate
from config import dataset_config

# TODO: let the process classes not to take config files as global variables.


def run():

    train_object = TrainAndEvaluate()
    train_object.train()


if __name__ == "__main__":
    run()
