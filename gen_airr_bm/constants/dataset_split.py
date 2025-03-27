from enum import Enum


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"

    @classmethod
    def train_and_test(cls):
        return [cls.TRAIN, cls.TEST]
