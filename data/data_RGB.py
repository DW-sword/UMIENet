import os
from .dataset_RGB import DataLoaderTrain, DataLoaderVal, DataLoaderTest


def get_training_data(rgb_dir, inp, target, dep, real_dir ,img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, inp, target,dep, real_dir , img_options)


def get_validation_data(rgb_dir, inp, target, dep, real_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, inp, target, dep,real_dir, img_options)


def get_testing_data(rgb_dir, inp, dep, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, inp, dep, img_options)