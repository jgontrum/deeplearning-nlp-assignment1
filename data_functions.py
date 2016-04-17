import csv
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from fuel.datasets import IndexableDataset
from collections import OrderedDict
from pandas import get_dummies, DataFrame
from sklearn.feature_extraction import DictVectorizer

mushroom_file = "data/mushrooms/agaricus-lepiota.data"


def get_mushroom_data():
    # See http://stackoverflow.com/a/28554340/4587312
    # This methods seems to be the fastest for CSV -> NP

    with open(mushroom_file, 'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter=",")

        data = [({row[0]: 1}, {feature_index: category
                    for feature_index, category in enumerate(row[1:])})
                for row in data_iter]

    # It's always good to randomize the order
    random.shuffle(data)

    # Get the labels and the features as lists
    labels_, features_ = zip(*data)

    features = DictVectorizer(sparse=False, dtype=np.int8).fit_transform(features_)
    labels = DictVectorizer(sparse=False, dtype=np.int64).fit_transform(labels_)

    # Split into train and test.
    split_at = int(len(data) * 0.9)
    X_training, X_test = features[:split_at,:], features[split_at:,:]
    y_training, y_test = labels[:split_at,:], labels[split_at:,:]

    print(y_training.shape)
    print(X_training.shape)
    print(y_training.dtype)
    print(X_training.dtype)


    train = IndexableDataset(
        indexables=OrderedDict([('features', X_training), ('targets', y_training)]),
        axis_labels=OrderedDict([('features', ('batch', 'height', 'width')),
                                 ('targets', ('batch', 'index'))]))

    test = IndexableDataset(
        indexables=OrderedDict([('features', X_test), ('targets', y_test)]),
        axis_labels=OrderedDict([('features', ('batch', 'height', 'width')),
                                 ('targets', ('batch', 'index'))]))

    return train, test


if __name__ == '__main__':
    get_mushroom_data()
    # print(len(read_data()[0][1][0]))
