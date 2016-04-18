import csv
import numpy as np
import random
from fuel.datasets import IndexableDataset
from sklearn.feature_extraction import DictVectorizer

mushroom_file = "data/mushrooms/agaricus-lepiota.data"


def get_mushroom_data():
    with open(mushroom_file, 'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter=",")

        # Prepare the data as dictionaries. Its not very neat, but it works.
        data = [
            (
                {row[0]: 1},
                {
                    feature_index: category
                    for feature_index, category in enumerate(row[1:])
                }
            ) for row in data_iter
        ]

    # It's always good to randomize the order
    random.shuffle(data)

    # Get the labels and the features as lists
    labels_, features_ = zip(*data)

    # Create features and labels as numpy arrays with one hot encoding
    features = DictVectorizer(
        sparse=False, dtype=np.uint8).fit_transform(features_)
    labels = DictVectorizer(
        sparse=False, dtype=np.uint8).fit_transform(labels_)

    # Split into train and test.
    split_at = int(len(data) * 0.7)
    X_training, X_test = features[:split_at, :], features[split_at:, :]
    y_training, y_test = labels[:split_at, :], labels[split_at:, :]

    train = IndexableDataset({
        'features': X_training.astype(np.uint8),
        'targets': y_training.astype(np.uint8)
    })
    test = IndexableDataset({
        'features': X_test.astype(np.uint8),
        'targets': y_test.astype(np.uint8)
    })

    return train, test


if __name__ == '__main__':
    get_mushroom_data()
    # print(len(read_data()[0][1][0]))
