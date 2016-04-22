import csv
import numpy as np
import random
from fuel.datasets import IndexableDataset
from sklearn.feature_extraction import DictVectorizer
from pprint import pprint


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

    feature_vectorizer = DictVectorizer(sparse=False, dtype=np.uint8)

    label_vectorizer = DictVectorizer(sparse=False, dtype=np.uint8)

    # Create features and labels as numpy arrays with one hot encoding
    features = feature_vectorizer.fit_transform(features_)
    labels = label_vectorizer.fit_transform(labels_)

    # Print the meaning of the one-hot encoded features
    # for i, f in enumerate(feature_vectorizer.get_feature_names()):
    #     print(f)
    #
    # for i, f in enumerate(label_vectorizer.get_feature_names()):
    #     print(f)

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
    train, test = get_mushroom_data()
