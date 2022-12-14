import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.
        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.X_train = None
        self.y_train = None
        self.k = k
        self.p = p
        self.ids = (213272644, 325482768)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """

        # TODO - your code here
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """

        y_test = list()
        for x in X:
            dist_matrix = list()
            # for each observation in X_train, calculate the distance to x by using order p norm
            # add the distance to the distance matrix
            for i in range(self.X_train.shape[0]):
                dist_matrix.append((np.linalg.norm(self.X_train[i] - x, ord=self.p), self.y_train[i]))
            dist_np = np.array(dist_matrix)
            # find the indices order for the sorted distance matrix. if the distance of 2 points is the same,
            # sort by lexicographic order of the label
            indices = np.lexsort((dist_np[:, 1], dist_np[:, 0]))
            # count appearances of each label in order to find the mode
            freq = np.bincount(self.y_train[indices[:self.k]])
            most_common_label = list()
            most_occurrences = np.amax(freq)
            for i in range(freq.shape[0]):
                # if the label i has the same appearances as the mode, add to the modes list
                if freq[i] == most_occurrences:
                    most_common_label.append(i)
            # if there is more than one mode, choose by the label of the nearest neighbor with a common label
            if len(most_common_label) > 1:
                k_labels_list = self.y_train[indices[:self.k]]
                i = 0
                while k_labels_list[i] not in most_common_label:
                    i += 1
                y_test.append(k_labels_list[i])
            # if there is only 1 mode, take it as the label for x
            else:
                y_test.append(most_common_label[0])
        return np.array(y_test)


def main():
    print("*" * 20)
    print("Started HW1_ID1_ID2.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
