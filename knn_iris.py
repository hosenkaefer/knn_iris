import numpy as np 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class KnnCf:
    """
    K-Nearest Neighbors Classifier für Iris Dataset
    """
    def __init__(self, num_neighbors):
        self.num_neighbors = num_neighbors
    # num_neighbours sind die Ks
    

    def euclidean_distance(self, p, q):
        return np.linalg.norm(p - q)


    def fit(self, train_features, class_labels):
        self.train_features = train_features
        self.class_labels = class_labels
    

    def predict(self, test_features):
        predictions = []
        for x in test_features:
            distances = [self.euclidean_distance(x, X_train) for X_train in self.train_features]
            k_indices = np.argsort(distances)[:self.num_neighbors]
            k_labels = self.class_labels[k_indices]
            unique, counts = np.unique(k_labels, return_counts = True)
            majority = unique[np.argmax(counts)]
            predictions.append(majority)
        return np.array(predictions)
    

iris = load_iris()
iris_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
iris_df['target'] = iris.target
print(iris_df.head(5))
print(iris_df.tail(5))
print(iris_df['target'].value_counts())

X = iris.data
Y = iris.target


# 67% Training, 33% Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)


def accuracy(pred, true_labels):
    return np.sum(pred == true_labels) / len(true_labels)


for k in range(1, 4):
    clf = KnnCf(num_neighbors = k)
    clf.fit(X_train, Y_train)

    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)

    acc_train = accuracy(pred_train, Y_train)
    acc_test = accuracy(pred_test, Y_test)

    conf_mat = confusion_matrix(Y_test, pred_test)

    print(f"k = {k}: Train acc = {acc_train:.3f}, Test acc = {acc_test:.3f}")
    print(conf_mat)
