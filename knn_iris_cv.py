import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

class KnnCf:
    """
    K-Nearest Neighbors Classifier for Iris Dataset with implemented crossvalidation
    """
    def __init__(self, num_neighbors):
        self.num_neighbors = num_neighbors
    

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
    
    def get_params(self, deep=True):
        return {"num_neighbors": self.num_neighbors}
    

iris = load_iris()
X = iris.data
Y = iris.target

cv_scores = [] 

folds = 10
# here 10-fold-cv 
# # changeable

ks = list(range(1,int(len(X) * ((folds - 1)/folds))))

ks = [k for k in ks if k % 3 != 0]
# Throw out all multiples of three 
# -> Since Iris dataset contains 3 species of 50 individuals each
# ->> Prevents a tie with regard to the number of k-neighbors

for k in ks:
    knn = KnnCf(num_neighbors = k) 
    scores = cross_val_score(knn, X, Y, cv=folds, scoring='accuracy') 
    mean = scores.mean()
    cv_scores.append(mean)
    print(f"K = {k}, Ø Genauigkeit = {mean}") 

# Find the best K, with the highest Ø accuracy
optimal_k_macc = ks[cv_scores.index(max(cv_scores))]
print(f"Optimaler K, basierend auf Ø Genauigkeit = {optimal_k_macc}")

# Plot of each K + corresponding Ø Accuracy
plt.plot(ks, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Ø Genauigkeit')
plt.show()