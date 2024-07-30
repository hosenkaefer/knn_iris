k-Nearest Neighbors (k-NN) Classifier in Python using the numpy library. 
Creating a class function `knn_classifier` which takes as input: `train_features` (feature set of the training data), `class_labels` (associated class labels), `test_features` (features of the testing dataset), 
and `num_neighbors` (size of the neighborhood). 
Using `numpy.linalg.norm` and `numpy.argsort` to find the k closest neighbors, and using `numpy.unique` to determine the majority vote within a neighborhood.

Applying the Iris dataset from the sklearn library to evaluate the classifier with different k values. Measuring accuracy on both training and testing sets, using 67% of the data for training.

Additionally: an implemented k-fold cross-validation to optimize the choice of k. Dividing the training data into k equal segments, using k-1 folds for training and the remaining fold for validation. 
Repeating this process k times, each time using a different fold for validation. Calculating the average accuracy across all k folds to determine the optimal k value.
