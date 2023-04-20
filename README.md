# Optimal-K-value-for-Breast-Cancer-Prediction-Model
*Finding the optimal k-value for Breast Cancer Prediction Model based on KNN algorithm which ensures the highest accuracy of the model.*

The Jupyter Notebook implements a K-Nearest Neighbors (KNN) algorithm to predict if a patient has benign or malignant breast cancer. It aims to find the optimal k-value for the model to achieve the highest accuracy.

The notebook imports various libraries such as sklearn.datasets, train_test_split, KNeighborsClassifier, and matplotlib.pyplot. It also loads the breast cancer dataset from scikit-learn's datasets package using load_breast_cancer().

The dataset is split into a training set and a test set using train_test_split(). A KNN model with n_neighbors=3 is created and trained on the training data using classifier.fit(X_train, y_train). The accuracy of the model for k=3 is then evaluated using classifier.score(X_test, y_test) and found to be 94.73%.

To find the optimal value of k, the model's accuracy is tested for different k-values ranging from 1 to 100. The accuracy scores are then appended to an empty list outside the loop using a for loop. Then, a line chart of k-values against validation accuracies is plotted using matplotlib.pyplot.

The plot shows that k-values around 22-24 and 58-59 produce the highest accuracy for the model.
