import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math
# Load data
data = pd.read_csv('D:/OxygenStatus/covid_data_finalised.csv', usecols=[0, 1], names=['Text', 'Label'], encoding='unicode_escape')

# Specify the number of random state values and iterations
random_states =[509, 906, 331, 172, 729, 250, 762, 629, 926, 392]

# Initialize lists to store evaluation metrics for each iteration
precision_scores = []
recall_scores = []
f1_scores = []
prediction_times =[]
best_iteration = -1
best_precision = -1
best_params = None
best_conf_matrix = None
# Perform multiple train-test splits with different random state values
for i, random_state in enumerate(random_states):
    # Split data into training and testing sets
    X_train, X_test, Y_Train, y_test = train_test_split(data["Text"], data["Label"], test_size=0.2, random_state=random_state)

    # Vectorize text data using CountVectorizer
    vectorizer = TfidfVectorizer(max_features=300)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }

    # Perform GridSearchCV for hyperparameter tuning
    svm_classifier = SVC()
    grid_search = GridSearchCV(svm_classifier, param_grid, cv=5)
    grid_search.fit(X_train_vectorized, Y_Train)

    # Save the results to a CSV file
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(f'svm_grid_search_randomsplit_new_{random_state}.csv', index=False)
    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print(best_params)

    # Train SVM model with best parameters
    clf = SVC(**grid_search.best_params_, random_state=random_state)
    clf.fit(X_train_vectorized, Y_Train)

    # Make predictions on test set
    import time
    start_time = time.time()

    y_pred = clf.predict(X_test_vectorized)

    # Record the end time
    end_time = time.time()
    prediction_time = end_time - start_time
    prediction_times.append(prediction_time)

    # Calculate evaluation metrics for each iteration
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Store the evaluation metrics
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    # Get the best hyperparameters for the current iteration
    current_best_params = grid_search.best_params_

    # Check if the precision for the current iteration is the best so far
    if precision > best_precision:
        best_precision = precision
        best_iteration = i
        best_params = current_best_params

        # Train SVM model with the best parameters for the current iteration
        best_clf = SVC(**best_params, random_state=random_state)
        best_clf.fit(X_train_vectorized, Y_Train)

        # Make predictions on the test set for the best iteration
        best_y_pred = best_clf.predict(X_test_vectorized)

        # Calculate and store the confusion matrix for the best iteration
        best_conf_matrix = confusion_matrix(y_test, best_y_pred)
# Calculate the mean and standard deviation for evaluation metrics

mean_precision = np.mean(precision_scores)
std_precision = np.std(precision_scores)

mean_recall = np.mean(recall_scores)
std_recall = np.std(recall_scores)

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

metrics_df = pd.DataFrame({
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1 Score': f1_scores,
    'prediction_time': prediction_times
})

# Save the DataFrame to a CSV file
metrics_df.to_csv('new_svm_metrics_scores.csv', index=False)

# Calculate the mean prediction time
mean_prediction_time = np.mean(prediction_times)
print(f"Average Prediction Time: {mean_prediction_time} seconds")

# Print the results
print(
    f'Precision: Mean={mean_precision}, Std={std_precision}, St_error= {std_precision / math.sqrt(10)}, CI={mean_precision - St_error*1.96, mean_precision + St_error*1.96}')
print(
    f'Recall: Mean={mean_recall}, Std={std_recall}, St_error= {std_recall / math.sqrt(10)}, CI={mean_recall - St_error*1.96, mean_recall + St_error*1.96}')
print(
    f'F1 Score: Mean={mean_f1}, Std={std_f1}, St_error= {std_f1 / math.sqrt(10)}, CI={mean_f1 - St_error*1.96, mean_f1 + St_error*1.96}')

best_iteration = np.argmax(precision_scores)
# Print the best hyperparameters
print("Best Hyperparameters for the Highest Precision (Iteration {}):".format(best_iteration + 1))
print("Best Hyperparameters:", best_params)
print("Best Precision:", best_precision)
print("Best Recall:", recall_scores[best_iteration])
print("Best F1 Score:", f1_scores[best_iteration])

# Print the confusion matrix for the best iteration
print('Confusion Matrix for the Best Model (Iteration {}):'.format(best_iteration + 1))
print(best_conf_matrix)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_test_vectorized.toarray())
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_test.values)
plt.show()

