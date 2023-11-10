import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Input, Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import math

# Load data
data = pd.read_csv('D:/OxygenStatus/covid_data_finalised.csv', usecols=[0, 1], names=['Text', 'Label'], encoding='unicode_escape')

# Specify the number of random state values and iterations
random_states = [509, 906, 331, 172, 729, 250, 762, 629, 926, 392]

# Initialize lists to store evaluation metrics for each iteration
precision_scores = []
recall_scores = []
f1_scores = []
prediction_times = []
best_iteration = -1
best_precision = -1
best_params = None
best_conf_matrix = None

# Perform multiple train-test splits with different random state values
for i, random_state in enumerate(random_states):
    max_word = 3000
    max_len = 30
    tok = Tokenizer(num_words=max_word)
    tok.fit_on_texts(data['Text'])

    # Split data into training and testing sets
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(data["Text"], data["Label"], test_size=0.2,
                                                        random_state=random_state)
    Encoder = LabelEncoder()
    Y_Train = Encoder.fit_transform(Y_Train)
    Y_Test = Encoder.fit_transform(Y_Test)

    # Split the remaining data to train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(X_Train, Y_Train, test_size=0.2, random_state=random_state)

    # Sequence padding
    train_sequences = tok.texts_to_sequences(X_train)
    train_sequences_matrix = pad_sequences(train_sequences, maxlen=max_len)
    Val_sequences = tok.texts_to_sequences(X_val)
    Val_sequences_matrix = pad_sequences(Val_sequences, maxlen=max_len)
    test_sequences = tok.texts_to_sequences(X_Test)
    test_sequences_matrix = pad_sequences(test_sequences, maxlen=max_len)

    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0001, restore_best_weights=True)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')


    def create_model(learn_rate=0.001, batch_size=128, epochs=10, filters=100):
        inputs = Input(name='inputs', shape=[max_len])
        layer = Embedding(max_word, 300, input_length=max_len)(inputs)
        layer = LSTM(128, dropout=0.5, recurrent_dropout=0.2)(layer)
        layer = Dense(1, name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs, outputs=layer)
        model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
        return model


    keras_clf = KerasClassifier(build_fn=create_model, verbose=1)

    # Define the grid search parameters
    param_grid = {
        'learn_rate': [0.001, 0.01, 0.1],
        'batch_size': [64, 128, 256],
        'epochs': [10, 20, 30],
        'filters': [100, 200, 300]
    }

    # Create the grid search
    grid_search_lstm = GridSearchCV(estimator=keras_clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)

    # Fit the grid search
    grid_search_lstm.fit(train_sequences_matrix, Y_train, validation_data=(Val_sequences_matrix, Y_val),
                         callbacks=[earlyStopping, reduce_lr_loss])

    # Save the results to a CSV file
    results_df = pd.DataFrame(grid_search_lstm.cv_results_)
    results_df.to_csv(f'lstm_grid_search_randomsplit{random_state}.csv', index=False)

    # Get the best hyperparameters
    best_params = grid_search_lstm.best_params_
    print(best_params)

    # Train the model with the best parameters
    model = create_model(**best_params)
    model.fit(train_sequences_matrix, Y_train)

    # Make predictions on the test set
    import time

    start_time = time.time()
    y_pred = model.predict(test_sequences_matrix)
    end_time = time.time()
    prediction_time = end_time - start_time
    prediction_times.append(prediction_time)

    # Calculate evaluation metrics for each iteration
    precision = precision_score(Y_Test, y_pred, average='weighted')
    recall = recall_score(Y_Test, y_pred, average='weighted')
    f1 = f1_score(Y_Test, y_pred, average='weighted')

    # Store the evaluation metrics
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    # Get the best hyperparameters for the current iteration
    current_best_params = grid_search_lstm.best_params_

    # Check if the precision for the current iteration is the best so far
    if precision > best_precision:
        best_precision = precision
        best_iteration = i
        best_params = current_best_params

        # Train the model with the best parameters for the current iteration
        best_model = create_model(**best_params)
        best_model.fit(train_sequences_matrix, Y_train)

        # Make predictions on the test set for the best iteration
        best_y_pred = best_model.predict(test_sequences_matrix)

        # Calculate and store the confusion matrix for the best iteration
        best_conf_matrix = confusion_matrix(Y_Test, best_y_pred)

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
metrics_df.to_csv('LSTM_metrics_scores.csv', index=False)

lower_bound_precision = np.percentile(precision_scores, 2.5)
upper_bound_precision = np.percentile(precision_scores, 97.5)
lower_bound_recall = np.percentile(recall_scores, 2.5)
upper_bound_recall = np.percentile(recall_scores, 97.5)
lower_bound_f1_score = np.percentile(f1_scores, 2.5)
upper_bound_f1_score = np.percentile(f1_scores, 97.5)

# Calculate the mean prediction time
mean_prediction_time = np.mean(prediction_times)
print(f"Average Prediction Time: {mean_prediction_time} seconds")

# Print the results
print(
    f'Precision: Mean={mean_precision}, Std={std_precision}, St_error= {std_precision / math.sqrt(10)}, CI={lower_bound_precision, upper_bound_precision}')
print(
    f'Recall: Mean={mean_recall}, Std={std_recall}, St_error= {std_recall / math.sqrt(10)},CI={lower_bound_recall, upper_bound_recall}')
print(
    f'F1 Score: Mean={mean_f1}, Std={std_f1}, St_error= {std_f1 / math.sqrt(10)}, CI={lower_bound_f1_score, upper_bound_f1_score}')

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
