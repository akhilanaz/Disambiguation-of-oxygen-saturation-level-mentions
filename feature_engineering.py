from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding,  Conv1D, GlobalMaxPooling1D, Dense, Flatten, Bidirectional, LSTM,Attention
from keras.utils import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
import math
from lime.lime_text import LimeTextExplainer
from lime import lime_tabular
from sklearn.feature_selection import RFE

data = pd.read_csv('covid_data_finalised.csv', usecols=[0, 1], names=['Text', 'Label'], encoding='unicode_escape')

# Tokenize and pad sequences
max_words = 3000
max_len = 30
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(data['Text'])
sequences = tok.texts_to_sequences(data['Text'])
sequences_matrix = pad_sequences(sequences, maxlen=max_len)
# Specify the number of random state values and iterations
random_states =[509, 906, 331, 172, 729, 250, 762, 629, 926, 392]
# Perform multiple train-test splits with different random state values
all_top_words = []
for i, random_state in enumerate(random_states):
    # Encode labels
    encoder = LabelEncoder()
    labels = encoder.fit_transform(data['Label'])
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(data['Text'], labels, test_size=0.2, random_state=random_state)
    X_train, X_test, Y_train, Y_test = train_test_split(sequences_matrix, labels, test_size=0.2, random_state=random_state)


    #########################################
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_Train)
    print(X_train_vectorized.shape)
    X_test_vectorized = vectorizer.transform(X_Test)
    print(X_test_vectorized.shape)
    feature_names = vectorizer.get_feature_names_out()
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'C': [10],
        'kernel': [ 'linear'],
        'gamma': ['scale']
    }

    # Perform GridSearchCV for hyperparameter tuning
    svm_classifier = SVC()
    grid_search = GridSearchCV(svm_classifier, param_grid, cv=5)
    grid_search.fit(X_train_vectorized, Y_train)


    best_params = grid_search.best_params_
    clf = SVC(**grid_search.best_params_, random_state=906)
    clf.fit(X_train_vectorized, Y_Train)
    y_pred_svm = clf.predict(X_test_vectorized)

    feature_names = vectorizer.get_feature_names_out()
    from collections import Counter
    # Specify the number of top features you want
    top_n = 10

    # Create a list to store all features from X_test
    all_features = []

    # Loop through all documents in X_test
    for document_vector in X_test_vectorized:
        # Get the nonzero feature indices for the current document
        nonzero_feature_indices = document_vector.nonzero()[1]

        # Extract the corresponding feature names
        features = [feature_names[i] for i in nonzero_feature_indices]

        # Append the features for the current document to the list
        all_features.extend(features)

    # Find the overall top features
    counter = Counter(all_features)
    top_features_overall = counter.most_common(top_n)

    # Print the overall top features
    print("Overall Top Features:")
    print(top_features_overall)
    all_top_words.append(top_features_overall)
    #######################################################################################

    param_grid_rf = {
        'n_estimators': [100],
        'max_depth': [None],
        'max_features': ['sqrt']
    }

    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier()

    # Create a GridSearchCV object to find the best hyperparameters
    grid_search_rf = GridSearchCV(rf_classifier, param_grid_rf, cv=5)
    grid_search_rf.fit(X_train_vectorized, Y_Train)
    best_params_rf = grid_search_rf.best_params_

    # Train RF model with best parameters
    rf_model = RandomForestClassifier(**best_params_rf, random_state=906)
    rf_model.fit(X_train_vectorized, Y_Train)
    y_pred_rf = rf_model.predict(X_test_vectorized)

    # Get feature importances from the trained RF model
    feature_importances_rf = rf_model.feature_importances_

    # Get the indices of the top features
    top_features_indices_rf = np.argsort(feature_importances_rf)[::-1][:10]

    # Get the feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Print the top features
    top_features_rf = feature_names[top_features_indices_rf]
    print("Top features for Random Forest:")
    print(top_features_rf)
    all_top_words.append(top_features_rf)

    ########################################################################################
    def create_lstm_model(units=128, dropout=0.2, recurrent_dropout=0.2):
        model = Sequential()
        model.add(Embedding(input_dim=max_words, output_dim=300, input_length=max_len))
        model.add(LSTM(units=units, dropout=dropout, recurrent_dropout=recurrent_dropout))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
        return model

    lstm_model = KerasClassifier(build_fn=create_lstm_model, epochs=10, batch_size=128, verbose=0)
    param_grid = {
        'units': [128],
        'dropout': [0.2],
        'recurrent_dropout': [0.2]
    }
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    grid_search = GridSearchCV(estimator=lstm_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)
    grid_search.fit(X_train, Y_train, validation_data=(X_test, Y_test), callbacks=[early_stopping])
    # Get the best hyperparameters
    best_params = grid_search.best_params_
    best_lstm_model = create_lstm_model(units=best_params['units'], dropout=best_params['dropout'],
                                        recurrent_dropout=best_params['recurrent_dropout'])
    best_lstm_model.fit(X_train, Y_train, epochs=10, batch_size=128)
    Y_pred = best_lstm_model.predict(X_test)
    Y_pred_binary_lstm = (Y_pred > 0.5).astype(int)

    # Get the embeddings from the LSTM or CNN model
    embeddings_lstm = Model(inputs=best_lstm_model.input, outputs=best_lstm_model.get_layer(index=0).output)

    # Get the embedded sequences
    embedded_sequences_lstm = embeddings_lstm.predict(sequences_matrix)

    # Average the embeddings across each sequence to get a representation for the entire text
    average_embedding_lstm = np.mean(embedded_sequences_lstm, axis=1)

    # Identify the top words contributing to the average embeddings
    sorted_indices = np.argsort(np.abs(average_embedding_lstm.mean(axis=0)))[::-1]
    top_words_lstm = [tok.index_word[i] for i in sorted_indices[:10] if i < len(tok.index_word)]

    # Print the top words
    print("Top words for LSTM:")
    print(top_words_lstm)
    all_top_words.append(top_words_lstm)
    ####################################################################################
    def create_bilstm_model(units=128, dropout=0.2, recurrent_dropout=0.2):
        model = Sequential()
        model.add(Embedding(input_dim=max_words, output_dim=300, input_length=max_len))
        model.add(Bidirectional(LSTM(units=units, dropout=dropout, recurrent_dropout=recurrent_dropout)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
        return model

    lstm_model = KerasClassifier(build_fn=create_bilstm_model, epochs=10, batch_size=128, verbose=0)
    param_grid = {
        'units': [128],
        'dropout': [0.2],
        'recurrent_dropout': [0.2]
    }
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    grid_search = GridSearchCV(estimator=lstm_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)
    grid_search.fit(X_train, Y_train, validation_data=(X_test, Y_test), callbacks=[early_stopping])

    best_params = grid_search.best_params_
    best_bilstm_model = create_bilstm_model(units=best_params['units'], dropout=best_params['dropout'],
                                        recurrent_dropout=best_params['recurrent_dropout'])
    best_bilstm_model.fit(X_train, Y_train, epochs=10, batch_size=128)
    Y_pred = best_bilstm_model.predict(X_test)
    Y_pred_binary_bilstm = (Y_pred > 0.5).astype(int)

    # Get the embeddings from the LSTM or CNN model
    embeddings_lstm = Model(inputs=best_bilstm_model.input, outputs=best_bilstm_model.get_layer(index=0).output)
    # Get the embedded sequences
    embedded_sequences_lstm = embeddings_lstm.predict(sequences_matrix)
    # Average the embeddings across each sequence to get a representation for the entire text
    average_embedding_lstm = np.mean(embedded_sequences_lstm, axis=1)
    # Identify the top words contributing to the average embeddings
    top_words_bilstm = [tok.index_word[i] for i in np.argsort(np.abs(average_embedding_lstm.mean(axis=0)))[::-1][:10]]
    # Print the top words
    print("Top words for Bi-LSTM:")
    print(top_words_bilstm)
    all_top_words.append(top_words_bilstm)
    ###########################################################################
    def create_cnn_model(filters=64, kernel_size=3):
        model = Sequential()
        model.add(Embedding(input_dim=max_words, output_dim=300, input_length=max_len))
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
        return model

    cnn_model = KerasClassifier(build_fn=create_cnn_model, epochs=10, batch_size=128, verbose=0)
    param_grid = {
        'filters': [256],
        'kernel_size': [5]}
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    grid_search = GridSearchCV(estimator=cnn_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)
    grid_search.fit(X_train, Y_train, validation_data=(X_test, Y_test), callbacks=[early_stopping])
    best_params = grid_search.best_params_
    best_cnn_model = create_cnn_model(filters=best_params['filters'], kernel_size=best_params['kernel_size'])
    best_cnn_model.fit(X_train, Y_train, epochs=10, batch_size=128)
    Y_pred = best_cnn_model.predict(X_test)
    Y_pred_binary_cnn = (Y_pred > 0.5).astype(int)

    # Get the embeddings from the LSTM or CNN model
    embeddings_lstm = Model(inputs=best_cnn_model.input, outputs=best_cnn_model.get_layer(index=0).output)

    # Get the embedded sequences
    embedded_sequences_lstm = embeddings_lstm.predict(sequences_matrix)

    # Average the embeddings across each sequence to get a representation for the entire text
    average_embedding_lstm = np.mean(embedded_sequences_lstm, axis=1)

    # Identify the top words contributing to the average embeddings
    top_words_cnn = [tok.index_word[i] for i in np.argsort(np.abs(average_embedding_lstm.mean(axis=0)))[::-1][:10]]

    # Print the top words
    print("Top words for CNN:")
    print(top_words_cnn)
    all_top_words.append(top_words_cnn)

import csv
# Specify the file path
csv_file_path = 'output.csv'

# Open the file in write mode with a CSV writer
with open(csv_file_path, 'w', newline='') as file:
    # Create a CSV writer object
    csv_writer = csv.writer(file)

    # Write the header if needed
    csv_writer.writerow(['Feature', 'Count'])

    # Write each list element as a new row
    csv_writer.writerows(all_top_words)

