from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding,  Conv1D, GlobalMaxPooling1D, Dense, Flatten, Bidirectional, LSTM
from keras.utils import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
import math


data = pd.read_csv('covid_data_finalised.csv', usecols=[0, 1], names=['Text', 'Label'], encoding='unicode_escape')

# Tokenize and pad sequences
max_words = 3000
max_len = 30
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(data['Text'])
sequences = tok.texts_to_sequences(data['Text'])
sequences_matrix = pad_sequences(sequences, maxlen=max_len)

# Encode labels
encoder = LabelEncoder()
labels = encoder.fit_transform(data['Label'])
random_states =[509, 906, 331, 172, 729, 250, 762, 629, 926, 392]
for random_state in random_states:
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(data['Text'], labels, test_size=0.2, random_state=random_state)
    X_train, X_test, Y_train, Y_test = train_test_split(sequences_matrix, labels, test_size=0.2, random_state=random_state)


    #########################################
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_Train)
    X_test_vectorized = vectorizer.transform(X_Test)

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'C': [10],
        'kernel': [ 'rbf'],
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
        'units': [ 128],
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




    conf_matrix_model_SVM = confusion_matrix(Y_Test, y_pred_svm)
    print('cm for svm',conf_matrix_model_SVM)
    conf_matrix_model_RF = confusion_matrix(Y_Test, y_pred_rf)
    print('cm for rf',conf_matrix_model_RF)
    conf_matrix_model_LSTM = confusion_matrix(Y_test, Y_pred_binary_lstm)
    print('cm for LSTM',conf_matrix_model_LSTM)
    conf_matrix_model_BiLSTM = confusion_matrix(Y_test, Y_pred_binary_bilstm)
    print('cm for BILSTM',conf_matrix_model_BiLSTM)
    conf_matrix_model_CNN = confusion_matrix(Y_test, Y_pred_binary_cnn)
    print('cm for CNN',conf_matrix_model_CNN)
    #
    # Extract indices of false positives and false negatives for each model
    FP_indices_svm = np.where((Y_Test == 0) & (y_pred_svm == 1))[0]
    FN_indices_svm = np.where((Y_Test == 1) & (y_pred_svm == 0))[0]

    FP_indices_rf = np.where((Y_Test == 0) & (y_pred_rf == 1))[0]
    FN_indices_rf = np.where((Y_Test == 1) & (y_pred_rf == 0))[0]

    FP_indices_lstm = np.where((Y_Test == 0) & (Y_pred_binary_lstm == 1))[0]
    FN_indices_lstm = np.where((Y_Test == 1) & (Y_pred_binary_lstm == 0))[0]

    FP_indices_bilstm = np.where((Y_Test == 0) & (Y_pred_binary_bilstm == 1))[0]
    FN_indices_bilstm = np.where((Y_Test == 1) & (Y_pred_binary_bilstm == 0))[0]

    FP_indices_cnn = np.where((Y_Test == 0) & (Y_pred_binary_cnn == 1))[0]
    FN_indices_cnn = np.where((Y_Test == 1) & (Y_pred_binary_cnn == 0))[0]

    # Find common false positives and false negatives
    common_FP = set(FP_indices_svm) & set(FP_indices_rf) & set(FP_indices_lstm) & set(FP_indices_bilstm) & set(FP_indices_cnn)
    common_FN = set(FN_indices_svm) & set(FN_indices_rf) & set(FN_indices_lstm) & set(FN_indices_bilstm) & set(FN_indices_cnn)

    # Extract texts for common false positives and false negatives
    common_FP_texts = X_Test.iloc[list(common_FP)].tolist()
    common_FN_texts = X_Test.iloc[list(common_FN)].tolist()

    # Save common false positives and false negatives to CSV files
    df_common_FP = pd.DataFrame({'Text': common_FP_texts, 'True Label': 0, 'Predicted Label': 1})
    df_common_FN = pd.DataFrame({'Text': common_FN_texts, 'True Label': 1, 'Predicted Label': 0})

    df_common_FP.to_csv(f'common_false_positives_{random_state}.csv', index=False)
    df_common_FN.to_csv(f'common_false_negatives_{random_state}.csv', index=False)
