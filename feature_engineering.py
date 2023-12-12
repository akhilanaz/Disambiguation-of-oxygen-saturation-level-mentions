from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

def explain_model(explainer, predict_fn, X_Test, class_labels, top_features=10):
    explanations = []
    positive_features_lists = {label: [] for label in class_labels}

    for class_label in class_labels:
        explanation_list = []
        positive_features = []

        for i in range(len(X_Test)):
            print(i)
            print(X_Test.iloc[i])

            exp = explainer.explain_instance(X_Test.iloc[i], predict_fn, num_features=top_features, labels=[class_label])

            print(exp)

            positive_features = [(feature, weight) for feature, weight in exp.as_list(label=class_label) if weight > 0]
            sorted_positive_features = sorted(positive_features, key=lambda x: x[1], reverse=True)
            top_positive_features_list = sorted_positive_features[:top_features]
            explanation_list.append(top_positive_features_list)

            positive_features_lists[class_label].extend(top_positive_features_list)

        explanations.append(explanation_list)

        # Convert the positive features to a DataFrame
        df_class = pd.DataFrame(positive_features_lists[class_label], columns=['Feature', 'Weight'])
        # Save the DataFrame to a CSV file
        df_class.to_csv(f'class_{class_label}_positive_features.csv', index=False)

    return explanations

# Example usage:

# For SVM
# svm_model = YourSVMModel()  # Replace YourSVMModel() with the actual SVM model instance
# svm_explainer = LimeTextExplainer(class_names=["0", "1"])
# svm_explanations_result = explain_model(svm_explainer, predict_fn_RF, X_Test, class_labels=["0", "1"])

# For RF
# rf_model = YourRFModel()  # Replace YourRFModel() with the actual RF model instance
# rf_explainer = LimeTextExplainer(class_names=["0", "1"])
# rf_explanations_result = explain_model(rf_explainer, predict_fn_RF, X_Test, class_labels=["0", "1"])

# For LSTM or BiLSTM
# lstm_model = YourLSTMModel()  # Replace YourLSTMModel() with the actual LSTM model instance
# lstm_explainer = LimeTextExplainer(class_names=["0", "1"])
# lstm_explanations_result = explain_model(lstm_explainer, predict_fn, X_Test, class_labels=["0", "1"])

# For CNN
# cnn_model = YourCNNModel()  # Replace YourCNNModel() with the actual CNN model instance
# cnn_explainer = LimeTextExplainer(class_names=["0", "1"])
# cnn_explanations_result = explain_model(cnn_explainer, predict_fn, X_Test, class_labels=["0", "1"])
