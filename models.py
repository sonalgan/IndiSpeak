from sklearn.svm import SVC
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

class SVMClassifier:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        self.model = SVC()  # Initialize a Support Vector Classifier
        self.model.fit(X, y)  # Fit the model to the training data

    def predict(self, X):
        return self.model.predict(X)  # Predict the labels for the given data

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)  # Save the trained model to a file

    def load_model(self, filepath):
        self.model = joblib.load(filepath)  # Load a pre-trained model from a file

class RFClassifier:
    def __init__(self):
        self.classifier = RandomForestClassifier()  # Initialize a Random Forest Classifier

    def fit(self, embeddings, labels):
        self.classifier.fit(embeddings, labels)  # Fit the classifier to the training data

    def predict(self, embeddings):
        predictions = self.classifier.predict(embeddings)  # Predict the labels for the given embeddings
        return predictions

    def save_model(self, model_path='rf_model.pkl'):
        joblib.dump(self.classifier, model_path)  # Save the trained model to a file

    def load_model(self, model_path='rf_model.pkl'):
        self.classifier = joblib.load(model_path)  # Load a pre-trained model from a file


