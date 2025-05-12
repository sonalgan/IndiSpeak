from sklearn.svm import SVC
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

class SVMClassifier:
    """
    A wrapper class for the Support Vector Classifier (SVC).
    Supports training, prediction, and model serialization.
    """
    def __init__(self):
        """
        Initializes an empty SVM model.
        """
        self.model = None

    def fit(self, X, y):
        """
        Trains an SVM classifier on the provided data.
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Corresponding labels.
        """
        self.model = SVC() 
        self.model.fit(X, y) 

    def predict(self, X):
        """
        Predicts labels for the given feature matrix.
        
        Args:
            X (np.ndarray): Feature matrix.
        
        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X) 

    def save_model(self, filepath):
        """
        Saves the trained SVM model to the specified file path.
        
        Args:
            filepath (str): Path where the model will be saved.
        """
        joblib.dump(self.model, filepath)  

    def load_model(self, filepath):
        """
        Loads a pre-trained SVM model from the specified file path.
        
        Args:
            filepath (str): Path to the saved model file.
        """
        self.model = joblib.load(filepath)  
        
class RFClassifier:
    """
    A wrapper class for the Random Forest classifier.
    Supports training, prediction, and model serialization.
    """
    def __init__(self):
        """
        Initializes a Random Forest classifier with default settings.
        """
        self.classifier = RandomForestClassifier()  

    def fit(self, embeddings, labels):
        """
        Trains the Random Forest classifier on the provided embeddings and labels.
        
        Args:
            embeddings (np.ndarray): Feature matrix.
            labels (np.ndarray): Corresponding labels.
        """
        self.classifier.fit(embeddings, labels)  

    def predict(self, embeddings):
        """
        Predicts labels for the provided embeddings.
        
        Args:
            embeddings (np.ndarray): Feature matrix.
        
        Returns:
            np.ndarray: Predicted labels.
        """
        predictions = self.classifier.predict(embeddings)  
        return predictions

    def save_model(self, model_path='rf_model.pkl'):
        """
        Saves the trained Random Forest model to the specified file.
        
        Args:
            model_path (str): Path to save the model.
        """
        joblib.dump(self.classifier, model_path) 

    def load_model(self, model_path='rf_model.pkl'):
        """
        Loads a pre-trained Random Forest model from the specified file.
        
        Args:
            model_path (str): Path to the saved model file.
        """
        self.classifier = joblib.load(model_path)  


