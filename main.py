import os
import argparse
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import models
import utils
from feature_extractor import FeatureExtractor
import tensorflow as tf
import sys
import time
import gc
import psutil

class Main:
    def __init__(self, audio_folder, checkpoint_path, pca_params_path, total_samples, samples_per_folder, batch_size, training_size, model, run_name):
        self.audio_folder = audio_folder
        self.checkpoint_path = checkpoint_path
        self.pca_params_path = pca_params_path
        self.total_samples = total_samples
        self.samples_per_folder = samples_per_folder
        self.batch_size = batch_size
        self.training_size = training_size
        self.run_name = run_name
        self.feature_extractor = FeatureExtractor(self.checkpoint_path, self.pca_params_path, self.run_name)
        self.model = model

    def load_data(self):
        self.feature_extractor.load_model()

    def train(self):
        print("Training started.")

        audio_generator = self.feature_extractor.audio_embeddings_generator(
            self.audio_folder,
            total_samples=self.total_samples,
            samples_per_folder=self.samples_per_folder,
            training_size=self.training_size,
            mode="train"
        )

        train_embeddings = []
        train_labels = []
        
        for embeddings_batch, labels_batch in tqdm(audio_generator, desc="Generating Train Embeddings", unit="batch"):
            embeddings_batch = embeddings_batch.reshape(embeddings_batch.shape[0], -1)  # Reshape to 2D
            train_embeddings.append(embeddings_batch)
            train_labels.append(labels_batch)

        if len(train_embeddings) == 0:
            print("No valid embeddings found for training.")
            sys.exit(1)
        train_embeddings = np.concatenate(train_embeddings, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        # print("After concatenation:")
        # print('Train Embeddings shape:', train_embeddings.shape)
        # print('Train Labels shape:', train_labels.shape)

        self.feature_extractor.fit_label_encoder(train_labels)  # Fit the label encoder with training labels
        
        encoded_train_labels = self.feature_extractor.encode_labels(train_labels)
        print('Train Embeddings shape:', train_embeddings.shape)
        print('Train Labels shape:', encoded_train_labels.shape)

        try:
            print("Fitting the model...")
            # print('Train Embeddings dtype:', train_embeddings.dtype)
            # print('Train Labels dtype:', encoded_train_labels.dtype)
            self.model.fit(train_embeddings, encoded_train_labels)
            print("Model fitting completed.")

            # Save the label encoder and training files
            label_encoder_path = f'misc/label_encoder_{self.run_name}.pkl'
            joblib.dump(self.feature_extractor.label_encoder, label_encoder_path)
            print("Label encoder saved.")
        except Exception as e:
            print("Error fitting model:", e)
            sys.exit(1)

    def evaluate(self):

        audio_generator = self.feature_extractor.audio_embeddings_generator(
            self.audio_folder,
            total_samples=self.total_samples,
            samples_per_folder=self.samples_per_folder,
            training_size=self.training_size,
            mode="test"
        )

        test_embeddings = []
        test_labels = []

        for embeddings_batch, labels_batch in tqdm(audio_generator, desc="Generating Test Embeddings", unit="batch"):
            embeddings_batch = embeddings_batch.reshape(embeddings_batch.shape[0], -1)  # Reshape to 2D
            test_embeddings.append(embeddings_batch)
            test_labels.append(labels_batch)

        if len(test_embeddings) > 0:
            test_embeddings = np.concatenate(test_embeddings, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)

            encoded_labels = self.feature_extractor.encode_labels(test_labels)

            if self.model is not None:
                # Reshape the embeddings to 2D before prediction
                test_embeddings = test_embeddings.reshape(test_embeddings.shape[0], -1)
                print('Test Embeddings shape:', test_embeddings.shape)

                predictions = self.model.predict(test_embeddings)

                # Compute evaluation metrics
                accuracy = accuracy_score(encoded_labels, predictions)
                precision = precision_score(encoded_labels, predictions, average="weighted")
                recall = recall_score(encoded_labels, predictions, average="weighted")
                f1 = f1_score(encoded_labels, predictions, average="weighted")
                confusion = confusion_matrix(encoded_labels, predictions)
                classification = classification_report(encoded_labels, predictions)
                # Save the classification report and confusion matrix
                classification_report_save_path = f'runs/classification_report_{self.run_name}.png'
                confusion_matrix_save_path = f'runs/confusion_matrix_{self.run_name}.png'

                # Print evaluation metrics
                print("Accuracy:", accuracy)
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1 Score:", f1)
                print("Confusion Matrix:")
                print(confusion)
                print("Classification Report:")
                print(classification)

                utils.save_classification_report(classification, classification_report_save_path)
                utils.save_confusion_matrix(confusion, self.feature_extractor.label_encoder.classes_,
                                             confusion_matrix_save_path)

            else:
                print("Model not found.")
        else:
            print("No test embeddings found.")

    def run(self):
        # Memory and RAM usage before execution
        process = psutil.Process(os.getpid())
        memory_usage_before = process.memory_info().rss / (1024 ** 2)  # Memory usage in MB
        ram_usage_before = psutil.virtual_memory().used / (1024 ** 2)  # RAM usage in MB

        start = time.time()
        gc.enable()
        label_encoder_path = f'misc/label_encoder_{self.run_name}.pkl'
        model_path = f'misc/model_{self.run_name}.pkl'

        if os.path.isfile(label_encoder_path) and os.path.isfile(model_path):
            # Pre-trained model and files exist, load them
            print("Loaded label encoder and model.")
            self.model.load_model(model_path)
            self.feature_extractor.label_encoder = joblib.load(label_encoder_path)
        else:
            print("Pre-trained model and files don't exist, start training")
            self.load_data()
            self.train()
            self.model.save_model(model_path)

        self.evaluate()
        end = time.time()
        
        # Memory and RAM usage after execution
        memory_usage_after = process.memory_info().rss / (1024 ** 2)  # Memory usage in MB
        ram_usage_after = psutil.virtual_memory().used / (1024 ** 2)  # RAM usage in MB

        print("Execution Time: ", end - start, "seconds")
        print("Memory Usage Before Execution:", memory_usage_before, "MB")
        print("RAM Usage Before Execution:", ram_usage_before, "MB")
        print("Memory Usage After Execution:", memory_usage_after, "MB")
        print("RAM Usage After Execution:", ram_usage_after, "MB")

        # Collect the garbage
        collected = gc.collect()
        print(f"Garbage collected: {collected} objects")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Classification")
    parser.add_argument("--audio_folder", type=str, default=r"/kaggle/input/audio-dataset-with-10-indian-languages/Language Detection Dataset",
                        help="Path to the main directory containing language folders.")
    parser.add_argument("--checkpoint_path", type=str, default="vggish_model.ckpt",
                        help="Path to the VGGish checkpoint file.")
    parser.add_argument("--pca_params_path", type=str, default="vggish_pca_params.npz",
                        help="Path to the VGGish PCA parameters file.")
    parser.add_argument("--total_samples", type=int, default=1000,
                        help="Total number of samples to consider.")
    parser.add_argument("--samples_per_folder", type=int, default=100,
                        help="Samples for each language")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for generating embeddings.")
    parser.add_argument("--training_size", type=float, default=0.8,
                        help="Proportion of samples to use for training.")
    parser.add_argument("--model", type=str, default="svm",
                        choices=["svm", "random_forest"],
                        help="Model to use for classification.")
    parser.add_argument("--run_name", type=str, default="run0",
                        help="Name of the run.")
    args = parser.parse_args()

    if args.model == "svm":
        model = models.SVMClassifier()
    elif args.model == "random_forest":
        model = models.RFClassifier()
    else:
        raise ValueError("Invalid model choice.")

    main = Main(args.audio_folder, args.checkpoint_path, args.pca_params_path, args.total_samples, args.samples_per_folder, args.batch_size, args.training_size, model, args.run_name)
    print("Run Name: ", args.run_name)
    print("ML Model: ", args.model)
    
    # Check if a GPU is available
    if tf.test.is_gpu_available():
        print("GPU is available")
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    else:
        print("GPU is not available")
    main.run()
