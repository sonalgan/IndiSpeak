import os
import numpy as np
import soundfile as sf
import tensorflow.compat.v1 as tf
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
from sklearn.preprocessing import LabelEncoder
import random
import joblib
import warnings
warnings.filterwarnings("ignore")

class FeatureExtractor:
    def __init__(self, checkpoint_path, pca_params_path, run_name):
        self.checkpoint_path = checkpoint_path
        self.pca_params_path = pca_params_path
        self.sess = tf.Session(graph=tf.Graph())
        self.pproc = None
        self.features_tensor = None
        self.embedding_tensor = None
        self.label_encoder = LabelEncoder()
        self.run_name = run_name

    def load_model(self):
        tf.disable_v2_behavior()
        tf.reset_default_graph()
        with self.sess.graph.as_default():
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(self.sess, self.checkpoint_path)
            self.features_tensor = self.sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            self.embedding_tensor = self.sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
            self.pproc = vggish_postprocess.Postprocessor(self.pca_params_path)

    def extract_features(self, audio_file):
        try:
            examples = vggish_input.wavfile_to_examples(audio_file)
            [embedding_batch] = self.sess.run([self.embedding_tensor], feed_dict={self.features_tensor: examples})
            postprocessed_batch = self.pproc.postprocess(embedding_batch)
            return postprocessed_batch
        except Exception as e:
            print(f"Error extracting features from {audio_file}: {e}")
            return None

    def fit_label_encoder(self, labels):
        self.label_encoder.fit(labels)

    def encode_labels(self, labels):
        encoded_labels = self.label_encoder.transform(labels)
        return encoded_labels

    def audio_embeddings_generator(self, audio_folder, total_samples, samples_per_folder=100, training_size=0.8, mode="train"):
        audio_files = []
        labels = []

        for folder_name in os.listdir(audio_folder):
            folder_path = os.path.join(audio_folder, folder_name)
            if os.path.isdir(folder_path):
                audio_files_in_folder = [
                    os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.mp3')
                ]
                audio_files_in_folder = np.random.choice(audio_files_in_folder, int(samples_per_folder), replace=False)
                audio_files.extend(audio_files_in_folder)
                label = self.extract_label_from_audio_file(audio_files_in_folder[0])  # Extract label from the first file
                labels.extend([label for _ in range(len(audio_files_in_folder))])
                labels.extend([self.extract_label_from_audio_file(file) for file in audio_files_in_folder])

        indices = np.random.permutation(len(audio_files))
        audio_files = np.array(audio_files)[indices]
        labels = np.array(labels)[indices]
        num_train_samples = int(total_samples * training_size)
        testfiles_path = f"misc/testfiles_{self.run_name}.pkl"
        testlabels_path = f"misc/testlabels_{self.run_name}.pkl"

        if mode == "train":
            train_files = audio_files[:num_train_samples]
            train_labels = labels[:num_train_samples]

            joblib.dump(audio_files[num_train_samples:], testfiles_path)  # Save testing files
            joblib.dump(labels[num_train_samples:], testlabels_path)  # Save testing labels

            train_embeddings = []
            train_labels_batch = []
            max_length = 0  # Maximum length among all embeddings

            for audio_file, label in zip(train_files, train_labels):
                embedding = self.extract_features(audio_file)
                if embedding is not None:  # Check if the embedding is not empty
                    train_embeddings.append(embedding)
                    train_labels_batch.append(label)
                    max_length = max(max_length, embedding.shape[0])

            padded_train_embeddings = []
            for embedding in train_embeddings:
                padding_length = max_length - embedding.shape[0]
                padded_embedding = np.pad(embedding, [(0, padding_length), (0, 0)], mode='constant')
                padded_train_embeddings.append(padded_embedding)

            train_embeddings = np.stack(padded_train_embeddings, axis=0)
            train_labels_batch = np.array(train_labels_batch)

            yield train_embeddings, train_labels_batch

        elif mode == "test":
            if os.path.isfile(testfiles_path) and os.path.isfile(testlabels_path):
                test_files = joblib.load(testfiles_path)
                test_labels = joblib.load(testlabels_path)
            else:
                test_files = audio_files[num_train_samples:]
                test_labels = labels[num_train_samples:]

            test_embeddings = []
            test_labels_batch = []
            max_length = 0  # Maximum length among all embeddings
            for audio_file, label in zip(test_files, test_labels):
                embedding = self.extract_features(audio_file)
                if embedding is not None:  # Check if the embedding is not empty
                    test_embeddings.append(embedding)
                    test_labels_batch.append(label)
                    max_length = max(max_length, embedding.shape[0])

            padded_test_embeddings = []
            for embedding in test_embeddings:
                padding_length = max_length - embedding.shape[0]
                padded_embedding = np.pad(embedding, [(0, padding_length), (0, 0)], mode='constant')
                padded_test_embeddings.append(padded_embedding)

            if test_embeddings:  # Handle empty test embeddings
                test_embeddings = np.stack(padded_test_embeddings, axis=0)
                test_labels_batch = np.array(test_labels_batch)
                yield test_embeddings, test_labels_batch
            else:
                print("No test embeddings found.")
        else:
            raise ValueError("Invalid mode. Mode must be either 'train' or 'test'.")

    @staticmethod
    def extract_label_from_audio_file(audio_file):
        label = os.path.basename(os.path.dirname(audio_file))
        return label
