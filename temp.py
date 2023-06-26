import numpy as np
import os
import tensorflow.compat.v1 as tf
import tf_slim as slim
import soundfile as sf
from collections import defaultdict
from tensorflow.python.platform import flags
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import csv
import tqdm
import warnings
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from feature_extractor import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

flags = tf.app.flags

flags.DEFINE_boolean(
    'train_vggish', True,
    'If True, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_string(
    'audio_folder', '/kaggle/input/audio-dataset-with-10-indian-languages/Language Detection Dataset',
    'Path to the folder containing audio files.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_integer(
    'batch_size', 32,
    'Number of audio files to process in each batch.')
flags.DEFINE_integer('num_batches', 32, 'Number of batches')

FLAGS = flags.FLAGS

_NUM_CLASSES = 10

def generate_audio_file_paths(audio_folder, num_files=1000, files_per_folder=100, csv_file='audio_files.csv'):
    # Check if the CSV file already exists
    if os.path.isfile(csv_file):
        print(f"CSV file '{csv_file}' already exists. Skipping audio file path generation.")
        return

    # Walk over the specified number of files in each folder
    audio_files = []
    for root, dirs, files in os.walk(audio_folder):
        count = 0
        for file in files:
            audio_files.append(os.path.join(root, file))
            count += 1
            if count >= files_per_folder:
                break
        if len(audio_files) >= num_files:
            break

    # Save the audio file paths to a CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Audio File Path'])  # Write the header
        writer.writerows([[path] for path in audio_files])  # Write each audio file path
        
def _get_examples_batch(audio_files, batch_size):
    features = []
    labels = []

    for audio_file in audio_files[:batch_size]:
        waveform, sr = sf.read(audio_file)
        examples = vggish_input.waveform_to_examples(waveform, sr)
        features.extend(examples)
        label = extract_label_from_audio_file(audio_file)
        labels.extend([label] * len(examples))

    return np.array(features), np.array(labels)


def main(_):
    # Generate audio file paths and save to a CSV file
    generate_audio_file_paths(FLAGS.audio_folder, num_files=1000, files_per_folder=100, csv_file='audio_files.csv')

    # Load audio file paths from CSV
    audio_files = []
    with open('audio_files.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            audio_files.append(row[0])

    # Initialize the feature extractor and load the model
    feature_extractor = FeatureExtractor(FLAGS.checkpoint, FLAGS.pca_params)
    feature_extractor.load_model()

    # Extract features and labels from audio files
    features = []
    labels = []
    for audio_file in tqdm.tqdm(audio_files, desc='Extracting features'):
        waveform, sr = sf.read(audio_file)
        waveform = np.asarray(waveform)  # Convert waveform to NumPy array
        features.append(feature_extractor.extract_features(waveform))
        labels.append(extract_label_from_audio_file(audio_file))

    
    # Apply one-hot encoding to the labels
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    labels = label_encoder.fit_transform(labels)
    labels = labels.reshape(len(labels), 1)
    labels = onehot_encoder.fit_transform(labels)

    features = np.array(features)
    labels = np.array(labels)

    print('Features shape:', features.shape)
    print('Labels shape:', labels.shape)

    # Train the model
    #train_model(features, labels)

if __name__ == '__main__':
    tf.compat.v1.app.run(main)
