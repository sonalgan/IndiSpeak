import csv
import numpy as np
import soundfile as sf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tqdm
import tensorflow.compat.v1 as tf
from feature_extractor import FeatureExtractor

# Define the necessary flags
tf.compat.v1.flags.DEFINE_string('audio_folder', '/kaggle/input/audio-dataset-with-10-indian-languages/Language Detection Dataset', 'Path to the folder containing audio samples')
tf.compat.v1.flags.DEFINE_string('checkpoint', '/kaggle/working/vggish_model.ckpt', 'Path to the VGGish checkpoint file')
tf.compat.v1.flags.DEFINE_string('pca_params', '/kaggle/working/vggish_pca_params.npz', 'Path to the VGGish PCA parameters file')
tf.compat.v1.flags.DEFINE_integer('samples_per_folder', 100, 'Number of samples to include per folder')

FLAGS = tf.compat.v1.flags.FLAGS


def main(_):
    feature_extractor = FeatureExtractor(FLAGS.checkpoint, FLAGS.pca_params)
    feature_extractor.load_model()

    embeddings = []
    labels = []
    progress_bar = tqdm.tqdm(feature_extractor.audio_embeddings_generator(FLAGS.audio_folder, FLAGS.samples_per_folder), desc='Extracting embeddings')
    for batch_embeddings, batch_labels in progress_bar:
        embeddings.append(batch_embeddings)
        labels.append(batch_labels)

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    print('Embeddings shape:', embeddings.shape)
    print('Labels shape:', labels.shape)

    # Train models


if __name__ == '__main__':
    tf.compat.v1.app.run(main)
