import os
import numpy as np
import soundfile as sf
import tensorflow.compat.v1 as tf
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

class FeatureExtractor:
    
    def __init__(self, checkpoint_path, pca_params_path):
        self.checkpoint_path = checkpoint_path
        self.pca_params_path = pca_params_path
        self.sess = tf.Session(graph=tf.Graph())
        self.pproc = None
        self.features_tensor = None
        self.embedding_tensor = None

    def load_model(self):
        with self.sess.graph.as_default():
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(self.sess, self.checkpoint_path)
            self.features_tensor = self.sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            self.embedding_tensor = self.sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
            self.pproc = vggish_postprocess.Postprocessor(self.pca_params_path)

    def extract_features(self, waveform):
        examples = vggish_input.waveform_to_examples(waveform, vggish_params.SAMPLE_RATE)
        [embedding_batch] = self.sess.run([self.embedding_tensor],
                                          feed_dict={self.features_tensor: examples})
        postprocessed_batch = self.pproc.postprocess(embedding_batch)
        return postprocessed_batch

    def audio_embeddings_generator(self, audio_folder, samples_per_folder=100):
        audio_files = []
        labels = []

        for folder_name in os.listdir(audio_folder):
            folder_path = os.path.join(audio_folder, folder_name)
            if os.path.isdir(folder_path):
                audio_files_in_folder = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.mp3')]
                audio_files_in_folder = audio_files_in_folder[:samples_per_folder]
                audio_files.extend(audio_files_in_folder)
                labels.extend([self.extract_label_from_audio_file(file) for file in audio_files_in_folder])

        batch_embeddings = []
        batch_labels = []
        for audio_file, label in zip(audio_files, labels):
            waveform, _ = sf.read(audio_file)
            waveform = np.asarray(waveform)
            embedding = self.extract_features(waveform)
            batch_embeddings.append(embedding)
            batch_labels.append(label)
            if len(batch_embeddings) == samples_per_folder:
                yield np.array(batch_embeddings), np.array(batch_labels)
                batch_embeddings = []
                batch_labels = []

        if batch_embeddings:
            yield np.array(batch_embeddings), np.array(batch_labels)

    @staticmethod
    def extract_label_from_audio_file(audio_file):
        label = os.path.basename(os.path.dirname(audio_file))
        return label
