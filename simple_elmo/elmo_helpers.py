# /bin/env python3
# coding: utf-8

import sys
import os
import numpy as np
import tensorflow as tf
import json
import zipfile
import logging
from simple_elmo.data import Batcher
from simple_elmo.model import BidirectionalLanguageModel
from simple_elmo.elmo import weight_layers


class ElmoModel:
    """
    Embeddings from Language Models (ELMo)
    """

    def __init__(self):
        self.batcher = None
        self.sentence_character_ids = None
        self.elmo_sentence_input = None
        self.sentence_embeddings_op = None
        self.batch_size = None
        self.max_chars = None
        self.vector_size = None
        self.n_layers = None

        # We do not use eager execution from TF 2.0
        tf.compat.v1.disable_eager_execution()

        logging.basicConfig(
            format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)

    def load(self, directory, max_batch_size=32, limit=100):
        # Loading a pre-trained ELMo model:
        # You can call load with top=True to use only the top ELMo layer
        """
        :param directory: directory or a ZIP archive with an ELMo model
        ('*.hdf5' and 'options.json' files must be present)
        :param max_batch_size: the maximum allowable batch size during inference
        :param limit: cache only the first <limit> words from the vocabulary file
        :return: ELMo batcher, character id placeholders, op object
        """
        if not os.path.exists(directory):
            raise SystemExit(f"Error: path  not found for {directory}!")
        self.batch_size = max_batch_size
        self.logger.info(f"Loading model from {directory}...")

        if os.path.isfile(directory) and directory.endswith(".zip"):
            message = """
            Assuming the model is a ZIP archive downloaded from the NLPL vector repository.
            Loading a model from a ZIP archive directly is slower than from the extracted files,
            but does not require additional disk space
            and allows to load from directories without write permissions.
            """
            self.logger.info(message)
            if sys.version_info.major < 3 or sys.version_info.minor < 7:
                raise SystemExit(
                    "Error: loading models from ZIP archives requires Python >= 3.7."
                )
            zf = zipfile.ZipFile(directory)
            vocab_file = zf.open("vocab.txt")
            options_file = zf.open("options.json")
            weight_file = zf.open("model.hdf5")
            m_options = json.load(options_file)
            options_file.seek(0)
        elif os.path.isdir(directory):
            # We have all the files already extracted in a separate directory
            if os.path.isfile(os.path.join(directory, "vocab.txt.gz")):
                vocab_file = os.path.join(directory, "vocab.txt.gz")
            elif os.path.isfile(os.path.join(directory, "vocab.txt")):
                vocab_file = os.path.join(directory, "vocab.txt")
            else:
                self.logger.info("No vocabulary file found in the model.")
                vocab_file = None
            if os.path.exists(os.path.join(directory, "model.hdf5")):
                weight_file = os.path.join(directory, "model.hdf5")
            else:
                weight_files = [
                    fl for fl in os.listdir(directory) if fl.endswith(".hdf5")
                ]
                if not weight_files:
                    raise SystemExit(
                        f"Error: no HDF5 model files found in the {directory} directory!"
                    )
                weight_file = os.path.join(directory, weight_files[0])
                self.logger.info(
                    f"No model.hdf5 file found. Using {weight_file} as a model file."
                )
            options_file = os.path.join(directory, "options.json")
            with open(options_file, "r") as of:
                m_options = json.load(of)
        else:
            raise SystemExit(
                "Error: either provide a path to a directory with the model "
                "or to the model in a ZIP archive."
            )

        max_chars = m_options["char_cnn"]["max_characters_per_token"]
        self.max_chars = max_chars
        if m_options["char_cnn"]["n_characters"] == 261:
            raise SystemExit(
                "Error: invalid number of characters in the options.json file: 261. "
                "Set n_characters to 262 for inference."
            )

        # Create a Batcher to map text to character ids.
        self.batcher = Batcher(vocab_file, max_chars, limit=limit)

        # Input placeholders to the biLM.
        self.sentence_character_ids = tf.compat.v1.placeholder(
            "int32", shape=(None, None, max_chars)
        )

        # Build the biLM graph.
        bilm = BidirectionalLanguageModel(
            options_file, weight_file, max_batch_size=max_batch_size
        )

        self.vector_size = int(bilm.options["lstm"]["projection_dim"] * 2)
        self.n_layers = bilm.options["lstm"]["n_layers"] + 1

        # Get ops to compute the LM embeddings.
        self.sentence_embeddings_op = bilm(self.sentence_character_ids)

        return "The model is now loaded."

    def get_elmo_vectors(self, texts, warmup=True, layers="average"):
        """
        :param texts: list of sentences (lists of words)
        :param warmup: warm up the model before actual inference (by running it over the 1st batch)
        :param layers: ["top", "average", "all"].
        Yield the top ELMo layer, the average of all layers, or all layers as they are.
        :return: embedding tensor for all sentences
        (number of used layers by max word count by vector size)
        """
        max_text_length = max([len(t) for t in texts])

        # Creating the matrix which will eventually contain all embeddings from all batches:
        if layers == "all":
            final_vectors = np.zeros((len(texts), self.n_layers, max_text_length, self.vector_size))
        else:
            final_vectors = np.zeros((len(texts), max_text_length, self.vector_size))

        with tf.compat.v1.Session() as sess:
            # Get an op to compute ELMo vectors (a function of the internal biLM layers)
            self.elmo_sentence_input = weight_layers("input", self.sentence_embeddings_op,
                                                     use_layers=layers)

            # It is necessary to initialize variables once before running inference.
            sess.run(tf.compat.v1.global_variables_initializer())

            if warmup:
                self.warmup(sess, texts)

            # Running batches:
            chunk_counter = 0
            for chunk in divide_chunks(texts, self.batch_size):
                # Converting sentences to character ids:
                sentence_ids = self.batcher.batch_sentences(chunk)
                self.logger.info(f"Texts in the current batch: {len(chunk)}")

                # Compute ELMo representations.
                elmo_vectors = sess.run(
                    self.elmo_sentence_input["weighted_op"],
                    feed_dict={self.sentence_character_ids: sentence_ids},
                )
                # Updating the full matrix:
                first_row = self.batch_size * chunk_counter
                last_row = first_row + elmo_vectors.shape[0]
                if layers == "all":
                    final_vectors[first_row:last_row, :, : elmo_vectors.shape[2], :] = elmo_vectors
                else:
                    final_vectors[first_row:last_row, : elmo_vectors.shape[1], :] = elmo_vectors
                chunk_counter += 1

            return final_vectors

    def get_elmo_vector_average(self, texts, warmup=True, layers="average"):
        """
        :param texts: list of sentences (lists of words)
        :param warmup: warm up the model before actual inference (by running it over the 1st batch)
        :param layers: ["top", "average", "all"].
        Yield the top ELMo layer, the average of all layers, or all layers as they are.
        :return: matrix of averaged embeddings for all sentences
        """

        if layers == "all":
            average_vectors = np.zeros((len(texts), self.n_layers, self.vector_size))
        else:
            average_vectors = np.zeros((len(texts), self.vector_size))

        counter = 0

        with tf.compat.v1.Session() as sess:
            # Get an op to compute ELMo vectors (a function of the internal biLM layers)
            self.elmo_sentence_input = weight_layers("input", self.sentence_embeddings_op,
                                                     use_layers=layers)

            # It is necessary to initialize variables once before running inference.
            sess.run(tf.compat.v1.global_variables_initializer())

            if warmup:
                self.warmup(sess, texts)

            # Running batches:
            for chunk in divide_chunks(texts, self.batch_size):
                # Converting sentences to character ids:
                sentence_ids = self.batcher.batch_sentences(chunk)
                self.logger.info(f"Texts in the current batch: {len(chunk)}")

                # Compute ELMo representations.
                elmo_vectors = sess.run(
                    self.elmo_sentence_input["weighted_op"],
                    feed_dict={self.sentence_character_ids: sentence_ids},
                )

                self.logger.debug(f"ELMo sentence input shape: {elmo_vectors.shape}")

                if layers == "all":
                    elmo_vectors = elmo_vectors.reshape((len(chunk), elmo_vectors.shape[2],
                                                         self.n_layers, self.vector_size))
                for sentence in range(len(chunk)):
                    if layers == "all":
                        sent_vec = np.zeros((elmo_vectors.shape[1], self.n_layers,
                                             self.vector_size))
                    else:
                        sent_vec = np.zeros((elmo_vectors.shape[1], self.vector_size))
                    for nr, word_vec in enumerate(elmo_vectors[sentence]):
                        sent_vec[nr] = word_vec
                    semantic_fingerprint = np.sum(sent_vec, axis=0)
                    semantic_fingerprint = np.divide(
                        semantic_fingerprint, sent_vec.shape[0]
                    )
                    query_vec = semantic_fingerprint / np.linalg.norm(
                        semantic_fingerprint
                    )

                    average_vectors[counter] = query_vec
                    counter += 1

        return average_vectors

    def warmup(self, sess, texts):
        for chunk0 in divide_chunks(texts, self.batch_size):
            self.logger.info(f"Warming up ELMo on {len(chunk0)} sentences...")
            sentence_ids = self.batcher.batch_sentences(chunk0)
            _ = sess.run(
                self.elmo_sentence_input["weighted_op"],
                feed_dict={self.sentence_character_ids: sentence_ids},
            )
            break
        self.logger.info("Warming up finished.")


def divide_chunks(data, n):
    for i in range(0, len(data), n):
        yield data[i: i + n]
