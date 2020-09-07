# /bin/env python3
# coding: utf-8

import sys
import re
import os
import numpy as np
import tensorflow as tf
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
from sklearn import preprocessing
import json
import zipfile


def tokenize(string):
    """
    :param string: well, text string
    :return: list of tokens
    """
    token_pattern = re.compile(r'(?u)\w+')
    tokens = [t.lower() for t in token_pattern.findall(string)]
    return tokens


def get_elmo_vectors(sess, texts, batcher, sentence_character_ids, elmo_sentence_input):
    """
    :param sess: TensorFlow session
    :param texts: list of sentences (lists of words)
    :param batcher: ELMo batcher object
    :param sentence_character_ids: ELMo character id placeholders
    :param elmo_sentence_input: ELMo op object
    :return: embedding matrix for all sentences (max word count by vector size)
    """

    # Create batches of data.
    sentence_ids = batcher.batch_sentences(texts)
    print(f'Sentences in this batch: {len(texts)}', file=sys.stderr)

    # Compute ELMo representations.
    elmo_sentence_input_ = sess.run(elmo_sentence_input['weighted_op'],
                                    feed_dict={sentence_character_ids: sentence_ids})

    return elmo_sentence_input_


def get_elmo_vector_average(sess, texts, batcher, sentence_character_ids, elmo_sentence_input):
    vectors = []

    # Create batches of data.
    sentence_ids = batcher.batch_sentences(texts)
    print(f'Sentences in this chunk: {len(texts)}', file=sys.stderr)
    # Compute ELMo representations.
    elmo_sentence_input_ = sess.run(elmo_sentence_input['weighted_op'],
                                    feed_dict={sentence_character_ids: sentence_ids})
    print(f'ELMo sentence input shape: {elmo_sentence_input_.shape}', file=sys.stderr)
    for sentence in range(len(texts)):
        sent_vec = np.zeros((elmo_sentence_input_.shape[1], elmo_sentence_input_.shape[2]))
        for word_vec in enumerate(elmo_sentence_input_[sentence, :, :]):
            sent_vec[word_vec[0], :] = word_vec[1]
        semantic_fingerprint = np.sum(sent_vec, axis=0)
        semantic_fingerprint = np.divide(semantic_fingerprint, sent_vec.shape[0])
        query_vec = preprocessing.normalize(semantic_fingerprint.reshape(1, -1), norm='l2')
        vectors.append(query_vec.reshape(-1))
    return vectors


def load_elmo_embeddings(directory, top=False, max_batch_size=128):
    """
    :param directory: directory or a ZIP archive with an ELMo model
    ('model.hdf5', 'options.json' and 'vocab.txt*' files must be present)
    :param top: use only top ELMo layer
    :param max_batch_size: the maximum allowable batch size during inference
    :return: ELMo batcher, character id placeholders, op object
    """

    if os.path.isfile(directory) and directory.endswith(".zip"):
        message = """
        Assuming the model is a ZIP archive downloaded from the NLPL vector repository.
        Loading a model from a ZIP archive directly is much slower than from the extracted files,
        but does not require additional disk space
        and allows to load from directories without write permissions.
        """
        print(message, file=sys.stderr)
        if sys.version_info.major < 3 or sys.version_info.minor < 7:
            raise SystemExit('Error: loading ELMo from ZIP archives requires Python >= 3.7.')
        zf = zipfile.ZipFile(directory)
        vocab_file = zf.open("vocab.txt")
        options_file = zf.open("options.json")
        weight_file = zf.open("model.hdf5")
        m_options = json.load(options_file)
        options_file.seek(0)
    else:
        # We have all the files already extracted in a separate directory
        if os.path.isfile(os.path.join(directory, 'vocab.txt.gz')):
            vocab_file = os.path.join(directory, 'vocab.txt.gz')
        elif os.path.isfile(os.path.join(directory, 'vocab.txt')):
            vocab_file = os.path.join(directory, 'vocab.txt')
        else:
            raise SystemExit('Error: no vocabulary file found in the directory.')
        options_file = os.path.join(directory, 'options.json')
        weight_file = os.path.join(directory, 'model.hdf5')
        with open(options_file, 'r') as of:
            m_options = json.load(of)

    max_chars = m_options['char_cnn']['max_characters_per_token']

    # Create a Batcher to map text to character ids.
    batcher = Batcher(vocab_file, max_chars)

    # Input placeholders to the biLM.
    sentence_character_ids = tf.compat.v1.placeholder('int32', shape=(None, None, max_chars))

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(options_file, weight_file, max_batch_size=max_batch_size)

    # Get ops to compute the LM embeddings.
    sentence_embeddings_op = bilm(sentence_character_ids)

    # Get an op to compute ELMo (weighted average of the internal biLM layers)
    elmo_sentence_input = weight_layers('input', sentence_embeddings_op, use_top_only=top)

    return batcher, sentence_character_ids, elmo_sentence_input


def divide_chunks(data, n):
    for i in range(0, len(data), n):
        yield data[i:i + n]
