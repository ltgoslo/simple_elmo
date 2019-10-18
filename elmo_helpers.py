# python3
# coding: utf-8

import sys
import re
import os
import tensorflow as tf
from bilm import Batcher, BidirectionalLanguageModel, weight_layers


def tokenize(string):
    """
    :param string: well, text string
    :return: list of tokens
    """
    token_pattern = re.compile('(?u)\w+')
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
    print('Sentences in this batch:', len(texts), file=sys.stderr)

    # Compute ELMo representations.
    elmo_sentence_input_ = sess.run(elmo_sentence_input['weighted_op'],
                                    feed_dict={sentence_character_ids: sentence_ids})

    return elmo_sentence_input_


def load_elmo_embeddings(directory, top=False):
    """
    :param directory: directory with an ELMo model ('model.hdf5', 'options.json' and 'vocab.txt.gz')
    :param top: use ony top ELMo layer
    :return: ELMo batcher, character id placeholders, op object
    """
    vocab_file = os.path.join(directory, 'vocab.txt.gz')
    options_file = os.path.join(directory, 'options.json')
    weight_file = os.path.join(directory, 'model.hdf5')

    # Create a Batcher to map text to character ids.
    batcher = Batcher(vocab_file, 50)

    # Input placeholders to the biLM.
    sentence_character_ids = tf.placeholder('int32', shape=(None, None, 50))

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(options_file, weight_file, max_batch_size=300)

    # Get ops to compute the LM embeddings.
    sentence_embeddings_op = bilm(sentence_character_ids)

    # Get an op to compute ELMo (weighted average of the internal biLM layers)
    elmo_sentence_input = weight_layers('input', sentence_embeddings_op, use_top_only=top)
    return batcher, sentence_character_ids, elmo_sentence_input
