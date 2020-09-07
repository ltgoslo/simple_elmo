# python3
# coding: utf-8

import argparse
from elmo_helpers import *
from smart_open import open
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', '-i', help='Path to input text, one sentence per line', required=True)
    arg('--elmo', '-e', help='Path to ELMo model', required=True)

    args = parser.parse_args()
    data_path = args.input

    raw_sentences = []

    with open(data_path, 'r') as f:
        for line in f:
            res = line.strip()
            raw_sentences.append(res)
    sentences = [tokenize(s) for s in raw_sentences]

    print('=====')
    print('%d sentences total' % len(sentences))
    print('=====')

    # We do not use eager execution from TF 2.0
    tf.compat.v1.disable_eager_execution()

    # Loading a pre-trained ELMo model:
    # You can call load_elmo_embeddings() with top=True to use only the top ELMo layer
    batcher, sentence_character_ids, elmo_sentence_input = load_elmo_embeddings(args.elmo)

    # Actually producing ELMo embeddings for our data:
    with tf.compat.v1.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.compat.v1.global_variables_initializer())
        elmo_vectors = get_elmo_vectors(
            sess, sentences, batcher, sentence_character_ids, elmo_sentence_input)

    print('ELMo embeddings for your input are ready')
    print('Tensor shape:', elmo_vectors.shape)

    # Due to batch processing, the above code produces for each sentence
    # the same number of token vectors, equal to the length of the longest sentence
    # (the 2nd dimension of the elmo_vector tensor).
    # If a sentence is shorter, the vectors for non-existent words are filled with zeroes.
    # Let's make a version without these redundant vectors:
    cropped_vectors = []
    for vect, sent in zip(elmo_vectors, sentences):
        cropped_vector = vect[:len(sent), :]
        cropped_vectors.append(cropped_vector)

    # A quick test:
    # in each sentence, we find the tokens most similar to the 2nd token of the first sentence
    query_nr = 2
    query_word = sentences[0][query_nr]
    print('Query sentence:', sentences[0])
    print('Query:', query_word)
    query_vec = cropped_vectors[0][query_nr, :]

    for sent_nr, sent in enumerate(sentences):
        if sent_nr == 0:
            continue
        print('======')
        print(sent)
        sims = {}
        for nr, word in enumerate(sent):
            w_vec = cropped_vectors[sent_nr][nr, :]
            sims[word] = np.dot(query_vec, w_vec)

        for k in sorted(sims, key=sims.get, reverse=True):
            print(k, sims[k])
