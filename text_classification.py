# /bin/env python3
# coding: utf-8

import argparse
import warnings
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from elmo_helpers import *
import pandas as pd
import numpy as np

# You can use this code to perform document pair classification
# (like in text entailment or paraphrase detection).

# Example datasets for Russian:
# https://rusvectores.org/static/testsets/paraphrases.tsv.gz
# https://rusvectores.org/static/testsets/paraphrases_lemm.tsv.gz
# (adapted from http://paraphraser.ru/)


def classify(data_file, elmo=None, algo="logreg", batch_size=300):
    data = pd.read_csv(data_file, sep="\t", compression="gzip")
    print(data.head())

    train0 = []
    train1 = []
    y = data.label.values
    batcher, sentence_character_ids, elmo_sentence_input = elmo
    sentences0 = [t.split() for t in data.text0]
    sentences1 = [t.split() for t in data.text1]
    print("=====")
    print(f"{len(sentences0)} sentences total")
    print("=====")
    # Here we divide all the sentences into several chunks to reduce the batch size
    with tf.compat.v1.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.compat.v1.global_variables_initializer())
        for chunk in divide_chunks(sentences0, batch_size):
            train0 += get_elmo_vector_average(
                sess, chunk, batcher, sentence_character_ids, elmo_sentence_input
            )
        for chunk in divide_chunks(sentences1, batch_size):
            train1 += get_elmo_vector_average(
                sess, chunk, batcher, sentence_character_ids, elmo_sentence_input
            )

    classes = Counter(y)
    print(f"Distribution of classes in the whole sample: {dict(classes)}")

    x_train = [[np.dot(t0, t1)] for t0, t1 in zip(train0, train1)]
    print(f"Train shape: {len(x_train)}")

    if algo == "logreg":
        clf = LogisticRegression(
            solver="lbfgs", max_iter=2000, multi_class="auto", class_weight="balanced"
        )
    else:
        clf = MLPClassifier(hidden_layer_sizes=(200,), max_iter=500)
    dummy = DummyClassifier(strategy="stratified")

    scoring = ["precision_macro", "recall_macro", "f1_macro"]
    # some splits are containing samples of one class, so we split until the split is OK
    counter = 0

    cv_scores = None
    cv_scores_dummy = None

    while True:
        try:
            cv_scores = cross_validate(clf, x_train, y, cv=10, scoring=scoring)
            cv_scores_dummy = cross_validate(dummy, x_train, y, cv=10, scoring=scoring)
        except ValueError:
            counter += 1
            if counter > 500:
                print("Impossible to find a good split!")
                exit()
            continue
        else:
            # No error; stop the loop
            break

    scores = [
        cv_scores["test_precision_macro"].mean(),
        cv_scores["test_recall_macro"].mean(),
        cv_scores["test_f1_macro"].mean(),
    ]
    dummy_scores = [
        cv_scores_dummy["test_precision_macro"].mean(),
        cv_scores_dummy["test_recall_macro"].mean(),
        cv_scores_dummy["test_f1_macro"].mean(),
    ]
    print("Real scores:")
    print("=====")
    print(f"Precision: {scores[0]:.3f}")
    print(f"Recall: {scores[1]:.3f}")
    print(f"F1: {scores[2]:.3f}")

    print("Random choice scores:")
    print("=====")
    print(f"Precision: {dummy_scores[0]:.3f}")
    print(f"Recall: {dummy_scores[1]:.3f}")
    print(f"F1: {dummy_scores[2]:.3f}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--input", help="Path to tab-separated file with input data", required=True)
    arg("--elmo", required=True, help="Path to ELMo model")
    arg("--batch", type=int, help="Max batch size", default=300)

    args = parser.parse_args()
    data_path = args.input
    max_batch_size = args.batch

    # We do not use eager execution from TF 2.0
    tf.compat.v1.disable_eager_execution()

    emb_model = load_elmo_embeddings(
        args.elmo, top=False, max_batch_size=max_batch_size
    )
    eval_scores = classify(data_path, elmo=emb_model, batch_size=max_batch_size)
