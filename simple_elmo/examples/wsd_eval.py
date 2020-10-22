# /bin/env python3
# coding: utf-8

import argparse
import csv
import numpy as np
from collections import Counter
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
import operator
import re
from simple_elmo import ElmoModel

# You can use this code to evaluate word sense disambiguation abilities of ELMo models
# Requires scikit-learn

# Example WSD datasets for English (adapted from Senseval 3):
# https://rusvectores.org/static/testsets/senseval3.tsv
# https://rusvectores.org/static/testsets/senseval3_lemm.tsv (lemmatized)

# Example WSD datasets for Russian (adapted from RUSSE'18):
# https://rusvectores.org/static/testsets/russe_wsd.tsv
# https://rusvectores.org/static/testsets/russe_wsd_lemm.tsv (lemmatized)


def tokenize(string, limit=None):
    """
    :param string: well, text string
    :param limit: max tokens to output
    :return: list of tokens
    """
    token_pattern = re.compile(r"(?u)\w+")
    tokens = [t for t in token_pattern.findall(string)]
    tokens = tokens[:limit]
    return tokens


def load_dataset(data_file, max_tokens=350):
    lens = []
    data = csv.reader(open(data_file), delimiter="\t")
    _ = next(data)
    data_set = {}
    cur_lemma = None
    word_set = []
    senses_dic = {}
    mfs_dic = {}
    for row in data:
        i, lemma, sense_id, left, word, right, senses = row
        if lemma in senses_dic:
            try:
                senses_dic[lemma][sense_id] += 1
            except KeyError:
                senses_dic[lemma][sense_id] = 1
        else:
            senses_dic[lemma] = {sense_id: 1}
        if lemma != cur_lemma:
            if len(word_set) > 0:
                data_set[cur_lemma] = word_set
            word_set = []
            cur_lemma = lemma
        sent = " ".join([left, word, right])
        tok_sent = tokenize(sent)
        tok_sent = tok_sent[
            :max_tokens
        ]  # We keep only the first tokens, to reduce batch size
        sent = " ".join(tok_sent)
        sent_len = len(tok_sent)
        lens.append(sent_len)
        cl = int(sense_id)
        num = len(tokenize(left))
        word_set.append((sent, num, cl))
    data_set[cur_lemma] = word_set
    print("Dataset loaded")
    print(f"Sentences: {len(lens)}")
    print(f"Max length: {np.max(lens)}")
    print(f"Average length: {np.average(lens):.2f}")
    print(f"Standard deviation: {np.std(lens):.2f}")
    for word in senses_dic:
        mfs = max(senses_dic[word].items(), key=operator.itemgetter(1))[0]
        mfs_dic[word] = mfs
    return data_set, mfs_dic


def classify(data_file, elmo):
    data, mfs_dic = load_dataset(data_file)
    scores = []

    # data looks like {w1 = [[w1 context1, w1 context2, ...], [w2 context1, w2 context2, ...]], ...}
    for word in data:
        x_train = []
        sentences = [tokenize(el[0]) for el in data[word]]
        nums = [el[1] for el in data[word]]
        y = [el[2] for el in data[word]]
        print("=====")
        print(f"{len(sentences)} sentences total for {word}")
        print("=====")

        # Actually producing ELMo embeddings for our data:
        elmo_vectors = elmo.get_elmo_vectors(sentences)

        for sentence, nr in zip(range(len(sentences)), nums):
            query_vec = elmo_vectors[sentence, nr, :]
            query_vec = query_vec / np.linalg.norm(query_vec)
            x_train.append(query_vec)

        classes = Counter(y)
        print(f"Distribution of classes in the whole sample: {dict(classes)}")
        clf = LogisticRegression(
            solver="lbfgs", max_iter=1000, multi_class="auto", class_weight="balanced"
        )

        averaging = True  # Do you want to average the cross-validate metrics?

        scoring = ["precision_macro", "recall_macro", "f1_macro"]
        # some splits are containing samples of one class, so we split until the split is OK
        counter = 0
        cv_scores = None
        while True:
            try:
                cv_scores = cross_validate(clf, x_train, y, cv=5, scoring=scoring)
            except ValueError:
                counter += 1
                if counter > 500:
                    raise SystemExit("Impossible to find a good split!")
                continue
            else:
                # No error; stop the loop:
                break
        scores.append(
            [
                cv_scores["test_precision_macro"].mean(),
                cv_scores["test_recall_macro"].mean(),
                cv_scores["test_f1_macro"].mean(),
            ]
        )
        if averaging:
            print(
                f"Average Precision on 5-fold cross-validation: "
                f"{cv_scores['test_precision_macro'].mean():.3f} "
                f"(+/- {cv_scores['test_precision_macro'].std() * 2:.3f})"
            )
            print(
                f"Average Recall on 5-fold cross-validation: "
                f"{cv_scores['test_recall_macro'].mean():.3f} "
                f"(+/- {cv_scores['test_recall_macro'].std() * 2:.3f})"
            )
            print(
                f"Average F1 on 5-fold cross-validation: "
                f"{cv_scores['test_f1_macro'].mean():.3f} "
                f"(+/- {cv_scores['test_f1_macro'].std() * 2:.3f})"
            )
        else:
            print("Precision values on 5-fold cross-validation:")
            print(cv_scores["test_precision_macro"])
            print("Recall values on 5-fold cross-validation:")
            print(cv_scores["test_recall_macro"])
            print("F1 values on 5-fold cross-validation:")
            print(cv_scores["test_f1_macro"])

        print("\n")

    print("=====")
    print(
        f"Average precision value for all words: {float(np.mean([x[0] for x in scores])):.3f} "
        f"(+/- {np.std([x[0] for x in scores]) * 2:.3f})"
    )
    print(
        f"Average recall value for all words: {float(np.mean([x[1] for x in scores])):.3f} "
        f"(+/- {np.std([x[1] for x in scores]) * 2:.3f})"
    )
    print(
        f"Average F1 value for all words: {float(np.mean([x[2] for x in scores])):.3f}"
        f"(+/- {np.std([x[2] for x in scores]) * 2:.3f})"
    )
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--input", help="Path to tab-separated file with WSD data", required=True)
    arg("--elmo", help="Path to ELMo model", required=True)

    args = parser.parse_args()
    data_path = args.input

    model = ElmoModel()
    model.load(args.elmo, top=True)

    eval_scores = classify(data_path, model)
