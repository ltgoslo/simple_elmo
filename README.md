# Simple ELMo
Minimal Python code to get vectors from pre-trained ELMo models in TensorFlow.

Heavily based on https://github.com/allenai/bilm-tf.
Requires Python >= 3.6

The main changes:
- more convenient data loading (including from compressed files)
- code adapted to recent TensorFlow versions (including TF 2.0).

# Usage example
If the model is a ZIP archive downloaded from the [NLPL vector repository](http://vectors.nlpl.eu/repository/),
unzip it first.

`python3 get_elmo_vectors.py -i test.txt -e ~/PATH_TO_ELMO/`

`PATH_TO_ELMO` is a directory containing 3 files:
- `model.hdf5`, pre-trained ELMo weights in HDF5 format;
- `options.json`, description of the model architecture in JSON;
- `vocab.txt`/`vocab.txt.gz`, one-word-per-line vocabulary of the most frequent words you would like to cache during inference
(not really necessary, the model will infer embeddings for OOV words from their characters).

Use the `elmo_vectors` tensor for your downstream tasks. 
Its dimensions are: (number of sentences, the length of the longest sentence, ELMo dimensionality).

# Text classification

Use this code to perform document pair classification (like in text entailment or paraphrase detection).

Simple average of ELMo embeddings for all words in a document is used;
then, the cosine similarity between two documents is calculated and used as a classifier feature.

Example datasets for Russian (adapted from http://paraphraser.ru/):
- https://rusvectores.org/static/testsets/paraphrases.tsv.gz
- https://rusvectores.org/static/testsets/paraphrases_lemm.tsv.gz (lemmatized)

`python3 text_classification.py --input paraphrases_lemm.tsv.gz --elmo ~/PATH_TO_ELMO/`


