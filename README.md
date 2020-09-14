# Simple ELMo
_simple_elmo_ is a Python library to work with pre-trained [ELMo embeddings](https://allennlp.org/elmo) in TensorFlow.

This is a significantly updated wrapper to the [original ELMo implementation](https://github.com/allenai/bilm-tf).
The main changes are:
- more convenient and transparent data loading (including from compressed files)
- code adapted to modern TensorFlow versions (including TensorFlow 2).

# Usage

`pip install simple_elmo`

 `model = ElmoModel()`

 `model.load(PATH_TO_ELMO)`

 `elmo_vectors = model.get_elmo_vectors(SENTENCES)`
 
  `averaged_vectors = model.get_elmo_vector_average(SENTENCES)`

`PATH_TO_ELMO` is a ZIP archive downloaded from the [NLPL vector repository](http://vectors.nlpl.eu/repository/),
OR a directory containing 3 files extracted from such an archive:
- `model.hdf5`, pre-trained ELMo weights in HDF5 format;
- `options.json`, description of the model architecture in JSON;
- `vocab.txt`/`vocab.txt.gz`, one-word-per-line vocabulary of the most frequent words you would like to cache during inference
(not really necessary, the model will infer embeddings for OOV words from their characters).

`SENTENCES` is a list of sentences (lists of words).

Use the `elmo_vectors` and `averaged_vectors` tensors for your downstream tasks. 

`elmo_vectors` contains contextualized word embeddings. Its shape is: (number of sentences, the length of the longest sentence, ELMo dimensionality).

`averaged_vectors` contains one vector per each input sentence, 
constructed by averaging individual contextualized word embeddings. 
It is a list of vectors (the shape is (ELMo dimensionality)).


# Example scripts

We provide two example scripts to make it easier to start using _simple_elmo_ right away:
- [Token embeddings](https://github.com/ltgoslo/simple_elmo/blob/master/simple_elmo/get_elmo_vectors.py)
 
`python3 get_elmo_vectors.py -i test.txt -e ~/PATH_TO_ELMO/`

- [Text classification](https://github.com/ltgoslo/simple_elmo/blob/master/simple_elmo/text_classification.py)

`python3 text_classification.py -i paraphrases_lemm.tsv.gz -e ~/PATH_TO_ELMO/`

The second script can be used to perform document pair classification (like in text entailment or paraphrase detection).

Simple average of ELMo embeddings for all words in a document is used;
then, the cosine similarity between two documents is calculated and used as a classifier feature.

Example paraphrase datasets for Russian (adapted from http://paraphraser.ru/):
- https://rusvectores.org/static/testsets/paraphrases.tsv.gz
- https://rusvectores.org/static/testsets/paraphrases_lemm.tsv.gz (lemmatized)


The library requires Python >= 3.6.

