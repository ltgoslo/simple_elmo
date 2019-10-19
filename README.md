# Simple ELMo
Minimal Python code to get vectors from pre-trained ELMo models in TensorFlow.

Heavily based on https://github.com/allenai/bilm-tf

# Usage example
If the model is a ZIP archive downloaded from the [NLPL vector repository](http://vectors.nlpl.eu/repository/),
unzip it first.

`python3 get_elmo_vectors.py -i test.txt -e ~/PATH_TO_ELMO/`

`PATH_TO_ELMO` is a directory containing 3 files:
- `model.hdf5`, pre-trained ELMo weights in HDF5 format;
- `options.json`, description of the model architecture in JSON;
- `vocab.txt`, one-word-per-line vocabulary of the most frequent words you would like to cache during inference
(not really necessary, the model will infer embeddings for OOV words from their characters).

Use the `elmo_vectors` tensor for your downstream tasks. 
Its dimensions are: (number of sentences, the length of the longest sentence, ELMo dimensionality).
