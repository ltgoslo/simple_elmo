_Simple_elmo_ is a Python library to work with pre-trained [ELMo embeddings](https://allennlp.org/elmo) in TensorFlow.

This is a significantly updated wrapper to the [original ELMo implementation](https://github.com/allenai/bilm-tf).
The main changes are:
- more convenient and transparent data loading (including from compressed files)
- code adapted to modern TensorFlow versions (including TensorFlow 2).

# Installation

`pip install --upgrade simple_elmo`

Make sure to update the package regularly, we are actively developing.

# Usage

 `from simple_elmo import ElmoModel`

 `model = ElmoModel()`

## Loading
 First, let's load a pretrained model from disk:

 `model.load(PATH_TO_ELMO)`

### Required arguments

 **PATH_TO_ELMO** is either a ZIP archive downloaded from the [NLPL vector repository](http://vectors.nlpl.eu/repository/),
OR a directory containing 2 files:
- `*.hdf5`, pre-trained ELMo weights in HDF5 format (_simple_elmo_ assumes the file is named `model.hdf5`;
if it is not found, the first existing file with the `.hdf5` extension will be used);
- `options.json`, description of the model architecture in JSON;

One can also provide a `vocab.txt`/`vocab.txt.gz` file in the same directory: 
a one-word-per-line vocabulary of words to be cached (as character id representations) before inference.
Even if it is not present at all, ELMo will still process all words normally.
However, providing the vocabulary file can slightly increase inference speed when working with very large corpora (by reducing the amount of word to char ids conversions).

### Optional arguments
- **max_batch_size**: *integer, default 32*;

   the maximum number of sentences/documents in a batch during inference;
   your input will be automatically split into chunks of the respective size;
   if your computational resources allow, you might want to increase this value.
- **limit**: *integer, default 100*;

    the number of words from the vocabulary file to actually cache (counted from the first line).
    Increase the default value if you are sure these words occur in your training data much more often than 1 or 2 times.
- **full**: *boolean, default False*;

    if True, will try to load the full model from TensorFlow checkpoints, together with the vocabulary.
    Models loaded this way can be used for language modeling.

## Working with models
 Currently, we provide three methods for loaded models (will be expanded in the future):

 - `model.get_elmo_vectors(SENTENCES)`
 
 - `model.get_elmo_vector_average(SENTENCES)`

 - `model.get_elmo_substitutes(RAW_SENTENCES)`

`SENTENCES` is a list of input sentences (lists of words).
`RAW_SENTENCES` is a list of input sentences as strings.

The `get_elmo_vectors()` method produces a tensor of contextualized word embeddings.
Its shape is (number of sentences, the length of the longest sentence, ELMo dimensionality).

The `get_elmo_vector_average()` method produces a tensor with one vector per each input sentence,
constructed by averaging individual contextualized word embeddings. 
Its shape is (number of sentences, ELMo dimensionality).

Both these methods can be used with the **layers** argument, which takes one of the three values:
- *average* (default): return the average of all ELMo layers for each word;
- *top*: return only the top (last) layer for each word;
- *all*: return all ELMo layers for each word 
(an additional dimension appears in the produced tensor, with the shape equal to the number of layers in the model, 3 as a rule)

Use these tensors for your downstream tasks.

Another argument for these methods is **session**. 
It defaults to **None** which means a new TensorFlow session is created automatically when the method is called.
This is convenient, since one does not have to worry about initializing the computational graph.
However, in some cases, you might want to re-use an existing session (for example, to call the method multiple times without the initialization overhead).

For this to work, one must do all the initialization manually before the method is called, for example:

```
import tensorflow as tf
from simple_elmo import ElmoModel

graph = tf.Graph()
with graph.as_default() as elmo_graph:
    elmo_model = ElmoModel()
    elmo_model.load(PATH_TO_ELMO)
...
with elmo_graph.as_default() as current_graph:
    tf_session = tf.compat.v1.Session(graph=elmo_graph)
        with tf_session.as_default() as sess:
            elmo_model.elmo_sentence_input = simple_elmo.elmo.weight_layers("input", elmo_model.sentence_embeddings_op)
            sess.run(tf.compat.v1.global_variables_initializer())
...
elmo_model.get_elmo_vectors(SENTENCES, session=tf_session)
elmo_model.get_elmo_vectors(SENTENCES2, session=tf_session)
...
```

The `get_elmo_substitutes()` method currently works only with the models loaded  with `full=True`.
For each input sentence, it  produces a list of lexical substitutes (LM predictions) for each word token in the sentence, produced by the forward and backward ELMo language models.
The substitutes are yielded as dictionaries containing the vocabulary identifiers of the most probable LM predictions, their lexical forms and their logit scores.
NB: this method is still experimental!

# Example scripts

We provide three example scripts to make it easier to start using _simple_elmo_ right away:
## [Inferring token embeddings](https://github.com/ltgoslo/simple_elmo/blob/master/simple_elmo/examples/get_elmo_vectors.py)
 
`python3 get_elmo_vectors.py -i test.txt -e ~/PATH_TO_ELMO/`

This script simply returns contextualized ELMo embeddings for the words in your input sentences.

## [Text pairs classification](https://github.com/ltgoslo/simple_elmo/blob/master/simple_elmo/examples/text_classification.py)

`python3 text_classification.py -i paraphrases_lemm.tsv.gz -e ~/PATH_TO_ELMO/`

This script can be used to perform document pair classification (like in text entailment or paraphrase detection).
Simple average of ELMo embeddings for all words in a document is used;
then, the cosine similarity between two documents is calculated and used as a single classifier feature.
Evaluated with macro F1 score and 10-fold cross-validation.

Example paraphrase dataset for English (adapted from [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398)):
- https://rusvectores.org/static/testsets/mrpc.tsv.gz

Example paraphrase datasets for Russian (adapted from http://paraphraser.ru/):
- https://rusvectores.org/static/testsets/paraphrases.tsv.gz
- https://rusvectores.org/static/testsets/paraphrases_lemm.tsv.gz (lemmatized)

## [Word sense disambiguation](https://github.com/ltgoslo/simple_elmo/blob/master/simple_elmo/examples/wsd_eval.py)

`python3 wsd_eval.py -i senseval3.tsv -e ~/PATH_TO_ELMO/`

This script takes as an input a word sense disambiguation (WSD) dataset and a pre-trained ELMo model.
It extracts token embeddings for ambiguous words and trains a simple Logistic Regression classifier  to predict word senses.
Averaged macro F1 score across all words in the test set is used as the evaluation measure (with 5-fold cross-validation).

Example WSD datasets for English (adapted from [Senseval 3](https://web.eecs.umich.edu/~mihalcea/senseval/senseval3/)):
- https://rusvectores.org/static/testsets/senseval3.tsv
- https://rusvectores.org/static/testsets/senseval3_lemm.tsv (lemmatized)

Example WSD datasets for Russian (adapted from [RUSSE'18](https://toloka.yandex.ru/datasets/)):
- https://rusvectores.org/static/testsets/russe_wsd.tsv
- https://rusvectores.org/static/testsets/russe_wsd_lemm.tsv (lemmatized)

# Frequently Asked Questions
### Where can I find pre-trained ELMo models?

Several repositories are available where one can download ELMo models compatible with _simple_elmo_:
- [NLPL vector repository](http://vectors.nlpl.eu/repository/)
(except _ELMoForManyLangs_ models trained on the CoNLL17 corpus; see below)
- [AllenNLP](https://allennlp.org/elmo)
- [CLARIN](https://www.clarin.si/repository/xmlui/handle/11356/1277)

### Can I load [ELMoForManyLangs](https://github.com/HIT-SCIR/ELMoForManyLangs) models?

Unfortunately not. These models are trained using a slightly different architecture.
Therefore, they are [not compatible](https://github.com/HIT-SCIR/ELMoForManyLangs/issues/1#issuecomment-427668469) neither with _AllenNLP_ nor with _simple_elmo_.
You should use the original [ELMoForManyLangs](https://github.com/HIT-SCIR/ELMoForManyLangs) code to work with these models.

### I see a lot of warnings about deprecated methods

This is normal. The  _simple_elmo_ library is based on the [original ELMo implementation](https://github.com/allenai/bilm-tf) which was aimed at the versions of TensorFlow which are very outdated today.
We significantly updated the code and fixed many warnings - but not all of them yet. The work continues (and will eventually lead to a complete switch to TensorFlow 2).

Meaniwhile, these warnings can be ignored: they do not harm the resulting embeddings in any way.

### Can I train my own ELMo with this library?

Currently we provide ELMo training code (updated and improved in the same way compared to the original implementation)
in a [separate repository](https://github.com/ltgoslo/simple_elmo_training).
It will be integrated into the _simple_elmo_ package in the nearest future.
