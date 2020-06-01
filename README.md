# BTP
This repository provides an implementation of three models for the question answering task of MS MARCO dataset - 1. [Deep Cascade Model](https://arxiv.org/abs/1811.11374), 2. [Masque](https://arxiv.org/abs/1901.02262v2) and 3. a model using passage ranking and extractive summarization.

### Dependencies
The packages required to run the codes are - allennlp, nltk, tqdm and pandas.


### Downloading the Dataset and Embeddings

Download MS MARCO Question Answering V2.1 dataset from https://microsoft.github.io/msmarco/ and save the files in data/msmarco/

Download Glove embeddings from http://nlp.stanford.edu/data/glove.6B.zip and save the files in data/glove/

Download 256-dimensional Elmo embedding [weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5) and [options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json) and save the files in data/elmo/

### Running Maque Model

To create the vocabulary, run the file Masque/create_vocab.py -
~~~
python create_vocab.py
~~~
To train the model, run the file Masque/train.py -
~~~
python train.py
~~~
To do the prediction, run the file Masque/predict.py
~~~
python predict.py
~~~
Use the official evaluaton [script](https://github.com/microsoft/MSMARCO-Question-Answering/tree/master/Evaluation) to evaluate.


### Running Deep Cascade Model

To preprocess the training set and development set data, run the file DeepCascadeModel/preprocess.py -
~~~
python preprocess.py --data [train|dev]
~~~
To train the model, run the file DeepCascadeModel/train.py -
~~~
python train.py
~~~
To do the prediction, run the file DeepCascade/predict.py
~~~
python predict.py
~~~
Use the official evaluaton [script](https://github.com/microsoft/MSMARCO-Question-Answering/tree/master/Evaluation) to evaluate.

### Running the Model with Passage Ranking and Extractive Summarization
#### Running Passage Ranking
To train the model, run the file passage_ranking_extractive_summarization/passage_ranking/train.py
~~~
python train.py
~~~
To do the prediction for training set(required for extractive summarization) and development set, run the file passage_ranking_extractive_summarization/passage_ranking/predict.py
~~~
python predict.py --data [train|dev]
~~~
To evaluate, run the file passage_ranking_extractive_summarization/passage_ranking/evaluate.py
~~~
python evaluate.py
~~~
#### Running Extractive Summarization
To train the model, run the file passage_ranking_extractive_summarization/extractive_summarization/train.py
~~~
python train.py
~~~
To do the prediction, run the file passage_ranking_extractive_summarization/extractive_summarization/predict.py
~~~
python predict.py
~~~
Use the official evaluaton [script](https://github.com/microsoft/MSMARCO-Question-Answering/tree/master/Evaluation) to evaluate.
