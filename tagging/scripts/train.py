"""Train a sequence tagger.

Usage:
  train.py [-m <model>] [-n <n>] [-a <addone>] [-c <classifier>] -o <file>
  train.py -h | --help

Options:
  -m <model>        Model to use [default: base]:
                      base: Baseline
                      hmm: Hidden Markov Model
                      mlhmm: Hidden Markov Model with Maximum Likelihood
                      memm: Maximum Entropy Markov Model
  -n <n>            Order of the model for memm [default: None]
  -a <addone>       Whether to use addone smoothing for mlhmm [default: True]
  -c <classifier>   Classifier to use in memm [default: LogisticRegression]:
                      LogisticRegression: Logistic Regression
                      LinearSVC: Linear Support Vector Classification
                      MultinomialNB: Multinomial Naive Bayes
  -o <file>         Output model file.
  -h --help         Show this screen.
"""
from docopt import docopt
import pickle

from corpus.ancora import SimpleAncoraCorpusReader
from tagging.baseline import BaselineTagger


models = {
    'base': BaselineTagger,
    'mlhmm': MLHMM,
    'memm': MEMM,
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)
    sents = list(corpus.tagged_sents())

    # train the model
    model_type = opts['-m']
    if model_type == 'base':
        model = models[model_type](sents)
    elif:


    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
