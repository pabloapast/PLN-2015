"""Train a sequence tagger.

Usage:
  train.py [-m <model>] [-n <n>] [-a <addone>] [-c <classifier>] -o <file>
  train.py -h | --help

Options:
  -m <model>        Model to use [default: base]:
                      base: Baseline
                      mlhmm: Hidden Markov Model with Maximum Likelihood
                      memm: Maximum Entropy Markov Model
  -n <n>            Order of the model for memm [default: 1]
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
from tagging.hmm import MLHMM
from tagging.memm import MEMM


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)
    sents = list(corpus.tagged_sents())

    # train the model
    model = None
    model_type = opts['-m']
    if model_type == 'mlhmm':
        if opts['-a'] in ['false', 'False']:
            opts['-a'] = False
        else:
            opts['-a'] = True
        model = MLHMM(n=int(opts['-n']), tagged_sents=sents, addone=opts['-a'])
    elif model_type == 'mlhmm':
        model = MEMM(n=int(opts['-n']), tagged_sents=sents, classifier=opts['-c'])
    else:
        model = BaselineTagger(tagged_sents=sents)

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
