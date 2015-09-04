"""Evaulate a language model using the test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
"""
import pickle

from ngram import EvalModel

WORKPATH = '/home/pablo/facu/2015/pln/PLN-2015'


if __name__ == '__main__':
    opts = docopt(__doc__)
    
    # Load n-gram model
    f = None
    model_file = opts['-i']
    try:
        f = open(model_file, 'rb')
    except IOError:
        print('Cannot open', model_file)

    model = pickle.load(f)
    
    # Load test corpus
    test = open(os.path.join(WORKPATH, 'corpus/books_corpus_test.txt'), 'r')
    sents = test.read()
    sents_list = sents.split('\n')[: - 1]

    # Evaluating model
    evalModel = EvalModel(model, sents_list)

    print('Log-Probability', evalModel.log_prob())
    print('Corss-Entropy', cross_entropy())
    print('Perplexity', perplexity())
