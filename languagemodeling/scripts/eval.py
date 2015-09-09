"""Evaulate a language model using the test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import RegexpTokenizer

from ngram import EvalModel

pattern = '''(?ix)    # set flag to allow verbose regexps
      (sr\.|sra\.|dr\.|dra\.)
    | ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
    | \w+(-\w+)*        # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
    | \.\.\.            # ellipsis
    | [][<>|\{}.,;"'“”«»¡!¿?():-_`]  # these are separate tokens; includes ], [
'''

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
    tokenizer = RegexpTokenizer(pattern)
    test_corpus = PlaintextCorpusReader('./corpus', 'books_corpus_test.txt',\
                                        word_tokenizer=tokenizer)
    test_sents = test_corpus.sents()

    # Evaluating model
    evalModel = EvalModel(model, test_sents)

    print('Log-Probability', evalModel.log_prob())
    print('Corss-Entropy', evalModel.cross_entropy())
    print('Perplexity', evalModel.perplexity())
