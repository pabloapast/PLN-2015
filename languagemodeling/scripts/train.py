"""Train an n-gram model.

Usage:
  train.py -n <n> [-m <model>] [-g <gamma>] [-b <beta>] [-a <addone>] -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
                  interpolated: N-grams with linear interpolation smoothing.
                  backoff: N-grams with backoff smoothing.
  -g <gamma>    Interpolation hyper-parameter, if not given is estimated using
                held-out data [default: None].
  -b <beta>     Discounting hyper-parameter, if not given is estimated using
                held-out data [default: None].
  -a <addone>   Whether to use addone smoothing [default: True].
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import RegexpTokenizer

from languagemodeling.ngram import NGram, AddOneNGram, InterpolatedNGram,\
                                   BackOffNGram

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

    # load the data
    tokenizer = RegexpTokenizer(pattern)
    corpus = PlaintextCorpusReader('./corpus', 'books_corpus_train.txt',\
                                   word_tokenizer=tokenizer)
    sents = corpus.sents()

    # train the model
    n = int(opts['-n'])
    m = (opts['-m'])
    g = (opts['-g'])
    b = (opts['-b'])
    a = (opts['-a'])

    if g == 'None':
        g = None
    else:
        g = int(g)
    if b == 'None':
        b = None
    else:
        b = int(b)
    if a in ['false', 'False']:
        a = False
    else:
        a = True

    model = None
    if m == 'addone':
        model = AddOneNGram(n, sents)
    elif m == 'interpolated':        
        model = InterpolatedNGram(n, sents, gamma=g, addone=a)
    elif m == 'backoff':
        model = BackOffNGram(n, sents, beta=b, addone=a)
    else:
        model = NGram(n, sents)

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
