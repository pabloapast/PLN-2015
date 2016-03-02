"""Train components of wikify
Usage:
  train.py -m <model> -i <file> [--vocabulary <file>] [-n <n>] [--surround <s>]
           [--ratio <r>] -o <file>
  train.py -h | --help
Options:
  -m <model>            Model to train:
                          keyphraseness: Train the 'keyphraseness' ranking
                          method.
                          disambiguation: Train the disambiguation method.
  -i <file>             XML dump to train the model.
  --vocabulary <file>   Load vocabulary learned in the keyphraseness method.
  -n <n>                N-gram range (1, n) [default: 3].
  --surround <s>        Amount of sorround words used in disambiguation
                        features [default: 6].
  --ratio <r>           Relation between number of words in an article and
                        number of keywords, used in Keyphraseness
                        [default: 0.04].
  -o <file>             Output model file.
  -h --help             Show this screen.
"""
from docopt import docopt
import pickle

from wikify.keyphraseness import Keyphraseness
from wikify.disambiguation import Disambiguation


if __name__ == '__main__':
    opts = docopt(__doc__)

    # XML corpus
    xml = opts['-i']

    # Train the model
    if opts['-m'] == 'keyphraseness':
        model = Keyphraseness(xml=xml, n=eval(opts['-n']),
                              ratio=eval(opts['--ratio']))

    elif opts['-m'] == 'disambiguation':
        with open(opts['--vocabulary'], 'rb') as f:
            keyphraseness = pickle.load(f)
            keyphraseness_vocabulary = keyphraseness.vocabulary

        model = Disambiguation(xml=xml, vocabulary=keyphraseness_vocabulary,
                               surround=eval(opts['--surround']))

    # Save it
    with open(opts['-o'], 'wb') as f:
        pickle.dump(model, f)
