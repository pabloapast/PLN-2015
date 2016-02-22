"""Train components of wikify
Usage:
  train.py -m <model> -i <file> [-r <r>] [-c <c>] -o <file>
  train.py -h | --help
Options:
  -m <model>    Model to train:
                  keyphraseness: Train the 'keyphraseness' ranking method
                  disambiguation: Train the disambiguation method
  -i <file>     XML dump to train the model
  -r <r>        Ratio between number of words in an article and number of
                  keywords, used in Keyphraseness [default: 6%]
  -c <c>        Number of cpu cores to use [default: 4].
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from wikify.keyphraseness import Keyphraseness
from wikify.disambiguation import Disambiguation


if __name__ == '__main__':
    opts = docopt(__doc__)

    # Corpus
    xml = opts['-i']

    # Train the model
    if opts['-m'] == 'keyphraseness':
        model = Keyphraseness(xml=xml, processes=int(opts['-c']))
    elif opts['-m'] == 'disambiguation':
        model = Disambiguation(xml=xml)

    # Save it
    with open(opts['-o'], 'wb') as f:
        pickle.dump(model, f)
