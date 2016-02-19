"""Train components of wikify
Usage:
  train.py -m <model> -i <xml> [-r <ratio>] [-c <cores>] -o <file>
  train.py -h | --help
Options:
  -m <model>    Model to use:
                  keyphraseness: Train the 'keyphraseness' ranking method
                  disambiguation: Train the disambiguation method
  -r <ratio>    Ratio between number of words in an article and number of
                keywords, used in Keyphraseness [default: 6%]
  -c <cores>    Number of cpu cores to use [default: 4].
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from wikify.keyphraseness import Keyphraseness


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    file = './wiki-dump/mini/enwiki-train5gb'

    # train the model
    model = Keyphraseness(file)

    # save it
    filename = './wiki-dump/mini/Keyphraseness'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
