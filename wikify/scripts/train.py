# from docopt import docopt
import pickle

from wikify.keyphraseness import Keyphraseness


if __name__ == '__main__':

    # load the data
    file = './wiki-dump/mini/enwiki-test1gb'

    # train the model
    model = Keyphraseness(file)

    # save it
    filename = './wiki-dump/mini/Keyphraseness'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
