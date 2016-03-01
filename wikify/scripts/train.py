"""Train components of wikify
Usage:
  train.py -m <model> -i <file> [--vocabulary <file>] [--classes <file>] [-n <n>]
           [--surround <s>] [--top <t>] [-c <classifier>] [--ratio <r>]
           [--saveVocabulary] [--saveClasses] -o <file>
  train.py -h | --help
Options:
  -m <model>            Model to train:
                          keyphraseness: Train the 'keyphraseness' ranking method.
                          disambiguation: Train the disambiguation method.
  -i <file>             XML dump to train the model.
  --vocabulary <file>   Vocabulary learned in the keyphraseness method.
  --classes <file>      All the possible classes that we need to disambiguate.
  -n <n>                N-gram range (1, n) [default: 3].
  --surround <s>        Amount of sorround words used in disambiguation features [default: 6].
  --top <t>             Amount of top words used in disambiguation features [default: 10].
  -c <classifier>       Classfier to use in word disambiguation [default: SGDClassifier]:
                          MultinomialNB: Multinomial Naive Bayes
                          LogisticRegression: Logistic Regression
                          LinearSVC: Linear Support Vector Classification
                          SGDClassifier
  --ratio <r>           Ratio between number of words in an article and number of
                        keywords, used in Keyphraseness [default: 0.04].
  --saveVocabulary      Save vocabulary of keyphraseness method [default: False].
  --saveClasses         Save classes of disambiguation method [default: False].
  -o <file>             Output model file.
  -h --help             Show this screen.
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
        model = Keyphraseness(xml=xml, n=eval(opts['-n']),
                              ratio=eval(opts['--ratio']))
        if opts['--saveVocabulary']:
            with open(opts['-o'] + '-vocabulary', 'wb') as f:
                pickle.dump(model._vocabulary, f)

    elif opts['-m'] == 'disambiguation':
        with open(opts['--vocabulary'], 'rb') as f:
            keyphraseness_vocabulary = pickle.load(f)

        classes = None
        if opts['--classes'] is not None:
            with open(opts['--classes'], 'rb') as f:
                classes = pickle.load(f)

        model = Disambiguation(xml=xml, vocabulary=keyphraseness_vocabulary,
                               n=eval(opts['-n']),
                               surround=eval(opts['--surround']),
                               top=eval(opts['--top']))

    # Save it
    with open(opts['-o'], 'wb') as f:
        pickle.dump(model, f)
