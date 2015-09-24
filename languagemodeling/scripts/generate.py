"""Generate natural language sentences using a language model.

Usage:
  generate.py -i <file> -n <n>
  generate.py -h | --help

Options:
  -i <file>     Language model file.
  -n <n>        Number of sentences to generate.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from languagemodeling.ngram import NGramGenerator


if __name__ == '__main__':
    opts = docopt(__doc__)

    f = None
    model_file = opts['-i']
    n = int(opts['-n'])

    try:
        f = open(model_file, 'rb')
    except IOError:
        print('Cannot open', model_file)

    model = pickle.load(f)
    generator = NGramGenerator(model)

    for i in range(n):
        sent = generator.generate_sent()
        print('* ' + ' '.join(sent))
