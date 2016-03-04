"""Text wikification
Usage:
  wikify.py -i <file> -o <file> -k <file> [--ratio <r>] -d <file>
  wikify.py -h | --help
Options:
  -i <file>     Input text file.
  -o <file>     Output text after wikification process.
  -k <file>     Keyphraseness trained model.
  --ratio <r>   Relation between number of words in an article and
                number of keywords, used in Keyphraseness [default: 0.04].
  -d <file>     Disambiguation trained model.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from wikify.utils import extract_surround_words, extract_n_grams


if __name__ == '__main__':
    opts = docopt(__doc__)

    # Keyphraseness model
    with open(opts['-k'], 'rb') as f:
        keyphraseness = pickle.load(f)
    # Ratio
    ratio = eval(opts['--ratio'])

    # Disambiguation model
    with open(opts['-d'], 'rb') as f:
        disambiguation = pickle.load(f)

    # Input text
    with open(opts['-i'], 'r') as f:
        text = f.read()

    sent = text.split()
    clean_to_original = extract_n_grams(sent, keyphraseness.n)

    # Rank keywords
    keywords = keyphraseness.rank(text, ratio)

    # Disambiguate
    for keyword in keywords:
        original_keyword = clean_to_original[keyword]
        original_k_index = text.index(original_keyword)
        # l_words = text[-150:original_k_index]
        # r_words = text[original_k_index + len(original_keyword):150]
        l_words, r_words = extract_surround_words(original_keyword, text)
        s = ' '.join([l_words, keyword, r_words])

        keyword_id = disambiguation.disambiguate(keyword, s)
        final_keyword = '[[' + keyword_id + '|' + original_keyword + ']]'

        text = text.replace(original_keyword, final_keyword, 1)

    with open(opts['-o'], 'w') as f:
        f.write(text)
