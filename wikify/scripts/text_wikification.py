"""Text wikification
Usage:
  text_wikification.py -i <file> -o <file> -k <file> [--ratio <r>] -d <file>
  text_wikification.py -h | --help
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
    # Delete keywords contained in another keyword
    keywords_copy = keywords.copy()
    for key1 in keywords_copy:
        for key2 in keywords_copy:
            if key1 != key2 and key1 in key2:
                try:
                    keywords.remove(key1)
                except:
                    pass

    # Disambiguate
    for keyword in keywords:
        if keyword in clean_to_original:
            original_keyword = clean_to_original[keyword]
            original_k_index = text.index(original_keyword)
            l_words, r_words = extract_surround_words(original_keyword, text)
            s = ' '.join([l_words, keyword, r_words])

            keyword_id = disambiguation.disambiguate(keyword, s)
            final_keyword = '[[' + keyword_id + '|' + original_keyword + ']]'

            text = text.replace(original_keyword, final_keyword, 1)

    # Save output text
    with open(opts['-o'], 'w') as f:
        f.write(text)

