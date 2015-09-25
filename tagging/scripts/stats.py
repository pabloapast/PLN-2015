"""Print corpus statistics.

Usage:
  stats.py
  stats.py -h | --help

Options:
  -h --help     Show this screen.
"""
from collections import defaultdict
from docopt import docopt

from corpus.ancora import SimpleAncoraCorpusReader


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/')
    sents = corpus.tagged_sents()

    # compute the statistics
    sents_count = 0
    words_count = 0
    tags_total_count = 0
    vocabulary_count = 0
    tags_count = 0
    vocabulary = set()
    tags = set()
    tags_dict = defaultdict(int)
    tags_words_dict = defaultdict(defaultdict)

    for sent in sents:
        sents_count += 1
        for word, tag in sent:
            words_count += 1
            tags_total_count += 1
            vocabulary.add(word)
            tags.add(tag)
            tags_dict[tag] += 1
            try:
                tags_words_dict[tag][word] += 1
            except KeyError:
                tags_words_dict[tag][word] = 1
    vocabulary_count = len(vocabulary)
    tags_count = len(tags)

    print('Sents:', sents_count)
    print('Words:', words_count)
    print('Vocabulary:', vocabulary_count)
    print('Tags set:', tags_count, end='\n\n')

    sorted_tags = sorted(tags_dict.items(), key=lambda tup: tup[1],
                         reverse=True)[:10]
    print('\n--- FREQUENT TAGS ---')
    for tag, t_count in sorted_tags:
        print('\n* ', tag, ': ', t_count, ', ',
              "{:0.2f}".format((t_count * 100) / tags_total_count), '%', sep='')

        sorted_words = sorted(tags_words_dict[tag].items(),
                              key=lambda tup: tup[1], reverse=True)[:5]
        for word, w_count in sorted_words:
            print('   - ', word, ': ', w_count, sep='')
