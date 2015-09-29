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

    # Compute the statistics
    words_list = []  # All the words
    tags_list = []  # All the tags
    # For each word, returns a list of their tags
    tags_by_word = defaultdict(set)
    # For each tag, returns a dict with their words and occurrence
    words_by_tags = defaultdict(lambda : defaultdict(int))
    word_frequence = defaultdict(int)
    for sent in sents:
        # words, tags = zip(*sent)
        for w, t in sent:
            tags_by_word[w].add(t)
            words_by_tags[t][w] += 1
            word_frequence[w] += 1
            words_list.append(w)
            tags_list.append(t)
    words_set = set(words_list)
    tags_set = set(tags_list)

    # Basic stats
    print('--- Basic Stats ---')
    print('Sents: {}'.format(len(sents)))
    print('Words: {}'.format(len(words_list)))
    print('Vocabulary: {}'.format(len(words_set)))
    print('Tags: {}'.format(len(tags_set)))
    print('')

    # Frequent tags
    top10_tags = sorted(words_by_tags.keys(),
                        key=lambda k: sum(words_by_tags[k].values()),
                        reverse=True)[:10]
    print('--- Frequent Tags ---')
    print(' Tag \tFrequence')
    for tag in top10_tags:
        frequence = sum(words_by_tags[tag].values())
        frequent_words = sorted(words_by_tags[tag].keys(),
                                key=lambda k: words_by_tags[tag][k],
                                reverse=True)[:5]
        print(' {} \t{} - {:2.2f}% = {}'.format(tag, frequence,
              (frequence / len(tags_list)) * 100, frequent_words))
    print('')

    # Ambiguity levels
    print('--- Ambiguity Levels ---')
    print(' Ambiguity \tWords')
    for level in range(1, 10):
        filtered_words = dict((word, t_list) for word, t_list in
                              tags_by_word.items() if level == len(t_list))
        top5_words = sorted(filtered_words.keys(),
                            key=lambda w: word_frequence[w], reverse=True)[:5]
        print(' {} tag  \t{} - {:2.2f}% = {}'.format(level, len(filtered_words),
              (len(filtered_words) / len(words_set)) * 100, top5_words))
