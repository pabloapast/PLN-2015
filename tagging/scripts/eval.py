"""Evaulate a tagger.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Tagging model file.
  -h --help     Show this screen.
"""
from collections import defaultdict
from docopt import docopt
import pickle
import sys

from corpus.ancora import SimpleAncoraCorpusReader


def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)
    sents = list(corpus.tagged_sents())

    hits, hits_k, hits_u, total, total_k, total_u = 0, 0, 0, 0, 0, 0
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    words_by_tags = defaultdict(lambda: defaultdict(int))
    n = len(sents)
    for i, sent in enumerate(sents):
        word_sent, gold_tag_sent = zip(*sent)

        model_tag_sent = model.tag(word_sent)
        assert len(model_tag_sent) == len(gold_tag_sent), i

        # global score
        hits_sent = [m == g for m, g in zip(model_tag_sent, gold_tag_sent)]
        hits_known = [m == g for j, (m, g) in
                      enumerate(zip(model_tag_sent, gold_tag_sent))
                      if not model.unknown(word_sent[j])]

        # Make confusion matrix
        for m, g in zip(model_tag_sent, gold_tag_sent):
            confusion_matrix[g][m] += m != g

        # For each tag, make a dict with their words and occurrence
        for word, tag in sent:
            words_by_tags[tag][word] += 1

        hits += sum(hits_sent)
        total += len(sent)
        hits_k += sum(hits_known)
        total_k += len(hits_known)

        acc = float(hits) / total
        acc_known = float(hits_k) / total_k
        acc_unknown = (float(hits) - float(hits_k)) / (total - total_k)
        progress('{:3.1f}% ({:2.2f}% / {:2.2f}% / {:2.2f}%)'.format(
                 float(i) * 100 / n, acc * 100, acc_known * 100,
                 acc_unknown * 100))

    acc = float(hits) / total
    acc_known = float(hits_k) / total_k
    acc_unknown = (float(hits) - float(hits_k)) / (total - total_k)

    print('')
    print('Accuracy: {:2.2f}%'.format(acc * 100))
    print('Accuracy known words: {:2.2f}%'.format(acc_known * 100))
    print('Accuracy unknown words: {:2.2f}%'.format(acc_unknown * 100))

    # Frequent tags
    top10_tags = sorted(words_by_tags.keys(),
                        key=lambda k: sum(words_by_tags[k].values()),
                        reverse=True)[:10]

    print(('     ' + '{}     '*len(top10_tags)).format(*top10_tags))
    for gold_tag in top10_tags:
        print(gold_tag + '  ', end='')
        for model_tag in top10_tags:
            if gold_tag == model_tag:
                print('  {}    '.format('-'), end='')
            else:
                print('{:2.2f}%  '.format(confusion_matrix.get(gold_tag).
                      get(model_tag, 0) / total), end='')
        print('')
