"""Evaulate a parser.

Usage:
  eval.py -i <file> [-m <m>] [-n <n>]
  eval.py -h | --help

Options:
  -i <file>     Parsing model file.
  -m <m>        Parse only sentences of length <= <m>.
  -n <n>        Parse only <n> sentences (useful for profiling).
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import sys

from corpus.ancora import SimpleAncoraCorpusReader

from parsing.util import spans


def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()


if __name__ == '__main__':
    opts = docopt(__doc__)

    print('Loading model...')
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    print('Loading corpus...')
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-2.0/', files)

    parsed_sents = None
    m = opts['-m']
    if m != '':
        m = eval(m)
        parsed_sents = [sent for sent in corpus.parsed_sents()
                        if len(sent) <= m]
    else:
        parsed_sents = list(corpus.parsed_sents())

    print('Parsing...')
    hits, total_gold, total_model = 0, 0, 0
    n = len(parsed_sents)

    max_parsed_sents = opts['-n']
    if max_parsed_sents != '':
        max_parsed_sents = eval(max_parsed_sents)
    else:
        max_parsed_sents = n

    format_str = '{:3.1f}% ({}/{}) (P={:2.2f}%, R={:2.2f}%, F1={:2.2f}%)'
    progress(format_str.format(0.0, 0, n, 0.0, 0.0, 0.0))
    for i, gold_parsed_sent in enumerate(parsed_sents):
        if i <= max_parsed_sents:
            tagged_sent = gold_parsed_sent.pos()

            # parse
            model_parsed_sent = model.parse(tagged_sent)

            # compute labeled scores
            gold_spans = spans(gold_parsed_sent, unary=False)
            model_spans = spans(model_parsed_sent, unary=False)
            hits += len(gold_spans & model_spans)
            total_gold += len(gold_spans)
            total_model += len(model_spans)

            # compute labeled partial results
            prec = float(hits) / total_model * 100
            rec = float(hits) / total_gold * 100
            f1 = 2 * prec * rec / (prec + rec)

            # compute unlabeled scores
            gold_interval = [i, j for n, i, j in gold_spans]
            model_interval = [i, j for n, i, j in model_spans]
            u_hits += len(gold_interval & model_interval)

            # compute unlabeled partial results
            u_prec = float(u_hits) / total_model * 100
            u_rec = float(u_hits) / total_gold * 100
            u_f1 = 2 * u_prec * u_rec / (u_prec + u_rec)

            progress(format_str.format(float(i+1) * 100 / n, i+1, n, prec, rec, f1, u_prec, u_rec, u_f1))
        else:
            break

    print('')
    print('Parsed {} sentences'.format(n))
    print('Labeled')
    print('  Precision: {:2.2f}% '.format(prec))
    print('  Recall: {:2.2f}% '.format(rec))
    print('  F1: {:2.2f}% '.format(f1))
    print('Unlabeled')
    print('  Precision: {:2.2f}% '.format(u_prec))
    print('  Recall: {:2.2f}% '.format(u_rec))
    print('  F1: {:2.2f}% '.format(u_f1))







