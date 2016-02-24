"""Eval perfomance of disambiguation method
Usage:
  eval_disambiguation.py -i <file> -d <file>
  eval_disambiguation.py -h | --help
Options:
  -i <file>     Trained model to evaluate
  -d <file>     XML dump to test prediction
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle


def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load trained model
    with open(opts['-i'], 'rb') as f:
        model = pickle.load(f)

    hits, total = 0, 0
    for i, _, elem in enumerate(etree.iterparse(opts['-d'] , tag='article')):
        text_iterator = elem.iterchildren(tag='text')
        text = next(text_iterator).text

        count_words = Counter([word for word in text.split()
                               if not word.isdigit() and word not in STOPWORDS])
        top_words, _ = zip(*count_words.most_common(self.top))
        top_words = list(top_words)

        keyword_iterator = elem.iterchildren(tag='keyword')

        context_list = model.text_context(keyword_iterator, top_words)

        gold_key_id = model.text_ids(keyword_iterator)

        predicted_key_id = [model.desambiguate(c) for c in context_list]

        hits_article = [g == p for g, p in zip(gold_key_id, predicted_key_id)]

        hits += sum(hits_article)
        total += len(gold_key_id)

        presicion = hits / total

        progress('{} articles processed - precision: {:2.2f}%'.format(i,
                  presicion * 100))
