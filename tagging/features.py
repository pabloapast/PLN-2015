from collections import namedtuple

from featureforge.feature import Feature


# sent -- the whole sentence.
# prev_tags -- a tuple with the n previous tags.
# i -- the position to be tagged.
History = namedtuple('History', 'sent prev_tags i')


def word_lower(h):
    """Feature: current lowercased word.

    h -- a history.
    """
    sent, i = h.sent, h.i
    return sent[i].lower()

def word_istitle(h):
    """Feature: check if current word is a title.

    h -- a history.
    """
    sent, i = h.sent, h.i
    return sent[i].istitle()

def word_isupper(h):
    """Feature: check if current word is upper.

    h -- a history.
    """
    sent, i = h.sent, h.i
    return sent[i].isupper()

def word_isdigit(h):
    """Feature: check if current word is digit.

    h -- a history.
    """
    sent, i = h.sent, h.i
    return sent[i].isdigit()

def prev_tags(h):
    """Feature: return the previous tags.

    h -- a history.
    """
    prev_tags = h.prev_tags
    return prev_tags


class NPrevTags(Feature):

    def __init__(self, n):
        """Feature: n previous tags tuple.

        n -- number of previous tags to consider.
        """
        self.n = n

    def _evaluate(self, h):
        """n previous tags tuple.

        h -- a history.
        """
        prev_tags = h.prev_tags
        return prev_tags[- self.n:]


class PrevWord(Feature):

    def __init__(self, f):
        """Feature: the feature f applied to the previous word.

        f -- the feature.
        """
        self.f = f

    def _evaluate(self, h):
        """Apply the feature to the previous word in the history.

        h -- the history.
        """
        sent, prev_tags, i = h.sent, h.prev_tags, h.i
        value = None
        if i > 0:
            prev_word_history = History(sent, prev_tags, i - 1)
            value = str(self.f(prev_word_history))
        else:
            value = 'BOS'

        return value
