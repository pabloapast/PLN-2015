from collections import namedtuple

from featureforge.feature import Feature


# keyword: current keyword
# left_words
# right_words
# top_words
Context = namedtuple('Context', 'keyword left_words right_words top_words')


def current_keyword(c):
    return c.keyword


class NLeftWord(Feature):

    def __init__(self, n):
        self.n = n

    def _evaluate(self, c):
        c.left_words[self.n]


class NRightWord(Feature):

    def __init__(self, n):
        self.n = n

    def _evaluate(self, c):
        c.right_words[self.n]


class NTopWord(Feature):

    def __init__(self, n):
        self.n = n

    def _evaluate(self, c):
        c.top_words[self.n]
