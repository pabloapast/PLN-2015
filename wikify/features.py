from collections import namedtuple

from featureforge.feature import Feature


# keyword: current keyword
# left_words
# right_words
# top_words
Context = namedtuple('Context', 'keyword l_words r_words top_words')


def current_keyword(c):
    keyword = c.keyword
    return keyword


class NLeftWord(Feature):

    def __init__(self, n):
        self.n = n

    def _evaluate(self, c):
        l_words = c.l_words
        return l_words[self.n]


class NRightWord(Feature):

    def __init__(self, n):
        self.n = n

    def _evaluate(self, c):
        r_words = c.r_words
        return r_words[self.n]


class NTopWord(Feature):

    def __init__(self, n):
        self.n = n

    def _evaluate(self, c):
        top_words = c.top_words
        return top_words[self.n]
