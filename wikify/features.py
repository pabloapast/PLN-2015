from collections import namedtuple

from featureforge.feature import Feature


# keyword: current keyword
# left_words
# right_words
# top5_words
Context = namedtuple('Context', 'keyword left_words right_words top_words')


def current_keyword(c):
    return c.keyword

# def left_word(c):
#     return c.left_words


# def right_word(c):
#     return c.right_words


# def top5_words(c):
#     return c.top5_words


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
