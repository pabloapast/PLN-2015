from collections import namedtuple

from featureforge.feature import Feature


# keyword: current keyword
# left_words
# right_words
# top5_words
Context = namedtuple('Context', 'keyword left_words right_words top5_words')


def current_keyword(c):
    return c.keyword


def n_left_word(c, n):
    return c.left_words[n]


def n_right_word(c, n):
    return c.right_words[n]


def n_top5_words(c, n):
    return c.top5_words[n]
