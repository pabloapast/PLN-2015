from nltk.tree import Tree
import time


class CKYParser:

    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG.
        """
        assert grammar.is_binarised()

        self._grammar = _grammar = grammar
        self._start = _grammar.start().symbol()
        # List of unary productions
        self._uni_prods = _uni_prods = [p for p in _grammar.productions()
                                        if len(p.rhs()) == 1]
        # List of binary productions
        self._bi_prods = _bi_prods = [p for p in _grammar.productions()
                                      if len(p.rhs()) == 2]

        self._prods_dict = _prods_dict = {}
        for p in _uni_prods:
            lhs, rhs = p.lhs().symbol(), p.rhs()
            if rhs not in _prods_dict.keys():
                _prods_dict[rhs] = {}
            _prods_dict[rhs][lhs] = p.logprob()

        for p in _bi_prods:
            lhs, rhs = p.lhs().symbol(),\
                       (p.rhs()[0].symbol(), p.rhs()[1].symbol())
            if rhs not in _prods_dict.keys():
                _prods_dict[rhs] = {}
            _prods_dict[rhs][lhs] = p.logprob()

    def parse(self, sent):
        """Parse a sequence of terminals.

        sent -- the sequence of terminals.
        """
        _grammar = self._grammar
        _uni_prods = self._uni_prods
        _bi_prods = self._bi_prods
        _prods_dict = self._prods_dict
        _start = self._start

        self._pi = _pi = {}
        self._bp = _bp = {}
        # self._sent = sent

        # start1 = time.time()
        for i, word in enumerate(sent, start=1):
            _pi[i, i] = _prods_dict.get((word,))
            _bp[i, i] = {}
            for lhs in _pi[i, i].keys():
                _bp[i, i][lhs] = Tree(lhs, [word])
        # print('start1 = {}'.format(start1 - time.time()))

        # start2 = time.time()
        n = len(sent)
        for l in range(1, n):
            for i in range(1, n - l + 1):
                j = i + l
                _pi[i, j] = {}
                _bp[i, j] = {}

                for s in range(i, j):
                    for A in _pi[i, s].keys():
                        for B in _pi[s + 1, j].keys():
                            for X, logprob in _prods_dict.get((A, B), {}).items():
                                new_prob = logprob + _pi[i, s][A] + _pi[s + 1, j][B]
                                if X not in _pi[i, j] or new_prob > _pi[i, j][X]:
                                    _pi[i, j][X] = new_prob
                                    _bp[i, j][X] = Tree(X, [_bp[i, s][A], _bp[s+1, j][B]])
        # print('start2 = {}'.format(start2 - time.time()))


        return _pi[1, n].get(_start, float('-inf')), _bp[1, n].get(_start)
