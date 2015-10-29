from nltk.tree import Tree


class CKYParser:

    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG.
        """
        assert grammar.is_binarised()

        self.grammar = grammar
        self.start = str(self.grammar.start())
        # self.productions = {}


    def parse(self, sent):
        """Parse a sequence of terminals.

        sent -- the sequence of terminals.
        """
        self._pi = _pi = {}
        self._bp = _bp = {}
        grammar = self.grammar
        n = len(sent)

        for i, word in enumerate(sent, start=1):
            _pi[i, i] = {}
            _bp[i, i] = {}
            productions = grammar.productions()
            for prod in productions:
                if word in prod.rhs():
                    X = str(prod.lhs())
                    _pi[i, i][X] = prod.logprob()
                    _bp[i, i][X] = Tree(X, [word])

        for l in range(1, n):
            for i in range(1, n - l + 1):
                j = i + l
                _pi[i, j] = {}
                _bp[i, j] = {}
                productions = grammar.productions()
                for prod in productions:
                    X = str(prod.lhs())
                    if len(prod.rhs()) == 2:
                        Y, Z = prod.rhs()
                        Y, Z = str(Y), str(Z)
                        logprob = prod.logprob()
                        for s in range(i, j):  # OJO: revisar este range
                            if Y in _pi[i, s].keys() and Z in _pi[s + 1, j].keys():
                                new_prob = logprob + _pi[i, s][Y] + _pi[s + 1, j][Z]
                                if X not in _pi[i, j] or new_prob > _pi[i, j][X]:
                                    _pi[i, j][X] = new_prob
                                    t = list(_bp[i, s].values()) + list(_bp[s + 1, j].values())
                                    _bp[i, j][X] = Tree(X, t)

        return _pi[1, n][self.start], _bp[1, n][self.start]
