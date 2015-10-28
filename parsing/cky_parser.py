

class CKYParser:

    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG.
        """
        assert grammar.is_binarised()

        self.grammar = grammar

    def parse(self, sent):
        """Parse a sequence of terminals.

        sent -- the sequence of terminals.
        """
        self._pi = _pi = {}
        self._bp = _bp = {}
        grammar = self.grammar
        n = len(sent)

        for i, word in enumerate(sent):
            _pi[(i, i)] = {}
            productions = grammar.productions(rhs=word)
            for prod in productions:
                X = prod.lhs()
                _pi[(i, i)][X] = prod.logprob()  # CHECKEAR: esta bien logprob?

        for l in range(n - 1):  # OJO: revisar si aca no falta un -1
            for i in range(n - l):  # OJO: revisar si aca no falta un -1
                j = i + l
                _pi[(i, j)] = {}
                productions = grammar.productions()
                for prod in productions:
                    X = prod.lhs()
                    rhs = [YZ, logprob for YZ, logprob in
                           zip(prod.rhs(), prod.logprob()) if len(YZ) == 2]
                    for Y, Z, logprob in rhs:
                        for s in range(i, j):  # OJO: revisar este range
                            new_prob = logprob + _pi[(i, s)][Y] +\
                                       _pi[(s + 1, j)][Z]
                            if X not in _pi[(i, j)] or\
                               new_prob > _pi[(i, j)][X]:
                               _pi[(i, j)][X] = new_prob
                    # Falta agregar bp(i, j, X)








