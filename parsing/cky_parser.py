from nltk.tree import Tree
import time


class CKYParser:

    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG.
        """
        assert grammar.is_binarised()

        self._grammar = _grammar = grammar

        # For each token returns a dict of the nonterminal associated
        # and his log probability
        self._lex_prods = _lex_prods = {}
        # For each token returns a dict of trees with the woken as his leaf
        # and the nonterminal as the parent node
        self._lex_trees = _lex_trees = {}
        for p in _grammar.productions():
            if p.is_lexical():
                lhs, rhs = p.lhs().symbol(), p.rhs()[0]
                _lex_prods[rhs], _lex_trees[rhs] = {}, {}

                _lex_prods[rhs][lhs] = p.logprob()
                _lex_trees[rhs][lhs] = Tree(lhs, [rhs])

        self._non_lex_prods = [prod for prod in _grammar.productions()
                               if prod.is_nonlexical()]

    def parse(self, sent):
        """Parse a sequence of terminals.

        sent -- the sequence of terminals.
        """
        _grammar = self._grammar
        _lex_prods = self._lex_prods
        _lex_trees = self._lex_trees
        _non_lex_prods = self._non_lex_prods
        start = _grammar.start().symbol()

        self._pi = _pi = {}
        self._bp = _bp = {}

        n = len(sent)

        start1 = time.time()  # ~0.00019 segundos
        for i, word in enumerate(sent, start=1):
            _pi[i, i] = _lex_prods.get(word)
            _bp[i, i] = {key: item.copy(deep=True)
                         for key, item in _lex_trees.get(word).items()}
        print('start1 = ', time.time() - start1)

        start2 = time.time()  # 25~2 segundos
        for l in range(1, n):
            for i in range(1, n - l + 1):
                j = i + l
                _pi[i, j] = {}
                _bp[i, j] = {}
                for prod in _non_lex_prods:
                    X = prod.lhs().symbol()
                    if len(prod.rhs()) == 2:  # Que pasa si es 1 ?! Deberia existir con 1?
                    # assert len(prod.rhs()) == 2, prod
                        Y, Z = prod.rhs()[0].symbol(), prod.rhs()[1].symbol()
                        logprob = prod.logprob()
                        for s in range(i, j):
                            # Y_logprob = _pi[i, s].get(Y)
                            # Z_logprob = _pi[s + 1, j].get(Z)
                            # if Y_logprob is not None and Z_logprob is not None:
                            #     new_prob = logprob + Y_logprob +\
                            #                Z_logprob
                            if Y in _pi[i, s].keys() and Z in _pi[s + 1, j].keys():
                                new_prob = logprob + _pi[i, s][Y] +\
                                           _pi[s + 1, j][Z]
                                if X not in _pi[i, j] or new_prob > _pi[i, j][X]:
                                    _pi[i, j][X] = new_prob
                                    sub_trees = list(_bp[i, s].values()) +\
                                                list(_bp[s + 1, j].values())
                                    _bp[i, j][X] = Tree(X, sub_trees)
        print('start2 = ', time.time() - start2)

        # print('Pi = ' , _pi, '\n\n')
        # print('Bp = ', _bp, '\n\n')

        # Que valor de pi retornar si no encontro un parsing ?!
        return _pi[1, n].get(start), _bp[1, n].get(start)
