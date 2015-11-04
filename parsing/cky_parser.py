from nltk.tree import Tree
import time


class CKYParser:

    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG.
        """
        assert grammar.is_binarised()

        self._grammar = _grammar = grammar

        self._uni_prods = _uni_prods = [p for p in _grammar.productions() if len(p.rhs()) == 1]
        self._bi_prods = _bi_prods = [p for p in _grammar.productions() if len(p.rhs()) == 2]
        # For each token returns a dict of the nonterminal associated
        # and his log probability
        self._uni_prods_d = _uni_prods_d = {}
        # For each token returns a dict of trees with the woken as his leaf
        # and the nonterminal as the parent node
        self._uni_trees_d = _uni_trees_d = {}
        self._bi_prods_d = _bi_prods_d = {}
        self._bi_trees_d = _bi_trees_d = {}
        for p in _uni_prods:
            lhs, rhs = p.lhs().symbol(), p.rhs()
            _uni_prods_d[rhs], _uni_trees_d[rhs] = {}, {}
            _uni_prods_d[rhs][lhs] = p.logprob()
            _uni_trees_d[rhs][lhs] = Tree(lhs, [rhs[0]])
        for p in _bi_prods:
            lhs, rhs = p.lhs().symbol(), (p.rhs()[0].symbol(), p.rhs()[1].symbol())
            _bi_prods_d[rhs], _bi_trees_d[rhs] = {}, {}
            _bi_prods_d[rhs][lhs] = p.logprob()
            # _bi_trees[rhs][lhs] = Tree(lhs, [rhs])

        # self._non_bi_prods = [prod for prod in _grammar.productions()
        #                        if prod.is_nonlexical()]

    def parse(self, sent):
        """Parse a sequence of terminals.

        sent -- the sequence of terminals.
        """
        _grammar = self._grammar
        _uni_prods = self._uni_prods
        _bi_prods = self._bi_prods
        _uni_prods_d = self._uni_prods_d
        _bi_prods_d = self._bi_prods_d
        _uni_trees_d = self._uni_trees_d
        # _bi_trees_d = self._bi_trees_d
        # _non_bi_prods = self._non_bi_prods
        start = _grammar.start().symbol()

        self._pi = _pi = {}
        self._bp = _bp = {}

        n = len(sent)

        # start1 = time.time()  # ~0.00019 segundos
        for i, word in enumerate(sent, start=1):
            _pi[i, i] = _uni_prods_d.get((word,))
            _bp[i, i] = {key: item.copy(deep=True)
                         for key, item in _uni_trees_d.get((word,)).items()}
        # print('start1 = ', time.time() - start1)

        # start2 = time.time()  # 25~2 segundos

        # print('_bi_prods_d = ', _bi_prods_d, '\n\n')

        for l in range(1, n):
            for i in range(1, n - l + 1):
                j = i + l
                _pi[i, j] = {}
                _bp[i, j] = {}

                for s in range(i, j):
                    for A in _pi[i, s].keys():
                        for B in _pi[s + 1, j].keys():
                            # print('Pi = ' , _pi)
                            # print('Bp = ', _bp)
                            # print('_pi[i, s] = ', _pi[i, s])
                            # print('_pi[s+1, j] = ', _pi[s+1, j])
                            # print('A, B = ', A, B)
                            # print('_bi_prods_d.keys() = ', _bi_prods_d.keys())
                            # print('\n\n')
                            # print('Pi = ' , _pi, '\n\n')
                            # print('Bp = ', _bp, '\n\n')
                            if (A, B) in _bi_prods_d.keys():
                                for X, logprob in _bi_prods_d[A, B].items():
                                    # print('_bi_prods_d[A, B] = ', _bi_prods_d[A, B], '\n')
                                    new_prob = logprob + _pi[i, s][A] + _pi[s + 1, j][B]

                                    if X not in _pi[i, j] or new_prob > _pi[i, j][X]:
                                        _pi[i, j][X] = new_prob
                                        subtrees = list(_bp[i, s].values()) +\
                                                    list(_bp[s + 1, j].values())
                                        _bp[i, j][X] = Tree(X, subtrees)


                # for prod in _bi_prods:
                #     X = prod.lhs().symbol()
                #     # if len(prod.rhs()) == 2:  # Que pasa si es 1 ?! Deberia existir con 1?
                #     # assert len(prod.rhs()) == 2, prod
                #     Y, Z = prod.rhs()[0].symbol(), prod.rhs()[1].symbol()
                #     logprob = prod.logprob()
                #     for s in range(i, j):
                #         # Y_logprob = _pi[i, s].get(Y)
                #         # Z_logprob = _pi[s + 1, j].get(Z)
                #         # if Y_logprob is not None and Z_logprob is not None:
                #         #     new_prob = logprob + Y_logprob +\
                #         #                Z_logprob
                #         if Y in _pi[i, s].keys() and Z in _pi[s + 1, j].keys():
                #             new_prob = logprob + _pi[i, s][Y] +\
                #                        _pi[s + 1, j][Z]
                #             if X not in _pi[i, j] or new_prob > _pi[i, j][X]:
                #                 _pi[i, j][X] = new_prob
                #                 sub_trees = list(_bp[i, s].values()) +\
                #                             list(_bp[s + 1, j].values())
                #                 _bp[i, j][X] = Tree(X, sub_trees)
        # print('start2 = ', time.time() - start2)

        print('Pi = ' , _pi, '\n\n')
        print('Bp = ', _bp, '\n\n')

        # Que valor de pi retornar si no encontro un parsing ?!
        return _pi[1, n].get(start), _bp[1, n].get(start)
