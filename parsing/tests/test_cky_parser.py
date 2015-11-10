# https://docs.python.org/3/library/unittest.html
from unittest import TestCase
from math import log2

from nltk.tree import Tree
from nltk.grammar import PCFG

from parsing.cky_parser import CKYParser


class TestCKYParser(TestCase):

    def test_parse(self):
        grammar = PCFG.fromstring(
            """
                S -> NP VP              [1.0]
                NP -> Det Noun          [0.6]
                NP -> Noun Adj          [0.4]
                VP -> Verb NP           [1.0]
                Det -> 'el'             [1.0]
                Noun -> 'gato'          [0.9]
                Noun -> 'pescado'       [0.1]
                Verb -> 'come'          [1.0]
                Adj -> 'crudo'          [1.0]
            """)

        parser = CKYParser(grammar)

        lp, t = parser.parse('el gato come pescado crudo'.split())

        # check chart
        pi = {
            (1, 1): {'Det': log2(1.0)},
            (2, 2): {'Noun': log2(0.9)},
            (3, 3): {'Verb': log2(1.0)},
            (4, 4): {'Noun': log2(0.1)},
            (5, 5): {'Adj': log2(1.0)},

            (1, 2): {'NP': log2(0.6 * 1.0 * 0.9)},
            (2, 3): {},
            (3, 4): {},
            (4, 5): {'NP': log2(0.4 * 0.1 * 1.0)},

            (1, 3): {},
            (2, 4): {},
            (3, 5): {'VP': log2(1.0) + log2(1.0) + log2(0.4 * 0.1 * 1.0)},

            (1, 4): {},
            (2, 5): {},

            (1, 5): {'S':
                     # rule S -> NP VP
                     log2(1.0) +
                     # left part
                     log2(0.6 * 1.0 * 0.9) +
                     # right part
                     log2(1.0) + log2(1.0) + log2(0.4 * 0.1 * 1.0)},
        }
        self.assertEqualPi(parser._pi, pi)

        # check partial results
        bp = {
            (1, 1): {'Det': Tree.fromstring("(Det el)")},
            (2, 2): {'Noun': Tree.fromstring("(Noun gato)")},
            (3, 3): {'Verb': Tree.fromstring("(Verb come)")},
            (4, 4): {'Noun': Tree.fromstring("(Noun pescado)")},
            (5, 5): {'Adj': Tree.fromstring("(Adj crudo)")},

            (1, 2): {'NP': Tree.fromstring("(NP (Det el) (Noun gato))")},
            (2, 3): {},
            (3, 4): {},
            (4, 5): {'NP': Tree.fromstring("(NP (Noun pescado) (Adj crudo))")},

            (1, 3): {},
            (2, 4): {},
            (3, 5): {'VP': Tree.fromstring(
                "(VP (Verb come) (NP (Noun pescado) (Adj crudo)))")},

            (1, 4): {},
            (2, 5): {},

            (1, 5): {'S': Tree.fromstring(
                """(S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                   )
                """)},
        }
        self.assertEqual(parser._bp, bp)

        # check tree
        t2 = Tree.fromstring(
            """
                (S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                )
            """)
        self.assertEqual(t, t2)

        # check log probability
        lp2 = log2(1.0 * 0.6 * 1.0 * 0.9 * 1.0 * 1.0 * 0.4 * 0.1 * 1.0)
        self.assertAlmostEqual(lp, lp2)

    def test_ambigous_sent(self):
        grammar = PCFG.fromstring(
            """
                NP -> D Ñ           [0.6]
                NP -> D NN          [0.4]

                Ñ -> JJ Ñ           [0.2]
                Ñ -> JJ NN          [0.08]
                Ñ -> NN Ñ           [0.07]
                Ñ -> NN NN          [0.09]
                Ñ -> Ñ Ñ            [0.11]
                Ñ -> Ñ NN           [0.13]
                Ñ -> NN Ñ           [0.15]
                Ñ -> NN NN          [0.17]

                D -> 'the'          [1.0]
                JJ -> 'fast'        [1.0]
                NN -> 'car'         [0.6]
                NN -> 'mechanic'    [0.4]
            """)

        parser = CKYParser(grammar)

        lp, t = parser.parse('the fast car mechanic'.split())

        # check chart
        pi = {
            (1, 1): {'D': log2(1.0)},
            (2, 2): {'JJ': log2(1.0)},
            (3, 3): {'NN': log2(0.6)},
            (4, 4): {'NN': log2(0.4)},

            (1, 2): {},
            (2, 3): {'Ñ': log2(0.6 * 1.0 * 0.08)},
            (3, 4): {'Ñ': log2(0.4 * 0.6 * 0.17)},

            (1, 3): {'NP': log2(1.0 * (0.6 * 1.0 * 0.08) * 0.6)},
            (2, 4): {'Ñ': log2(1.0 * (0.17 * 0.6 * 0.4) * 0.2)},

            (1, 4): {'NP': log2(0.6 * (0.17 * 0.6 * 0.4) * 0.2 * 1.0)},
        }
        self.assertEqualPi(parser._pi, pi)

        # check partial results
        bp = {
            (1, 1): {'D': Tree.fromstring("(D the)")},
            (2, 2): {'JJ': Tree.fromstring("(JJ fast)")},
            (3, 3): {'NN': Tree.fromstring("(NN car)")},
            (4, 4): {'NN': Tree.fromstring("(NN mechanic)")},

            (1, 2): {},
            (2, 3): {'Ñ': Tree.fromstring("(Ñ (JJ fast) (NN car))")},
            (3, 4): {'Ñ': Tree.fromstring("(Ñ (NN car) (NN mechanic))")},

            (1, 3): {'NP': Tree.fromstring("""(NP
                                                  (D the)
                                                  (Ñ (JJ fast) (NN car)))""")},
            (2, 4): {'Ñ': Tree.fromstring("""(Ñ
                                                 (JJ fast)
                                                 (Ñ (NN car)
                                                    (NN mechanic)))""")},

            (1, 4): {'NP': Tree.fromstring("""(NP
                                                  (D the)
                                                  (Ñ
                                                     (JJ fast)
                                                     (Ñ
                                                        (NN car)
                                                        (NN mechanic))))""")},
        }
        self.assertEqual(parser._bp, bp)

        t2 = Tree.fromstring(
            """
                (NP
                    (D the)
                    (Ñ
                        (JJ fast)
                        (Ñ
                            (NN car)
                            (NN mechanic)
                        )
                    )
                )
            """)
        self.assertEqual(t, t2)

        lp2 = log2(0.6 * (0.17 * 0.6 * 0.4) * 0.2 * 1.0)
        self.assertAlmostEqual(lp, lp2)

    def assertEqualPi(self, pi1, pi2):
        self.assertEqual(set(pi1.keys()), set(pi2.keys()))

        for k in pi1.keys():
            d1, d2 = pi1[k], pi2[k]
            self.assertEqual(d1.keys(), d2.keys(), k)
            for k2 in d1.keys():
                prob1 = d1[k2]
                prob2 = d2[k2]
                self.assertAlmostEqual(prob1, prob2)
