from nltk.grammar import induce_pcfg, Nonterminal
from nltk.tree import Tree

from parsing.cky_parser import CKYParser
from parsing.util import lexicalize, unlexicalize


class UPCFG:
    """Unlexicalized PCFG.
    """

    def __init__(self, parsed_sents, start='sentence', horzMarkov=None):
        """
        parsed_sents -- list of training trees.
        """
        # self.countFlat = 0
        # List of productions in Chomsky Normal Form
        cnf_prods = []
        # Convert each tree to CNF and collapse unary
        for sent in parsed_sents:
            cnf_tree = sent.copy(deep=True)
            cnf_tree = unlexicalize(cnf_tree)
            cnf_tree.collapse_unary(collapsePOS=True)
            cnf_tree.chomsky_normal_form()
            cnf_prods += cnf_tree.productions()
        # cnf_prods = set(cnf_prods)
        self._grammar = induce_pcfg(Nonterminal(start), cnf_prods)  # UPCFG grammar
        self._parser = CKYParser(self._grammar)  # CKY parser

    def productions(self):
        """Returns the list of UPCFG probabilistic productions.
        """
        return self._grammar.productions()

    def parse(self, tagged_sent):
        """Parse a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        _grammar = self._grammar
        _parser = self._parser

        # self._tagged_sent = tagged_sent

        sent, tags = zip(*tagged_sent)
        log_p, tree = _parser.parse(tags)

        if tree is None:  # Build a flat tree
            subtrees = []
            for word, tag in tagged_sent:
                subtrees.append(Tree(tag, [word]))
            tree = Tree(_grammar.start().symbol(), subtrees)
            # self.countFlat += 1
        else:
            tree.un_chomsky_normal_form()
            tree = lexicalize(tree, sent)

        return tree
