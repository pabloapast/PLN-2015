from nltk.grammar import induce_pcfg, Nonterminal
from nltk.tree import Tree

from parsing.cky_parser import CKYParser
from parsing.util import lexicalize, unlexicalize


class UPCFG:
    """Unlexicalized PCFG.
    """

    def __init__(self, parsed_sents, start='sentence'):
        """
        parsed_sents -- list of training trees.
        """
        # List of productions in Chomsky Normal Form
        cnf_prods = []
        # Convert each tree to CNF and collapse unary
        for sent in parsed_sents:
            cnf_tree = sent.copy(deep=True)
            cnf_tree = unlexicalize(cnf_tree)
            cnf_tree.chomsky_normal_form()
            cnf_tree.collapse_unary(collapsePOS=True)
            cnf_prods += cnf_tree.productions()

        self._grammar = induce_pcfg(Nonterminal(start), cnf_prods)  # UPCFG grammar

    def productions(self):
        """Returns the list of UPCFG probabilistic productions.
        """
        return self._grammar.productions()

    def parse(self, tagged_sent):
        """Parse a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        _grammar = self._grammar
        parser = CKYParser(_grammar)  # CKY parser
        sent, tags = zip(*tagged_sent)
        log_p, tree = parser.parse(tags)

        if tree is None:  # Build a flat tree
            subtrees = []
            for word, tag in tagged_sent:
                subtrees.append(Tree(tag, [word]))
            tree = Tree(_grammar.start().symbol(), subtrees)
        else:
            tree = lexicalize(tree, sent)

        return tree
