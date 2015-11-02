from nltk.grammar import induce_pcfg, nonterminals

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
            cnf_tree.collapse_unary()
            cnf_prods += cnf_tree.productions()

        start = nonterminals(start)[0]
        self.grammar = induce_pcfg(start, cnf_prods)  # UPCFG grammar

    def productions(self):
        """Returns the list of UPCFG probabilistic productions.
        """
        return self.grammar.productions()

    def parse(self, tagged_sent):
        """Parse a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        sent, tags = zip(*tagged_sent)
        parser = CKYParser(self.grammar)  # CKY parser
        log_p, tree = parser.parse(tags)
        tree = lexicalize(tree, sent)

        return tree

