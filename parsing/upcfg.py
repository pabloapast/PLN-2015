from collections import defaultdict


class UPCFG:
    """Unlexicalized PCFG.
    """

    def __init__(self, parsed_sents):
        """
        parsed_sents -- list of training trees.
        """
        # List of productions in Chomsky Normal Form
        self.cnf_sents = cnf_sents = [] # no guardar
        # Convert each tree to CNF and collapse unary
        for sent in parsed_sents:
            cnf_sent = sent.copy()
            cnf_sent = unlexicalize(cnf_sent)
            cnf_sent.chomsky_normal_form()
            cnf_sent.collapse_unary()
            cnf_sents.append(cnf_sent)

        # usar induce pcfg
        self.counts = counts = defaultdict(lambda: defaultdict(int))
        self.prods_list = prods_list = [sents.productions()
                                        for sent in parsed_sents]
        for prods in prods_list:
            for prod in prods:
                counts[prod.lhs()][prod.rhs()] += 1


    def productions(self):
        """Returns the list of UPCFG probabilistic productions.
        """
        p_prods = []  # List of probabilistic productions
        for tree in self.parsed_sents:



    def parse(self, tagged_sent):
        """Parse a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
