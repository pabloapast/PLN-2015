from collections import defaultdict
from math import log2

START = '<s>'
STOP = '</s>'


class HMM:

    def __init__(self, n, tagset, trans, out):
        """
        n -- n-gram size.
        tagset -- set of tags.
        trans -- transition probabilities dictionary.
        out -- output probabilities dictionary.
        """
        self.n = n
        self.tags = tagset
        self.trans = trans
        self.out = out

    def tagset(self):
        """Returns the set of tags.
        """
        return self.tags

    def trans_prob(self, tag, prev_tags):
        """Probability of a tag.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        tag_prob = -1
        if prev_tags is None or prev_tags == []:
            assert self.n == 1
            tag_prob = self.trans.get(tag, 0)
        else:
            tags_with_probs = self.trans.get(tuple(prev_tags))  # PUEDE DEVOLVER NONE??
            tag_prob = tags_with_probs.get(tag, 0)

        assert tag_prob >= 0
        return tag_prob

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        return (self.out.get(tag)).get(word, 0)

    def tag_prob(self, y):
        """
        Probability of a tagging.
        Warning: subject to underflow problems.

        y -- tagging.
        """
        prev_tags = [START] * (self.n - 1)
        tagging = prev_tags + list(y)
        p = 0
        for i in range(self.n - 1, len(tagging)):
            p *= self.trans_prob(tagging[i], tagging[i - self.n + 1:i])
        return p

    def prob(self, x, y):
        """
        Joint probability of a sentence and its tagging.
        Warning: subject to underflow problems.

        x -- sentence.
        y -- tagging.
        """
        assert len(x) == len(y)

        p_y = self.tag_prob(y)
        p_x = 0
        for i in len(x):
            p_x *= self.out_prob(x[i], y[i])
        return p_y * p_x

    def tag_log_prob(self, y):
        """
        Log-probability of a tagging.

        y -- tagging.
        """
        prev_tags = [START] * (self.n - 1)
        tagging = prev_tags + list(y)
        p = 0
        for i in range(self.n - 1, len(tagging)):
            p += log2(self.trans_prob(tagging[i], tagging[i - self.n + 1:i]))
        return p

    def log_prob(self, x, y):
        """
        Joint log-probability of a sentence and its tagging.

        x -- sentence.
        y -- tagging.
        """
        assert len(x) == len(y)

        p_y = self.tag_log_prob(y)
        p_x = 0
        for i in len(x):
            p_x += log2(self.out_prob(x[i], y[i]))
        return p_y + p_x


    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        tagger = ViterbiTagger(self)
        return tagger.tag(sent)


class ViterbiTagger:

    def __init__(self, hmm):
        """
        hmm -- the HMM.
        """
        self.hmm = hmm

    def K_n(self, k):
        if k - self.n <= 0:
            return set([START])
        else:
            return self.hmm.tags

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        # llenar pi[0]
        # for i in range(1, n+1):
        #     llenar pi[i][t1, t2]
        #     en terminos de pi[i-1]
        #     # Opcion 1
        #     for t1, t2 in tags^2:
        #         llenar pi[i][t1,t2] en termino de pi[i-1]
        #     # Opcion 2 - la buena
        #     for t1, t2, w in pi[i-1].items()
        #         llenar los pi[i] que correspondan (!= 0)
        N = len(sent)
        tag_secuence = defaultdict()
        pi = defaultdict(defaultdict)
        # bp = defaultdict(defaultdict)

        pi[0][(START,) * self.n] = 1
        for k in range(1, N + 1):
            for tag in self.K_n(k):
                max_prob = 0
                for prev_tags, value in pi[k - 1].items():
                    # tag = v, prev_tags = (w, u)
                    # self.pi[k][prev_tags + tuple([tag])] = 0
                    partial_prob = (value * self.trans_prob(tag, prev_tags) *
                                    self.out_prob(sent[k], (tag,)))
                    if max_prob < partial_prob:
                        max_prob = partial_prob
                        pi[k][prev_tags[1:] + (v,)] = max_prob
                        tag_secuence[k] = prev_tags[0]
                        # bp[k][prev_tags[1:] + (v,)] = prev_tags[0]

        max_prob = 0
        for prev_tags, value in pi[N].items():
            partial_prob = (value * self.trans_prob(STOP, prev_tags))
            if max_prob < partial_prob:
                max_prob = partial_prob
                for i, tag in enumerate(prev_tags):
                    tag_secuence[N + 1 - self.n - i] = tag

        return tag_secuence.values()
