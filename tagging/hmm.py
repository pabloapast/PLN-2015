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
            tags_with_probs = self.trans.get(tuple(prev_tags), dict())
            tag_prob = tags_with_probs.get(tag, 0)

        assert tag_prob >= 0
        return tag_prob

    def trans_log_prob(self, tag, prev_tags):
        """Log probability of a tag.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        tag_prob = self.trans_prob(tag, prev_tags)
        if tag_prob == 0:
            return float('-inf')
        else:
            return log2(tag_prob)

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        # print(self.out, self.out.get(tag[0]))
        return self.out.get(tag, dict()).get(word, 0)

    def out_log_prob(self, word, tag):
        """Log probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        word_prob = self.out_prob(word, tag)
        # print(word, tag, word_prob)
        if word_prob == 0:
            return float('-inf')
        else:
            return log2(word_prob)

    def tag_prob(self, y):
        """
        Probability of a tagging.
        Warning: subject to underflow problems.

        y -- tagging.
        """
        prev_tags = [START] * (self.n - 1)
        tagging = prev_tags + y
        p = 1
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
        p_x = 1
        for i in range(len(x)):
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
            p += self.trans_log_prob(tagging[i], tagging[i - (self.n - 1):i])
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
        for i in range(len(x)):
            p_x += self.out_log_prob(x[i], y[i])
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
        self._pi = defaultdict(defaultdict)

    def K_n(self, k):
        if k - self.hmm.n <= 0:
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
        #     for t1, t2, w in pi[i-1].items()
        #         llenar los pi[i] que correspondan (!= 0)
        N = len(sent)
        # Initialization: pi(0, *, ..., *) = 1
        self._pi[0][(START,) * (self.hmm.n - 1)] = (log2(1), [])

        for k in range(1, N + 1):
            for tag in self.hmm.tags:
                max_prob = float('-inf')
                for prev_tags, (prob, tag_seq) in self._pi[k - 1].items():
                    partial_prob = (prob +
                                    self.hmm.trans_log_prob(tag, prev_tags) +
                                    self.hmm.out_log_prob(sent[k - 1], tag))

                    # print(prev_tags, tag, sent[k - 1])
                    # print(partial_prob, prob, self.hmm.trans_log_prob(tag, prev_tags),
                          # self.hmm.out_log_prob(sent[k - 1], tag))

                    if max_prob < partial_prob:
                        max_prob = partial_prob
                        try:
                            self._pi[k][prev_tags[1:] + (tag,)][0] = max_prob
                            self._pi[k][prev_tags[1:] + (tag,)][1].append(tag)
                        except KeyError:
                            self._pi[k][prev_tags[1:] + (tag,)] = (max_prob,
                                self._pi[k - 1][prev_tags][1].append(tag))
            print(self._pi)

        return max(self._pi[N].items(), key=lambda tup: tup[1][0])[1]


class MLHMM:

    def __init__(self, n, tagged_sents, addone=True):  # TODO AddOne
        """
        n -- order of the model.
        tagged_sents -- training sentences, each one being a list of pairs.
        addone -- whether to use addone smoothing (default: True).
        """
        self.n = n
        self.trans = defaultdict(defaultdict)
        self.out = defaultdict(defaultdict)

        self.counts = counts = defaultdict(int)
        self.vocabulary = vocabulary = []
        self.vocabulary_size = vocabulary_size = 0
        self.tags_set = tags_set = []
        self.tagset_size = tagset_size = 0
        for sent in sents:
            words, tags = zip(*sent)
            vocabulary += words
            tags_set += tags
            tags = ([START] * (n - 1)) + tags
            tags.append(STOP)
            for i in range(len(tags) - n + 1):
                ngram = tuple(tags[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1
        vocabulary = set(vocabulary)
        vocabulary_size = len(vocabulary)
        tags_set = set(tags_set)
        tagset_size = len(tags_set)

        self.compute_trans()
        self.compute_out()

    def compute_trans(self):  # TODO
        """
        Compute transition probabilities dictionary.
        """
        # for key, value in model.counts.items():
        #     if len(key) == self.n:
        #         prev_tokens = key[:-1]
        #         token = key[-1]
        #         self.probs[prev_tokens][token] = model.prob(token,
        #                                                     list(prev_tokens))

        # for key, value in self.probs.items():
        #     self.sorted_probs[key] = sorted(list(value.items()),
        #                                     key=lambda tup: (tup[1], tup[0]),
        #                                     reverse=True)
        pass

    def compute_out(self):  # TODO
        """
        Compute output probabilities dictionary.
        """
        pass

    def tcount(self, tokens):
        """Count for an k-gram for k <= n.

        tokens -- the k-gram tuple.
        """
        return self.counts.get(tuple(tokens), 0)

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w in self.vocabulary

    def tagset(self):
        """Returns the set of tags.
        """
        return self.tags_set

    def trans_prob(self, tag, prev_tags=None):
        """Probability of a tag.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        if prev_tags is None:
            prev_tags = []
        assert len(prev_tags) == self.n - 1

        tags = prev_tags + [tag]
        return self.tcount(tags) / self.tcount(prev_tags)

    def trans_log_prob(self, tag, prev_tags):  # TODO
        """Log probability of a tag.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        pass

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        return self.out.get(tag, dict()).get(word, 0)

    def out_log_prob(self, word, tag):  # TODO
        """Log probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        pass

    def tag_prob(self, y):
        """
        Probability of a tagging.
        Warning: subject to underflow problems.

        y -- tagging.
        """
        tagging = [START] * (self.n - 1) + list(y) + [STOP]
        p = 1
        for i in range(self.n - 1, len(tagging)):
            p *= self.trans_prob(tagging[i], tagging[i - (self.n - 1):i])
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
        p_x = 1
        for i in range(len(x)):
            p_x *= self.out_prob(x[i], y[i])
        return p_y * p_x

    def tag_log_prob(self, y):
        """
        Log-probability of a tagging.

        y -- tagging.
        """
        tagging = [START] * (self.n - 1) + list(y) + [STOP]
        p = 0
        for i in range(self.n - 1, len(tagging)):
            p += self.trans_log_prob(tagging[i], tagging[i - (self.n - 1):i])
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
        for i in range(len(x)):
            p_x += self.out_log_prob(x[i], y[i])
        return p_y + p_x

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        tagger = ViterbiTagger(self)
        return tagger.tag(sent)
