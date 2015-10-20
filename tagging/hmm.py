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
        self.tags_set = tagset
        self.trans = trans
        self.out = out

    def tagset(self):
        """Returns the set of tags.
        """
        return self.tags_set

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
        return self.out.get(tag, dict()).get(word, 0)

    def out_log_prob(self, word, tag):
        """Log probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        word_prob = self.out_prob(word, tag)
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
        tagging = prev_tags + y + [STOP]
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
        for word, tag in zip(x, y):
            p_x *= self.out_prob(word, tag)
        return p_y * p_x

    def tag_log_prob(self, y):
        """
        Log-probability of a tagging.

        y -- tagging.
        """
        prev_tags = [START] * (self.n - 1)
        tagging = prev_tags + list(y) + [STOP]
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
        for word, tag in zip(x, y):
            p_x += self.out_log_prob(word, tag)
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
        self._hmm = hmm

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        # N = len(sent)
        # hmm = self._hmm
        # # Initialization: pi(0, *, ..., *) = 1
        # pi = self._pi = defaultdict(defaultdict)
        # pi[0][(START,) * (hmm.n - 1)] = (log2(1), [])

        # for k in range(1, N + 1):
        #     for tag in hmm.tags_set:
        #         max_prob = float('-inf')
        #         for prev_tags, (prob, tag_seq) in pi[k - 1].items():
        #             assert len(prev_tags) == hmm.n - 1
        #             partial_prob = (prob +
        #                             hmm.trans_log_prob(tag, prev_tags) +
        #                             hmm.out_log_prob(sent[k - 1], tag))
        #             # print('max_prob {:2.4f} - partial_prob {:2.4f}'.format(max_prob, partial_prob))
        #             if max_prob < partial_prob:
        #                 max_prob = partial_prob
        #                 pi[k][prev_tags[1:] + (tag,)] = (max_prob,
        #                         pi[k - 1][prev_tags][1] + [tag])

        # print(pi)
        # return max(pi[N].items(), key=lambda t: t[1][0])[1][1]

        m = len(sent)
        hmm = self._hmm
        n = hmm.n
        tagset = hmm.tagset()

        self._pi = pi = {}
        pi[0] = {
            ('<s>',) * (n - 1): (0.0, [])
        }

        for i, w in zip(range(1, m + 1), sent):
            pi[i] = {}

            # iterate over tags that can follow with out_prob > 0.0
            tag_out_probs = [(t, hmm.out_prob(w, t)) for t in tagset]
            for t, out_p in [(t, p) for t, p in tag_out_probs if p > 0.0]:
                # iterate over non-zeros in the previous column
                for prev, (lp, tag_sent) in pi[i - 1].items():
                    trans_p = hmm.trans_prob(t, prev)
                    if trans_p > 0.0:
                        new_prev = (prev + (t,))[1:]
                        new_lp = lp + log2(out_p) + log2(trans_p)
                        # is it the max?
                        if new_prev not in pi[i] or new_lp > pi[i][new_prev][0]:
                            # XXX: what if equal?
                            pi[i][new_prev] = (new_lp, tag_sent + [t])

        # last step: generate STOP
        max_lp = float('-inf')
        result = None
        print(pi)
        for prev, (lp, tag_sent) in pi[m].items():
            p = hmm.trans_prob('</s>', prev)
            if p > 0.0:
                new_lp = lp + log2(p)
                if new_lp > max_lp:
                    max_lp = new_lp
                    result = tag_sent
        print(result)
        return result


class MLHMM(HMM):

    def __init__(self, n, tagged_sents, addone=True):
        """
        n -- order of the model.
        tagged_sents -- training sentences, each one being a list of pairs.
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        assert tagged_sents is not None

        self.n = n
        self.addone = addone
        self.counts = counts = defaultdict(int)
        self.out = defaultdict(defaultdict)
        self.vocabulary = vocabulary = []
        self.tags_set = tags_set = []

        # Compute counts of tags and words, vocabulary and tags_set
        pairs_count = defaultdict(defaultdict)  # Used to compute out prob
        for tagged_sent in tagged_sents:
            words, tags = [], []
            for word, tag in tagged_sent:
                words.append(word)
                tags.append(tag)
                try:
                    pairs_count[tag][word] += 1
                except KeyError:
                    pairs_count[tag][word] = 1

            vocabulary += words
            tags_set += tags
            tags = [START] * (n - 1) + tags + [STOP]
            for i in range(len(tags) - n + 1):
                ngram = tuple(tags[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1

        # Compute out probabilities
        for tag, words_with_count in pairs_count.items():
            total_words_count = sum(words_with_count.values())
            for word, word_count in words_with_count.items():
                self.out[tag][word] = word_count / total_words_count

        vocabulary = set(vocabulary)
        tags_set = set(tags_set)
        self.vocabulary_size = len(vocabulary)
        self.tagset_size = len(tags_set)  # XXX DEBERIA CONTAR STOP??

    def tcount(self, tokens):
        """Count for an k-gram for k <= n.

        tokens -- the k-gram tuple.
        """
        assert len(tokens) <= self.n
        return self.counts.get(tuple(tokens), 0)

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self.vocabulary

    def trans_prob(self, tag, prev_tags=None):
        """Probability of a tag.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        if prev_tags is None:
            prev_tags = []
        assert len(prev_tags) == self.n - 1, prev_tags

        tags = list(prev_tags) + [tag]
        prob = -1
        if self.addone:
            prob = (self.tcount(tags) + 1) /\
                   (self.tcount(prev_tags) + self.tagset_size)
        else:
            prob = self.tcount(tags) / self.tcount(prev_tags)
        assert prob >= 0
        return prob

    def trans_log_prob(self, tag, prev_tags):  # TODO
        """Log probability of a tag.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        tag_prob = self.trans_prob(tag, prev_tags)
        prob = 0
        if tag_prob == 0:
            prob = float('-inf')
        else:
            prob = log2(tag_prob)
        return prob

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        prob = 0
        if self.unknown(word):
            prob = 1.0 / self.vocabulary_size
        else:
            prob = self.out.get(tag).get(word, 0)
            # assert prob != 0
        return prob

    def out_log_prob(self, word, tag):
        """Log probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        word_prob = self.out_prob(word, tag)
        prob = 0
        if word_prob == 0:
            prob = float('-inf')
        else:
            prob = log2(word_prob)
        return prob

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        tagger = ViterbiTagger(self)
        return tagger.tag(sent)
