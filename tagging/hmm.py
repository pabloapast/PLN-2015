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

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        return self.out.get(tag, dict()).get(word, 0)

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
            trans_p = self.trans_prob(tagging[i], tagging[i - (self.n - 1):i])
            if trans_p > 0:
                p += log2(trans_p)
            else:
                p = float('-inf')
                break
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
            out_p = self.out_prob(word, tag)
            if out_p > 0:
                p_x += log2(out_p)
            else:
                p_x = float('-inf')
                break
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
        hmm = self._hmm
        n = hmm.n
        tagset = hmm.tagset()
        N = len(sent)

        # Initialization: pi(0, *, ..., *) = 1
        pi = self._pi = {}
        pi[0] = {(START,) * (n - 1): (log2(1), [])}

        for k in range(1, N + 1):
            pi[k] = {}

            word = sent[k - 1]
            tags_whit_out_prob = [(t, hmm.out_prob(word, t)) for t in tagset
                                  if hmm.out_prob(word, t) > 0]
            for tag, out_prob in tags_whit_out_prob:
                max_prob = float('-inf')
                for prev_tags, (prob, tag_seq) in pi[k - 1].items():
                    assert len(prev_tags) == n - 1
                    trans_p = hmm.trans_prob(tag, prev_tags)
                    if trans_p > 0:
                        new_prob = (prob + log2(trans_p) + log2(out_prob))
                        new_prev_tags = (prev_tags + (tag,))[1:]
                        if new_prev_tags not in pi[k] or\
                           new_prob > pi[k][new_prev_tags][0]:
                            pi[k][new_prev_tags] = (new_prob, tag_seq + [tag])

        tagging = None
        max_prob = float('-inf')
        for prev_tags, (prob, tag_seq) in pi[N].items():
            if max_prob < prob:
                max_prob = prob
                tagging = tag_seq

        assert tagging is not None
        return tagging


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

        # Used to compute out prob
        pairs_count = defaultdict(lambda: defaultdict(int))
        # Compute counts of tags and words, vocabulary and tags_set
        for tagged_sent in tagged_sents:
            words, tags = [], []
            for word, tag in tagged_sent:
                words.append(word)
                tags.append(tag)
                pairs_count[tag][word] += 1

            vocabulary += words
            tags_set += tags
            tags = [START] * (n - 1) + tags + [STOP]
            for i in range(len(tags) - n + 1):
                ngram = tuple(tags[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1

        self.vocabulary = set(vocabulary)
        self.tags_set = set(tags_set)
        self.vocabulary_size = len(self.vocabulary)
        self.tagset_size = len(self.tags_set)

        # Compute out probabilities
        for tag, words_with_count in pairs_count.items():
            total_words_count = sum(words_with_count.values())
            for word, word_count in words_with_count.items():
                self.out[tag][word] = word_count / total_words_count

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

        tags = list(prev_tags) + [tag]
        prob = -1
        if self.addone:
            prob = (self.tcount(tags) + 1) /\
                   (self.tcount(prev_tags) + self.tagset_size)
        else:
            prob = self.tcount(tags) / self.tcount(prev_tags)
        assert prob >= 0
        return prob

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        prob = 0
        if self.unknown(word):
            prob = 1 / self.vocabulary_size
        else:
            prob = self.out.get(tag).get(word, 0)
        return prob

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        tagger = ViterbiTagger(self)
        return tagger.tag(sent)
