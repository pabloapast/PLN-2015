# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from math import floor, log2
import random

START = '<s>'
STOP = '</s>'


class NGram(object):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)
        self.words = set()

        for sent in sents:
            self.words = self.words.union(set(sent))
            sent = ([START] * (n - 1)) + sent
            sent.append(STOP)
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1

    def prob(self, token, prev_tokens=None):
        n = self.n
        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]
        return float(self.counts[tuple(tokens)]) / self.counts[tuple(prev_tokens)]

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.
 
        tokens -- the n-gram or (n-1)-gram tuple.
        """
        c = 0
        key = tuple(tokens)
        if key in self.counts:
            c = self.counts[key]
        return c
 
    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
 
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        p = -1
        n = self.n
        if n == 1:  # Caso para unigramas
            p = self.prob(token)
        else:
            assert prev_tokens != None
            if self.count(prev_tokens) > 0:
                p = self.prob(token, prev_tokens)
            else:
                p = 0
        assert p >= 0
        return p

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.
 
        sent -- the sentence as a list of tokens.
        """
        n = self.n
        p = 1
        sent = ([START] * (n - 1)) + sent
        sent.append(STOP)
        for i in range(n - 1, len(sent)):
            p_i = self.cond_prob(sent[i], sent[i - (n - 1) : i])
            p *= p_i
        return p
 
    def sent_log_prob(self, sent):
        """Log-probability of a sentence.
 
        sent -- the sentence as a list of tokens.
        """
        n = self.n
        p = 0
        sent = ([START] * (n - 1)) + sent
        sent.append(STOP)
        for i in range(n - 1, len(sent)):
            p_i = self.cond_prob(sent[i], sent[i - (n - 1) : i])
            if p_i == 0:
                return float('-inf')
            else:
                p += log2(p_i)
        return p


class NGramGenerator(object):
 
    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self.n = model.n
        self.probs = defaultdict(defaultdict)
        self.sorted_probs = defaultdict(list)

        for key, value in model.counts.items():
            if len(key) == self.n:
                prev_tokens = key[:-1]
                token = key[-1]
                self.probs[prev_tokens][token] = model.prob(token,
                                                    list(prev_tokens))

        for key, value in self.probs.items():
            self.sorted_probs[key] = sorted(list(value.items()),
                                        key=lambda tup: (tup[1], tup[0]), reverse=True)

    def generate_sent(self):
        """Randomly generate a sentence."""
        prev_tokens = [START] * (self.n - 1)
        sent = []
        token = self.generate_token(prev_tokens)
        while token != STOP:
            sent.append(token)
            if self.n > 1:
                prev_tokens.pop(0)
                prev_tokens.append(token)
            token = self.generate_token(prev_tokens)
        return sent

 
    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.
 
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        token = ''
        rand = random.random()  # random value in the range [0, 1)
        token_prob_tuples = ()

        if prev_tokens == None or prev_tokens == []:
            token_prob_tuples = self.sorted_probs[()]
        else:
            token_prob_tuples = self.sorted_probs[tuple(prev_tokens)]

        p0 = 0.0
        for t, p in token_prob_tuples:
            if rand <= p0 + p:
                token = t
                break
            else:
                p0 += p
        return token


class AddOneNGram(NGram):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        NGram.__init__(self, n, sents)
 
    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
 
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == self.n - 1

        tokens = prev_tokens + [token]

        return float((self.count(tuple(tokens))) + 1) / \
                     (self.count(tuple(prev_tokens)) + self.V())
 
    def V(self):
        """Size of the vocabulary.
        """
        # lenght of the vocabulary plus </s>
        return len(self.words) + 1


class EvalModel:

    def __init__(self, model, test_sents):
        """
        model -- n-gram model.
        test -- list of sentences for test
        """
        self.model = model
        self.test_sents = test_sents
        self.M = sum(len(sent) for sent in self.test_sents)

    def log_prob(self):
        """ Log probability of the model
        """
        return sum(self.model.sent_log_prob(sent) for sent in self.test_sents)

    def cross_entropy(self):
        """ Cross-Entropy or Average log probability of the model
        """
        return self.log_prob() / self.M

    def perplexity(self):
        """ Perplexity of the model
        """
        return 2 ** (- self.cross_entropy())


class InterpolatedNGram:
 
    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        pass


class BackOffNGram(NGram):
 
    def __init__(self, n, sents, beta=None, addone=True):
        """
        Back-off NGram model with discounting as described by Michael Collins.
 
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        beta -- discounting hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)
        self.words = set()       
        self.beta = beta
        self.addone = addone

        train = sents[: floor(len(sents) * 0.9)]
        test = sents[floor(len(sents) * 0.9) :]
        assert len(train) + len(test) == len(sents)
        
        for sent in train:
            self.words = self.words.union(set(sent))
            sent = ([START] * (n - 1)) + sent
            sent.append(STOP)
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                for j in range(n):
                    counts[ngram[:-j]] += 1

        if not self.beta:
            counts_test = defaultdict(int)
            for sent in test:
                sent = ([START] * (n - 1)) + sent
                sent.append(STOP)
                for i in range(len(sent) - n + 1):
                    counts_test[tuple(sent[i: i + n])] += 1

            betas = [x / 10 for x in range(1, 10)]
            sum_log_probs = []
            for b in betas:
                self.beta = b
                s = sum(count * log2(self.cond_prob(tokens[-1], tokens[:-1]))\
                        for tokens, count in counts_test.items())
                sum_log_probs.append(s)

            self.beta = betas.index(max(sum_log_probs))


    def disc_count(self, tokens):
        """Dicounted Count for an n-gram
 
        tokens -- the n-gram tuple.
        """
        return self.count(tokens) - self.beta
 
    def A(self, tokens):
        """Set of words with counts > 0 for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
        a = set()
        for word in self.words:
            if self.count(list(tokens).append(word)) > 0:
                a.add(word)
        return a
 
    def alpha(self, tokens):
        """Missing probability mass for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
        return 1 - sum(self.disc_count(tokens.append(word)) / self.count(tokens)\
                       for word in self.A(tokens))
 
    def denom(self, tokens):
        """Normalization factor for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
        norm = -1
        if len(tokens) == 1:
            norm = sum(self.prob(token) for token in self.words if not token in self.A)
        else:
            norm = sum(self.cond_prob(token, tokens[1:]) for token in self.words\
                       if not token in self.A(tokens))
        assert norm >= 0
        return norm

    def prob(self, token, prev_tokens=None):
        if not prev_tokens:
            prev_tokens = []
        tokens = prev_tokens + [token]
        return self.count(tokens) / self.count(prev_tokens)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
 
        token -- the token.
        prev_tokens -- the previous n-1 tokens.
        """
        p = -1
        if not prev_tokens:
            p = self.count([token]) / self.count([])
        elif token in self.A(prev_tokens):
            p = self.disc_count(prev_tokens.append(token)) / self.count(prev_tokens)
        else:
            if len(prev_tokens) == 1:
                p = self.alpha(prev_tokens) * (self.prob(token) / self.denom(prev_tokens))
            else:
                p = self.alpha(prev_tokens) * (self.cond_prob(token, prev_tokens[1:]) / \
                    self.denom(prev_tokens))
        assert p >= 0
        return p
