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
        self.words = list()

        for sent in sents:
            self.words += sent
            sent = ([START] * (n - 1)) + sent
            sent.append(STOP)
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1
        self.words.append(STOP)  # Vocabulary + </s>
        self.words = set(self.words)

    def prob(self, token, prev_tokens=None):
        n = self.n
        if prev_tokens == None:
            prev_tokens = []
        assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]
        return float(self.counts[tuple(tokens)]) /\
                     self.counts[tuple(prev_tokens)]

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.
 
        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self.counts.get(tuple(tokens), 0)
 
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

    def M(self, test_sents):
        return sum(len(sent) for sent in test_sents)

    def log_prob(self, test_sents):
        """ Log probability of the model
        """
        return sum(self.sent_log_prob(sent) for sent in test_sents)

    def cross_entropy(self, test_sents):
        """ Cross-Entropy or Average log probability of the model
        """
        return self.log_prob(test_sents) / self.M(test_sents)

    def perplexity(self, test_sents):
        """ Perplexity of the model
        """
        return 2 ** (- self.cross_entropy(test_sents))


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
            self.sorted_probs[key] = sorted(list(value.items()),\
                                            key=lambda tup: (tup[1], tup[0]),\
                                            reverse=True)

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
        if prev_tokens == None:
            prev_tokens = []
        assert len(prev_tokens) == self.n - 1

        tokens = prev_tokens + [token]

        return float((self.count(tuple(tokens))) + 1) / \
                     (self.count(tuple(prev_tokens)) + self.V())
 
    def V(self):
        """Size of the vocabulary.
        """
        # lenght of the vocabulary plus </s>
        return len(self.words)


class InterpolatedNGram(NGram):
 
    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self.n = n

        train = sents
        self.gamma = gamma
        if self.gamma == None:
            # Train: 90% of sents
            train = sents[: floor(len(sents) * 0.9)]
            # Test: 10% of sents
            test = sents[floor(len(sents) * 0.9) :]
            assert len(train) + len(test) == len(sents)

        self.models = None
        if addone:
            self.models = [AddOneNGram(1, train)]  # AddOne for unigrams
            self.models += [NGram(i, train) for i in range(2, self.n + 1)]
        else:
            self.models = [NGram(i, train) for i in range(1, self.n + 1)]
        self.models.reverse()  # Order: Ngram, N-1gram, ..., 1gram

        if self.gamma == None:
            # Gamma aproximation
            gammas = [x for x in range(0, 50000, 500)]
            min_perplexity = float('inf')
            min_gamma = None
            for g in gammas:
                self.gamma = g
                if self.perplexity(test) < min_perplexity:
                    min_perplexity = self.perplexity(test)
                    min_gamma = self.gamma
                print(self.gamma, self.perplexity(test))
            self.gamma = min_gamma
            assert self.gamma != None
            print('-- Gamma --', self.gamma)

    def set_lambdas(self, tokens):
        lambdas = list()
        # Lambdas for Ngram, N-1gram, ..., 2gram
        for i in range(self.n - 1):
            l = (1 - sum(lambdas)) * (self.models[i].count(tokens[i :]) /\
                (self.models[i].count(tokens[i :]) + self.gamma))
            lambdas.append(l)
        # Lambda for 1gram
        l = (1 - sum(lambdas))
        lambdas.append(l)
        return lambdas

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
 
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        lambdas = self.set_lambdas(prev_tokens)
        
        p = 0
        if prev_tokens == None:
            prev_tokens = []
        for i in range(len(lambdas)):
            p += lambdas[i] * self.models[i].cond_prob(token, prev_tokens[i :])

        return p


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
        self.beta = beta
        self.addone = addone
        self.counts = counts = defaultdict(int)
        self.words = list()
        self.alphas = defaultdict(int)
        self.denoms = defaultdict(int)
        self.As = defaultdict(set)
        train = sents
        test = None
        
        if self.beta == None:
            # Train: 90% of sents
            train = sents[: floor(len(sents) * 0.9)]
            # Test: 10% of sents
            test = sents[floor(len(sents) * 0.9) :]
            assert len(train) + len(test) == len(sents)

        for sent in train:
            self.words += sent
            sent_c = sent.copy()
            sent_c.append(STOP)
            for i in range(1, n + 1):
                if i > 1:
                    sent_c = [START] + sent_c
                for j in range(len(sent_c) - i + 1):
                    ngram = tuple(sent_c[j: j + i])
                    counts[ngram] += 1
                    prev = ngram[:-1]
                    if i == 1 or prev == (START,) * (i - 1):
                        counts[prev] += 1
        self.words.append(STOP)  # Vocabulary + </s>
        self.words = set(self.words)


        # for sent in train:
        #     self.words += sent
        #     sent.append(STOP)
        #     for i in range(1, n + 1):
        #         if i > 1:
        #             sent = [START] + sent
        #         for j in range(len(sent) - i + 1):
        #             ngram = tuple(sent[j: j + i])
        #             counts[ngram] += 1
        #             prev = ngram[:-1]
        #             if i == 1 or prev == (START,)*(i-1):
        #                 counts[prev] += 1
        # self.words.append(STOP)  # Vocabulary + </s>
        # self.words = set(self.words)

        self.prevs = sorted([k for k in self.counts.keys()\
                             if 0 < len(k) < self.n and not STOP in k],\
                            key=lambda k: len(k))
        # print('Prevs =', self.prevs, '\n\n')
        # print(self.counts)

        if self.beta == None:
            self.beta = 0

        # print('Counts = ', self.counts, '\n')
        self.compute_A()
        # print('As =', self.As, '\n')
        self.compute_alphas()
        # import pdb; pdb.set_trace()
        # print('alphas =', self.alphas, '\n')
        self.compute_denoms()
        # print('denoms =', self.denoms, '\n')

    def disc_count(self, tokens):
        """Dicounted Count for an n-gram
 
        tokens -- the n-gram tuple.
        """
        return self.count(tokens) - self.beta

    def V(self):
        """Size of the vocabulary.
        """
        return len(self.words)

    def prob(self, token, prev_tokens=None):
        if prev_tokens == None:
            prev_tokens = []
        tokens = prev_tokens + [token]

        p = 0
        # Caso para addone en unigrama
        if len(prev_tokens) == 0 and self.addone:
            p = (self.count(tokens) + 1) / (self.count(prev_tokens) + self.V())
        else:
            p = self.count(tokens) / self.count(prev_tokens) 
        return p

    def A(self, tokens):
        """Set of words with counts > 0 for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
        tokens = list(tokens)
        return {word for word in self.words\
                if self.count(tokens + [word]) > 0}

    def alpha(self, tokens):
        """Missing probability mass for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
        tokens = list(tokens)
        print(tokens, self.counts.get(tuple(tokens)), [self.disc_count(tokens + [word]) for word in self.As[tuple(tokens)]])
        return 1 - sum(self.disc_count(tokens + [word]) / self.counts.get(tuple(tokens), 1) for word in self.As[tuple(tokens)])
 
    def denom(self, tokens):
        """Normalization factor for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
        tokens = list(tokens)
        return 1 - sum(self.cond_prob(word, tokens[1:]) for word in self.As[tuple(tokens)])

    def compute_A(self):
        for tokens in self.prevs:
            self.As[tokens] = self.A(tokens)
            assert self.As[tokens] != set(), tokens

    def compute_alphas(self):
        for tokens in self.prevs:
            self.alphas[tokens] = self.alpha(tokens)
            assert self.alphas[tokens] >= 0, (tokens, self.alphas[tokens])

    def compute_denoms(self):
        for tokens in self.prevs:           
            self.denoms[tokens] = self.denom(tokens)
            assert self.denoms[tokens] > 0, (tokens, self.denoms[tokens])

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
 
        token -- the token.
        prev_tokens -- the previous n-1 tokens.
        """

        p = -1
        # if prev_tokens == None:
        #     prev_tokens = []
        # if token in self.A(prev_tokens):
        #     p = self.disc_count(prev_tokens + [token]) / self.count(prev_tokens)
        # else:
        if prev_tokens == None or len(prev_tokens) == 0:
            p = self.prob(token)
        elif token in self.As.get(tuple(prev_tokens), {}):
            p = self.disc_count(list(prev_tokens) + [token]) / self.counts.get(tuple(prev_tokens), 1)
        else:
            p = self.alphas[tuple(prev_tokens)] *\
                (self.cond_prob(token, prev_tokens[1:]) /\
                 self.denoms.get(tuple(prev_tokens), 1))
        if p < 0:
            print(prev_tokens, token, p)
        assert p >= 0
        return p
