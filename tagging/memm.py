from featureforge.vectorizer import Vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tagging.features import History, word_lower, word_istitle, word_isupper,
                             word_isdigit, prev_tags, NPrevTags, PrevWord
from tagging.hmm import START


class MEMM:

    def __init__(self, n, tagged_sents):
        """
        n -- order of the model.
        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        self.n = n
        self.vocabulary = vocabulary = []

        for tagged_sent in tagged_sents:
            sent, tags = zip(*tagged_sent)
            vocabulary += sent
        vocabulary = set(vocabulary)

        vect = Vectorizer([word_lower, word_istitle, word_isupper,
                           word_isdigit, prev_tags])
        self.hist_clf = hist_clf = Pipeline([('vect', vect),
                                             ('clf', LogisticRegression()),
                                            ])
        hist_clf = hist_clf.fit(self.sents_histories(tagged_sents),
                                self.sents_tags(tagged_sents))

    def sents_histories(self, tagged_sents):
        """
        Iterator over the histories of a corpus.

        tagged_sents -- the corpus (a list of sentences)
        """
        histories = []
        for tagged_sent in tagged_sents:
            histories += sent_histories(tagged_sent)

        return histories

    def sent_histories(self, tagged_sent):
        """
        Iterator over the histories of a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        histories = []
        sent, tags = zip(*tagged_sents):
        tags = (START,) * (self.n - 1) + tags
        for i in len(sent):
            histories.append(History(sent, tags[i:self.n - 1], i))

        return histories

    def sents_tags(self, tagged_sents):
        """
        Iterator over the tags of a corpus.

        tagged_sents -- the corpus (a list of sentences)
        """
        tags_list = []
        for tagged_sent in tagged_sents:
            tags_list += sent_tags(tagged_sent)

        return tags_list

    def sent_tags(self, tagged_sent):
        """
        Iterator over the tags of a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        sent, tags = zip(*tagged_sent)
        return list(tags)


    def tag(self, sent):  # TODO
        """Tag a sentence.

        sent -- the sentence.
        """

    def tag_history(self, h):  # TODO
        """Tag a history.

        h -- the history.
        """
        predicted = self.hist_clf.predict(h)


    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self.vocabulary
