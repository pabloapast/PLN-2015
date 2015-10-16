from featureforge.vectorizer import Vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from tagging.features import History, word_lower, word_istitle, word_isupper,\
                             word_isdigit, prev_tags, NPrevTags, PrevWord
from tagging.hmm import START


class MEMM:

    def __init__(self, n, tagged_sents, classifier='LogisticRegression'):
        """
        n -- order of the model.
        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        self.n = n
        self.vocabulary = vocabulary = []

        classifiers = {'LogisticRegression': LogisticRegression(),
                       'LinearSVC': LinearSVC(),
                       'MultinomialNB': MultinomialNB(),
                       }

        for tagged_sent in tagged_sents:
            sent, tags = zip(*tagged_sent)
            vocabulary += sent
        vocabulary = set(vocabulary)

        basic_features = [word_lower, word_istitle, word_isupper, word_isdigit,
                          prev_tags]
        features = basic_features.copy()
        for i in range(1, self.n):
            features.append(NPrevTags(i))
        # All the basic features except prev_tags
        for ft in basic_features[:-1]:  # REVISAR ESTO
            features.append(PrevWord(ft))

        self.hist_clf = hist_clf = Pipeline([('vect', Vectorizer(features)),
                                             ('clf', classifiers[classifier]),
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
            histories += self.sent_histories(tagged_sent)

        return histories

    def sent_histories(self, tagged_sent):
        """
        Iterator over the histories of a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        histories = []
        sent, tags = zip(*tagged_sent)
        sent = list(sent)
        tags = (START,) * (self.n - 1) + tags
        for i in range(len(sent)):
            histories.append(History(sent, tags[i:i + self.n - 1], i))

        return histories

    def sents_tags(self, tagged_sents):
        """
        Iterator over the tags of a corpus.

        tagged_sents -- the corpus (a list of sentences)
        """
        tags_list = []
        for tagged_sent in tagged_sents:
            tags_list += self.sent_tags(tagged_sent)

        return tags_list

    def sent_tags(self, tagged_sent):
        """
        Iterator over the tags of a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        sent, tags = zip(*tagged_sent)
        return list(tags)


    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        tagging = [START] * (self.n - 1)
        for i in range(len(sent)):
            hist = History(sent, tagging[i:i + self.n - 1], i)
            tagging.append(self.tag_history(hist))

        return tagging[self.n - 1:]

    def tag_history(self, h):
        """Tag a history.

        h -- the history.
        """
        return self.hist_clf.predict([h])[0]

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self.vocabulary
