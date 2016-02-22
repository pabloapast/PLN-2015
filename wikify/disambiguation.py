from collections import Counter

from wikify.const import PAGE_TAG, TEXT_TAG, NAMESPACE_TAG, ARTICLE_ID, CLEAN_REGEX
from wikify.features import Context, current_keyword, n_left_word, n_right_word,\
                            n_top5_words
from wikify.utils import clean_keywords

from featureforge.vectorizer import Vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


class Disambiguation:

    def __init__(self, xml):
        features = []
        features.append(current_keyword)
        for i in range(5):
            features.append(n_left_word[i])
            features.append(n_right_word[i])
            features.append(n_top5_words[i])

        self.clf = clf = Pipeline([('vect', Vectorizer(features)),
                                   ('clf', MultinomialNB)])
        clf = clf.fit(self.texts_context(xml), self.texts_ids(xml))


    def texts_context(self, xml):
        context_list = []
        xml_reader = etree.iterparse(xml, tag=PAGE_TAG)

        for event, elem in xml_reader:
            iterator = elem.iterchildren(tag=NAMESPACE_TAG)
            namespace_id = next(iterator).text

            if namespace_id == ARTICLE_ID:
                # Text in the article
                iterator = elem.iterdescendants(tag=TEXT_TAG)
                text = next(iterator).text
                context_list += self.text_context(text)

        return context_list



    def text_context(self, text):
        context_list = []
        sents = text.split('.')

        for sent in sents:
            keywords = extract_regex.findall(sent)

            for key in keywords:
                left_words, right_words = sent.split('[[' + key + ']]')
                key_name = clean_keywords(key)

                if not key_name == '':
                    left_words = CLEAN_REGEX.tokenize(left_words)
                    right_words = CLEAN_REGEX.tokenize(right_words)

                    left_words = [stemmer.stem(word) for word in left_words
                                  if word not in stopwords and
                                  not word.isdigit()][-6:]
                    right_words = [stemmer.stem(word) for word in right_words
                                   if word not in stopwords and
                                   not word.isdigit()][:6]

                    left_words = ['']*(6 - len(left_words)) + left_words
                    right_words = right_words + ['']*(6 - len(right_words))

                    count_words = Counter(text.lower().split())
                    top5_words, counts = zip(*count_words.most_common(5))

                    context_list.append(Context(key_name, left_words,
                                                right_words, top5_words))

        return context_list


    def texts_ids(self, xml):
        key_id_list = []
        xml_reader = etree.iterparse(xml, tag=PAGE_TAG)

        for event, elem in xml_reader:
            iterator = elem.iterchildren(tag=NAMESPACE_TAG)
            namespace_id = next(iterator).text

            if namespace_id == ARTICLE_ID:
                # Text in the article
                iterator = elem.iterdescendants(tag=TEXT_TAG)
                text = next(iterator).text
                key_id_list += self.text_ids(text)

        return key_id_list


    def text_ids(self, text):
        key_id_list = []
        sents = text.split('.')

        for sent in sents:
            keywords = extract_regex.findall(sent)

            for key in keywords:
                left_words, right_words = sent.split('[[' + key + ']]')
                key_name = clean_keywords(key)

                if not key_name == '':
                    key_id = key.split('|')[0]
                    key_id_list.append(key_id)

        return key_id_list


    def desambiguate(self, c):
        return self.clf.predict([c])
