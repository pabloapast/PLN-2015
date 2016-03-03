from collections import defaultdict, Counter

from lxml import etree
from nltk.stem.snowball import EnglishStemmer
from nltk.wsd import lesk

from wikify.const import ARTICLE_TAG
from wikify.utils import article_keywords, clear_xml_node


class Disambiguation:

    def __init__(self, xml, vocabulary, surround=10):
        # Vocabulary learned in the keyphraseness method
        self.vocabulary = set(vocabulary)
        # Number of sorrounding words to take from the context
        self._surround = surround
        # Dict mapping lesk synsets to wikipedia articles
        self._lesk_defs = _lesk_defs = dict()
        # Dict mapping words (from vocabulary) to their most
        # frequent linked wikipedia article
        self._key_name_id = _key_name_id = defaultdict(Counter)
        self._stemmer = EnglishStemmer()

        for _, article in etree.iterparse(xml, tag=ARTICLE_TAG):
            # Extract keywords from article and keep only
            # those are in the vocabulary
            keywords = article_keywords(article)
            keywords = [keyword for keyword in keywords
                        if keyword.attrib['name'] in self.vocabulary]

            for (key_name, sent), key_id in zip(self.keyword_context(keywords),
                                                self.keyword_ids(keywords)):
                synset = lesk(sent, key_name)
                if synset is not None:
                    _lesk_defs[synset.name()] = key_id

                _key_name_id[key_name].update([key_id])

            # Clear data read
            clear_xml_node(article)

        for k, v in _key_name_id.items():
            _key_name_id[k] = v.most_common(1)[0][0]

    def parse_context(self, l_words, r_words):
        """Extract surrounding words

        l_words -- string containing words to the left of the keyword
        r_words -- string containing words to the right of the keyword
        """
        # Take amount of words specified by the surround number
        l_words = l_words.split()[-self._surround:]
        r_words = r_words.split()[:self._surround]
        # Apply stemming
        # l_words = [self._stemmer.stem(w) for w in l_words]
        # r_words = [self._stemmer.stem(w) for w in r_words]

        return l_words, r_words

    def keyword_context(self, keywords):
        """Return a list of tuples (keyword_name, sentence)

        keywords -- list of keywords nodes
        """
        context_list = []

        for keyword in keywords:
            key_name = keyword.attrib['name']
            l_words = keyword.attrib['l_words']
            r_words = keyword.attrib['r_words']
            l_words, r_words = self.parse_context(l_words, r_words)

            context_list.append((key_name,
                                 ' '.join(l_words + [key_name] + r_words)))

        return context_list

    def keyword_ids(self, keywords):
        """Return a list of keywords ids
        (id = article title linked to the keyword)

        keywords -- list of keywords nodes
        """
        key_id_list = []

        for keyword in keywords:
            key_id = keyword.attrib['id']
            key_id_list.append(key_id)

        return key_id_list

    def disambiguate(self, key_name, sent):
        """Given a keyword name and the sentence where it appears, returns
        the keyword id

        key_name -- string, keyword name
        sent -- string, sentence containing the keyword
        """
        result = 'None'

        surround = sent.split(key_name)
        l_words, r_words = surround[0], surround[-1]
        l_words, r_words = self.parse_context(l_words, r_words)

        sent = ' '.join(l_words + [key_name] + r_words)
        synset = lesk(sent, key_name)
        if synset is not None and synset.name() in self._lesk_defs.keys():
            result = self._lesk_defs[synset.name()]
        else:
            result = self._key_name_id[key_name]

        return result
