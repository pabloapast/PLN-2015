from lxml import etree
from nltk.stem.snowball import EnglishStemmer
from nltk.wsd import lesk


class Disambiguation:

    def __init__(self, xml, vocabulary, n=3, surround=6, top=10):
        self._vocabulary = set(vocabulary)
        self._n = n
        self._surround = surround
        self._top = top

        self._stemmer = _stemmer = EnglishStemmer(ignore_stopwords=False)
        self._lesk_defs = _lesk_defs = dict()
        self._key_name_id = _key_name_id = dict()

        for _, article in etree.iterparse(xml, tag='article'):
            childrens = article.getchildren()

            keywords = [keyword for keyword in childrens[1:]
                        if keyword.attrib['name'] in self._vocabulary]

            for keyword in keywords:
                key_name = keyword.attrib['name']
                key_id = keyword.attrib['id']
                l_words = keyword.attrib['l_words'].split()[-self._surround:]
                r_words = keyword.attrib['r_words'].split()[:self._surround]

                l_words = [_stemmer.stem(w) for w in l_words]
                r_words = [_stemmer.stem(w) for w in r_words]

                sent = ' '.join(l_words + [key_name] + r_words)
                synset = lesk(sent, key_name)

                if synset is not None:
                    _lesk_defs[synset.name()] = key_id

                _key_name_id[key_name] = key_id

            # Clear data read
            article.clear()
            while article.getprevious() is not None:
                del article.getparent()[0]


    def disambiguate(self, keyword, sent):
        surround = sent.split(keyword)
        l_words, r_words = surround[0], surround[-1]
        l_words = [self._stemmer.stem(w) for w in l_words]
        r_words = [self._stemmer.stem(w) for w in r_words]

        sent = ' '.join(l_words + [keyword] + r_words)

        res = 'None'
        synset = lesk(sent, keyword)
        if synset is not None and synset.name() in self._lesk_defs.keys():
            res = self._lesk_defs[synset.name()]
        else:
            res = self._key_name_id[keyword]

        return res


    def text_context(self, keywords, top_words=None):
        context_list = []

        for keyword in keywords:
            key_name = keyword.attrib['name']
            l_words = keyword.attrib['l_words'].split()[-self._surround:]
            r_words = keyword.attrib['r_words'].split()[:self._surround]

            context_list.append((key_name, ' '.join(l_words + [key_name] + r_words)))

        return context_list


    def text_ids(self, keywords):
        key_id_list = []

        for keyword in keywords:
            key_id = keyword.attrib['id']
            key_id_list.append(key_id)

        return key_id_list

