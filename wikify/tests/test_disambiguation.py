from unittest import TestCase

from lxml import etree

from wikify.const import ARTICLE_TAG
from wikify.disambiguation import Disambiguation
from wikify.utils import article_keywords


class TestDisambiguation(TestCase):

    def setUp(self):
        xml = './tests/mini_dump.xml'
        vocabulary = ['calculation', 'data processing']
        self.model = Disambiguation(xml, vocabulary)

        _, article = next(etree.iterparse(xml, tag=ARTICLE_TAG))
        self.keywords = article_keywords(article)

    def test_mappings(self):
        lesk_defs_dict = {
            'calculation.n.03': 'calculation'
            }
        key_name_id_dict = {
            'calculation': 'calculation',
            'data processing': 'data processing',
            }

        self.assertEqual(lesk_defs_dict, self.model._lesk_defs)
        self.assertEqual(key_name_id_dict, self.model._key_name_id)

    def test_parse_context(self):
        keyword_context = {
            'calculation': (
                ['self', 'contained', 'step', 'step', 'set',
                 'operations', 'performed', 'algorithms', 'exist',
                 'perform'],
                ['data', 'processing', 'automated', 'reasoning',
                 'algorithm', 'effective', 'method', 'expressed',
                 'within', 'finite']
                ),
            'data processing': (
                ['contained', 'step', 'step', 'set',
                 'operations', 'performed', 'algorithms', 'exist',
                 'perform', 'calculation'],
                ['automated', 'reasoning', 'algorithm', 'effective',
                 'method', 'expressed', 'within', 'finite',
                 'amount', 'space'],
                ),
            }

        for keyword in self.keywords:
            key_name = keyword.attrib['name']
            l_words = keyword.attrib['l_words']
            r_words = keyword.attrib['r_words']

            self.assertEqual(keyword_context[key_name],
                             self.model.parse_context(l_words, r_words))

    def test_disambiguate(self):
        key1, sent1 = 'calculation', 'self contained step step set operations \
                                      performed algorithms exist perform \
                                      calculation data processing automated \
                                      reasoning algorithm effective method \
                                      expressed within finite'
        key2, sent2 = 'data processing', 'contained step step set operations \
                                          performed algorithms exist perform \
                                          calculation data processing \
                                          automated reasoning algorithm \
                                          effective method expressed within \
                                          finite amount space'

        self.assertEqual('calculation',
                         self.model.disambiguate(key1, sent1))
        self.assertEqual('data processing',
                         self.model.disambiguate(key2, sent2))
