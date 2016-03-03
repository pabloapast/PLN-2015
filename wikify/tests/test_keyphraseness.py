from unittest import TestCase

from lxml import etree

from wikify.const import ARTICLE_TAG
from wikify.keyphraseness import Keyphraseness


class TestKeyphraseness(TestCase):

    def setUp(self):
        self.xml = './tests/mini_dump.xml'
        self.model = Keyphraseness(self.xml, m=1)
        self.keywords = ['calculation', 'data processing']
        self.vocabulary = ['calculation', 'data processing']

    def test_extract_keywords(self):
        _, article = next(etree.iterparse(self.xml, tag=ARTICLE_TAG))
        keywords = self.model.extract_keywords(article)

        self.assertEqual(keywords, self.keywords)

    def test_vocabulary(self):
        self.assertEqual(self.model.vocabulary.sort(), self.vocabulary.sort())

    def test_rank(self):
        text = """all computer programs including programs that do not
                  perform numeric calculation
               """
        gold_rank = ['calculation']
        predicted_rank = self.model.rank(text)

        self.assertEqual(gold_rank, predicted_rank)
