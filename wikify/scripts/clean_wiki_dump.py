"""Parse wikipedia corpus, extract keywords and normalize article text
Usage:
  clean_wiki_dump.py -i <file> -o <file>
  clean_wiki_dump.py -h | --help
Options:
  -i <file>     XML dump to clean.
  -o <file>     Output cleaned xml.
  -h --help     Show this screen.
"""
from docopt import docopt

from lxml import etree

from wikify.utils import clear_xml_node, extract_node_text, extract_keywords,\
                         clean_text, extract_surround_words,\
                         extract_keyword_id, extract_keyword_name,\
                         clean_keyword_name


NAMESPACE = '{*}'  # Wildcard
ARTICLE_ID = '0'  # Id = 0 is assigned to wikipedia articles
NAMESPACE_TAG = NAMESPACE + 'ns'
PAGE_TAG = NAMESPACE + 'page'
TITLE_TAG = NAMESPACE + 'title'
REDIRECT_TAG = NAMESPACE + 'redirect'
TEXT_TAG = NAMESPACE + 'text'
IGNORED_STARTWITH = ('image:', 'file:', 'category:', 'wikipedia:')


if __name__ == '__main__':
    opts = docopt(__doc__)

    xml_header = b'<enwiki>\n'
    xml_tail = b'</enwiki>\n'

    out = open(opts['-o'], 'ab')
    out.write(xml_header)

    # There is a lot of keywords with an old id (title of article),
    # this dictionary maps old ids to the correct new ones
    title_unique_id = dict()
    for _, elem in etree.iterparse(opts['-i'], tag=PAGE_TAG):
        # Extract page ID (page type), we only want to parse articles
        namespace_id = extract_node_text(elem, NAMESPACE_TAG)
        # Check if is a redirect article, we want to exclude this article
        # because doesn't have valuable information
        try:
            redirect_title = extract_node_text(elem, REDIRECT_TAG)
            article_title = extract_node_text(elem, TITLE_TAG)
            title_unique_id[redirect_title.lower()] = article_title
        except StopIteration:
            article_title = extract_node_text(elem, TITLE_TAG)
            title_unique_id[article_title.lower()] = article_title
        # Clear xml node
        clear_xml_node(elem)

    # Iterates over each wikipedia page in the xml
    for _, elem in etree.iterparse(opts['-i'], tag=PAGE_TAG):
        # Extract page ID (page type), we only want to parse articles
        namespace_id = extract_node_text(elem, NAMESPACE_TAG)
        # Check if is a redirect article, we want to exclude this article
        # because doesn't have valuable information
        try:
            _ = extract_node_text(elem, REDIRECT_TAG)
            is_redirect = True
        except StopIteration:
            article_title = extract_node_text(elem, TITLE_TAG)
            is_redirect = False

        if namespace_id == ARTICLE_ID and not is_redirect:
            # Text in the article
            text = extract_node_text(elem, TEXT_TAG)

            if text is not None:
                # Find keywords, they are between '[[' ']]'
                keywords = [key for key in extract_keywords(text)
                            if not key.lower().startswith(IGNORED_STARTWITH)]
                # Clean text
                cleaned_text = clean_text(text)

                if len(keywords) > 0 and len(cleaned_text) > 0:
                    # Build xml nodes
                    # Article node
                    article_node = etree.Element('article',
                                                 title=article_title)

                    # Text node
                    text_node = etree.Element('text')
                    text_node.text = cleaned_text
                    article_node.append(text_node)

                    # Keywords nodes
                    for keyword in keywords:
                        key_id = extract_keyword_id(keyword)
                        key_name = extract_keyword_name(keyword)
                        # Clean keyword name
                        key_name = clean_keyword_name(key_name)
                        # Assign the correct id
                        key_id = title_unique_id[key_id.lower()]
                        # Left and right words
                        l_words, r_words = extract_surround_words(keyword,
                                                                  text)
                        # Build node
                        keyword_node = etree.Element('keyword', id=key_id,
                                                     name=key_name,
                                                     l_words=l_words,
                                                     r_words=r_words)
                        article_node.append(keyword_node)

                    out.write(etree.tostring(article_node, pretty_print=True))

        # Clear xml node
        clear_xml_node(elem)

    out.write(xml_tail)
    out.close()
