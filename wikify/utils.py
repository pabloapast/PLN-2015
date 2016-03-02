

def article_text(article):
    text_node = article.getchildren()[0]

    return text_node.text


def clear_xml_node(xml_node):
    xml_node.clear()
    while xml_node.getprevious() is not None:
        del xml_node.getparent()[0]




# # Iteration over xml with low ram consumption
# def fast_xml_iter(xml, func, dest, tag=None):
#     context = etree.iterparse(xml, tag=tag)

#     for event, elem in context:
#         # Execute operations over xml node
#         func(elem, dest)

#         # Clear data read
#         elem.clear()
#         while elem.getprevious() is not None:
#             del elem.getparent()[0]


# def clean_keywords(keyword):
#     cleaned_keyword = ''
#     keyword = keyword.split('|')[-1].lower()
#     if keyword not in STOPWORDS and not keyword.startswith(IGNORED_KEYWORDS):
#         cleaned_keyword = ' '.join(CLEAN_REGEX.tokenize(keyword))
#         if len(cleaned_keyword.split()) > 3:
#             cleaned_keyword = ''

#     return cleaned_keyword


# def clean_text(text):
#     tokens = CLEAN_REGEX.tokenize(text)
#     cleaned_tokens = [token for token in tokens if token not in STOPWORDS and
#                       not token.isdigit()]
#     return cleaned_tokens
