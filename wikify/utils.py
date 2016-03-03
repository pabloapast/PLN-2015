
def article_keywords(article):
    keywords = article.getchildren()[1:]

    return keywords


def article_text(article):
    text_node = article.getchildren()[0]

    return text_node.text


def clear_xml_node(xml_node):
    xml_node.clear()
    while xml_node.getprevious() is not None:
        del xml_node.getparent()[0]
