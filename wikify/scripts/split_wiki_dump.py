"""Split wikipedia corpus in test and train
Usage:
  split_wiki_dump.py -i <file> [-p <p>] -o <path>
  split_wiki_dump.py -h | --help
Options:
  -i <file>     XML dump to split.
  -p <p>        Percentage of corpus for train. [default: 0.8]
  -o <path>     Path of directory to save output.
  -h --help     Show this screen.
"""
from docopt import docopt
from os.path import basename, join, splitext
import random

from lxml import etree

from wikify.utils import clear_xml_node


if __name__ == '__main__':
    opts = docopt(__doc__)

    xml_header = b'<enwiki>\n'
    xml_tail = b'</enwiki>\n'

    input_file = opts['-i']

    # open output files
    train_out = open(join(opts['-o'],
                     basename(splitext(input_file)[0]) + '-train.xml'),
                     'ab')
    test_out = open(join(opts['-o'],
                    basename(splitext(input_file)[0]) + '-test.xml'),
                    'ab')

    # write xml header
    train_out.write(xml_header)
    test_out.write(xml_header)

    prob = eval(opts['-p'])
    assert 0 < prob < 1

    for _, article in etree.iterparse(input_file, tag='article'):
        # splits wiki dump choosing randomly the articles of each part
        article = etree.tostring(article)
        if random.random() < prob:
            train_out.write(article)
        else:
            test_out.write(article)

        # clear data read
        clear_xml_node(article)

    # write xml tail
    train_out.write(xml_tail)
    test_out.write(xml_tail)

    # close files
    train_out.close()
    test_out.close()
