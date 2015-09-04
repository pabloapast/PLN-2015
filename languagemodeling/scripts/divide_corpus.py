from math import ceil
import os

WORKPATH = '/home/pablo/facu/2015/pln/PLN-2015'


f = open(os.path.join(WORKPATH, 'corpus/books_corpus.txt'), 'r')

r = f.read()

n = len(r.split('\n')[:-1])

train_list = r.split('\n')[: ceil(n * 0.9)]

test_list = r.split('\n')[- ceil(n * 0.1) : ][: - 1]

# Checkear que son disjuntos
for line in train_list:
    assert not line in test_list

# Checkear que esta el corpus original completo
assert len(train_list) + len(test_list) == n


# Pasar de listas a un unico string
train_string = ("\n").join(train_list)
test_string = ("\n").join(test_list)

# Guardar
train = open(os.path.join(WORKPATH, 'corpus/books_corpus_train.txt'), 'w')
train.write(train_string)
train.close()

test = open(os.path.join(WORKPATH, 'corpus/books_corpus_test.txt'), 'w')
test.write(test_string)
test.close()
