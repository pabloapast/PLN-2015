PLN 2015 - Trabajo Práctico 2: Etiquetado de Secuencias
===================================================
Pablo Pastore


Ejercicio 1: Corpus Ancora: Estadísticas de etiquetas POS
---------------------------------------------------------

El objetivo de este ejercicio es hacer un script que muestre estadísticas del
corpus que usamos (Ancora)

1. Uso del script::

    $ python tagging/scripts/stats.py

2. Resultado::

    --- Basic Stats ---
    Sents: 17379
    Words: 517268
    Vocabulary: 46482
    Tags: 48

    --- Frequent Tags ---
     Tag    Frequence
     nc     92002 - 17.79% = ['años', 'presidente', 'millones', 'equipo', 'partido']
     sp     79904 - 15.45% = ['de', 'en', 'a', 'del', 'con']
     da     54552 - 10.55% = ['la', 'el', 'los', 'las', 'El']
     vm     50609 - 9.78% = ['está', 'tiene', 'dijo', 'puede', 'hace']
     aq     33904 - 6.55% = ['pasado', 'gran', 'mayor', 'nuevo', 'próximo']
     fc     30148 - 5.83% = [',']
     np     29113 - 5.63% = ['Gobierno', 'España', 'PP', 'Barcelona', 'Madrid']
     fp     21157 - 4.09% = ['.', '(', ')']
     rg     15333 - 2.96% = ['más', 'hoy', 'también', 'ayer', 'ya']
     cc     15023 - 2.90% = ['y', 'pero', 'o', 'Pero', 'e']

    --- Ambiguity Levels ---
     Ambiguity  Words
     1 tag      44109 - 94.89% = [',', 'el', 'en', 'con', 'por']
     2 tag      2194 - 4.72% = ['la', 'y', '"', 'los', 'del']
     3 tag      153 - 0.33% = ['.', 'a', 'un', 'no', 'es']
     4 tag      19 - 0.04% = ['de', 'dos', 'este', 'tres', 'todo']
     5 tag      4 - 0.01% = ['que', 'mismo', 'cinco', 'medio']
     6 tag      3 - 0.01% = ['una', 'como', 'uno']
     7 tag      0 - 0.00% = []
     8 tag      0 - 0.00% = []
     9 tag      0 - 0.00% = []


Ejercicio 2: Baseline Tagger
----------------------------

Programar un etiquetador baseline, que elija para cada palabra su etiqueta
más frecuente observada en entrenamiento


Ejercicio 3: Entrenamiento y Evaluación de Taggers
--------------------------------------------------

1. Entrenar etiquetador baseline::

    $ python tagging/scripts/train.py -m base -o tagging/models/baseline

2. Evaluar accuracy del etiquetador::

    $ python tagging/scripts/eval.py -i tagging/models/baseline

    (Mostrar matriz de confusion)
    $ python tagging/scripts/eval.py -c -i tagging/models/baseline

3. Resultados::

    - baseline
    100.0% (89.00% / 95.32% / 31.80%)
    Accuracy: 89.00%
    Accuracy known words: 95.32%
    Accuracy unknown words: 31.80%
         nc     sp     vm     da     aq     fc     fp     rg     np     cc
    nc    -    0.00%  0.00%  0.00%  0.00%  0.00%  0.00%  0.00%  0.00%  0.00%
    sp  0.00%    -    0.00%  0.00%  0.00%  0.00%  0.00%  0.00%  0.00%  0.00%
    vm  0.02%  0.00%    -    0.00%  0.00%  0.00%  0.00%  0.00%  0.00%  0.00%
    da  0.00%  0.00%  0.00%    -    0.00%  0.00%  0.00%  0.00%  0.00%  0.00%
    aq  0.02%  0.00%  0.00%  0.00%    -    0.00%  0.00%  0.00%  0.00%  0.00%
    fc  0.00%  0.00%  0.00%  0.00%  0.00%    -    0.00%  0.00%  0.00%  0.00%
    fp  0.00%  0.00%  0.00%  0.00%  0.00%  0.00%    -    0.00%  0.00%  0.00%
    rg  0.00%  0.00%  0.00%  0.00%  0.00%  0.00%  0.00%    -    0.00%  0.00%
    np  0.02%  0.00%  0.00%  0.00%  0.00%  0.00%  0.00%  0.00%    -    0.00%
    cc  0.00%  0.00%  0.00%  0.00%  0.00%  0.00%  0.00%  0.00%  0.00%    -


Ejercicio 4: Hidden Markov Models y Algoritmo de Viterbi
--------------------------------------------------------

En esta parte debíamos implementar un Hidden Markov Model que recibe como parámetro
las probabilidades de transición entre etiquetas (tags) y de emisión de palabras dado un tag.
En otra clase se implemento el algoritmo de Viterbi que calcula el etiquetado más probable de una oración.


Ejercicio 5: HMM POS Tagger
---------------------------

Se implemento un Hidden Markov Model donde los parámetros son estimados usando
Maximum Likelihood

1. Entrenar etiquetador (caso n = 1, usando addone)::

    $ python tagging/scripts/train.py -m mlhmm -n 1 -a -o tagging/models/hmm1

2. Evaluar accuracy del etiquetador::

    $ python tagging/scripts/eval.py -i tagging/models/hmm1

3. Resultados::

    - n = 1
    100.0% (89.01% / 95.32% / 31.80%)
    Accuracy: 89.01%
    Accuracy known words: 95.32%
    Accuracy unknown words: 31.80%

    real    0m12.388s
    user    0m12.216s
    sys 0m0.136s

    - n = 2
    100.0% (92.72% / 97.61% / 48.42%)
    Accuracy: 92.72%
    Accuracy known words: 97.61%
    Accuracy unknown words: 48.42%

    real    0m24.211s
    user    0m24.056s
    sys 0m0.108s

    - n = 3
    100.0% (93.16% / 97.67% / 52.36%)
    Accuracy: 93.16%
    Accuracy known words: 97.67%
    Accuracy unknown words: 52.36%

    real    1m31.388s
    user    1m31.096s
    sys 0m0.212s

    - n = 4
    100.0% (93.13% / 97.43% / 54.13%)
    Accuracy: 93.13%
    Accuracy known words: 97.43%
    Accuracy unknown words: 54.13%

    real    8m23.666s
    user    8m22.432s
    sys 0m0.860s


Ejercicio 6: Features para Etiquetado de Secuencias
---------------------------------------------------

Implementación de algunos features que luego vamos a usar para entrenar un tagger
usando scikit-learn


Ejercicio 7: Maximum Entropy Markov Models
------------------------------------------

Implementé un Maximum Entropy Markov Model usando un pipeline de scikit-learn de
la siguiente forma:
- Vectorizador (featureforge.vectorizer.Vectorizer) con los features definidos
en el ejercicio anterior
- Clasificador de máxima entropía (sklearn.linear_model.LogisticRegression). También
usé como clasificadores el Multinomial Naive Bayes y Linear Support Vector Classification
(sklearn.naive_bayes.MultinomialNB, sklearn.svm.LinearSVC)

Para el algoritmo de tagging se usó beam inference con un beam de tamaño 1.

1. Entrenar etiquetador (caso n = 1, usando Logistic Regression)::

    $ python tagging/scripts/train.py -m memm -n 1 -o tagging/models/memm1-maxent

2. Evaluar accuracy del etiquetador::

    $ python tagging/scripts/eval.py -i tagging/models/memm1-maxent

3. Resultados::

    - n = 1, Logistic Regression
    100.0% (92.70% / 95.28% / 69.32%)
    Accuracy: 92.70%
    Accuracy known words: 95.28%
    Accuracy unknown words: 69.32%

    real    0m38.174s
    user    0m37.928s
    sys 0m0.168s

    - n = 2, Logistic Regression
    100.0% (91.97% / 94.54% / 68.76%)
    Accuracy: 91.97%
    Accuracy known words: 94.54%
    Accuracy unknown words: 68.76%

    real    0m39.820s
    user    0m39.612s
    sys 0m0.156s

    - n = 3, Logistic Regression
    100.0% (92.17% / 94.71% / 69.18%)
    Accuracy: 92.17%
    Accuracy known words: 94.71%
    Accuracy unknown words: 69.18%

    real    0m42.837s
    user    0m42.624s
    sys 0m0.160s

    - n = 4, Logistic Regression
    100.0% (92.23% / 94.72% / 69.65%)
    Accuracy: 92.23%
    Accuracy known words: 94.72%
    Accuracy unknown words: 69.65%

    real    0m45.175s
    user    0m44.932s
    sys 0m0.188s

    - n = 1, Linear Support Vector
    100.0% (94.43% / 97.04% / 70.82%)
    Accuracy: 94.43%
    Accuracy known words: 97.04%
    Accuracy unknown words: 70.82%

    real    0m37.903s
    user    0m37.712s
    sys 0m0.136s

    - n = 2, Linear Support Vector
    100.0% (94.29% / 96.91% / 70.57%)
    Accuracy: 94.29%
    Accuracy known words: 96.91%
    Accuracy unknown words: 70.57%

    real    0m40.580s
    user    0m40.372s
    sys 0m0.156s

    - n = 3, Linear Support Vector
    100.0% (94.40% / 96.94% / 71.40%)
    Accuracy: 94.40%
    Accuracy known words: 96.94%
    Accuracy unknown words: 71.40%

    real    0m42.919s
    user    0m42.724s
    sys 0m0.140s

    - n = 4, Linear Support Vector
    100.0% (94.46% / 96.96% / 71.81%)
    Accuracy: 94.46%
    Accuracy known words: 96.96%
    Accuracy unknown words: 71.81%

    real    0m44.930s
    user    0m44.720s
    sys 0m0.156s

    - n = 1, Multinomial Naive Bayes
    100.0% (82.18% / 85.85% / 48.89%)
    Accuracy: 82.18%
    Accuracy known words: 85.85%
    Accuracy unknown words: 48.89%

    real    29m57.565s
    user    29m49.024s
    sys 0m2.944s

    - n = 2, Multinomial Naive Bayes
    100.0% (66.64% / 69.96% / 36.50%)
    Accuracy: 66.64%
    Accuracy known words: 69.96%
    Accuracy unknown words: 36.50%

    real    43m55.288s
    user    40m29.836s
    sys 0m13.684s

    - n = 3, Multinomial Naive Bayes
    100.0% (66.07% / 69.25% / 37.25%)
    Accuracy: 66.07%
    Accuracy known words: 69.25%
    Accuracy unknown words: 37.25%

    real    27m53.838s
    user    27m30.280s
    sys 0m14.288s

    - n = 4, Multinomial Naive Bayes
    100.0% (64.12% / 66.81% / 39.75%)
    Accuracy: 64.12%
    Accuracy known words: 66.81%
    Accuracy unknown words: 39.75%

    real    30m40.390s
    user    30m19.452s
    sys 0m15.508s
