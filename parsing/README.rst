PLN 2015 - Trabajo Práctico 3: Análisis Sintáctico
===================================================
Pablo Pastore


Ejercicio 1: Evaluación de Parsers
----------------------------------

1. Entrenar modelos baseline:

    - Flat::

        $ python parsing/scripts/train.py -m flat -o parsing/models/flat

    - RBranch::

        $ python parsing/scripts/train.py -m rbranch -o parsing/models/rbranch

    - LBranch::

        $ python parsing/scripts/train.py -m lbranch -o parsing/models/lbranch

2. Evaluar modelos entrenados (para oraciones de largo <= 20):

    - Flat::

        $ time python parsing/scripts/eval.py -i parsing/models/flat -m 20

    - RBranch::

        $ time python parsing/scripts/eval.py -i parsing/models/rbranch -m 20

    - LBranch::

        $ time python parsing/scripts/eval.py -i parsing/models/lbranch -m 20

    - Para evaluar, por ejemplo, solo 100 oraciones::

        $ time python parsing/scripts/eval.py -i parsing/models/lbranch -m 20 -n 100

3. Resultados de la evaluacion:

    - Flat::

        100.0% (1444/1444) (P=99.93%, R=14.57%, F1=25.43%)
        Parsed 1444 sentences
        Labeled
          Precision: 99.93%
          Recall: 14.57%
          F1: 25.43%
        Unlabeled
          Precision: 100.00%
          Recall: 14.58%
          F1: 25.45%

        real    0m6.727s
        user    0m6.555s
        sys 0m0.159s

    - RBranch::

        100.0% (1444/1444) (P=8.81%, R=14.57%, F1=10.98%)
        Parsed 1444 sentences
        Labeled
          Precision: 8.81%
          Recall: 14.57%
          F1: 10.98%
        Unlabeled
          Precision: 8.87%
          Recall: 14.68%
          F1: 11.06%

        real    0m7.438s
        user    0m7.262s
        sys 0m0.158s

    - LBranch::

        100.0% (1444/1444) (P=8.81%, R=14.57%, F1=10.98%)
        Parsed 1444 sentences
        Labeled
          Precision: 8.81%
          Recall: 14.57%
          F1: 10.98%
        Unlabeled
          Precision: 14.71%
          Recall: 24.33%
          F1: 18.33%

        real    0m7.405s
        user    0m7.231s
        sys 0m0.157s


Ejercicio 2: Algoritmo CKY
--------------------------

Para el test de gramatica con una oracion ambigua use la siguiente oracion: "the fast car mechanic"

Producciones probabilisticas asociadas en CNF::

    NP -> D Ñ           [0.6]
    NP -> D NN          [0.4]

    Ñ -> JJ Ñ           [0.2]
    Ñ -> JJ NN          [0.08]
    Ñ -> NN Ñ           [0.07]
    Ñ -> NN NN          [0.09]
    Ñ -> Ñ Ñ            [0.11]
    Ñ -> Ñ NN           [0.13]
    Ñ -> NN Ñ           [0.15]
    Ñ -> NN NN          [0.17]

    D -> 'the'          [1.0]
    JJ -> 'fast'        [1.0]
    NN -> 'car'         [0.6]
    NN -> 'mechanic'    [0.4]

+-----------------------------------------------------------------------------------------------------------------------------------------------+
| Tabla pi que deberia generar mi CKY:                                                                                                          |
+===================================+===================================+===================================+===================================+
| the                               | fast                              | car                               | mechanic                          |
+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
| D  1.0                            |                                   | NP  1.0*0.048*0.6 = 0.0288        | NP  0.6*1.0*0.2*0.0408 = 0.004896 |
+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
|                -                  | JJ  1.0                           | Ñ  0.6*1.0*0.08 = 0.048           | Ñ  1.0*0.0408*0.2 = 0.00816       |
+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
|                -                  |                -                  | NN  0.6                           | Ñ  0.4*0.6*0.17 = 0.0408          |
+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
|                -                  |                -                  |                -                  | NN  0.4                           |
+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+

Arbol que se espera obtener (el de mayor probabilidad)::

              NP
      ________|___
     |            Ñ
     |    ________|___
     |   |            Ñ
     |   |         ___|_____
     D   JJ       NN        NN
     |   |        |         |
    the fast     car     mechanic


Ejercicio 3: PCFGs No Lexicalizadas
-----------------------------------

1. Entrenar UPCFG::

    $ python parsing/scripts/train.py -m upcfg -o parsing/models/upcfg

2. Evaluar modelo entrenado::

    $ time python parsing/scripts/eval.py -i parsing/models/upcfg -m 20

3. Resultados de la evaluacion::

    100.0% (1444/1444) (P=73.14%, R=72.84%, F1=72.99%)
    Parsed 1444 sentences
    Labeled
      Precision: 73.14%
      Recall: 72.84%
      F1: 72.99%
    Unlabeled
      Precision: 75.25%
      Recall: 74.94%
      F1: 75.09%

    real    3m40.891s
    user    3m38.775s
    sys 0m1.059s


Ejercicio 4: Markovización Horizontal
-------------------------------------
