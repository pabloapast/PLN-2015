Wikify! Linking Documents to Encyclopedic Knowledge
===================================================
Pablo Pastore


Introducción:
-------------

Se conoce como "text wikification" la tarea de manualmente seleccionar las
palabras mas importantes de un articulo en wikipedia y linkearlas a otro
articulo dentro de dicha enciclopedia.

Dado el extenso y rico vocabulario de wikipedia, como así también una extensa
fuente de definiciones, se va a trabajar en la creación de un sistema que
automatice la tarea de text wikification.

El sistema consta de dos modulos principales:
    1. Extraer keywords importante
    2. Desambiguar keywords para mapearlas con su correspondiente articulo

Referencia principal: http://digital.library.unt.edu/ark%3A/67531/metadc31001/m2/1/high_res_d/Mihalcea-2007-Wikify-Linking_Documents_to_Encyclopedic.pdf


Dataset:
--------

El dataset usado es un dump de la wikipedia en ingles, se puede descargar desde: http://burnbit.com/torrent/445817/enwiki_20160113_pages_articles_xml_bz2


Limpiar Dataset:
----------------

Debido a la enorme inconsistencia de los datos y la cantidad de informacion redundante se creo un scrip para guardar una version modificada del corpus.

Para ejecutar::

    $ python scripts/clean_wiki_dump.py -i <file> -o <file>

    -i es para pasar el corpus descargado
    -o es el nombre del nuevo corpus "limpio"


Partir Dataset:
---------------

Luego de limpiar el corpus, queremos separar los articulos en dos conjuntos, uno para entrenar los modelos y otro para testearlos.

Para ejecutar::

    python scripts/split_wiki_dump.py -i <file> [-p <p>] -o <path>

    -i es para pasar el corpus a partir
    -p es la proporción asignada para entrenamiento, por default es 0.8 = 80%
    -o es el directorio donde se guardaran los corpus divididos


Entrenamiento:
--------------

    - Extracción y ranking de keywords::

        $ python scripts/train.py -m keyphraseness -i <file> [-n <n>] [--ratio <r>] -o <file>

        -i es para pasar el corpus de entrenamiento
        -n es el máximo rango de ngramas, por default es 3 (osea (1, 3))
        --ratio es el radio promedio entre keywords seleccionadas y cantidad de palabras en el articulo
        -o es el nombre del modelo entrenado

    - Desambiguación::

        $ python scripts/train.py -m disambiguation -i <file> --vocabulary <file> -o <file>

        -i es para pasar el corpus de entrenamiento
        --vocabulary es el modelo anterior entrenado del cual usaremos su vocabulario
        -o es el nombre del modelo entrenado


Evaluación:
-----------

    - Extracción y ranking de keywords::

        $ time python scripts/eval_keyphraseness.py -i models/Keyph-enwiki-clean-train -d wiki-dump/enwiki-clean-test.xml

        real    120m7.039s
        user    117m20.132s
        sys     1m43.468s

        297251 Articles - P: 62.92% - R: 58.11% - F1: 60.42%

    - Desambiguación::

        $ time python scripts/eval_disambiguation.py -i models/Disamb-enwiki-clean-train -d wiki-dump/enwiki-clean-test.xml -t models/enwiki-clean-titles

        real    28m8.273s
        user    27m14.456s
        sys     0m37.604s

        1010598 articles processed - precision: 76.82%


Text Wikification:
------------------

Luego de tener todos los modelos entrenados ya podemos automatizar la tarea de text wikification.

Para ejecutar::

    $ python scripts/text_wikification.py -i <file> -o <file> -k <file> [--ratio <r>] -d <file>

    -i es para pasar el texto de entrada
    -o es el nombre del texto de salida
    -k es el modelo keyphraseness entrenado
    --ratio es el radio explicado anteriormente
    -d es el modelo de desambiguación entrenado

Ejemplo:

    Texto original::

        Presidential contender Donald Trump has come under attack from his rivals at a Republican debate, after a day in which the party's veteran politicians urged voters to desert him.
        The front-runner in the Republican race was on the defensive in Detroit as Marco Rubio and Ted Cruz piled in.
        In a testy debate, Mr Trump admitted he had changed his stance on issues but said flexibility was a strength.
        Senior Republicans say Mr Trump is a liability who would lose the election.
        The debate hosted by Fox News began with Mr Trump being asked about an attack earlier in the day by Mitt Romney, the 2012 nominee, who accused the businessman of bullying, greed and misogyny.
        He also told the audience he reserved the right to be "flexible" and change his mind on issues if he felt like it. He was shown tapes of all the times he'd done just that - the Iraq war, the US involvement in Afghanistan, and on whether to accept Syrian refugees.
        Calling him a "phony" and a "fraud", the former standard-bearer of the party said Mr Trump's policies - like the deportation of undocumented migrants and banning Muslims from entering the US - would make the world less safe.
        Others like Paul Ryan, John McCain and a host of national security committee members have also attacked the New Yorker since he cemented his front-runner status earlier in the week on Super Tuesday.

    Ejecutamos el script::

        $ python scripts/text_wikification.py -i text -o text_parsed -k models/Keyph-enwiki-clean-train -d models/Disamb-enwiki-clean-train

    Texto resultante::

        Presidential contender [[Donald Trump|Donald Trump]] has come under attack from his rivals at a Republican debate, after a day in which the party's veteran politicians urged voters to desert him.
        The front-runner in the Republican race was on the defensive in Detroit as Marco Rubio and Ted Cruz piled in.
        In a testy debate, Mr Trump admitted he had changed his stance on issues but said flexibility was a strength.
        Senior Republicans say Mr Trump is a liability who would lose the election.
        The debate hosted by [[Fox News|Fox News]] began with Mr Trump being asked about an attack earlier in the day by [[Mitt Romney|Mitt Romney,]] the 2012 nominee, who accused the businessman of bullying, greed and [[misogyny|misogyny.]]
        He also told the audience he reserved the right to be "flexible" and change his mind on issues if he felt like it. He was shown tapes of all the times he'd done just that - the [[Iraq War|Iraq war,]] the US involvement in Afghanistan, and on whether to accept Syrian refugees.
        Calling him a "phony" and a "fraud", the former standard-bearer of the party said Mr Trump's policies - like the deportation of undocumented migrants and banning Muslims from entering the US - would make the world less safe.
        Others like [[Paul Ryan|Paul Ryan,]] [[John McCain|John McCain]] and a host of [[National Security Committee of the Republic of Kazakhstan|national security committee]] members have also attacked [[The New Yorker|the New Yorker]] since he cemented his front-runner status earlier in the week on [[Super Tuesday|Super Tuesday.]]
