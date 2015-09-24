PLN 2015 - Trabajo Práctico 1: Modelado de Lenguaje
===================================================
Pablo Pastore


Ejercicio 1: Corpus
-------------------

El corpus usado para los experimentos lo construí usando libros gratuitos y 
de descarga libre. Primero convertí todos los libros en formato pdf con la
herramienta pdftotext y luego los uní en un único archivo txt.
Este corpus se encuentra en languagemodeling/corpus/ con el nombre 
books_corpus.txt.
Finalmente modifique el script train.py para usar este corpus, agregando
también un patrón para la correcta tokenización de mi corpus.


Ejercicio 2: Modelo de n-gramas
-------------------------------

En esta parte implemente un modelo de n-gramas. Este se puede entrenar
corriendo el script train.py de la siguiente manera::

        python languagemodeling/scripts/train.py -n <n> -o <file>

donde la opcion -n <n> establece que el modelo es un n-grama y -o <file> es
para setear el nombre del modelo entrenado.


Ejercicio 3: Generación de Texto
--------------------------------

En este ejercicio implemente una clase que me permite generar oraciones a
partir de los modelos entrenados.
Para generar estas oraciones debemos correr el script generate.py de la 
siguiente manera::

        python languagemodeling/scripts/generate.py -i <file> -n <n>

donde -i <file> establece el modelo de n-gramas a usar (previamente entrenado)
y -n <n> la cantidad de oraciones a imprimir.

Resultados:
 - Unigramas:
    1.! no y cuando sin . ... no que el banda con del lado que para esto de El
      . presencia estas , quería mujer quizá prodigios , saber que segura Lydia
      blancas se
    2. director las quemaba por sobre contiguas lodo hablar ni en un otras
       del . profundidades pistola que especies a embargo la de imaginó sido
       Lydia no inmigrantes emplearlas ¡ sirven ninguna ancianos muy andaba lo
       casa sentaron , atribuir franqueó Llamó ! mi
    3. era embargo oh . un

 - Bigramas:
    1. Pero ahora , sintiéndome poseído de los postreros rayos no otros diez
       jóvenes : Dile que se alejan un pasillo que la especialización el calor,
       por diferentes especies nuevas emociones que creí que ese terreno muy
       extendidas , porque a las dudas .
    2. Seguro que su hermana sea , las causas políticas que colgaba el jueves
       en el autoritario que su marido abogaría en la inmensa cúpula del
       interior había arrastrado por pasos , trayéndolo en algún modo ,
       aprovechando la Coca Cola , que ha puesto desocupado , no tiene que
       media hora siguiente manera , cuyo rostro muy necesaria de nuestra
       línea no en el bolsillo .
    3. No me encargué , parece increíble que se hizo caso tenemos también
       podemos sacar panoramas magníficos trajes de tu estupidez de caballeros
       que se abrieron , y mi padre es mejor amigo que los animales , y
       solícito custodio de Tetuán !
 
 - Trigramas:
    1. necesito una docena de plantas introducidas que han descendido de una
       muerte cierta y le dijo : Aún tengo que contarte .
    2. y de la mano y contempló más detenidamente al azotador como sustituto .
    3. Y esto es lo mismo que ocurre es que , bordeando lagunas y hundiéndose ,
       sin faltar a todas las de las diferentes lenguas sobre palomas , 
       gallinas , que la vimos convertirse en un profundo problema , ¿ cómo has
       pasado regular , y , no habrán tenido ocasión de la tristeza con que
       había al pie de la boda le producía un efecto misterioso , pues cree
       que es muy fácil ... Pero no creáis que bebiendo se ha hundido muchas
       veces en una lista de todos los hechos que demuestran que son más
       tímidas que las muchas fuerzas del enemigo .
 
 - Cuatrigramas:
    1. y qué corazón tan generoso !
    2. Además , el propietario de la casa y de su corona !
    3. ¡ Y por cierto ¡ oh hijo mío !


Ejercicio 4: Suavizado "add-one"
--------------------------------

En esta parte herede la clase NGram en una clase AddOneNGram para implementar
n-gramas con suavizado add-one.
Este se puede entrenar corriendo el script train.py de la siguiente manera::

        python languagemodeling/scripts/train.py -n <n> -m addone -o <file>

donde la opción -n <n> establece que el modelo es un n-grama, -m addone para
usar suavizado add-one y -o <file> es para setear el nombre del modelo
entrenado.


Ejercicio 5: Evaluación de Modelos de Lenguaje
----------------------------------------------

Desde la consola python importe el corpus books_corpus.txt y lo separe dos
archivos, uno llamado books_corpus_train.txt, usado para entrenar los modelos
de n-gramas conteniendo las oraciones desde inicio del corpus original hasta
el 90% del tamaño de este; el otro archivo, books_corpus_test.txt usado para
testear el modelo entrenado, contiene las oraciones del 10% del final del
corpus original.
Cabe destacar que ambos archivos (books_corpus_train.txt y 
books_corpus_test.txt) son disjuntos.

Luego en el archivo ngram.py implemente dentro de la clase NGram funciones
para evaluar la perplejidad del modelo.

Para evaluar los modelos debemos correr el script eval.py de la siguiente
manera::

    python languagemodeling/scripts/eval.py -i <file>

donde -i <file> establece el modelo a evaluar.

- Resultados:
    N-grama       |    1    |     2   |     3    |   4
    Add-one       | 1786.61 | 6479.56 | 39676.04 | 58835.36
    Interpolation | 1803.97 | 778.93  | 772.61   | 783.41
    Back-off      | 1803.97 | 659.96  | 664.91   | 678.13


Ejercicio 6: Suavizado por Interpolación
----------------------------------------

Para implementar el suavizado por interpolación herede NGram en la clase
InterpolatedNgram.
Este modelo se puede entrenar corriendo el script train.py de la 
siguiente manera::

        python languagemodeling/scripts/train.py -n <n> -m interpolated
            [-g <gamma>] [-a <addone>] -o <file>

donde la opcion -n <n> establece que el modelo es un n-grama, -m interpolated
para usar suavizado por interpolación, -g <gamma> es opcional para setear el
valor de gamma (si no es dado este se calcula con barrido sobre datos held-out),
-a <addone> para usar suavizado add-one en el caso de unigramas y -o <file> es
para setear el nombre del modelo entrenado.


Ejercicio 7: Suavizado por Back-Off con Discounting
---------------------------------------------------

En la clase BackOffNGram herede de NGram y para hacer su computo mas eficiente
cree diccionarios que almacenen los valores posibles para alpha y para el
denominador normalizador.
Este modelo se puede entrenar corriendo el script train.py de la 
siguiente manera::

        python languagemodeling/scripts/train.py -n <n> -m backoff
            [-b <beta>] [-a <addone>] -o <file>

donde la opcion -n <n> establece que el modelo es un n-grama, -m backoff
para usar suavizado por back-off, -b <beta> es opcional para setear el
valor de beta (si no es dado este se calcula con barrido sobre datos held-out),
-a <addone> para usar suavizado add-one en el caso de unigramas y -o <file> es
para setear el nombre del modelo entrenado.
