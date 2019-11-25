# Wikify! Linking Documents to Encyclopedic Knowledge
> Pablo Pastore


## Introduction:

Given a text or hypertext document, we define “text wikification” as the task of automatically extracting the most important words and phrases in the document, and identifying for each such keyword the appropriate link to a Wikipedia article. This is the typical task performed by the Wikipedia users when contributing articles to the Wikipedia repository.

Given the amount and rich vocabulary at Wikipedia, we will work on a system being able to automatically perform the “text wikification” task.

Our system has two principal modules:
Extract the most important words
Disambiguate those words and link them to the corresponding article

Reference: http://digital.library.unt.edu/ark%3A/67531/metadc31001/m2/1/high_res_d/Mihalcea-2007-Wikify-Linking_Documents_to_Encyclopedic.pdf

## Dataset:

http://burnbit.com/torrent/445817/enwiki_20160113_pages_articles_xml_bz2

## Cleaning Dataset:

Given the inconstency on the data and the amout of redundace on the information, we use a script to clean our corpus.

Execute:

```
$ python scripts/clean_wiki_dump.py -i <file> -o <file>
```

## Splitting Dataset:

After corpus cleaning, we want to split the Wikipedia articles in two sets, one for training and another for testing.

Execute:

```
$ python scripts/split_wiki_dump.py -i <file> [-p <p>] -o <path>
```

## Training:

- Keyword extraction and ranking:

```
$ python scripts/train.py -m keyphraseness -i <file> [-n <n>] [--ratio <r>] -o <file>
```

- Word disambiguation:

```
$ python scripts/train.py -m disambiguation -i <file> --vocabulary <file> -o <file>
```

## Evaluation:

- Keyword extraction and ranking:

```
$ time python scripts/eval_keyphraseness.py -i models/Keyph-enwiki-clean-train -d wiki-dump/enwiki-clean-test.xml
```

Output
```
        real    23m52.089s
        user    15m39.196s
        sys     0m18.300s

        297251 Articles - P: 65.32% - R: 60.33% - F1: 62.72%
```

- Word disambiguation:

```
$ time python scripts/eval_disambiguation.py -i models/Disamb-enwiki-clean-train -d wiki-dump/enwiki-clean-test.xml -t models/enwiki-clean-titles
```

Output
```
        real    28m8.273s
        user    27m14.456s
        sys     0m37.604s

        1010598 articles processed - precision: 76.82%
```

## Text Wikification:

After we have our models trained, we can start using them for automatic text wikification:

Execute:

```
    $ python scripts/text_wikification.py -i <file> -o <file> -k <file> [--ratio <r>] -d <file>
```

### Example:

#### Original text:

        Presidential contender Donald Trump has come under attack from his rivals at a Republican debate, after a day in which the party's veteran politicians urged voters to desert him.
        The front-runner in the Republican race was on the defensive in Detroit as Marco Rubio and Ted Cruz piled in.
        In a testy debate, Mr Trump admitted he had changed his stance on issues but said flexibility was a strength.
        Senior Republicans say Mr Trump is a liability who would lose the election.
        The debate hosted by Fox News began with Mr Trump being asked about an attack earlier in the day by Mitt Romney, the 2012 nominee, who accused the businessman of bullying, greed and misogyny.
        He also told the audience he reserved the right to be "flexible" and change his mind on issues if he felt like it. He was shown tapes of all the times he'd done just that - the Iraq war, the US involvement in Afghanistan, and on whether to accept Syrian refugees.
        Calling him a "phony" and a "fraud", the former standard-bearer of the party said Mr Trump's policies - like the deportation of undocumented migrants and banning Muslims from entering the US - would make the world less safe.
        Others like Paul Ryan, John McCain and a host of national security committee members have also attacked the New Yorker since he cemented his front-runner status earlier in the week on Super Tuesday.

We excecute the following:

```
$ python scripts/text_wikification.py -i text -o text_parsed -k models/Keyph-enwiki-clean-train -d models/Disamb-enwiki-clean-train
```

#### Results:

        Presidential contender [[Donald Trump|Donald Trump]] has come under attack from his rivals at a Republican debate, after a day in which the party's veteran politicians urged voters to desert him.
        The front-runner in the Republican race was on the defensive in Detroit as Marco Rubio and Ted Cruz piled in.
        In a testy debate, Mr Trump admitted he had changed his stance on issues but said flexibility was a strength.
        Senior Republicans say Mr Trump is a liability who would lose the election.
        The debate hosted by [[Fox News|Fox News]] began with Mr Trump being asked about an attack earlier in the day by [[Mitt Romney|Mitt Romney,]] the 2012 nominee, who accused the businessman of bullying, greed and [[misogyny|misogyny.]]
        He also told the audience he reserved the right to be "flexible" and change his mind on issues if he felt like it. He was shown tapes of all the times he'd done just that - the [[Iraq War|Iraq war,]] the US involvement in Afghanistan, and on whether to accept Syrian refugees.
        Calling him a "phony" and a "fraud", the former standard-bearer of the party said Mr Trump's policies - like the deportation of undocumented migrants and banning Muslims from entering the US - would make the world less safe.
        Others like [[Paul Ryan|Paul Ryan,]] [[John McCain|John McCain]] and a host of [[National Security Committee of the Republic of Kazakhstan|national security committee]] members have also attacked [[The New Yorker|the New Yorker]] since he cemented his front-runner status earlier in the week on [[Super Tuesday|Super Tuesday.]]
