# Multilingual Latent Dirichlet Allocation (LDA) Pipeline

This project is for text clustering using the Latent Dirichlet Allocation (LDA) algorithm. It can be adapted to many languages provided that the [Snowball stemmer](http://snowball.tartarus.org/texts/stemmersoverview.html), a dependency of this project, supports it.

## Usage

```python

from lda_service.lda_service import train_lda_pipeline_default


FR_STOPWORDS = [
    "le", "les", "la", "un", "de", "en",
    "a", "b", "c", "s",
    "est", "sur", "tres", "donc", "sont",
    # even slang/texto stop words:
    "ya", "pis", "yer"]
# Note: this list of stop words is poor and is just as an example.

fr_comments = [
    "Un super-chat marche sur le trottoir",
    "Les super-chats aiment ronronner",
    "Les chats sont ronrons",
    "Un super-chien aboie",
    "Deux super-chiens",
    "Combien de chiens sont en train d'aboyer?"
]

transformed_comments, top_comments, _1_grams, _2_grams = train_lda_pipeline_default(
    fr_comments,
    n_topics=2,
    stopwords=FR_STOPWORDS,
    language='french')

print(transformed_comments)
print(top_comments)
print(_1_grams)
print(_2_grams)
```
Output:
```
array([[0.14218195, 0.85781805],
       [0.11032992, 0.88967008],
       [0.16960695, 0.83039305],
       [0.88967041, 0.11032959],
       [0.8578187 , 0.1421813 ],
       [0.83039303, 0.16960697]])

['Un super-chien aboie', 'Les super-chats aiment ronronner']

[[('chiens', 3.4911404011996545), ('super', 2.5000203653313933)],
 [('chats',  3.4911393765493255), ('super', 2.499979634668601 )]]

[[('super chiens', 2.4921035508342464)],
 [('super chats',  2.492102155345991 )]]
```

## How it works

See [Multilingual-LDA-Pipeline-Tutorial](https://github.com/ArtificiAI/Multilingual-Latent-Dirichlet-Allocation-LDA/blob/master/Multilingual-LDA-Pipeline-Tutorial.ipynb) for an exhaustive example (intended to be read from top to bottom, not skimmed through). For more explanations on the Inverse Lemmatization, see [Stemming-words-from-multiple-languages](https://github.com/ArtificiAI/Multilingual-Latent-Dirichlet-Allocation-LDA/blob/master/Stemming-words-from-multiple-languages.ipynb).

## Supported Languages

Those languages are supported:

- Danish
- Dutch
- English
- Finnish
- French
- German
- Hungarian
- Italian
- Norwegian
- Porter
- Portuguese
- Romanian
- Russian
- Spanish
- Swedish
- Turkish

You need to bring your own list of stop words. That could be achieved by computing the Term Frequencies on your corpus (or on a bigger corpus of the same language) and to use some of the most common words as stop words.

## Dependencies and their license

```
numpy==1.14.3           # BSD-3-Clause and BSD-2-Clause BSD-like and Zlib
scikit-learn==0.19.1    # BSD-3-Clause
PyStemmer==1.3.0        # BSD-3-Clause and MIT
snowballstemmer==1.2.1  # BSD-3-Clause and BSD-2-Clause
```

## Unit tests

Run pytest with `./run_tests.sh`. Coverage:

```
----------- coverage: platform linux, python 3.6.6-final-0 -----------
Name                                      Stmts   Miss  Cover
-------------------------------------------------------------
lda_service/__init__.py                       0      0   100%
lda_service/application/__init__.py           0      0   100%
lda_service/data/__init__.py                  0      0   100%
lda_service/data/load_sample_data.py          8      8     0%
lda_service/data_utils.py                    39      0   100%
lda_service/lda_service.py                   31      0   100%
lda_service/logic/__init__.py                 0      0   100%
lda_service/logic/count_vectorizer.py         9      0   100%
lda_service/logic/lda.py                     23      7    70%
lda_service/logic/letter_splitter.py         36      4    89%
lda_service/logic/stemmer.py                 60      3    95%
lda_service/logic/stop_words_remover.py      61      5    92%
-------------------------------------------------------------
TOTAL                                       267     27    90%
```

## License

This [project](https://github.com/ArtificiAI/Multilingual-Latent-Dirichlet-Allocation-LDA) is published under the [MIT License (MIT)](https://github.com/ArtificiAI/Multilingual-Latent-Dirichlet-Allocation-LDA/blob/master/LICENSE).

Copyright (c) [2018 Artifici online services inc](https://github.com/ArtificiAI).

Coded by [Guillaume Chevalier](https://github.com/guillaume-chevalier) at [Neuraxio Inc.](https://github.com/Neuraxio)
