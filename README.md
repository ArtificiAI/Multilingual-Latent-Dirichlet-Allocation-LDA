# Multilingual Latent Dirichlet Allocation (LDA) Pipeline

This project is for text clustering using the Latent Dirichlet Allocation (LDA) algorithm. It can be adapted to many languages provided that the [Snowball stemmer](http://snowball.tartarus.org/texts/stemmersoverview.html), a dependency of this project, supports it.

## Usage

```python

from artifici_lda.lda_service import train_lda_pipeline_default


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
numpy==1.26.3           # BSD-3-Clause and BSD-2-Clause BSD-like and Zlib
scikit-learn==1.4.0     # BSD-3-Clause
PyStemmer==2.2.0.1      # BSD-3-Clause and MIT
snowballstemmer==2.2.0  # BSD-3-Clause and BSD-2-Clause
translitcodec==0.7.0    # MIT License
scipy==1.12.0           # BSD-3-Clause and MIT-like
```

## Unit tests

Run pytest with `./run_tests.sh`. Coverage:

```
----------- coverage: platform linux, python 3.6.7-final-0 -----------
Name                                       Stmts   Miss  Cover
--------------------------------------------------------------
artifici_lda/__init__.py                       0      0   100%
artifici_lda/data_utils.py                    39      0   100%
artifici_lda/lda_service.py                   31      0   100%
artifici_lda/logic/__init__.py                 0      0   100%
artifici_lda/logic/count_vectorizer.py         9      0   100%
artifici_lda/logic/lda.py                     23      7    70%
artifici_lda/logic/letter_splitter.py         36      4    89%
artifici_lda/logic/stemmer.py                 60      3    95%
artifici_lda/logic/stop_words_remover.py      61      5    92%
--------------------------------------------------------------
TOTAL                                        259     19    93%
```

## License

This [project](https://github.com/ArtificiAI/Multilingual-Latent-Dirichlet-Allocation-LDA) is published under the [MIT License (MIT)](https://github.com/ArtificiAI/Multilingual-Latent-Dirichlet-Allocation-LDA/blob/master/LICENSE).

Copyright (c) [2018 Artifici online services inc](https://github.com/ArtificiAI).

Coded by [Guillaume Chevalier](https://github.com/guillaume-chevalier) at [Neuraxio Inc.](https://github.com/Neuraxio)
