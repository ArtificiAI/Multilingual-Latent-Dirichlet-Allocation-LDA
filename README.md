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
# Refer to snowball lemmatizer's documentation to see all supported languages:
# http://snowball.tartarus.org/texts/stemmersoverview.html

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

## Dependencies and their license

```
numpy==1.14.3           # BSD-3-Clause and BSD-2-Clause BSD-like and Zlib
scikit-learn==0.19.1    # BSD-3-Clause
PyStemmer==1.3.0        # BSD-3-Clause and MIT
snowballstemmer==1.2.1  # BSD-2-Clause
```

## How it works

See the file named `LDA-Pipeline-Tutorial.ipynb` for an example.

## License of this project

This project is published under the MIT License (MIT).

Copyright (c) [2018 Artifici online services inc](https://github.com/ArtificiAI).

Coded by [Guillaume Chevalier](https://github.com/guillaume-chevalier) at [Neuraxio Inc.](https://github.com/Neuraxio)
