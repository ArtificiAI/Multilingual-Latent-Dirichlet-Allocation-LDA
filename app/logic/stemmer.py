# Some code here is inspired from:
# https://github.com/snowballstem/pystemmer/blob/master/docs/quickstart_python3.txt
# For more information on PyStemmer's license, see: https://github.com/snowballstem/pystemmer
# (It's a mix of the MIT License and the BSD 3-Clause License)


import Stemmer as st

from string import punctuation


class Stemmer:
    def __init__(self, language='french'):
        """
        Create a stemmer (a.k.a. lemmatizer) for a specific language supported by snowball's algorithms.
        It's a wrapper to the PyStemmer (Stemmer) open-source library.
        """
        self.language = language
        self.words = []

    def lemmatize(self, documents):
        """
        Lemmatize a list of document. A document is a string.
        It may mess with punctuation and special characters.
        """
        lemmatized_documents = []

        for doc in documents:
            # Split on spaces and convert words to their
            # lowercase lemmas, whilst ignoring punctuation.
            doc = doc.lower()
            for punctuation_character in punctuation:
                doc = doc.replace(
                    punctuation_character, " {} ".format(punctuation_character)
                )
            doc = doc.replace("  ", " ").replace("  ", " ").strip()

            words_or_punct = doc.split(" ")

            stemmer = st.Stemmer(self.language)
            stemmed_words = stemmer.stemWords(words_or_punct)

            lemmatized_document = " ".join(stemmed_words)

            lemmatized_documents.append(lemmatized_document)

        return lemmatized_documents

    def inverse_stem(self, words):
        pass
