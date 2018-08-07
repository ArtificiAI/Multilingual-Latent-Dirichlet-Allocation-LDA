# Some code here is inspired from:
# https://github.com/snowballstem/pystemmer/blob/master/docs/quickstart_python3.txt


import Stemmer as st

from string import punctuation


class Stemmer:
    def __init__(self, language='french'):
        self.language = language
        self.words = []

    def lemmatize(self, documents):
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
