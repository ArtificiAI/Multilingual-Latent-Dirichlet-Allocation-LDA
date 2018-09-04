# Some code here is inspired from:
# https://github.com/snowballstem/pystemmer/blob/master/docs/quickstart_python3.txt
# For more information on PyStemmer's license, see: https://github.com/snowballstem/pystemmer
# (It's a mix of the MIT License and the BSD 3-Clause License)

from string import punctuation
import unidecode
import sys

from sklearn.base import TransformerMixin
import Stemmer as st

FRENCH = 'french'
ENGLISH = 'english'


# More languages:
# ['danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian',
#  'norwegian', 'porter', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish', 'turkish']

class Stemmer(TransformerMixin):
    def __init__(self, language=FRENCH):
        """
        Create a stemmer (a.k.a. lemmatizer) for a specific language supported by snowball's algorithms.
        It's a wrapper to the PyStemmer (Stemmer) open-source library.

        :param language: the language, refer to snowball lemmatizer's documentation for a list
            of languages. Example: 'french', 'english'. http://snowball.tartarus.org/texts/stemmersoverview.html
        """
        self.language = language

        self.stemmed_word_to_equiv_word_count = dict()

    def get_params(self, deep=True):
        """
        This function is implemented for the class to be usable by scikit-learn's Pipeline() behavior.
        """
        return {"language": self.language}

    def set_params(self, **parameters):
        """
        This function is implemented for the class to be usable by scikit-learn's Pipeline() behavior.
        """
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self

    def fit(self, X=None, y=None):
        """
        This function is implemented for the class to be usable by scikit-learn's Pipeline() behavior.
        y is ignored here, but required by convention.
        """

        # TODO: implement fit_transform which would be more optimized (in time) for this step.
        for document in X:
            self.stem_document(document, re_fit=True)

        return self

    def transform(self, documents):
        """
        This function is implemented for the class to be usable by scikit-learn's Pipeline() behavior.

        Stem all the words in a list of document. A document is a string.
        It may mess with punctuation and special characters.
        """
        stemmed_documents = []
        for doc in documents:
            stemmed_document = self.stem_document(doc, re_fit=False)
            stemmed_documents.append(stemmed_document)
        return stemmed_documents

    def stem_document(self, doc, re_fit):
        """
        Stem the documents, and prepare the stemmer for the inverse_transform by keeping track of
        a mapping back to the original expressions and their counts.

        :param doc: document string
        :param re_fit: boolean, if True, it will prepare the stemmer for the inverse_transform by saving state.
        :return: stemmed document string
        """
        # Ignore punctuation and split on spaces.
        for punctuation_character in punctuation:
            doc = doc.replace(
                # punctuation_character, " {} ".format(punctuation_character)
                punctuation_character, " ".format(punctuation_character)
            )
        doc = doc.replace("  ", " ").replace("  ", " ").strip()

        # words_or_punct = doc.split(" ")
        # stemmer = st.Stemmer(self.language)
        # stemmed_words = stemmer.stemWords(words_or_punct)

        # Stemmed words won't have accents nor capital letters anymore.
        transformed_words = [unidecode.unidecode(w).lower() for w in doc.split(" ")]
        words = doc.split(" ")
        stemmer = st.Stemmer(self.language)
        stemmed_words = stemmer.stemWords(transformed_words)

        if re_fit:
            # Keep track of things for inverse stemming: each word has its count.
            # But the inverse relationship is not deterministic: we need to count occurences
            # because we need the TOP equivalent word back.
            for (_word, _stemmed_word) in zip(words, stemmed_words):

                if _stemmed_word in self.stemmed_word_to_equiv_word_count:

                    if _word in self.stemmed_word_to_equiv_word_count[_stemmed_word]:
                        count_yet = self.stemmed_word_to_equiv_word_count[_stemmed_word][_word]
                        self.stemmed_word_to_equiv_word_count[_stemmed_word][_word] = count_yet + 1  # += 1
                    else:
                        self.stemmed_word_to_equiv_word_count[_stemmed_word][_word] = 1
                else:
                    self.stemmed_word_to_equiv_word_count[_stemmed_word] = {_word: 1}

        else:
            stemmed_document = " ".join(stemmed_words)
            return stemmed_document

    def inverse_transform(self, stemmed_documents):
        """
        Stemmed words to a guess of the original words. Documents are lists of words.

        :param stemmed_documents: a list of documents. Specially here, a document isn't a string, but a list of words.
        :return: a guess of unstemmed documents. Each document is a list of words, not a string.
        """

        all_reversed_words = []

        for stemmed_words in stemmed_documents:
            reversed_words = [self.find_orig_word(word) for word in stemmed_words]

            all_reversed_words.append(reversed_words)

        return all_reversed_words

    def find_orig_word(self, word):
        """
        Guess what was the original word from the state saved during the fit(...) method.

        :param word: a string
        :return: a string of a guess of the original word.
        """

        if " " in word:
            # If there is a space, this means some words were combined into n-grams.
            # So let's undo each word by itself recursively:
            return " ".join(
                [self.find_orig_word(w) for w in word.split(" ")]
            )

        try:
            # Note: a DefaultDict may be used here to remove this try/except.
            orig_words = self.stemmed_word_to_equiv_word_count[word]
        except:
            print("Warning, stemmer.py, find_orig_word('{}'): "
                  "word '{}' not found in vocabulary for inverse stemming.".format(word, word), file=sys.stderr)
            return ""
        # Compare original words on their count which is the last "-1" item of tuples from the inner dicts.
        return max(list(orig_words.items()), key=lambda item: item[-1])[0]
