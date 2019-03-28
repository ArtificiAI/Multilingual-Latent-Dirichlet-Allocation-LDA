from artifici_lda.logic.stemmer import Stemmer, FRENCH
from testing.const_utils import \
    CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS, \
    CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED, \
    CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_THEN_UNSTEMMED, \
    CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_LDA_TOPICS_INVERSE_TRANSFORM_2, \
    CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_LDA_TOPICS_INVERSE_TRANSFORM_3


def test_stopwords_removal_transform():
    st = Stemmer(language=FRENCH)

    result = st.transform(CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS)

    print("")
    print(result)
    print(CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED)
    assert CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED == result


def test_stopwords_inverse_transform():
    st = Stemmer(language=FRENCH)
    result = st.fit_transform(CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS)

    result = [r.split(" ") for r in result]  # Seems like we need to do this to support inverse topics in priority.
    inverted_undone = st.inverse_transform(result)

    print("")
    print(inverted_undone)
    print(CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_THEN_UNSTEMMED)

    assert CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_THEN_UNSTEMMED == inverted_undone


def test_stopwords_inverse_transform_topics():
    st = Stemmer(language=FRENCH)
    _ = st.fit(CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS)

    inverted_undone = st.inverse_transform(CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_LDA_TOPICS_INVERSE_TRANSFORM_2)

    print("")
    print(inverted_undone)
    print(CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_LDA_TOPICS_INVERSE_TRANSFORM_3)

    assert CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_LDA_TOPICS_INVERSE_TRANSFORM_3 == inverted_undone
