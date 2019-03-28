from artifici_lda.data_utils import get_params_from_prefix_dict
from artifici_lda.lda_service import LDA_PIPELINE_PARAMS_WORDS
from artifici_lda.logic.count_vectorizer import CountVectorizer
from testing.const_utils import \
    CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED, \
    CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_VECTORIZED, \
    CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_VECTORIZED_LDA_TOPICS_INVERSE_TRANSFORM_1, \
    CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_LDA_TOPICS_INVERSE_TRANSFORM_2


def test_count_vectorizer_transform():
    cv, vectorized = get_vectorized()

    assert (CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_VECTORIZED == vectorized).all()


def test_count_vectorizer_inverse_transform_topics():
    cv, vectorized = get_vectorized()
    topic_words = cv.inverse_transform(CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_VECTORIZED_LDA_TOPICS_INVERSE_TRANSFORM_1)

    #   unvectorized = cv.inverse_transform(vectorized)

    print("")
    print(topic_words)
    print(CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_LDA_TOPICS_INVERSE_TRANSFORM_2)
    assert CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_LDA_TOPICS_INVERSE_TRANSFORM_2 == topic_words


def get_vectorized():
    param_prefix = "count_vect__"
    count_vectorizer_params = get_params_from_prefix_dict(param_prefix, LDA_PIPELINE_PARAMS_WORDS)
    cv = CountVectorizer(**count_vectorizer_params)  # param dict to named arguments.
    vectorized = cv.fit_transform(CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED).toarray()
    return cv, vectorized
