from app.logic.stop_words_remover import StopWordsRemover
from tests.const_utils import \
    TEST_STOPWORDS, \
    CATS_DOGS_COMMENTS_IN_FRENCH_NORMAL, \
    CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS


def test_stopwords_removal_transform():
    swr = StopWordsRemover(stopwords=TEST_STOPWORDS)

    result = swr.fit_transform(CATS_DOGS_COMMENTS_IN_FRENCH_NORMAL)

    print("")
    print(result)
    print(CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS)
    assert CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS == result


def test_stopwords_inverse_transform():
    swr = StopWordsRemover(stopwords=TEST_STOPWORDS)
    swr.fit()
    result = swr.fit_transform(CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS)

    inverted_undone = swr.inverse_transform(result)

    print("")
    print(result)
    print(inverted_undone)
    assert result == inverted_undone  # TODO: behavior is unchanged. Could be improved?
