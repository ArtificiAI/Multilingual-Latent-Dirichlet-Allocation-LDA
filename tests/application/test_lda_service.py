from app.application.lda_service import train_lda_pipeline_failsafe_fallback_with_ngrams_on_letters
from tests.const_utils import CATS_DOGS_COMMENTS_IN_FRENCH_NORMAL, CATS_DOGS_LABELS_A, CATS_DOGS_LABELS_B, \
    CATS_TOP_WORDS_ORDERING, DOGS_TOP_WORDS_ORDERING


def test_lda_can_cluster_obvious_text():
    transformed_comments, _ = train_lda_pipeline_failsafe_fallback_with_ngrams_on_letters(CATS_DOGS_COMMENTS_IN_FRENCH_NORMAL)

    assert (
            (transformed_comments.argmax(-1) == CATS_DOGS_LABELS_A).all() or
            (transformed_comments.argmax(-1) == CATS_DOGS_LABELS_B).all()
    ), "Error. Got {}".format(transformed_comments, transformed_comments.argmax(-1))


def test_topics_and_words_are_as_expected():
    _, topics_and_words = train_lda_pipeline_failsafe_fallback_with_ngrams_on_letters(CATS_DOGS_COMMENTS_IN_FRENCH_NORMAL)

    # print(topics_and_words[0][0].values())

    # Words are sorted from most important to last important
    topic_a_words = [word for word, word_weight in topics_and_words[0]]
    topic_b_words = [word for word, word_weight in topics_and_words[1]]

    # The "or" is because we don't know which cluster is which.
    assert (
            ((topic_a_words.index(CATS_TOP_WORDS_ORDERING[0]) < topic_a_words.index(CATS_TOP_WORDS_ORDERING[1])) and
             (topic_b_words.index(DOGS_TOP_WORDS_ORDERING[0]) < topic_b_words.index(DOGS_TOP_WORDS_ORDERING[1]))
             ) or
            ((topic_b_words.index(CATS_TOP_WORDS_ORDERING[0]) < topic_b_words.index(CATS_TOP_WORDS_ORDERING[1])) and
             (topic_a_words.index(DOGS_TOP_WORDS_ORDERING[0]) < topic_a_words.index(DOGS_TOP_WORDS_ORDERING[1]))
             )
    ), ("ERROR. topic_a_words: {}, topic_b_words: {}".format(topic_a_words, topic_b_words))
