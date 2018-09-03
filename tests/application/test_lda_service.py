from lda_service.lda_service import \
    train_lda_pipeline_default, \
    train_lda_pipeline_on_words
from lda_service.logic.stemmer import FRENCH
from tests.const_utils import \
    CATS_DOGS_COMMENTS_IN_FRENCH_NORMAL, \
    CATS_DOGS_LABELS_A, \
    CATS_DOGS_LABELS_B, \
    CATS_TOP_WORDS, \
    DOGS_TOP_WORDS, \
    TEST_STOPWORDS


def test_lda_can_cluster_obvious_text():
    transformed_comments, _, _, _ = train_lda_pipeline_on_words(
        CATS_DOGS_COMMENTS_IN_FRENCH_NORMAL,
        n_topics=2,
        stopwords=TEST_STOPWORDS,
        language=FRENCH)

    assert (
            (transformed_comments.argmax(-1) == CATS_DOGS_LABELS_A).all() or
            (transformed_comments.argmax(-1) == CATS_DOGS_LABELS_B).all()
    ), "Error. Got {}".format(transformed_comments, transformed_comments.argmax(-1))


def test_topics_and_words_are_as_expected():
    _, _, topics_and_words_1_gram, _ = train_lda_pipeline_on_words(
        CATS_DOGS_COMMENTS_IN_FRENCH_NORMAL,
        n_topics=2,
        stopwords=TEST_STOPWORDS,
        language=FRENCH)

    # print(topics_and_words[0][0].values())

    # Words are sorted from most important to last important
    topic_a_words = set([word for word, word_weight in topics_and_words_1_gram[0]])
    topic_b_words = set([word for word, word_weight in topics_and_words_1_gram[1]])
    print(topic_a_words)
    print(topic_b_words)

    # The "or" is because we don't know which cluster is which.
    assert ((topic_a_words == CATS_TOP_WORDS and topic_b_words == DOGS_TOP_WORDS) or
            (topic_b_words == CATS_TOP_WORDS and topic_a_words == DOGS_TOP_WORDS))


def test_lda_can_cluster_text_with_no_important_words_but_based_on_letters():
    # No words repeating across two documents would remain after stopwords removal,
    # so this way the LDA receives no words at all:
    random_comments = [
        # b and a:
        "abababababa le abba du abababb les des abbabbaba",
        "ababababa le aba du ababab les des abbaba",
        # b and g:
        "ggbgbg bgbbbg du gbbbggbgb le bgbbbbgggbbg les bbbbggbbbggg des bggbgbbggb",
        "bgbg du gggb le bbbg les bgbggg des bgbgbg",
    ]
    transformed_comments, _, _, _ = train_lda_pipeline_default(
        random_comments,
        n_topics=2,
        stopwords=TEST_STOPWORDS,
        language=FRENCH)

    category = transformed_comments.argmax(-1)
    # At least, let's see if we find the 2 categories:
    assert category[0] == category[1]
    assert category[2] == category[3]
    assert category[0] == 1 - category[2]
    assert category[1] == 1 - category[3]
