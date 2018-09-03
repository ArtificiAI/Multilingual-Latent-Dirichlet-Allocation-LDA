import numpy as np

from lda_service.data_utils import get_params_from_prefix_dict
from lda_service.lda_service import LDA_PIPELINE_PARAMS_WORDS
from lda_service.logic.lda import LDA
from tests.const_utils import \
    CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_VECTORIZED, \
    CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_VECTORIZED_LDA_TOPICS_INVERSE_TRANSFORM_1, \
    CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_VECTORIZED_LDA_TOPICS_INVERSE_TRANSFORM_1_INV, \
    CATS_DOGS_LABELS_B, CATS_DOGS_LABELS_A


def test_lda_transform():
    lda, clusterized_comments = get_lda()

    print("")
    print(clusterized_comments)
    print(CATS_DOGS_LABELS_A)
    print(CATS_DOGS_LABELS_B)
    assert (
            (clusterized_comments.argmax(-1) == CATS_DOGS_LABELS_A).all() or
            (clusterized_comments.argmax(-1) == CATS_DOGS_LABELS_B).all()
    ), "Error. Got {}".format(clusterized_comments, clusterized_comments.argmax(-1))


def test_lda_inverse_transform_topics():
    lda, clusterized_comments = get_lda()
    expected_a = CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_VECTORIZED_LDA_TOPICS_INVERSE_TRANSFORM_1
    expected_b = CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_VECTORIZED_LDA_TOPICS_INVERSE_TRANSFORM_1_INV

    topic_features = lda.inverse_transform()

    print("")
    print(topic_features)
    print(expected_a)
    print(expected_b)
    assert (
            (np.array(expected_a) == np.array(topic_features)).all() or
            (np.array(expected_b) == np.array(topic_features)).all()
    )


def get_lda():
    param_prefix = "lda__"
    lda_params = get_params_from_prefix_dict(param_prefix, LDA_PIPELINE_PARAMS_WORDS)
    lda = LDA(**lda_params)  # param dict to named arguments.
    clusterized = lda.fit_transform(CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_VECTORIZED)
    return lda, clusterized
