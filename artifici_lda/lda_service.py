from artifici_lda.data_utils import link_topics_and_weightings, get_top_comments, split_1_grams_from_n_grams, \
    get_lda_params_with_specific_n_cluster_or_language, get_word_weightings
from artifici_lda.logic.letter_splitter import LetterSplitter
from artifici_lda.logic.stop_words_remover import StopWordsRemover
from artifici_lda.logic.stemmer import Stemmer, FRENCH
from artifici_lda.logic.lda import LDA
from artifici_lda.logic.count_vectorizer import CountVectorizer

from sklearn.pipeline import Pipeline

LDA_PIPELINE_PARAMS_WORDS = {
    'stopwords__stopwords': None,
    'stemmer__language': FRENCH,  # ENGLISH
    'count_vect__max_df': 0.98,
    'count_vect__min_df': 2,
    'count_vect__max_features': 10000,
    'count_vect__ngram_range': (1, 2),
    'count_vect__strip_accents': None,
    'lda__n_components': 2,
    'lda__max_iter': 750,
    'lda__learning_decay': 0.5,
    'lda__learning_method': 'online',
    'lda__learning_offset': 10,
    'lda__batch_size': 25,
    'lda__n_jobs': -1,  # Use all CPUs
}

LDA_PIPELINE_PARAMS_LETTERS = {
    'stopwords__stopwords': None,
    # No stemmer here, so no language 'stemmer__language' is set.
    'count_vect__max_df': 0.98,
    'count_vect__min_df': 2,
    'count_vect__max_features': 10000,
    'count_vect__ngram_range': (1, 2),
    'count_vect__strip_accents': None,
    'lda__n_components': 2,
    'lda__max_iter': 750,
    'lda__learning_decay': 0.5,
    'lda__learning_method': 'online',
    'lda__learning_offset': 10,
    'lda__batch_size': 25,
    'lda__n_jobs': -1,  # Use all CPUs
}


def train_lda_pipeline_default(comments, n_topics=2, language=FRENCH, stopwords=None):
    """
    Try to train a pipeline on ngrams of words, and if it fails (because no words were found), try on ngrams of letters.

    :param comments: a list of strings
    :param n_topics: the number of clusters (categories, groups, or topics) to find.
    :param language: the language, refer to snowball lemmatizer's documentation for a list
        of languages. Example: 'french', 'english'. http://snowball.tartarus.org/texts/stemmersoverview.html
    :return: a list containing the topic probabilities for each comment, and another list containing topics if it
        trained on words, where each topic is a list of tuples, where each of those tuples are of the form
        (str('word'), float(importance_of_word)), sorted by the importance of each word (most important comes first).
    """
    try:
        return train_lda_pipeline_on_words(comments, n_topics=n_topics, language=language, stopwords=stopwords)
    except:
        return train_lda_pipeline_on_letters(comments, n_topics=n_topics, stopwords=stopwords)


def train_lda_pipeline_on_words(comments, n_topics=2, language=FRENCH, stopwords=None):
    """
    Train an LDA and transform the comments.

    :param comments: a list of strings
        call to this method failed to extract word features from the CountVectorizer.
    :param n_topics: the number of clusters (categories, groups, or topics) to find.
    :param language: the language, refer to snowball lemmatizer's documentation for a list
        of languages. Example: 'french', 'english'. http://snowball.tartarus.org/texts/stemmersoverview.html
    :return: a list containing the topic probabilities for each comment, and another list containing topics, where each
        topic is a list of tuples, where each of those tuples are of the form (str('word'), float(importance_of_word)),
        sorted by the importance of each word (most important comes first).
    """
    params = get_lda_params_with_specific_n_cluster_or_language(
        LDA_PIPELINE_PARAMS_WORDS, n_topics=n_topics, language=language, stopwords=stopwords)

    lda_pipeline = Pipeline([
        ('stopwords', StopWordsRemover()),
        ('stemmer', Stemmer()),
        ('count_vect', CountVectorizer()),
        ('lda', LDA()),
    ]).set_params(**params)

    # Fit the data
    transformed_comments = lda_pipeline.fit_transform(comments)
    top_comments = get_top_comments(comments, transformed_comments)

    # Extract information about data
    topic_words = lda_pipeline.inverse_transform(X=None)
    topic_words_weighting = get_word_weightings(lda_pipeline)
    topics_words_and_weightings = link_topics_and_weightings(topic_words, topic_words_weighting)

    # Manipulations on the information for a clean return.
    _1_grams, _2_grams = split_1_grams_from_n_grams(topics_words_and_weightings)

    return transformed_comments, top_comments, _1_grams, _2_grams


def train_lda_pipeline_on_letters(comments, n_topics=2, stopwords=None):
    """
    Train an LDA and transform the comments.

    :param comments: a list of strings
    :param n_topics: the number of clusters (categories, groups, or topics) to find.
    :param language: the language, refer to snowball lemmatizer's documentation for a list
        of languages. Example: 'french', 'english'.
    :return: a list containing the topic probabilities for each comment, and another list that is empty but that
        would normally contain topics' descriptions.
    """
    params = get_lda_params_with_specific_n_cluster_or_language(
        LDA_PIPELINE_PARAMS_LETTERS, n_topics=n_topics, stopwords=stopwords)

    lda_pipeline = Pipeline([
        ('stopwords', StopWordsRemover()),
        ('letter_splitter', LetterSplitter()),
        ('count_vect', CountVectorizer()),
        ('lda', LDA()),
    ]).set_params(**params)

    # Fit the data
    transformed_comments = lda_pipeline.fit_transform(comments)
    # print("score:", lda_pipeline.score(comments))

    no_info_for_topics = [[] for _ in range(LDA_PIPELINE_PARAMS_LETTERS['lda__n_components'])]

    top_comments = get_top_comments(comments, transformed_comments)

    return transformed_comments, top_comments, no_info_for_topics, no_info_for_topics
