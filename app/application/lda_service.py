from app.application.data_utils import link_topics_and_weightings
from app.logic.letter_splitter import LetterSplitter
from app.logic.stop_words_remover import StopWordsRemover
from app.logic.stemmer import Stemmer, FRENCH
from app.logic.lda import LDA
from app.logic.count_vectorizer import CountVectorizer

from sklearn.pipeline import Pipeline

LDA_PIPELINE_PARAMS_WORDS = {
    'stemmer__language': FRENCH,
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
    # No language here, so no `'stemmer__language': FRENCH,`.
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


def train_lda_pipeline_default(comments):
    """
    Try to train a pipeline on ngrams of words, and if it fails (because no words were found), try on ngrams of letters.

    :param comments: a list of strings
    :return: a list containing the topic probabilities for each comment, and another list containing topics if it
        trained on words, where each topic is a list of tuples, where each of those tuples are of the form
        (str('word'), float(importance_of_word)), sorted by the importance of each word (most important comes first).
    """
    try:
        return train_lda_pipeline_on_words(comments)
    except:
        return train_lda_pipeline_on_letters(comments)


def train_lda_pipeline_on_words(comments):
    """
    Train an LDA and transform the comments.

    :param comments: a list of strings
        call to this method failed to extract word features from the CountVectorizer.
    :return: a list containing the topic probabilities for each comment, and another list containing topics, where each
        topic is a list of tuples, where each of those tuples are of the form (str('word'), float(importance_of_word)),
        sorted by the importance of each word (most important comes first).
    """
    lda_pipeline = Pipeline([
        ('stopwords', StopWordsRemover()),
        ('stemmer', Stemmer()),
        ('count_vect', CountVectorizer()),
        ('lda', LDA()),
    ]).set_params(**LDA_PIPELINE_PARAMS_WORDS)

    # Fit the data
    transformed_comments = lda_pipeline.fit_transform(comments)
    print("score:", lda_pipeline.score(comments))

    # Extract information about data
    lda = lda_pipeline.named_steps['lda']
    # features = lda_pipeline.named_steps['count_vect'].get_feature_names()
    topic_words = lda_pipeline.inverse_transform(X=None)
    topics = lda.components_
    topic_words_weighting = [list(reversed(sorted(t))) for t in topics]

    return transformed_comments, link_topics_and_weightings(topic_words, topic_words_weighting)


def train_lda_pipeline_on_letters(comments):
    """
    Train an LDA and transform the comments.

    :param comments: a list of strings
    :param use_letters_instead_of_words: boolean, set to True only if a first
        call to this method failed to extract word features from the CountVectorizer.
    :return: a list containing the topic probabilities for each comment, and another list that is empty but that
        would normally contain topics' descriptions.
    """
    lda_pipeline = Pipeline([
        ('stopwords', StopWordsRemover()),
        ('letter_splitter', LetterSplitter()),
        ('count_vect', CountVectorizer()),
        ('lda', LDA()),
    ]).set_params(**LDA_PIPELINE_PARAMS_LETTERS)

    # Fit the data
    transformed_comments = lda_pipeline.fit_transform(comments)
    print("score:", lda_pipeline.score(comments))

    no_info_for_topics = [[]] * LDA_PIPELINE_PARAMS_LETTERS['lda__n_components']
    return transformed_comments, no_info_for_topics
