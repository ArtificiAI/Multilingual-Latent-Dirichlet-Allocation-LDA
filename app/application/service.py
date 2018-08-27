from app.application.data_utils import topics_and_weightings_to_associated_dict
from app.logic.letter_splitter import LetterSplitter
from app.logic.stop_words_remover import StopWordsRemover
from app.logic.stemmer import Stemmer
from app.logic.lda import LDA
from app.logic.count_vectorizer import CountVectorizer

from sklearn.pipeline import Pipeline

lda_pipeline_params = {
    'stemmer__language': 'french',
    'count_vect__max_df': 0.98,
    'count_vect__min_df': 2,
    'count_vect__max_features': 10000,
    'count_vect__ngram_range': (1, 2),
    'count_vect__strip_accents': None,
    'lda__n_components': 2,
    'lda__max_iter': 1000,  # TODO: find good balance here. Lower?
    'lda__learning_decay': 0.5,
    'lda__learning_method': 'online',
    'lda__learning_offset': 10,
    'lda__batch_size': 25,
    # 'lda__n_jobs': -1,  # Use all CPUs
}


def train_lda_pipeline_failsafe_fallback_with_ngrams_on_letters(comments):
    try:
        return train_default_lda_pipeline(comments)
    except:
        return train_default_lda_pipeline(comments, use_letters_instead_of_words=True)


def train_default_lda_pipeline(comments, use_letters_instead_of_words=False):
    if use_letters_instead_of_words:
        lda_pipeline = Pipeline([
            ('stopwords', StopWordsRemover()),
            ('stemmer', Stemmer()),
            ('letter_splitter', LetterSplitter()),
            ('count_vect', CountVectorizer()),
            ('lda', LDA()),
        ])
    else:
        # Don't use the letter splitter here.
        lda_pipeline = Pipeline([
            ('stopwords', StopWordsRemover()),
            ('stemmer', Stemmer()),
            ('count_vect', CountVectorizer()),
            ('lda', LDA()),
        ])

    lda_pipeline.set_params(**lda_pipeline_params)

    # Fit the data
    transformed_comments = lda_pipeline.fit_transform(comments)
    print("score:", lda_pipeline.score(comments))

    # Extract information about data
    lda = lda_pipeline.named_steps['lda']
    # features = lda_pipeline.named_steps['count_vect'].get_feature_names()
    topic_words = lda_pipeline.inverse_transform(X=None)
    topics = lda.components_
    topic_words_weighting = [list(reversed(sorted(t))) for t in topics]

    return transformed_comments, topics_and_weightings_to_associated_dict(topic_words, topic_words_weighting)
