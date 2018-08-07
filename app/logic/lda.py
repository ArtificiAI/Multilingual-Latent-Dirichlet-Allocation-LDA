# Some code here is inspired from:
#     https://github.com/scikit-learn/scikit-learn/blob/master/examples/applications/plot_topics_extraction_with_nmf_lda.py
#     (Which is available under the following license: BSD 3-Clause)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class LDA:
    HYPERPARAMETERS = dict()

    def __init__(self, n_topics, max_iter=5, learning_method='online', learning_offset=50.):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.learning_method = learning_method
        self.learning_offset = learning_offset

        self._stemmed_documents = []
        self._clusterized_documents = []
        self._topics = []
        self._is_finished = False

        self.max_vocab_size = 10000
        self.ngram_range = (1, 2)  # 1-gram to 2-grams will be used.
        self.tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                             max_features=self.max_vocab_size,
                                             ngram_range=self.ngram_range)

    def fit(self, lemmatized_dataset):
        tf = self.tf_vectorizer.fit_transform(lemmatized_dataset)

        lda = LatentDirichletAllocation(n_components=self.n_topics, max_iter=self.max_iter,
                                        learning_method=self.learning_method,
                                        learning_offset=self.learning_offset,
                                        random_state=0)  # TODO: random state
        lda.fit(tf)

        print("Topics in the LDA model:")

        tf_feature_names = self.tf_vectorizer.get_feature_names()
        self.print_top_words(lda, tf_feature_names, n_top_words=3)

        self._is_finished = True

        return lda, tf_feature_names

    def print_top_words(self, model, feature_names, n_top_words):
        for i, topic in enumerate(model.components_):
            top_topics = topic.argsort()[:-n_top_words - 1:-1]
            escape = lambda x: "'" + x + "'"
            print("topic #{}:".format(i), " ".join([escape(feature_names[i]) for i in top_topics]))


    def is_finished(self):
        return self._is_finished

    def get_topics(self, document):
        return self._clusterized_documents[document]

    def get_top_expressions_for_topic(self, topic):
        return self._topics[topic]
