# Some code here is inspired from:
#     https://github.com/scikit-learn/scikit-learn/blob/master/examples/applications/plot_topics_extraction_with_nmf_lda.py
#     (Which is available under the following license: BSD 3-Clause)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class LDA:
    HYPERPARAMETERS = dict()

    def __init__(self, n_topics, **sklearn_params):
        """
        Create an LDA algorithm. Some hyperparameters in arguments are adjustable.
        For more info on the hyperparameters, visit:
        http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
        """
        self.n_topics = n_topics
        self.sklearn_params = sklearn_params

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
        """
        Fit the LDA on a dataset. The dataset is a list of documents. A document is a string.
        It's recommended that the documents be lemmatized and without stop words.
        """
        tf_features = self.tf_vectorizer.fit_transform(lemmatized_dataset)

        self.lda_sklearn = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=0,  # TODO: random state
            **self.sklearn_params)

        # print("Topics in the LDA model:")
        self.tf_feature_names = self.tf_vectorizer.get_feature_names()
        self.lda_sklearn.fit(tf_features)
        # self.print_top_words(lda, tf_feature_names, n_top_words=3)

        self._is_finished = True

        return self.lda_sklearn, self.tf_feature_names

    def transform(self, dataset):
        tf_features = self.tf_vectorizer.transform(dataset)
        self.lda_sklearn.transform(tf_features)

    def score(self, dataset):
        tf_features = self.tf_vectorizer.transform(dataset)
        return self.lda_sklearn.score(tf_features)

    def perplexity(self, dataset):
        tf_features = self.tf_vectorizer.transform(dataset)
        return self.lda_sklearn.perplexity(tf_features)

    def is_finished(self):
        """
        TODO: May parallelize the class (async calls).
        """
        return self._is_finished

    def get_topics(self, document):
        """
        Get the top topics for a document.
        """
        return self._clusterized_documents[document]

    def print_top_words(self, model, feature_names, n_top_words):
        """
        Get the best N words that represents each topic.
        """
        for i, topic in enumerate(model.components_):
            top_topics = topic.argsort()[:-n_top_words - 1:-1]
            escape = lambda x: "'" + x + "'"
            print("topic #{}:".format(i), " ".join([escape(feature_names[i]) for i in top_topics]))

    def get_top_expressions_for_topic(self, topic):
        """
        TODO
        """
        return self._topics[topic]
