# Some code here is inspired from:
#     https://github.com/scikit-learn/scikit-learn/blob/master/examples/applications/plot_topics_extraction_with_nmf_lda.py
#     (Which is available under the following license: BSD 3-Clause)


from sklearn.decomposition import LatentDirichletAllocation


class LDA(LatentDirichletAllocation):

    def score(self, X, y=None):
        """
        We make the score positive by inverting its sign here since the GridSearchCV seems to maximize it...
        """
        # TODO: review and fix this behavior.
        return LatentDirichletAllocation.score(self, X, y)

    def inverse_transform(self, documents=None):
        """
        This function is implemented for the class to be usable by scikit-learn's Pipeline() behavior.

        This will return the top features (words ids) for each topic, or returns the conversion of a document.
        """

        all_top_words = []
        if documents is None:
            # We want topics.
            for topic in self.components_:
                top_words_for_topic = topic.argsort()[::-1]  # arguments of the reverse sorting are the top words
                all_top_words.append(top_words_for_topic)
        else:
            # we return documents as they are: indexes.
            return documents

        return all_top_words

    def print_top_words(self, feature_names, n_top_words=None):
        """
        Get the best N words that represents each topic.

        feature_names: features-to-word 1D list or array such as `count_vectorizer.get_feature_names()`
        n_top_words: number of words. If None, will use half of feature_name's length.
        """

        if n_top_words is None:
            n_top_words = int((len(feature_names) + 0.5) / 2)

        for i, topic in enumerate(self.components_):
            top_topics = topic.argsort()[:-n_top_words - 1:-1]
            escape = lambda x: "'" + x + "'"
            print("topic #{}:".format(i), " ".join([escape(feature_names[i]) for i in top_topics]))

