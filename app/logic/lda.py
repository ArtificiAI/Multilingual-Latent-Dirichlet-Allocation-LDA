# Some code here is inspired from:
#     https://github.com/scikit-learn/scikit-learn/blob/master/examples/applications/plot_topics_extraction_with_nmf_lda.py
#     (Which is available under the following license: BSD 3-Clause)

import math

from sklearn.decomposition import LatentDirichletAllocation


class LDA(LatentDirichletAllocation):

    def inverse_transform(self, documents=None):
        """
        This function is implemented for the class to be usable by scikit-learn's Pipeline() behavior.

        This will return the top features (words ids) for each topic, or returns the conversion of a document.
        The top words returned are selectionned carefully such that the returned list isn't too big.
        """

        some_top_words = []
        if documents is None:
            # We want topics.
            for topic in self.components_:
                all_words_for_topic = topic.argsort()[::-1]  # arguments of the reverse sorting are the top words

                # We just want to keep the most pertinent word features: half of them for near-zero counts,
                # then sqrt of them as we approach 10 words. See:
                #   Formula: y=floor(1.05^-x * (x/2) + (1-(1.05^-x))*(sqrt(x))+0.5)
                #   http://www.wolframalpha.com/input/?i=y%3Dfloor(x%2F2%2B0.5),+y%3Dfloor(1.05%5E-x+*+(x%2F2)+%2B+(1-(1.05%5E-x))*(sqrt(x))%2B0.5),+from+x+%3D+0..25
                x = len(all_words_for_topic)
                exp = 1.05**(-x)  # transition from 1 to 0, fades slowly: half life of exp decay is near x=4.
                y = int(
                    exp * (x / 2) +  # at the beginning, we take half.
                    (1 - exp) * (math.sqrt(x)) +  # after some time `(1 - exp)`, we transition to square root of words.
                    0.5  # the 0.5 makes int() round up like if we dir round(). But let's avoid using `round()`.
                )
                half_or_sqrt_of_words = y

                top_words_for_topic = all_words_for_topic[:half_or_sqrt_of_words]
                some_top_words.append(top_words_for_topic)
        else:
            # we return documents as they are: indexes.
            return documents

        return some_top_words

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
