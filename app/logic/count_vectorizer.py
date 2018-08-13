from sklearn.feature_extraction.text import CountVectorizer as CV


class CountVectorizer(CV):

    def inverse_transform(self, X):
        """
        Note: this method overrides the original one to retain the order of the features passed as argument.

        Return a list of words for each document, keeping the order of the transformed words indexes.
        """

        all_undid = []  # Let's undo that.
        for doc in X:
            undid_doc = [self.get_feature_names()[i] for i in doc]
            all_undid.append(undid_doc)
        return all_undid

        self._check_vocabulary()
