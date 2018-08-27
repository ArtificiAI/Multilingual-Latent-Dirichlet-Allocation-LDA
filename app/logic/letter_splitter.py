from sklearn.base import TransformerMixin


class LetterSplitter(TransformerMixin):

    def __init__(self):
        pass

    def get_params(self, deep=True):
        """
        This function is implemented for the class to be usable by scikit-learn's Pipeline() behavior.
        """
        return dict()

    def set_params(self, **parameters):
        """
        This function is implemented for the class to be usable by scikit-learn's Pipeline() behavior.
        """
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self

    def fit(self, X=None, y=None):
        """
        This function is implemented for the class to be usable by scikit-learn's Pipeline() behavior.
        X & y are ignored here, but required by convention.
        """
        return self

    def transform(self, text):
        """
        This function is implemented for the class to be usable by scikit-learn's Pipeline() behavior.

        This function only recurse and then call the single "self.remove_from_string(...)".
        This means that this version is a vectorized version of "self.remove_from_string(...)"
        """
        return [self.ngrams(example) for example in text]

    def ngrams(self, strin):
        a = []
        b = []
        c = []

        r1 = ''
        r2 = ''
        for r in strin:
            aa = r
            bb = r1 + aa
            cc = r2 + bb

            a.append(aa)
            b.append(bb)
            c.append(cc)

            r2 = r1
            r1 = r
            # r = continue

        c = c[2:]

        # result = " xyz".join(a) + " xyz".join(b) + " xyz".join(c)
        result = " xyz".join(b) + " xyz".join(c * 3)
        result = result.replace("  ", " ")
        return result

    def inverse_transform(self, text):
        return ["".join(example).replace(" xyz", "") for example in text]
