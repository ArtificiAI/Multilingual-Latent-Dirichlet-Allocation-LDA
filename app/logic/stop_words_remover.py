import string
import unidecode


class StopWordsRemover:
    def __init__(self):
        """
        Get the stop words at creation of the StopWordsRemover to then 
        be able to remove them from strings or nested iterables of strings.

        This stop word remover is built so as to be very gentle with the
        input strings so as to keep their structure as much as possible.
        """
        with open("app/data/custom_FR_EN_stop_words.txt") as f:
            self.stopwords = f.read().split("\n")
        self.safe_stopwords = [unidecode.unidecode(w).lower() for w in self.stopwords]

    def remove_from_many_strings(self, text):
        """
        This function only recurse and then call the single "self.remove_from_string(...)"
        """

        if isinstance(text, str):
            return self.remove_from_string(text)

        else:
            # It should be an iterable, recurse to find text at a lower branch deep down.
            ret_list = []
            for inner in text:
                inner = self.remove_from_many_strings(inner)
                ret_list.append(inner)
            return ret_list

    def remove_from_string(self, text):
        """
        Remove stopwords from a string in the safest possible way to keep the text intact.
        """

        # In the following variables, text's characters will flow from bottom to top such as:
        # text --> last_word|last_punct --> past_text
        past_text = ""
        last_punct = ""
        last_word = ""

        text += "."  # add a last punctuation to loop 1 last time closing the sentence.
        for char in text:
            decoded_char = unidecode.unidecode(char).lower()

            char_is_letter = False
            if decoded_char in string.ascii_lowercase:  # Lowercase alphabet
                char_is_letter = True

            # We loop if it's part of a word.
            if char_is_letter:
                # We're building a word.
                # Loop
                last_word += char

            # Otherwise if it's punctuation, we're either somehow before or directly after a word.
            elif not char_is_letter:

                # We ignore N punctuations in a row before a word.
                if last_word == "":
                    # Move on.
                    last_punct += char
                # Otherwise we're closing a word. Let's process it now.
                else:
                    full_word = last_word
                    safe_full_word = unidecode.unidecode(full_word).lower()
                    if safe_full_word in self.safe_stopwords:
                        # We remove the word (and the following apostrophe or space if there is one)!
                        full_word = ""
                        if char in "’'‘’'' ":
                            char = ""

                    # Loop
                    past_text += last_punct + full_word
                    last_punct = char
                    last_word = ""

        past_text += last_punct
        return past_text[:-1]
