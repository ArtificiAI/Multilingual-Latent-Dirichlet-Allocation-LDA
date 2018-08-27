
def topics_and_weightings_to_associated_dict(topic_words, topic_words_weighting):
    """
    Pair every words with their weightings for topics into dicts, for each topic.

    :param topic_words: a 2D array of shape [topics, top_words]
    :param topic_words_weighting: a 2D array of shape [topics, top_words_weightings]
    :return: A list containing dicts of shape [topics, dict({top_word: top_words_weighting} for each word)]
    """

    topics_and_words = [
        [
            {word: weightings}
            for word, weightings in zip(word_list, weightings_list)
        ]
        # TODO: limit the number of words returned to half or a maximum:
        for word_list, weightings_list in zip(topic_words, topic_words_weighting)
    ]
    return topics_and_words