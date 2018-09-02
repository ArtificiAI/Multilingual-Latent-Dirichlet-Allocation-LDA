def link_topics_and_weightings(topic_words, topic_words_weighting):
    """
    Pair every words with their weightings for topics into dicts, for each topic.

    :param topic_words: a 2D array of shape [topics, top_words]
    :param topic_words_weighting: a 2D array of shape [topics, top_words_weightings]
    :return: A list containing dicts of shape [topics, dict({top_word: top_words_weighting} for each word)]
    """

    topics_and_words = [
        [
            (word, weightings)
            for word, weightings in zip(word_list, weightings_list)
        ]
        # TODO: limit the number of words returned to half or a maximum:
        for word_list, weightings_list in zip(topic_words, topic_words_weighting)
    ]
    return topics_and_words


def get_params_from_prefix_dict(param_prefix, lda_pipeline_params):
    """
    Strip away the param_prefix from the lda_pipeline_params' keys.

    :param param_prefix: string such as 'lda__' or 'stemmer__'.
    :param lda_pipeline_params: dict such as {'lda__learning_decay': 0.5, 'stemmer__language': 'french',}
    :return: the lda_pipeline_params with only the keys from the prefix, such as for example:
        {'learning_decay': 0.5} is returned from the example if the param_prefix was set to 'lda__'.
    """
    count_vectorizer_params = {
        param[len(param_prefix):]: val
        for (param, val) in lda_pipeline_params.items()
        if param[:len(param_prefix)] == param_prefix
    }
    return count_vectorizer_params
