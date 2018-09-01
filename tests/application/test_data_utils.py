from app.application.data_utils import link_topics_and_weightings


def test_link_topics_and_weightings_works():
    # prepare words and their weight
    topic_1_words = ['chats', 'chiens', 'ours']
    topic_3_words = ['ours', 'chats', 'chiens']
    topic_2_words = ['chiens', 'chats', 'ours']
    topic_1_weightings_for_words = [2.0, 0.75, 0.25]
    topic_2_weightings_for_words = [1.0, 0.5, 0.25]
    topic_3_weightings_for_words = [1.0, 0.15, 0.05]
    # inputs
    words = [topic_1_words, topic_2_words, topic_3_words]
    topics = [topic_1_weightings_for_words, topic_2_weightings_for_words, topic_3_weightings_for_words]
    # output
    expected_words_and_topics_linked = [
        # Note that `list(zip(...))` changes 2 lists into 1 list of pairs:
        list(zip(topic_1_words, topic_1_weightings_for_words)),
        list(zip(topic_2_words, topic_2_weightings_for_words)),
        list(zip(topic_3_words, topic_3_weightings_for_words))
    ]

    obtained_words_and_topics_linked = link_topics_and_weightings(words, topics)

    print("")
    print(expected_words_and_topics_linked)
    print(obtained_words_and_topics_linked)

    assert expected_words_and_topics_linked == obtained_words_and_topics_linked