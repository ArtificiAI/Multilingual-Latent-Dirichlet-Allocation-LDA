import numpy as np

CATS_DOGS_COMMENTS_IN_FRENCH_NORMAL = [
    "Les chats sont super",  # Cats
    "Un super-chat marche sur le trottoir",  # Cats
    "Les chats sont super ronrons",  # Cats
    "Un super-chien",  # Dogs
    "Deux super-chiens",  # Dogs
    "Combien de chiens?"  # Dogs
]

# FR-EN slang stopwords:
TEST_STOPWORDS = ["le", "les", "la", "un", "de",
                  "a", "b", "c", "s",
                  "est", "sur", "tres", "donc",
                  "the", "is",
                  "ya", "pis", "yer"]

CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS = [
    "chats sont super",
    "super-chat marche trottoir",
    "chats sont super ronrons",
    "super-chien",
    "Deux super-chiens",
    "Combien chiens?"
]

CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED = [
    'chat sont sup',
    'sup chat march trottoir',
    'chat sont sup ronron',
    'sup chien',
    'deux sup chien',
    'combien chien'
]

CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_THEN_UNSTEMMED = [
    ['chats', 'sont', 'super'],
    ['super', 'chats', 'marche', 'trottoir'],
    ['chats', 'sont', 'super', 'ronrons'],
    ['super', 'chiens'],
    ['Deux', 'super', 'chiens'],
    ['Combien', 'chiens']
]

CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_VECTORIZED = np.array([
    [1, 1, 0, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 1, 0],  # Note: here it's random that it's all 0 or 1, but we could have got 2 and more as values.
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 0, 0, 0, 0]
])

CATS_DOGS_LABELS_A = np.array([1, 1, 1, 0, 0, 0])  # Should cluster randomly: it will find one of those A or B labels.
CATS_DOGS_LABELS_B = np.array([0, 0, 0, 1, 1, 1])
CATS_TOP_WORDS_ORDERING = ['super', 'chats']  # There are more times the word "super" than "chat" in the 1st cluster.
DOGS_TOP_WORDS_ORDERING = ['chiens', 'super']  # There are more times the word "chiens" than "super" int he 2nd cluster.

CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_VECTORIZED_LDA_TOPICS_INVERSE_TRANSFORM_1 = [
    np.array([0, 5, 4, 3, 1, 2, 6]),
    np.array([2, 5, 6, 0, 4, 3, 1])
]
CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_VECTORIZED_LDA_TOPICS_INVERSE_TRANSFORM_1_INV = [
    np.array([2, 5, 6, 0, 4, 3, 1]),
    np.array([0, 5, 4, 3, 1, 2, 6])
]

CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_STEMMED_LDA_TOPICS_INVERSE_TRANSFORM_2 = [
    ['chat', 'sup', 'sont sup', 'sont', 'chat sont', 'chien', 'sup chien'],
    ['chien', 'sup', 'sup chien', 'chat', 'sont sup', 'sont', 'chat sont']
]

CATS_DOGS_COMMENTS_IN_FRENCH_WITHOUT_STOPWORDS_LDA_TOPICS_INVERSE_TRANSFORM_3 = [
    ['chats', 'super', 'sont super', 'sont', 'chats sont', 'chiens', 'super chiens'],
    ['chiens', 'super', 'super chiens', 'chats', 'sont super', 'sont', 'chats sont']
]
