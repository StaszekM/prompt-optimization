import spacy

nlp = spacy.load("pl_core_news_lg")


def get_word_coverage_score(string_a, string_b) -> float:

    a_doc = nlp(string_a)
    b_doc = nlp(string_b)

    a_words_set = {
        token.lemma_ for token in a_doc if (not token.is_stop) and (not token.is_punct)
    }
    b_words_set = {
        token.lemma_ for token in b_doc if (not token.is_stop) and (not token.is_punct)
    }

    if len(b_words_set) == 0 and len(a_words_set) == 0:
        return 1.0

    return len(a_words_set.intersection(b_words_set)) / len(
        a_words_set.union(b_words_set)
    )


def word_coverage_metric(example, prediction) -> float:
    """Penalizes mismatch between sets of words used by two strings (IoU on sets of word tokens that are not stop words)"""

    return get_word_coverage_score(
        example.reformulated_message, prediction.reformulated_question
    )
