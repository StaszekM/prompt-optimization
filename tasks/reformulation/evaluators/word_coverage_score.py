import dspy
import spacy

nlp = spacy.load("pl_core_news_lg")


def get_word_coverage_score(
    string_a, string_b, case_sensitive=False
) -> tuple[float, set[str], set[str]]:

    a_doc = nlp(string_a)
    b_doc = nlp(string_b)

    a_words_set = {
        token.lemma_ for token in a_doc if (not token.is_stop) and (not token.is_punct)
    }
    b_words_set = {
        token.lemma_ for token in b_doc if (not token.is_stop) and (not token.is_punct)
    }

    if not case_sensitive:
        a_words_set = set(map(str.lower, a_words_set))
        b_words_set = set(map(str.lower, b_words_set))

    if len(b_words_set) == 0 and len(a_words_set) == 0:
        return 1.0, set(), set()

    numeric_value = len(a_words_set.intersection(b_words_set)) / len(
        a_words_set.union(b_words_set)
    )

    a_diff_b = a_words_set.difference(b_words_set)
    b_diff_a = b_words_set.difference(a_words_set)

    return numeric_value, a_diff_b, b_diff_a


def word_coverage_metric(
    example, prediction, trace=None, pred_name=None, pred_trace=None
) -> dspy.Prediction:
    """Penalizes mismatch between sets of words used by two strings (IoU on sets of word tokens that are not stop words)"""

    coverage_value, a_diff_b, b_diff_a = get_word_coverage_score(
        example.reformulated_message, prediction.reformulated_question
    )

    if len(a_diff_b) == 0 and len(b_diff_a) == 0:
        return dspy.Prediction(
            score=1.0,
            feedback="Brak różnic w występowaniu słów istotnych (pomijając stop-words)",
        )

    feedback = "Wykryto rozbieżności w występowaniu słów istotnych -"

    if len(a_diff_b) > 0:
        feedback += f" słowa brakujące: {', '.join(a_diff_b)} -"

    if len(b_diff_a) > 0:
        feedback += f" słowa nadmiarowe: {', '.join(b_diff_a)} -"

    feedback = feedback.rstrip("-").strip()

    return dspy.Prediction(score=coverage_value, feedback=feedback)
