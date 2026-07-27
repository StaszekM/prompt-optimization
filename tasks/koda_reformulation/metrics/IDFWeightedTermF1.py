import math
from abc import ABC
from typing import Any

import numpy as np
import pandas as pd
from pandas import Series
from spacy.language import Language
from spacy.ml import Doc
from spacy.tokens import Token


class IDFMetric(ABC):
    def __init__(self, gold_questions_list: list[str], nlp: Language):
        self._nlp = nlp
        self._lemma_weight_lut = self._construct_lemma_weight_lut(gold_questions_list)
        if len(self._lemma_weight_lut) == 0:
            self._oov_weight = 1.0
        else:
            self._oov_weight = float(
                np.percentile(self._lemma_weight_lut.to_numpy(), 95)
            )

    def _construct_lemma_weight_lut(self, gold_questions_list: list[str]) -> Series:
        """Returns Pandas series with index `lemma` (lemma) and value `idf` (IDF for lemma, calculated for entire `gold_questions_list`)"""
        docs = [doc for doc in self._nlp.pipe(gold_questions_list)]

        lemmas_per_doc: list[set[str]] = [self._lemmatize_terms(doc) for doc in docs]
        lemmas_total: set[str] = set().union(*lemmas_per_doc)

        n_docs = len(gold_questions_list)

        lemmas_df = pd.DataFrame(lemmas_total, columns=["lemma"])
        lemmas_df["df"] = lemmas_df["lemma"].apply(
            lambda l: sum(l in lemmas_set for lemmas_set in lemmas_per_doc)
        )
        lemmas_df["idf"] = lemmas_df["df"].apply(
            lambda df: math.log((n_docs + 1) / (df + 1)) + 1
        )

        return lemmas_df.set_index("lemma")["idf"]

    def _lemmatize_terms(self, doc: Doc | str):
        if isinstance(doc, str):
            doc = self._nlp(doc)
        return {token.lemma_ for token in doc if self.is_term(token)}

    def _get_weight(self, lemma: str) -> float:
        """Gets the lemma weight with fallback to OOV weight"""
        if lemma in self._lemma_weight_lut.index:
            return float(self._lemma_weight_lut.loc[lemma])
        return self._oov_weight

    def _sum_weights(self, lemmas: set[str]) -> float:
        return float(sum(self._get_weight(lemma) for lemma in lemmas))

    @staticmethod
    def is_term(token: Token):
        if token.is_punct:
            return False

        if token.is_stop:
            return False

        if token.like_num:
            return False

        return token.pos_ in ["NOUN", "PROPN", "ADJ", "VERB", "ADV"]


class IDFWeightedTermF1(IDFMetric):
    r"""
    Compute an IDF-weighted term-level F1 score between a predicted query and a
    gold query.

    Both texts are converted into sets of content *terms*. The final score is an
    IDF-weighted F1 score computed over exact lemma matches.

    The metric rewards predictions that preserve important retrieval terms from
    the gold query and penalizes predictions that introduce unsupported or
    spurious terms.

    ## Term sets

    For a predicted query and a gold query, define:

    $$A = \operatorname{terms}(\text{predicted})$$

    $$G = \operatorname{terms}(\text{expected})$$

    $$C = A \cap G$$

    Here:

    - $A$ is the set of content lemmas extracted from the model output.
    - $G$ is the set of content lemmas extracted from the gold query.
    - $C$ is the set of exact lemma matches.

    The function $\operatorname{terms}(x)$ removes punctuation, stop words,
    numeric-like tokens, and tokens whose part-of-speech tag is not one of
    `NOUN`, `PROPN`, `ADJ`, `VERB`, or `ADV`.

    ## IDF weights

    Lemma weights are estimated from the complete list of gold queries passed to
    the constructor. Each gold query is treated as a separate document.

    For a lemma $t$, its document frequency is:

    $$\operatorname{df}(t) = \sum_{i=1}^{N} \mathbf{1}[t \in \operatorname{terms}(g_i)]$$

    Its smoothed inverse document frequency is:

    $$w(t) = \log\left(\frac{N + 1}{\operatorname{df}(t) + 1}\right) + 1$$

    Here, $N$ is the number of gold queries and $\mathbf{1}[\cdot]$ is the
    indicator function.

    Lemmas absent from the IDF lookup table are treated as out-of-vocabulary
    terms and assigned a fixed OOV weight:

    $$w_{\mathrm{OOV}} = P_{95}(w(t) \mid t \in V)$$

    Here, $V$ is the vocabulary of content lemmas observed in
    `gold_questions_list`.

    This prevents unseen predicted terms from being ignored when precision is
    calculated. Unsupported or hallucinated terms produced by the model therefore
    still increase the precision denominator.

    ## Score calculation

    The total weight of exact matches is:

    $$W_C = \sum_{t \in C} w(t)$$

    The total predicted-term weight is:

    $$W_A = \sum_{t \in A} w(t)$$

    The total gold-term weight is:

    $$W_G = \sum_{t \in G} w(t)$$

    Weighted precision is:

    $$P = \frac{W_C}{W_A}$$

    Weighted recall is:

    $$R = \frac{W_C}{W_G}$$

    The resulting F1 score is:

    $$F_1 = \frac{2PR}{P + R}$$

    The weight $w(t)$ is read from the IDF lookup table when available. Otherwise,
    it is set to $w_{\mathrm{OOV}}$.

    ## Feedback

    The feedback string reports asymmetric differences between the predicted and
    gold term sets.

    Missing terms are defined as:

    $$\operatorname{missing} = G \setminus A$$

    Spurious terms are defined as:

    $$\operatorname{spurious} = A \setminus G$$

    Missing terms are expected terms from the gold query that are absent from the
    model output.

    Spurious terms are terms produced by the model that are absent from the gold
    query.

    These diagnostics are intended for reflective prompt-optimization loops such
    as GEPA.

    ## Parameters

    - `gold_questions_list`:
      List of gold paraphrases used to construct the IDF lookup table. Each item
      is processed with `nlp` and contributes one document to the
      document-frequency estimate.

    - `nlp`:
      spaCy language pipeline used for tokenization, part-of-speech tagging,
      stop-word detection, and lemmatization.

    ## Returns

    A callable metric object.

    Calling the object with `predicted` and `expected` returns a
    `(score, feedback)` tuple:

    - `score`:
      IDF-weighted exact-lemma F1 score. The value is in the range `[0.0, 1.0]`
      when all IDF weights are non-negative.

    - `feedback`:
      Textual diagnostic message containing missing and spurious lemmas ordered
      by their IDF weights.

    ## Notes

    Predicted OOV terms are intentionally penalized through the precision
    denominator. Unseen terms in the model output may represent unsupported
    additions or hallucinated constraints.
    """

    def __init__(self, gold_questions_list: list[str], nlp: Language):
        super().__init__(gold_questions_list=gold_questions_list, nlp=nlp)

    def __call__(self, predicted: str, expected: str) -> tuple[float, str]:

        predicted_lemmatized = self._lemmatize_terms(predicted)
        expected_lemmatized = self._lemmatize_terms(expected)

        common_lemmas = predicted_lemmatized.intersection(expected_lemmatized)

        common_lemmas_wt_sum = self._sum_weights(common_lemmas)
        predicted_lemmas_wt_sum = self._sum_weights(predicted_lemmatized)
        expected_lemmas_wt_sum = self._sum_weights(expected_lemmatized)

        if predicted_lemmas_wt_sum == 0 and expected_lemmas_wt_sum == 0:
            return 1.0, "Brak rozbieżności w ważnych terminach"

        if predicted_lemmas_wt_sum == 0 or expected_lemmas_wt_sum == 0:
            return 0.0, "Jedna z parafraz nie zawiera żadnych terminów treściowych."

        precision = common_lemmas_wt_sum / predicted_lemmas_wt_sum
        recall = common_lemmas_wt_sum / expected_lemmas_wt_sum

        if precision + recall == 0:
            score = 0.0
        else:
            score = 2 * precision * recall / (precision + recall)

        missing_terms = expected_lemmatized.difference(predicted_lemmatized)
        spurious_terms = predicted_lemmatized.difference(expected_lemmatized)

        feedbacks = []
        if missing_terms:
            feedbacks.append(
                f"Brakujące ważne terminy: {self._format_terms_by_weight(missing_terms)}."
            )
        if spurious_terms:
            feedbacks.append(
                f"Nadmiarowe terminy: {self._format_terms_by_weight(spurious_terms)}."
            )

        if feedbacks:
            feedback_str = "\n".join(feedbacks)
        else:
            feedback_str = "Brak rozbieżności w ważnych terminach"

        return float(score), feedback_str

    def _format_terms_by_weight(self, terms: set[str], limit: int = 10) -> str:
        sorted_terms = sorted(terms, key=self._get_weight, reverse=True)
        limited_terms = sorted_terms[:limit]

        grouped_terms: dict[float, list[str]] = {}

        for term in limited_terms:
            weight = round(self._get_weight(term), 2)
            grouped_terms.setdefault(weight, []).append(term)

        formatted_groups = []
        for weight, group_terms in grouped_terms.items():
            terms_str = ", ".join(sorted(group_terms))
            formatted_groups.append(f"{terms_str} ({weight:.2f})")

        suffix = ""
        if len(sorted_terms) > limit:
            suffix = f" oraz {len(sorted_terms) - limit} więcej"

        return ", ".join(formatted_groups) + suffix


class ContextCarryoverRecall(IDFMetric):
    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)

    def __call__(
        self, predicted: str, expected: str, last_user_question: str
    ) -> tuple[float, str]:
        predicted_lemmatized = self._lemmatize_terms(predicted)
        expected_lemmatized = self._lemmatize_terms(expected)
        last_user_question_lemmatized = self._lemmatize_terms(last_user_question)

        context_terms = expected_lemmatized - last_user_question_lemmatized
        context_terms_idf_sum = self._sum_weights(context_terms)

        matching_terms = predicted_lemmatized.intersection(context_terms)
        matching_terms_idf_sum = self._sum_weights(matching_terms)

        if len(context_terms) == 0 or context_terms_idf_sum == 0.0:
            return (
                1.0,
                "Brak ważnych terminów do przeniesienia między ostatnim pytaniem użytkownika a parafrazą.",
            )

        metric_value = matching_terms_idf_sum / context_terms_idf_sum

        missing_context_terms = context_terms - matching_terms

        if len(missing_context_terms):
            formatted_missing_context_terms = ", ".join(
                f'"{x}"' for x in missing_context_terms
            )
            feedback = f"Parafraza nie dodaje ważnych terminów do ostatniego pytania użytkownika. Brakuje: {formatted_missing_context_terms}"
        else:
            feedback = "Nie brakuje żadnych ważnych terminów, które należy dodać do ostatniego pytania użytkownika"

        return metric_value, feedback
