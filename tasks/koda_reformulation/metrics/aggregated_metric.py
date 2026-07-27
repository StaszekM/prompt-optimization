from statistics import harmonic_mean
from typing import Any

import dspy
import pandas as pd
import spacy

from .IDFWeightedTermF1 import (
    ContextCarryoverRecall,
    IDFWeightedTermF1,
)


class AggregatedMetric:
    __name__ = "AggregatedMetric"

    def __init__(self, train_location, val_location):
        nlp = spacy.load("pl_core_news_lg")

        gold_answers_train = pd.read_json(train_location, lines=True)[
            "reformulated question"
        ].to_list()

        gold_answers_val = pd.read_json(val_location, lines=True)[
            "gold_paraphrase"
        ].to_list()

        gold_answers = [*gold_answers_train, *gold_answers_val]

        self.idf_weighted_term_f1 = IDFWeightedTermF1(
            gold_questions_list=gold_answers, nlp=nlp
        )
        self.context_carryover_recall = ContextCarryoverRecall(
            gold_questions_list=gold_answers, nlp=nlp
        )

    def __call__(
        self, example, prediction, trace=None, pred_name=None, pred_trace=None
    ) -> Any:
        f1_score, f1_feedback = self.idf_weighted_term_f1(
            predicted=prediction.reformulated_question,
            expected=example.reformulated_message,
        )

        carryover_score, carryover_feedback = self.context_carryover_recall(
            predicted=prediction.reformulated_question,
            expected=example.reformulated_message,
            last_user_question=example.last_user_message,
        )

        aggregated_score = harmonic_mean([f1_score, carryover_score])

        aggregated_feedback = f"Pokrycie słowami: {f1_feedback}\nPrzenoszenie kontekstu: {carryover_feedback}"

        return dspy.Prediction(score=aggregated_score, feedback=aggregated_feedback)
