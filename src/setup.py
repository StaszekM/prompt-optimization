from typing import Literal

import dotenv
import dspy
import wikipediaapi

from src.clarin_lm import create_clarin_lm

dotenv.load_dotenv()

CLARIN_MODEL_NAME = "gpt-4o"
lm = create_clarin_lm(model_name=CLARIN_MODEL_NAME)

dspy.configure(lm=lm)


Season = Literal[
    "spring",
    "summer",
    "autumn",
    "winter",
]


class HaikuBot(dspy.Signature):
    """
    Write a classical haiku given the provided inputs.
    """

    location: str = dspy.InputField(desc="The setting of the poem")
    mood: str = dspy.InputField()
    haiku: str = dspy.OutputField()
    season: Season = dspy.InputField()


# haiku_bot = dspy.ChainOfThought(HaikuBot)
# result = haiku_bot(location="a quiet library", mood="mysterious", season="winter")
# print(result.haiku)
# print(result.reasoning)
# print(dspy.inspect_history(n=1))


def wikipedia_search(query: str) -> list[str]:
    """Search Wikipedia for the given query and return a list of page titles."""
    wiki = wikipediaapi.Wikipedia(
        user_agent="MyProjectName (merlin@example.com)", language="en"
    )

    search_results = wiki.search(query=query)
    return [p.title for p in search_results.pages.values()]


def get_wikipedia_page(title: str) -> str:
    """Get the content of a Wikipedia page given its title."""
    wiki = wikipediaapi.Wikipedia(
        user_agent="MyProjectName (merlin@example.com)",
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )

    p_wiki = wiki.page(title)

    return p_wiki.text


# haiku_bot = dspy.ReAct(HaikuBot, tools=[wikipedia_search, get_wikipedia_page])
# result = haiku_bot(location="Camp Meeker", mood="pensive", season="summer")
# print(result.haiku)
# print(result.reasoning)
# print(dspy.inspect_history(n=1))


class HaikuEnsemble(dspy.Module):
    def __init__(self, n: int = 3):
        super().__init__()
        self.n = n
        # Module 1 generates several haikus
        self.writer = dspy.ReAct(
            "location, season, mood, num_haikus: int -> haikus: list[str]",
            tools=[wikipedia_search, get_wikipedia_page],
            max_iters=5,
        )
        # Module 2 picks the most evocative
        self.judge = dspy.ChainOfThought(
            "location, season, mood, candidates: list[str] -> most_evocative_index: int"
        )

    def forward(self, location: str, season: str, mood: str) -> dspy.Prediction:
        candidates = self.writer(
            location=location,
            season=season,
            mood=mood,
            num_haikus=self.n,
        ).haikus
        verdict = self.judge(
            location=location,
            season=season,
            mood=mood,
            candidates=candidates,
        )
        return dspy.Prediction(
            haiku=candidates[verdict.most_evocative_index],
            candidates=candidates,
            reasoning=verdict.reasoning,
        )


ensemble = HaikuEnsemble(n=5)
result = ensemble(location="Bodega Bay", season="autumn", mood="inspired")
print(result.haiku)
print(result.reasoning)
print(result.candidates)
