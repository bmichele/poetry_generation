import random
import re
from typing import List


def remove_punct(text: str) -> str:
    out = re.sub("[^\w\s]", " ", text)
    return " ".join(out.split())


def tokenize(text: str) -> List[str]:
    return text.split()


def filter_candidates(
    candidates: List[str], remove_duplicates: bool = True, n_max: int = None
):
    out = [
        candidate for candidate in candidates if remove_punct(candidate)
    ]  # make sure that lines containing only punctuation are excluded
    if remove_duplicates:
        out = list(set(out))
    if n_max > 0:
        out = random.sample(out, n_max)
    return out
