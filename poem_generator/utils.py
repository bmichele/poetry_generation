import re
from typing import List


def remove_punct(text: str) -> str:
    out = re.sub("[^\w\s]", " ", text)
    return " ".join(out.split())


def tokenize(text: str) -> List[str]:
    return text.split()


def filter_candidates(candidates: List[str]):
    out = [
        candidate for candidate in candidates if remove_punct(candidate)
    ]  # make sure that lines containing only punctuation are excluded
    return list(set(out))
