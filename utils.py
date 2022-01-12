import re
from typing import List


def remove_punct(text: str) -> str:
    out = re.sub("[^\w\s]", " ", text)
    return " ".join(out.split())


def tokenize(text: str) -> List[str]:
    return text.split()
