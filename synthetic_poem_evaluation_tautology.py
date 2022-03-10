import json
import logging
import os
from typing import Set, List
import string

import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize

logger = logging.getLogger("asdasdsad")
logger.setLevel(level=logging.INFO)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter("%(asctime)s - %(message)s")
fh.setFormatter(fh_formatter)
logger.addHandler(fh)


DATA_DIR = "/Users/miche/uniData/poetryGeneration/generated_poems/"


def get_word_set(line: str) -> Set[str]:
    line = line.translate(str.maketrans("", "", string.punctuation)).strip()
    line_words = set([token.lower() for token in word_tokenize(line)])
    return line_words


if __name__ == "__main__":

    results = {}

    training_steps = 75000

    generated_poem_files = [
        "mbart_en_gut_keywords_{}.jsonl".format(training_steps),
        "mbart_en_gut_rhyme_{}.jsonl".format(training_steps),
        "mbart_en_gut_mul_lines_{}.jsonl".format(training_steps),
        "mbart_en_gut_next_line_{}.jsonl".format(training_steps),
        "mbart_en_gut_mixed_lines_{}.jsonl".format(training_steps),
    ]

    # model_flavour = "next_line"
    #
    # generated_poem_files = [
    #     "mbart_en_gut_{}_75000.jsonl".format(model_flavour),
    #     # "mbart_en_gut_{}_60000.jsonl".format(model_flavour),
    #     "mbart_en_gut_{}_45000.jsonl".format(model_flavour),
    #     "mbart_en_gut_{}_30000.jsonl".format(model_flavour),
    #     "mbart_en_gut_{}_15000.jsonl".format(model_flavour),
    #
    # ]

    for generated_poem_file in generated_poem_files:
        logger.info("Processing {}".format(generated_poem_file))

        poem_jsonl_file = os.path.join(DATA_DIR, generated_poem_file)

        with open(poem_jsonl_file, "r") as f:
            data = [json.loads(line) for line in f.readlines()]

        # data = data[:100]

        tautology_scores = []
        candidate_similarity_scores = []

        for generation_outcome in tqdm(data):

            candidates = generation_outcome["candidates"]
            last_poem_line = generation_outcome["poem_state"][-1]
            last_line_words = get_word_set(last_poem_line)
            for i, candidate in enumerate(candidates):
                logger.debug(candidate)
                candidate_words = get_word_set(candidate)
                logger.debug(candidate_words)
                numerator = 2 * len(last_line_words.intersection(candidate_words))
                denominator = len(last_poem_line) + len(candidate_words)
                if denominator:
                    similarity = numerator / denominator
                    tautology_scores.append(similarity)

                # compute diversity
                for other_candidate in candidates[min(i + 1, len(candidates)) :]:
                    other_candidate_words = get_word_set(other_candidate)
                    numerator = 2 * len(
                        candidate_words.intersection(other_candidate_words)
                    )
                    denominator = len(candidate_words) + len(other_candidate_words)
                    if denominator:
                        candidate_similarity = numerator / denominator
                        candidate_similarity_scores.append(candidate_similarity)

        logger.info(
            "{}: Tautology {} (std {}) - Cand Diversity {} (std {})".format(
                generated_poem_file,
                np.mean(tautology_scores),
                np.std(tautology_scores),
                1 - np.mean(candidate_similarity_scores),
                np.std(candidate_similarity_scores),
            )
        )
