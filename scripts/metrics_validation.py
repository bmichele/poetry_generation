########################################################################################################################
# Semantic Coherence Measure Validation (see quality_estimation.coherence_estimator.SemanticCoherenceEstimator)
########################################################################################################################

import logging
import os
import random
from typing import Optional

import numpy as np
import spacy
from gensim import models
from tqdm import tqdm

from quality_estimation.coherence_estimator import (
    SemanticCoherenceEstimator,
    SyntacticAnnotator,
)

logging.basicConfig(level=logging.INFO)

########################################################################################################################
# Global Variables
########################################################################################################################

DATADIR = os.path.join(os.environ["DATADIR_UNI"], "kaggle_poemsdataset/forms")


########################################################################################################################
# Helper Functions
########################################################################################################################


def clean_poem(poem: str) -> str:
    poem_lines = [line.strip() for line in poem.split("\n")]
    poem_lines = [line for line in poem_lines if line]
    filtered_lines = []
    for line in poem_lines:
        logging.debug("#{}#".format(line))
        if (line[0] in ["(", "["]) or (line[-1] in [")", "]"]) or ("©" in line):
            logging.debug("Removing poem line: {}".format(line))
        else:
            filtered_lines.append(line)
    return "\n".join(filtered_lines)


def shuffle_lines(text: str, separator: Optional[str] = "\n") -> str:
    """
    Simple function to shuffle lines in a text.
    :param text:
    :param separator:
    :return:
    """
    lines = text.split(separator)
    random.shuffle(lines)
    return "\n".join(lines)


########################################################################################################################
# Main
########################################################################################################################

if __name__ == "__main__":
    ####################################################################################################################
    # Look into data
    # Dataset downloaded from kaggle: https://www.kaggle.com/michaelarman/poemsdataset
    # I am using this dataset instead of Gutenberg to avoid validating the metrics with data used for training the
    # models...
    ####################################################################################################################

    # load the data
    poems = []
    for category in [
        folder for folder in os.listdir(os.path.join(DATADIR)) if folder != ".DS_Store"
    ]:
        for file in os.listdir(os.path.join(DATADIR, category)):
            file_path = os.path.join(DATADIR, category, file)
            with open(file_path, "r") as f:
                poem = f.read()
            if poem:
                poems.append(poem)

    poems = [clean_poem(poem) for poem in poems]

    random.seed(42)
    # uncomment to reduce the number of poems
    # keep = [random.random() > 0.9 for _ in range(len(poems))]
    # poems = [poem for i, poem in enumerate(poems) if keep[i]]

    # load word2vec model
    model_dir = "/Users/miche/uniModels"
    w2v = models.KeyedVectors.load_word2vec_format(
        os.path.join(model_dir, "GoogleNews-vectors-negative300.bin"), binary=True
    )

    # load spacy model for annotating
    nlp = spacy.load("en_core_web_sm")
    annotator = SyntacticAnnotator(nlp)

    # initialize estimator
    estimator = SemanticCoherenceEstimator(w2v, annotator)

    fast_tokenize = True
    keep_stopwords = True
    remove_punct = False

    # get statistics about coherence scores for real poems
    scores = []
    for poem in tqdm(poems):
        poem_lines = poem.split("\n")
        for i in range(len(poem_lines) - 1):
            line_to_line_coherence = estimator.predict(
                poem_lines[i],
                poem_lines[i + 1],
                stopwords=keep_stopwords,
                fast_tokenize=fast_tokenize,
                remove_punct=remove_punct,
            )
            if line_to_line_coherence != float("inf"):
                scores.append(line_to_line_coherence)
    average_score = np.mean(scores)
    variance = np.var(scores)
    logging.info(
        "test set: average score {} {} ({} line pairs)".format(
            average_score, variance, len(scores)
        )
    )

    # shuffle the poems and compute average coherence
    shuffled_poems = [shuffle_lines(poem) for poem in poems]
    scores = []
    for poem in tqdm(shuffled_poems):
        poem_lines = poem.split("\n")
        for i in range(len(poem_lines) - 1):
            line_to_line_coherence = estimator.predict(
                poem_lines[i],
                poem_lines[i + 1],
                stopwords=keep_stopwords,
                fast_tokenize=fast_tokenize,
                remove_punct=remove_punct,
            )
            if line_to_line_coherence != float("inf"):
                scores.append(line_to_line_coherence)
    average_score = np.mean(scores)
    variance = np.var(scores)
    logging.info(
        "shuffled poem set: average score {} {} ({} line pairs)".format(
            average_score, variance, len(scores)
        )
    )

    # shuffle poems by mixing lines from different poems
    poem_lengths = [len(poem.split("\n")) for poem in poems]
    all_lines = [line for poem in poems for line in poem.split("\n")]
    random.shuffle(all_lines)
    mixed_poems = []
    for length in poem_lengths:
        mixed_poem_lines = [all_lines.pop(0) for _ in range(length)]
        mixed_poems.append("\n".join(mixed_poem_lines))

    scores = []
    for poem in tqdm(mixed_poems):
        poem_lines = poem.split("\n")
        for i in range(len(poem_lines) - 1):
            line_to_line_coherence = estimator.predict(
                poem_lines[i],
                poem_lines[i + 1],
                stopwords=keep_stopwords,
                fast_tokenize=fast_tokenize,
                remove_punct=remove_punct,
            )
            if line_to_line_coherence != float("inf"):
                scores.append(line_to_line_coherence)
    average_score = np.mean(scores)
    variance = np.var(scores)
    logging.info(
        "shuffled poem set: average score {} {} ({} line pairs)".format(
            average_score, variance, len(scores)
        )
    )
