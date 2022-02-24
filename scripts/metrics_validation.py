########################################################################################################################
# Semantic Coherence Measure Validation (see quality_estimation.coherence_estimator.SemanticCoherenceEstimator)
########################################################################################################################

import logging
import os
import random
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

import numpy as np
import spacy
from gensim import models
from tqdm import tqdm

from quality_estimation.coherence_estimator import (
    SemanticCoherenceEstimator,
    SyntacticAnnotator,
)

import warnings

warnings.simplefilter("error")

logger = logging.getLogger("metric_validation")
logger.setLevel(level=logging.INFO)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter("%(asctime)s - %(message)s")
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

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
        logger.debug("#{}#".format(line))
        if (line[0] in ["(", "["]) or (line[-1] in [")", "]"]) or ("©" in line):
            logger.debug("Removing poem line: {}".format(line))
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


def n_semantic_coherence(
    context: List[str],
    line: str,
    n: int,
    coherence_estimator: SemanticCoherenceEstimator,
    *args,
    **kwargs,
) -> float:

    if n == 1:
        return coherence_estimator.predict(
            context[-1],
            line,
            *args,
            **kwargs,
        )
    else:
        raise NotImplementedError


def poem_coherence_scores(
    poem: List[str],
    coherence_estimator: SemanticCoherenceEstimator,
    max_n: int,
    *args,
    **kwargs,
) -> np.array:
    """
    Given a poem, computes all n-coherence scores up to the max n value given as argument
    :param poem:
    :param coherence_estimator:
    :param max_n:
    :param args:
    :param kwargs:
    :return:
    """
    # convert all poem lines into vectors
    lines_as_word_seq = [
        coherence_estimator._line_words(line, *args, **kwargs) for line in poem
    ]
    line_vectors = []
    for word_seq in lines_as_word_seq:
        line_vector = np.array(
            [float("nan")] * coherence_estimator.w2v_model.vector_size
        )
        if word_seq:
            word_vectors = coherence_estimator._words_to_vectors(word_seq)
            if word_vectors:
                word_vectors = np.array(word_vectors)
                logger.debug("computing mean of line_vectors (1)")
                line_vector = word_vectors.mean(axis=0)
        line_vectors.append(line_vector)
    if line_vectors:
        line_vectors = np.array(line_vectors)  # get rid of nans
        # line_vectors = np.array(line_vectors)  # get rid of nans
        # print(line_vectors.shape)
        out = np.ones(
            (line_vectors.shape[0] - 1, min(max_n, line_vectors.shape[0] - 1))
        ) * float("nan")
        for i in range(line_vectors.shape[0]):
            for order in range(1, min(max_n, i) + 1):
                # print(i, order)
                logger.debug("computing mean of line_vectors (2)")
                context_vector = line_vectors[i - order : i].mean(axis=0)
                # print(line_vectors[i - order: i].shape)
                line_vector = line_vectors[i]
                # print(line_vector.shape)
                similarity = coherence_estimator._vector_similarity(
                    context_vector, line_vector
                )
                # print(similarity)
                out[i - 1, order - 1] = similarity
        return out
    else:
        return float("nan")


########################################################################################################################
# Main
########################################################################################################################

if __name__ == "__main__":
    # set random seed
    random.seed(42)

    ####################################################################################################################
    # Look into data
    # Dataset downloaded from kaggle: https://www.kaggle.com/michaelarman/poemsdataset
    # I am using this dataset instead of Gutenberg to avoid validating the metrics with data used for training the
    # models...
    ####################################################################################################################

    # load the data
    logger.info("Loading poems")
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

    # # uncomment to reduce the number of poems
    # logger.info("Dropping some poems for testing")
    # keep = [random.random() > 0.99 for _ in range(len(poems))]
    # poems = [poem for i, poem in enumerate(poems) if keep[i]]

    # create three datasets:
    #  * real poems as they are
    #  * poems with shuffled lines
    #  * poems obtained by mixing lines across poems

    logger.info("Creating validation data")
    poems = [clean_poem(poem) for poem in poems]

    poems_shuffled = poems.copy()
    poems_shuffled = [shuffle_lines(poem) for poem in poems_shuffled]

    poems_mixed = []
    poem_lengths = [len(poem.split("\n")) for poem in poems]
    all_lines = [line for poem in poems for line in poem.split("\n")]
    random.shuffle(all_lines)
    for length in poem_lengths:
        mixed_poem_lines = [all_lines.pop(0) for _ in range(length)]
        poems_mixed.append("\n".join(mixed_poem_lines))

    # load word2vec model
    logger.info("Loading word2vec model")
    model_dir = "/Users/miche/uniModels"
    w2v = models.KeyedVectors.load_word2vec_format(
        os.path.join(model_dir, "GoogleNews-vectors-negative300.bin"), binary=True
    )

    # load spacy model for annotation
    logger.info("Loading spacy model")
    nlp = spacy.load("en_core_web_sm")
    annotator = SyntacticAnnotator(nlp)

    # initialize estimator
    logger.info("Initialize coherence estimator")
    estimator = SemanticCoherenceEstimator(w2v, annotator)

    fast_tokenize = False
    keep_stopwords = False
    remove_punct = True

    max_context_lines = 10

    # get statistics about coherence scores for the three datasets of poems
    validation_datasets = {
        "real_poems": poems,
        "shuffled_poems": poems_shuffled,
        "mixed_poems": poems_mixed,
    }
    results = {}
    for poem_dataset_name, poem_dataset in validation_datasets.items():
        logger.info("Processing {} dataset".format(poem_dataset_name))
        scores = [np.array([])] * max_context_lines  # store here all the values for each order
        score_sums = np.zeros(max_context_lines)
        score_counts = np.zeros(max_context_lines)
        for poem in tqdm(poem_dataset):
            poem_lines = poem.split("\n")
            poem_scores = poem_coherence_scores(
                poem_lines,
                estimator,
                max_n=max_context_lines,
                stopwords=keep_stopwords,
                fast_tokenize=fast_tokenize,
                remove_punct=remove_punct,
            )
            for i in range(poem_scores.shape[1]):
                poem_order_scores = poem_scores[:, i]
                poem_order_scores = poem_order_scores[~np.isnan(poem_order_scores)]
                scores[i] = np.concatenate((scores[i], poem_order_scores))
                score_sums[i] += poem_order_scores.sum()
                score_counts[i] += poem_order_scores.size
            # for i in range(1, len(poem_lines)):
            #     line_to_line_coherence = n_semantic_coherence(
            #         poem_lines[:i],
            #         poem_lines[i],
            #         1,
            #         estimator,
            #         stopwords=keep_stopwords,
            #         fast_tokenize=fast_tokenize,
            #         remove_punct=remove_punct,
            #     )
            #     if line_to_line_coherence != float("inf"):
            #         scores.append(line_to_line_coherence)
        average_scores = score_sums / score_counts
        st_deviations = np.array([np.std(order_scores) for order_scores in scores])
        logger.info(
            "{} dataset: average score {}, std {} (over {} samples)".format(
                poem_dataset_name, average_scores, st_deviations, score_counts
            )
        )
        results[poem_dataset_name] = {
            "mean_n_coherence_values": average_scores,
            "example_counts": score_counts,
            "std": np.array([np.std(order_scores) for order_scores in scores])
        }

    # final_results = {
    #     "real_poems": {
    #         "mean_n_coherence_values": np.array(
    #             [
    #                 0.26981735,
    #                 0.32609856,
    #                 0.35921341,
    #                 0.38087973,
    #                 0.39456287,
    #                 0.40658103,
    #                 0.41404798,
    #                 0.42128864,
    #                 0.4258188,
    #                 0.43042484,
    #             ]
    #         ),
    #         "example_counts": np.array(
    #             [
    #                 151332.0,
    #                 142537.0,
    #                 134325.0,
    #                 126756.0,
    #                 119936.0,
    #                 113736.0,
    #                 108125.0,
    #                 102768.0,
    #                 98093.0,
    #                 93714.0,
    #             ]
    #         ),
    #     },
    #     "shuffled_poems": {
    #         "mean_n_coherence_values": np.array(
    #             [
    #                 0.24229989,
    #                 0.30308653,
    #                 0.33816893,
    #                 0.36161014,
    #                 0.37825468,
    #                 0.3905279,
    #                 0.39998995,
    #                 0.40761665,
    #                 0.41384682,
    #                 0.41889239,
    #             ]
    #         ),
    #         "example_counts": np.array(
    #             [
    #                 151063.0,
    #                 141912.0,
    #                 133423.0,
    #                 125700.0,
    #                 118767.0,
    #                 112492.0,
    #                 106784.0,
    #                 101440.0,
    #                 96660.0,
    #                 92220.0,
    #             ]
    #         ),
    #     },
    #     "mixed_poems": {
    #         "mean_n_coherence_values": np.array(
    #             [
    #                 0.18238005,
    #                 0.23476299,
    #                 0.26722008,
    #                 0.29007578,
    #                 0.30712293,
    #                 0.32034901,
    #                 0.33088417,
    #                 0.33954463,
    #                 0.34683808,
    #                 0.35311953,
    #             ]
    #         ),
    #         "example_counts": np.array(
    #             [
    #                 149900.0,
    #                 139414.0,
    #                 129544.0,
    #                 120349.0,
    #                 111976.0,
    #                 104334.0,
    #                 97349.0,
    #                 90794.0,
    #                 84847.0,
    #                 79337.0,
    #             ]
    #         ),
    #     },
    # }

    # ns = np.arange(max_context_lines)
    #
    # plt.plot(
    #     ns,
    #     results["real_poems"]["mean_n_coherence_values"],
    #     "g^",
    #     ns,
    #     results["shuffled_poems"]["mean_n_coherence_values"],
    #     "bs",
    #     ns,
    #     results["mixed_poems"]["mean_n_coherence_values"],
    #     "rd",
    # )
    # plt.ylabel("n-Semantic Coherence")
    # plt.xlabel("n")
    # plt.savefig("n_semanticCoherence.png")
