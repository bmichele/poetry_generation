import json
import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import spacy
from gensim import models
from tqdm import tqdm

from quality_estimation.coherence_estimator import SemanticCoherenceEstimator, SyntacticAnnotator

logger = logging.getLogger("metric_validation")
logger.setLevel(level=logging.INFO)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter("%(asctime)s - %(message)s")
fh.setFormatter(fh_formatter)
logger.addHandler(fh)


DATA_DIR = "/Users/miche/uniData/poetryGeneration/generated_poems/"


def poem_coherence_scores(
    poem: List[str],
    coherence_estimator: SemanticCoherenceEstimator,
    max_n: int,
    *args,
    **kwargs,
) -> np.array:
    """Given a poem, computes all n-coherence scores up to the max n value given as argument

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


if __name__ == "__main__":

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

    max_context_lines = 5

    results = {}

    generated_poem_files = [
        "mbart_en_gut_keywords_75000.jsonl",
        "mbart_en_gut_rhyme_75000.jsonl",
        "mbart_en_gut_mul_lines_75000.jsonl",
        "mbart_en_gut_next_line_75000.jsonl",
        "mbart_en_gut_mixed_lines_75000.jsonl",
    ]

    for generated_poem_file in generated_poem_files:
        logger.info("Processing {}".format(generated_poem_file))

        poem_jsonl_file = os.path.join(DATA_DIR, generated_poem_file)

        with open(poem_jsonl_file, "r") as f:
            data = [json.loads(line) for line in f.readlines()]

        # data = data[:100]

        # compute n-semantic coherence as done for the validation (duplicate codes... do not care atm)
        # consider only final poems
        poem_ids = set(data_line["id"] for data_line in data)
        poems = []
        for poem_identifier in poem_ids:
            final_poem = [data_line["poem_state"] for data_line in data if data_line["id"] == poem_identifier][-1]
            poems.append(final_poem)

        # logger.info("Processing {} dataset".format(poem_dataset_name))
        scores = [
            np.array([])
        ] * max_context_lines  # store here all the values for each order
        score_sums = np.zeros(max_context_lines)
        score_counts = np.zeros(max_context_lines)
        for poem in tqdm(poems):
            poem_scores = poem_coherence_scores(
                poem,
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

        average_scores = score_sums / score_counts
        st_deviations = np.array([np.std(order_scores) for order_scores in scores])
        logger.info(
            "{}: average score {}, std {} (over {} samples)".format(
                generated_poem_file, average_scores, st_deviations, score_counts
            )
        )
        results[generated_poem_file] = {
            "mean_n_coherence_values": average_scores,
            "example_counts": score_counts,
            "std": np.array([np.std(order_scores) for order_scores in scores]),
        }


    ns = np.arange(1, 6)

    training_steps = 75000

    plt.plot(
        ns,
        results["mbart_en_gut_keywords_{}.jsonl".format(training_steps)]["mean_n_coherence_values"],
        "g^",
        label="NL-Keywords",
    )
    plt.plot(
        ns,
        results["mbart_en_gut_next_line_{}.jsonl".format(training_steps)]["mean_n_coherence_values"],
        "mo",
        label="NL-Single",
    )
    plt.plot(
        ns,
        results["mbart_en_gut_mul_lines_{}.jsonl".format(training_steps)]["mean_n_coherence_values"],
        "kd",
        label="NL-Multi",
    )
    plt.plot(
        ns,
        results["mbart_en_gut_rhyme_{}.jsonl".format(training_steps)]["mean_n_coherence_values"],
        "rs",
        label="NL-Rhyme",
    )
    plt.plot(
        ns,
        results["mbart_en_gut_mixed_lines_{}.jsonl".format(training_steps)]["mean_n_coherence_values"],
        "b*",
        label="Mixed",
    )
    plt.legend(loc="lower right")
    plt.ylabel("n-Semantic Coherence")
    plt.xlabel("n")
    plt.xticks(ns)
    plt.grid(axis="y")
    plt.ylim([0, 1])
    # plt.show()
    plt.savefig("n_modelsSemanticCoherence.png")
