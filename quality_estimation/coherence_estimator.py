from __future__ import annotations

import logging
import math as m
from collections import Counter
from typing import List, Optional, Dict

import numpy as np
import spacy
from tqdm import tqdm
from copy import deepcopy

logger = logging.getLogger(__name__)


class AnnotatedToken:
    def __init__(self, text: str, is_stop: bool, annotation: str):
        self.text = text
        self.is_stop = is_stop
        self.annotation = annotation

    def __str__(self):
        return "AnnotatedToken({}, {}, {})".format(
            self.text, self.is_stop, self.annotation
        )

    def __repr__(self):
        return str(self)


class AnnotatedPhraseIterator:
    """Iterator class"""

    def __init__(self, annotated_tokens: AnnotatedPhrase):
        self._tokens = annotated_tokens
        self._index = 0

    def __next__(self):
        if self._index < len(self._tokens.to_list()):
            result = self._tokens.to_list()[self._index]
            self._index += 1
            return result
        raise StopIteration


class AnnotatedPhrase:
    def __init__(self, tokens: Optional[List[AnnotatedToken]]):
        self._tokens = tokens if tokens else list()

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, item):
        if item < len(self):
            return self._tokens[item]
        else:
            raise IndexError

    def __iter__(self):
        return AnnotatedPhraseIterator(self)

    def __str__(self):
        return "AnnotatedPhrase([{}])".format(
            ", ".join(str(token) for token in self._tokens)
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if type(other) == AnnotatedPhrase and str(other) == str(self):
            return True
        return False

    def to_list(self):
        return list(self._tokens)

    def contains_entity(self, entity: str):
        if entity in [token.text for token in self._tokens]:
            return True
        return False

    def get_annotation(self, entity: str):
        for token in self._tokens:
            if token.text == entity:
                return token.annotation
        return "-"


class SyntacticAnnotator:
    def __init__(self, spacy_model):
        self.spacy_model = spacy_model

    def annotate(self, text: str) -> AnnotatedPhrase:
        annotated_doc = self.spacy_model(text)
        entities_dict = {
            "subj": "S",
            "obj": "O",
        }
        return AnnotatedPhrase(
            [
                AnnotatedToken(
                    token.text, token.is_stop, entities_dict.get(token.dep_[1:], "X")
                )
                for token in annotated_doc
            ]
        )


class EntityGrid:
    def __init__(self, annotator: SyntacticAnnotator, separator: str = "\n"):
        self.array: np.array = np.array([])
        self.entities: list = []
        self.annotator = annotator
        self.separator = separator

    def train(self, paragraph: str, stopwords: bool = True):
        phrases = map(lambda x: x.strip(), paragraph.split(self.separator))
        phrases = [phrase for phrase in phrases if phrase]

        annotated_phrases = []
        for phrase in phrases:
            annotated_phrase = self.annotator.annotate(phrase)
            if not stopwords:
                annotated_phrase = AnnotatedPhrase(
                    [token for token in annotated_phrase if not token.is_stop]
                )
            annotated_phrases.append(annotated_phrase)

        self.entities = list(
            set(
                annotated_token.text
                for annotated_phrase in annotated_phrases
                for annotated_token in annotated_phrase
            )
        )
        rows = []
        for annotated_phrase in annotated_phrases:
            grid_row = [
                annotated_phrase.get_annotation(entity) for entity in self.entities
            ]
            rows.append(grid_row)
        self.array = np.array(rows)


class CoherenceEstimator:
    def __init__(self, annotator, max_ngram: int):
        self.annotator = annotator
        self.max_ngram = max_ngram
        self.grids: List[EntityGrid] = list()
        self.dict_count: dict = {}
        # self.dict_probabilities: dict = {}

    def compute_grids(self, paragraphs: List[str], stopwords: bool = True):
        for paragraph in tqdm(paragraphs):
            paragraph_grid = EntityGrid(self.annotator)
            paragraph_grid.train(paragraph, stopwords)
            self.grids.append(paragraph_grid)

    @staticmethod
    def get_ngrams(sequence: str, max_ngram_length) -> List[str]:
        ngrams = []
        for length in range(1, max_ngram_length + 1):
            # print("length {}".format(length))
            for start_index in range(len(sequence) - length + 1):
                ngram = sequence[start_index: start_index + length]
                ngrams.append(ngram)
        return ngrams

    def count_ngrams(self):
        all_ngrams = []
        for grid in self.grids:
            transposed_grid = grid.array.transpose()
            grid_columns = ["".join(grid_column.tolist()) for grid_column in transposed_grid]  # TODO: optimize by
            # using numpy array
            for column in grid_columns:
                # print(column)
                column_ngrams = self.get_ngrams(column, self.max_ngram)
                all_ngrams += column_ngrams
        counts = Counter(all_ngrams)
        self.dict_count = dict(counts)
        self.dict_count["*"] = sum([count for ngram, count in self.dict_count.items() if len(ngram) == 1])

    @staticmethod
    def probability(
        annotation: str, history: str, counter: Dict[str, int], unigrams_total_count: int
    ):
        if len(history) > 0:
            return counter.get(history + annotation, 0) / counter.get(history, 0)
        else:
            return counter.get(annotation) / unigrams_total_count

    # def compute_probabilities(self):
    #     unit_sequence_count = sum(
    #         count for sequence, count in self.dict_count.items() if len(sequence) == 1
    #     )
    #     for sequence in self.dict_count.keys():
    #         if len(sequence) > 1:
    #             self.dict_probabilities[sequence] = self.probability(
    #                 sequence[0], sequence[1:], self.dict_count
    #             )
    #         else:
    #             self.dict_probabilities[sequence] = (
    #                 self.dict_count[sequence] / unit_sequence_count
    #             )

    def train(self, paragraphs: List[str], stopwords: bool):
        logger.info("Computing entity grids of training paragraph")
        self.compute_grids(paragraphs, stopwords)
        logger.info("Computing probabilities")
        self.count_ngrams()
        # self.compute_probabilities()

    def predict(self, paragraph: str, stopwords: bool, max_ngram: int):
        assert max_ngram <= self.max_ngram, "max_ngram too high. Use a higher max_ngram value when training"
        paragraph_grid = EntityGrid(self.annotator)
        paragraph_grid.train(paragraph, stopwords)
        transposed_grid = paragraph_grid.array.transpose()
        log_probability = 0
        # print(transposed_grid.shape)
        # print(transposed_grid)
        grid_columns = ["".join(grid_column.tolist()) for grid_column in transposed_grid]
        for column in grid_columns:
            # print(column)
            for end_index in range(len(column)):
                # print(end_index)
                sliding_window = column[:end_index + 1][-max_ngram:]
                # print(sliding_window)
                log_probability += m.log(
                    self.probability(sliding_window[-1], sliding_window[:-1], self.dict_count, self.dict_count["*"])
                )

        return log_probability / paragraph_grid.array.size


if __name__ == "__main__":

    poem_1 = """
    Come when the nights are bright with stars
    Or when the moon is mellow;
    Come when the sun his golden bars
    Drops on the hay-field yellow.
    Come in the twilight soft and gray,
    Come in the night or come in the day,
    Come, O love, whene’er you may,
    And you are welcome, welcome.
    """
    poem_2 = """
    You are sweet, O Love, dear Love,
    You are soft as the nesting dove.
    Come to my heart and bring it rest
    As the bird flies home to its welcome nest.
    """
    poem_3 = """
    Come when my heart is full of grief
    Or when my heart is merry;
    Come with the falling of the leaf
    Or with the redd’ning cherry.
    Come when the year’s first blossom blows,
    Come when the summer gleams and glows,
    Come with the winter’s drifting snows,
    And you are welcome, welcome.
    """

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("merge_entities")

    my_annotator = SyntacticAnnotator(nlp)
    estimator = CoherenceEstimator(my_annotator, 4)
    estimator.train([poem_1, poem_2], False)
    out = estimator.predict(poem_3, False, 2)
    # print(out)
    # print(m.exp(out))
    #
    # test_grid = EntityGrid(my_annotator)
    # test_grid.train(poem_1, False)
