import os
import csv
import logging
from typing import List, Tuple
from utils import remove_punct, tokenize

logging.basicConfig(level=logging.DEBUG)

DATASET = "wikisource"
LANGUAGE = "fi"
DATA_DIR = os.path.join(os.environ["DATADIR_UNI"], "finnishPoetryGeneration")
TRAIN_DATA = os.path.join(DATA_DIR, "data_{}_{}_train.csv".format(DATASET, LANGUAGE))


def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str]]:
    if n > len(tokens):
        return []
    elif n == len(tokens):
        return [tuple(tokens)]
    else:
        out = []
        for i in range(len(tokens) - n + 1):
            out.append(tuple(tokens[i: i + n]))
        return out


train_examples = []
with open(TRAIN_DATA) as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        train_examples.append(row)

# check for duplicated
counter = 0
for example in train_examples[1:]:
    src = remove_punct(example[2]).lower()
    trg = remove_punct(example[3]).lower()
    if src == trg:
        logging.info("Found duplicate src - trg in line id {}".format(example[0]))
        counter += 1
    else:
        # compute total matching ngrams
        tokens_src = tokenize(src)
        tokens_trg = tokenize(trg)

        max_overlap_len = min(len(tokens_src), len(tokens_trg))

        if max_overlap_len > 1:
            n = max_overlap_len - 1
            while n > 0:
                # check number of overlapping ngrams of length n
                ngrams_1 = set(get_ngrams(tokens_src, n))
                ngrams_2 = set(get_ngrams(tokens_trg, n))
                matching_ngrams = ngrams_1.intersection(ngrams_2)
                if matching_ngrams and (n / max_overlap_len) > 0.7:
                    logging.debug("found {} matching {}-grams in line id {}".format(len(matching_ngrams), n, example[0]))
                    logging.debug(example)
                    counter += 1
                    break  # avoid double counting
                n -= 1

