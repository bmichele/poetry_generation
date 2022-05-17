# From the gutenberg-poetry corpus, I generate a csv to be fed to the model

import csv
import gzip
import json
import logging
import os
import random
from time import time
from typing import List, Tuple

# from tqdm import tqdm
import pronouncing
from gensim import models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

RUN_ID = str(round(time() * 1000))
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    filename="build_training_corpus_{}.log".format(RUN_ID),
    level=logging.INFO,
)

random.seed(42)

# DATA_DIR = os.environ["DATADIR_POETRY"]
DATA_DIR = "/Users/miche/uniData/poetryGeneration"
LANG = "en"
# OUT_DATASETS = ["single_line", "mul_lines", "mixed_lines", "rhyme", "keywords"]
OUT_DATASETS = ["keywords"]


def example_list_to_csv(examples: List[Tuple], columns: Tuple, filename: str):
    with open(filename, mode="w") as out_file:
        data_writer = csv.writer(out_file)
        data_writer.writerow(list(columns))
        for example in examples:
            data_writer.writerow(list(example))


# remove weird stuff
def check_string(text: str) -> bool:
    if set(text).intersection(
        {"Euro(TM)", "{", "}", "$", "@", "#", "%", "&", "*", "[", "]", "|"}
    ):
        return False
    return True


def multiline_examples_from_poem(
    poem: List[str], window: int, sep_token: str
) -> List[Tuple[str, str]]:
    examples = []
    line_count = len(poem)
    i = 0
    for i in range(1, min(window, line_count)):
        src = sep_token.join(poem[:i][-window:]) + sep_token
        tgt = poem[i]
        example = (src, tgt)
        examples.append(example)
    while i < line_count - 1:
        i += 1
        src = sep_token.join(poem[i - window : i]) + sep_token
        tgt = poem[i]
        example = (src, tgt)
        examples.append(example)
    return examples


def multiline_examples_from_split_poem(
    poem: List[str], max_poem_length: int, window: int, sep_token: str
) -> List[List[Tuple[str, str]]]:
    examples = []
    if len(poem) > max_poem_length:
        for i in range(0, len(poem), max_poem_length):
            poem_split = poem[i : i + max_poem_length]
            examples.append(multiline_examples_from_poem(poem_split, window, sep_token))
    else:
        examples.append(multiline_examples_from_poem(poem, window, sep_token))
    return examples


if __name__ == "__main__":
    # load the dataset
    logging.info("Reading raw data")
    all_lines = []
    data_files = {
        "en": "gutenberg-poetry-v001.ndjson.gz",
        "fi": "gutenberg-poetry-fi.ndjson.gz",
        "sv": "gutenberg-poetry-sv.ndjson.gz",
    }
    for i, line in enumerate(gzip.open(os.path.join(DATA_DIR, data_files[LANG]))):
        # if i >= 100000:
        #     break
        all_lines.append(json.loads(line.strip()))

    ####################################################################################################################
    # Building datasets #
    #####################
    # I build the following datasets:
    #  * next line: src is a line, tgt is the next line in the poem
    #  * multiple lines: src is a sequence of lines, tgt is the next line
    #  * mixed lines: src is a line, tgt is a random line
    #  * rhyme: src is a line and a word, tgt is a line with an ending word that rhymes with the input word
    ####################################################################################################################

    #####################
    # Next Line Dataset #
    #####################
    if "single_line" in OUT_DATASETS:
        logging.info("Building next_line dataset")
        out_examples = []
        for i, line in enumerate(all_lines[:-1]):
            if i % 100000 == 0:
                logging.info("processing line {}".format(i))
            next_line = all_lines[i + 1]
            if next_line["gid"] == line["gid"]:
                out_example = (i, line["gid"], line["s"], next_line["s"])
                if check_string(out_example[2] + out_example[3]):
                    out_examples.append(out_example)
            else:
                continue

        # check for duplicates
        if len(out_examples) != len(set(out_examples)):
            logging.info("duplicate examples found, you should clean the dataset")

        # split train/test
        logging.info("Splitting train/val data")
        examples_train = []
        examples_val = []
        for out_example in out_examples:
            if random.random() > 0.2:
                examples_train.append(out_example)
            else:
                examples_val.append(out_example)

        logging.info("Shuffling data")
        random.shuffle(examples_train)
        random.shuffle(examples_val)

        logging.info("Saving train data to file")
        example_list_to_csv(
            examples_train,
            ("id", "gutenberg_id", "line", "next_line"),
            os.path.join(DATA_DIR, "data.{}.{}.train.csv".format(LANG, "next_line")),
        )
        logging.info("Saving val data to file")
        example_list_to_csv(
            examples_val,
            ("id", "gutenberg_id", "line", "next_line"),
            os.path.join(DATA_DIR, "data.{}.{}.val.csv".format(LANG, "next_line")),
        )

    ##########################
    # Multiple Lines Dataset #
    ##########################
    if "mul_lines" in OUT_DATASETS:
        logging.info("Building multiple_line dataset")
        window_size = 3  # maximum number of lines in the source
        max_length = 20  # poems that have more lines that this are split
        separator = " >>>SEP<<< "

        logging.info("Split train and val")
        train_count = round(0.8 * len(all_lines))
        train_lines = all_lines[:train_count]
        val_lines = all_lines[train_count:]

        # Building Train data
        # get separate poems
        logging.info("Building train data")
        all_lines_with_id = [(line["gid"], line["s"]) for line in train_lines]
        gutenberg_ids = list(set([g_id for g_id, _ in all_lines_with_id]))
        examples_train = []
        logging.info("Processing {} gutenberg ids".format(len(gutenberg_ids)))
        for gutenberg_id in gutenberg_ids:
            poem_lines = [
                line for g_id, line in all_lines_with_id if g_id == gutenberg_id
            ]

            poem_examples = multiline_examples_from_split_poem(
                poem=poem_lines,
                window=window_size,
                max_poem_length=max_length,
                sep_token=separator,
            )
            examples_train += poem_examples
        # shuffle examples
        random.shuffle(examples_train)
        # flatten the list
        final_train_examples = []
        example_id = 0
        for poem_examples in examples_train:
            for singe_example in poem_examples:
                final_train_examples.append((example_id, *singe_example))
                example_id += 1

        logging.info("Saving train data to file")
        example_list_to_csv(
            final_train_examples,
            ("id", "src", "tgt"),
            os.path.join(DATA_DIR, "data.{}.{}.train.csv".format(LANG, "mul_lines")),
        )

        # Repeat the same Procedure for Validation Data
        # get separate poems
        logging.info("Building validation data")
        all_lines_with_id = [(line["gid"], line["s"]) for line in val_lines]
        gutenberg_ids = list(set([g_id for g_id, _ in all_lines_with_id]))
        examples_val = []
        logging.info("Processing {} gutenberg ids".format(len(gutenberg_ids)))
        for gutenberg_id in gutenberg_ids:
            poem_lines = [
                line for g_id, line in all_lines_with_id if g_id == gutenberg_id
            ]

            poem_examples = multiline_examples_from_split_poem(
                poem=poem_lines,
                window=window_size,
                max_poem_length=max_length,
                sep_token=separator,
            )
            examples_val += poem_examples
        # shuffle examples
        random.shuffle(examples_val)
        # flatten the list
        final_val_examples = []
        example_id = 0
        for poem_examples in examples_val:
            for singe_example in poem_examples:
                final_val_examples.append((example_id, *singe_example))
                example_id += 1

        logging.info("Saving val data to file")
        example_list_to_csv(
            final_val_examples,
            ("id", "src", "tgt"),
            os.path.join(DATA_DIR, "data.{}.{}.val.csv".format(LANG, "mul_lines")),
        )

    #######################
    # Mixed Lines Dataset #
    #######################
    if "mixed_lines" in OUT_DATASETS:
        logging.info("Building mixed_lines dataset")
        out_examples = []
        for i, line in enumerate(all_lines):
            if i % 100000 == 0:
                logging.info("processing line {}".format(i))
            next_line = random.choice(all_lines)
            out_example = (i, line["s"], next_line["s"])
            if check_string(out_example[1] + out_example[2]):
                out_examples.append(out_example)

        # check for duplicates
        if len(out_examples) != len(set(out_examples)):
            logging.info("duplicate examples found, you should clean the dataset")

        # split train/test
        logging.info("Splitting train/val data")
        examples_train = []
        examples_val = []
        for out_example in out_examples:
            if random.random() > 0.2:
                examples_train.append(out_example)
            else:
                examples_val.append(out_example)

        logging.info("Shuffling data")
        random.shuffle(examples_train)
        random.shuffle(examples_val)

        logging.info("Saving train data to file")
        example_list_to_csv(
            examples_train,
            ("id", "line", "next_line"),
            os.path.join(DATA_DIR, "data.{}.{}.train.csv".format(LANG, "mixed_lines")),
        )
        logging.info("Saving val data to file")
        example_list_to_csv(
            examples_val,
            ("id", "line", "next_line"),
            os.path.join(DATA_DIR, "data.{}.{}.val.csv".format(LANG, "mixed_lines")),
        )

    #################
    # Rhyme Dataset #
    #################
    if "rhyme" in OUT_DATASETS:
        logging.info("Building rhyme dataset")
        max_rhymes = 4  # number of examples to be built from a single line pair
        separator = " >>>SEP<<< "

        # the implementation is the same as for next line, but we add a rhyme to the last tgt word in the src text
        out_examples = []
        for i, line in enumerate(all_lines[:-1]):
            if i % 100000 == 0:
                logging.info("processing line {}".format(i))
            next_line = all_lines[i + 1]
            if next_line["gid"] == line["gid"]:
                out_example = (i, line["gid"], line["s"], next_line["s"])
                if check_string(out_example[2] + out_example[3]):
                    out_examples.append(out_example)
            else:
                continue

        # split train/test
        logging.info("Splitting train/val data")
        examples_train = []
        examples_val = []
        for out_example in out_examples:
            if random.random() > 0.2:
                examples_train.append(out_example)
            else:
                examples_val.append(out_example)

        logging.info("Shuffling data")
        random.shuffle(examples_train)
        random.shuffle(examples_val)

        logging.info("Adding rhymes to train data")

        def find_rhymes(poem_line: str) -> List[str]:
            ending_token = poem_line.split()[-1]
            return pronouncing.rhymes(ending_token)

        train_with_rhymes = []
        for i, example in enumerate(examples_train):
            if i % 100000 == 0:
                logging.info("processing line {}".format(i))
            tgt_line = example[-1]
            rhymes = find_rhymes(tgt_line)

            if rhymes:
                if len(rhymes) > max_rhymes:
                    rhymes = random.sample(rhymes, max_rhymes)
                for rhyme in rhymes:
                    src = rhyme + separator + example[-2]
                    train_with_rhymes.append((example[0], example[1], src, tgt_line))
        random.shuffle(train_with_rhymes)

        val_with_rhymes = []
        for i, example in enumerate(examples_val):
            if i % 100000 == 0:
                logging.info("processing line {}".format(i))
            tgt_line = example[-1]
            rhymes = find_rhymes(tgt_line)

            if rhymes:
                if len(rhymes) > max_rhymes:
                    rhymes = random.sample(rhymes, max_rhymes)
                for rhyme in rhymes:
                    src = rhyme + separator + example[-2]
                    val_with_rhymes.append((example[0], example[1], src, tgt_line))
        random.shuffle(val_with_rhymes)

        logging.info("Saving train data to file")
        example_list_to_csv(
            train_with_rhymes,
            ("id", "gutenberg_id", "line", "next_line"),
            os.path.join(DATA_DIR, "data.{}.{}.train.csv".format(LANG, "rhyme")),
        )
        logging.info("Saving val data to file")
        example_list_to_csv(
            val_with_rhymes,
            ("id", "gutenberg_id", "line", "next_line"),
            os.path.join(DATA_DIR, "data.{}.{}.val.csv".format(LANG, "rhyme")),
        )

    ####################
    # Keywords Dataset #
    ####################
    if "keywords" in OUT_DATASETS:

        logging.info("Building keywords dataset")
        separator = " >>>SEP<<< "

        out_examples = []
        for i, line in enumerate(all_lines[:-1]):
            if i % 100000 == 0:
                logging.info("processing line {}".format(i))
            next_line = all_lines[i + 1]
            if next_line["gid"] == line["gid"]:
                out_example = (i, line["gid"], line["s"], next_line["s"])
                if check_string(out_example[2] + out_example[3]):
                    out_examples.append(out_example)
            else:
                continue

        # split train/test
        logging.info("Splitting train/val data")
        examples_train = []
        examples_val = []
        for out_example in out_examples:
            if random.random() > 0.2:
                examples_train.append(out_example)
            else:
                examples_val.append(out_example)

        logging.info("Shuffling data")
        random.shuffle(examples_train)
        random.shuffle(examples_val)

        logging.info("Adding keywords to train data")

        model_dir = "/Users/miche/uniModels"
        w2v = models.KeyedVectors.load_word2vec_format(
            os.path.join(model_dir, "GoogleNews-vectors-negative300.bin"), binary=True
        )
        from gensim.similarities.annoy import AnnoyIndexer

        if os.path.exists("annoy_index"):
            logging.info("Loading annoy index")
            annoy_index = AnnoyIndexer()
            annoy_index.load("annoy_index")
            annoy_index.model = w2v
        else:
            logging.info("Building annoy index")
            annoy_index = AnnoyIndexer(w2v, 100)
            annoy_index.save("annoy_index")

        def find_related_words(
            poem_line: str,
            word2vec: models.keyedvectors.KeyedVectors,
            sample_k: int = None,
        ) -> List[str]:
            tokens = [
                token
                for token in word_tokenize(poem_line)
                if token.lower() not in stopwords.words("english")
            ]
            tokens = [token for token in tokens if token in word2vec]
            if tokens:
                related = word2vec.most_similar(tokens, topn=10, indexer=annoy_index)
                related = [word.replace("_", " ") for word, _ in related]
                if sample_k:
                    related = random.sample(related, sample_k)
                return related
            else:
                return []

        train_with_keywords = []
        for i, example in enumerate(examples_train):
            if i % 50000 == 0:
                logging.info("processing line {}".format(i))
            tgt_line = example[-1]
            related_words = find_related_words(tgt_line, w2v, sample_k=2)

            src = " ".join(related_words) + separator + example[-2]
            train_with_keywords.append((example[0], example[1], src, tgt_line))
        random.shuffle(train_with_keywords)

        val_with_keywords = []
        for i, example in enumerate(examples_val):
            if i % 50000 == 0:
                logging.info("processing line {}".format(i))
            tgt_line = example[-1]
            related_words = find_related_words(tgt_line, w2v, sample_k=2)

            src = " ".join(related_words) + separator + example[-2]
            val_with_keywords.append((example[0], example[1], src, tgt_line))
        random.shuffle(val_with_keywords)

        logging.info("Saving train data to file")
        example_list_to_csv(
            train_with_keywords,
            ("id", "gutenberg_id", "line", "next_line"),
            os.path.join(DATA_DIR, "data.{}.{}.train.csv".format(LANG, "keywords")),
        )
        logging.info("Saving val data to file")
        example_list_to_csv(
            val_with_keywords,
            ("id", "gutenberg_id", "line", "next_line"),
            os.path.join(DATA_DIR, "data.{}.{}.val.csv".format(LANG, "keywords")),
        )

    ####################
    # Antonyms Dataset #
    ####################

    logging.info("Done")
