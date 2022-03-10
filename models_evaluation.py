import logging
import os
import random
from random import sample

import jsonlines
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from poem_generator.generator import PoemGenerator
from poem_generator.io.candidates import PoemLine, PoemLineList
from poem_generator.io.config import (
    GenerationConfig,
    ModelConfig,
    PoemGeneratorConfiguration,
)

# setup logger
logger = logging.getLogger("my_module_name")
logger.setLevel(level=logging.INFO)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter("%(asctime)s - %(message)s")
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

DATADIR = os.path.join(os.environ["DATADIR_UNI"], "kaggle_poemsdataset/forms")
POEMS_PER_MODEL = 10
ITERATIONS_PER_POEM = 10
# POEMS_PER_MODEL = 1
# ITERATIONS_PER_POEM = 2

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


word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
response = requests.get(word_site)
WORDS = response.content.splitlines()


def src_builder_rhyme(poem_state: PoemLineList) -> str:
    if len(poem_state) < 2 or random.random() > 0.7:
        rhyme_word = random.sample(WORDS, 1)[0].decode("utf-8")
        # logger.info("sampling random rhyme {}".format(rhyme_word))
    else:
        rhyme_word = poem_state[-2].text.split()[-1]
    return rhyme_word + " >>>SEP<<< " + poem_state[-1].text


def src_builder_mul_lines(poem_state: PoemLineList) -> str:
    return (
        " >>>SEP<<< ".join(poem_line.text for poem_line in poem_state.to_list()[-3:])
        + " >>>SEP<<< "
    )


def src_builder_keywords(poem_state: PoemLineList) -> str:
    kwd_proxies = word_tokenize(poem_state[0].text)
    kwd_proxies = [word for word in kwd_proxies if word.lower() not in stopwords.words("english")]
    if kwd_proxies:
        kwd_proxies = random.sample(kwd_proxies, min(3, len(kwd_proxies)))
    out = " ".join(kwd_proxies) + " >>>SEP<<< " + poem_state[-1].text
    logger.debug(out)
    return out


if __name__ == "__main__":
    # load poem lines to be used as starting line
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

    poem_lines = [line for poem in poems for line in poem.split("\n")]
    poem_lines = [line.capitalize().strip() for line in poem_lines if len(line) >= 20]

    ##############################
    # Run the Generation Routine #
    ##############################

    configs = {
        "mbart_en_gut_next_line_60000": PoemGeneratorConfiguration(
            next_line_model_config=ModelConfig(
                base_model="facebook/mbart-large-cc25",
                model_file="models/mbart_en_gut_next_line/checkpoint-60000/pytorch_model.bin",
                lang="en",
            ),
            generation_config=GenerationConfig(
                src_builder=lambda poem_state: poem_state[-1].text,
                batch_multiply=10,
                remove_duplicate_candidates=False,
            ),
        ),
        "mbart_en_gut_mixed_lines_60000": PoemGeneratorConfiguration(
            next_line_model_config=ModelConfig(
                base_model="facebook/mbart-large-cc25",
                model_file="models/mbart_en_gut_mixed_lines/checkpoint-60000/pytorch_model.bin",
                lang="en",
            ),
            generation_config=GenerationConfig(
                src_builder=lambda poem_state: poem_state[-1].text,
                batch_multiply=10,
                remove_duplicate_candidates=False,
            ),
        ),
        "mbart_en_gut_mul_lines_60000": PoemGeneratorConfiguration(
            next_line_model_config=ModelConfig(
                base_model="facebook/mbart-large-cc25",
                model_file="models/mbart_en_gut_mul_lines/checkpoint-60000/pytorch_model.bin",
                lang="en",
                special_tokens=[">>>SEP<<<"],
            ),
            generation_config=GenerationConfig(
                src_builder=src_builder_mul_lines,
                batch_multiply=10,
                remove_duplicate_candidates=False,
            ),
        ),
        "mbart_en_gut_keywords_60000": PoemGeneratorConfiguration(
            next_line_model_config=ModelConfig(
                base_model="facebook/mbart-large-cc25",
                model_file="models/mbart_en_gut_keywords/checkpoint-60000/pytorch_model.bin",
                lang="en",
                special_tokens=[">>>SEP<<<"],
            ),
            generation_config=GenerationConfig(
                src_builder=src_builder_keywords,
                batch_multiply=10,
                remove_duplicate_candidates=False,
            ),
        ),
        "mbart_en_gut_rhyme_60000": PoemGeneratorConfiguration(
            next_line_model_config=ModelConfig(
                base_model="facebook/mbart-large-cc25",
                model_file="models/mbart_en_gut_rhyme/checkpoint-60000/pytorch_model.bin",
                lang="en",
                special_tokens=[">>>SEP<<<"],
            ),
            generation_config=GenerationConfig(
                src_builder=src_builder_rhyme,
                batch_multiply=10,
                remove_duplicate_candidates=False,
            ),
        ),
        # "mbart_en_gut_next_line_75000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_next_line/checkpoint-75000/pytorch_model.bin",
        #         lang="en",
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=lambda poem_state: poem_state[-1].text,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_mixed_lines_75000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_mixed_lines/checkpoint-75000/pytorch_model.bin",
        #         lang="en",
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=lambda poem_state: poem_state[-1].text,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_mul_lines_75000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_mul_lines/checkpoint-75000/pytorch_model.bin",
        #         lang="en",
        #         special_tokens=[">>>SEP<<<"],
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=src_builder_mul_lines,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_keywords_75000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_keywords/checkpoint-75000/pytorch_model.bin",
        #         lang="en",
        #         special_tokens=[">>>SEP<<<"],
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=src_builder_keywords,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_rhyme_75000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_rhyme/checkpoint-75000/pytorch_model.bin",
        #         lang="en",
        #         special_tokens=[">>>SEP<<<"],
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=src_builder_rhyme,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_next_line_30000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_next_line/checkpoint-30000/pytorch_model.bin",
        #         lang="en",
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=lambda poem_state: poem_state[-1].text,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_mixed_lines_30000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_mixed_lines/checkpoint-30000/pytorch_model.bin",
        #         lang="en",
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=lambda poem_state: poem_state[-1].text,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_mul_lines_30000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_mul_lines/checkpoint-30000/pytorch_model.bin",
        #         lang="en",
        #         special_tokens=[">>>SEP<<<"],
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=src_builder_mul_lines,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_keywords_30000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_keywords/checkpoint-30000/pytorch_model.bin",
        #         lang="en",
        #         special_tokens=[">>>SEP<<<"],
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=src_builder_keywords,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_rhyme_30000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_rhyme/checkpoint-30000/pytorch_model.bin",
        #         lang="en",
        #         special_tokens=[">>>SEP<<<"],
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=src_builder_rhyme,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_next_line_45000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_next_line/checkpoint-45000/pytorch_model.bin",
        #         lang="en",
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=lambda poem_state: poem_state[-1].text,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_mixed_lines_45000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_mixed_lines/checkpoint-45000/pytorch_model.bin",
        #         lang="en",
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=lambda poem_state: poem_state[-1].text,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_mul_lines_45000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_mul_lines/checkpoint-45000/pytorch_model.bin",
        #         lang="en",
        #         special_tokens=[">>>SEP<<<"],
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=src_builder_mul_lines,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_keywords_45000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_keywords/checkpoint-45000/pytorch_model.bin",
        #         lang="en",
        #         special_tokens=[">>>SEP<<<"],
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=src_builder_keywords,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_rhyme_45000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_rhyme/checkpoint-45000/pytorch_model.bin",
        #         lang="en",
        #         special_tokens=[">>>SEP<<<"],
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=src_builder_rhyme,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_next_line_15000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_next_line/checkpoint-15000/pytorch_model.bin",
        #         lang="en",
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=lambda poem_state: poem_state[-1].text,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_mixed_lines_15000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_mixed_lines/checkpoint-15000/pytorch_model.bin",
        #         lang="en",
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=lambda poem_state: poem_state[-1].text,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_mul_lines_15000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_mul_lines/checkpoint-15000/pytorch_model.bin",
        #         lang="en",
        #         special_tokens=[">>>SEP<<<"],
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=src_builder_mul_lines,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_keywords_15000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_keywords/checkpoint-15000/pytorch_model.bin",
        #         lang="en",
        #         special_tokens=[">>>SEP<<<"],
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=src_builder_keywords,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
        # "mbart_en_gut_rhyme_15000": PoemGeneratorConfiguration(
        #     next_line_model_config=ModelConfig(
        #         base_model="facebook/mbart-large-cc25",
        #         model_file="models/mbart_en_gut_rhyme/checkpoint-15000/pytorch_model.bin",
        #         lang="en",
        #         special_tokens=[">>>SEP<<<"],
        #     ),
        #     generation_config=GenerationConfig(
        #         src_builder=src_builder_rhyme,
        #         batch_multiply=10,
        #         remove_duplicate_candidates=False,
        #     ),
        # ),
    }

    for model_name, config in configs.items():
        logger.info("Initialising poem creator with model {}".format(model_name))
        generator = PoemGenerator(configs[model_name])
        logger.info("Generating poems with {} model".format(model_name))
        out_file = "{}.jsonl".format(model_name)
        f = open(out_file, "w")
        writer = jsonlines.Writer(f)
        for poem_id in range(POEMS_PER_MODEL):
            if poem_id % 20 == 0 and poem_id != 0:
                logger.info("{} - Saving to file".format(poem_id))
                writer.close()
                f.close()
                f = open(out_file, "a")
                writer = jsonlines.Writer(f)
            generator.state = PoemLineList()
            random_first_line = random.sample(poem_lines, 1)[0]
            # generator.add_line(PoemLine("Two roads diverged in a yellow wood,"))
            generator.add_line(PoemLine(random_first_line))
            for i in range(ITERATIONS_PER_POEM):
                candidates = generator.get_line_candidates()
                unique_candidates = list(set(candidate.text for candidate in candidates))
                logger.debug("Total candidates {}".format(len(candidates)))
                logger.debug("Unique candidates {}".format(len(unique_candidates)))
                unique_candidates = PoemLineList([PoemLine(candidate) for candidate in unique_candidates])
                writer.write(
                    {
                        "id": poem_id,
                        "poem_state": [poem_line.text for poem_line in generator.state],
                        "candidates": [
                            candidate.text for candidate in candidates.to_list()
                        ],
                    }
                )
                if not candidates:
                    logger.info("poem_id={}, iteration={} - no candidates found".format(poem_id, i))
                    break
                random_selection = sample(unique_candidates.to_list(), 1)[0]
                generator.add_line(random_selection)

        writer.close()
        f.close()
        del generator

    logger.info("Done")
