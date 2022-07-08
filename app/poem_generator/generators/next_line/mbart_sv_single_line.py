import logging

import torch
from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
)

from poem_generator.io.candidates import PoemLine, PoemLineList
from poem_generator.utils import filter_candidates

# This file contains the code for generating the next line with all supported models.
# In future, it can become a package with several file (one for each implementation)

BASE_MODEL = "facebook/mbart-large-50"
MODEL_FILE = "models/poetry-generation-nextline-mbart-ws-sv-test/pytorch_model.bin"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_tokenizer_and_model():
    tokenizer = MBart50TokenizerFast.from_pretrained(
        BASE_MODEL,
        src_lang="sv_SE",
        tgt_lang="sv_SE",
    )

    logging.info("Loading base model {}".format(BASE_MODEL))
    model = MBartForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.decoder_start_token_id = tokenizer.lang_code_to_id["sv_SE"]

    model.resize_token_embeddings(len(tokenizer))  # is this really necessary here?
    logging.info("Model vocab size is {}".format(model.config.vocab_size))
    model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device(DEVICE)))
    model.to(DEVICE)

    return tokenizer, model


def generate(poem_state: PoemLineList, tokenizer, model) -> PoemLineList:
    """
    Implementation of next line poem generator using mbart for finnish language
    :return:
    """
    source = poem_state[-1].text
    encoded = tokenizer.encode(
        source, padding="max_length", max_length=32, truncation=True
    )
    encoded = torch.tensor([encoded] * 10).to(DEVICE)

    sample_outputs = model.generate(
        encoded,
        do_sample=True,
        max_length=16,
        temperature=5.0,
        top_k=5,
    )

    candidates = [
        tokenizer.decode(sample_output, skip_special_tokens=True)
        for sample_output in sample_outputs
    ]
    logging.info("Generated candidates {}".format(candidates))

    return PoemLineList(
        [
            PoemLine(text=candidate)
            for candidate in filter_candidates(candidates, n_max=4)
        ]
    )
