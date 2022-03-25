import logging
from typing import List

import torch
from transformers import (
    MBartTokenizer,
    MBartForConditionalGeneration,
)

from poem_generator.io.candidates import PoemLine, PoemLineList
from poem_generator.utils import remove_punct
from poem_generator.utils import filter_candidates

# This file contains the code for generating the next line with all supported models.
# In future, it can become a package with several file (one for each implementation)

BASE_MODEL = "facebook/mbart-large-cc25"
MODEL_FILE = "models/1-2-3-to-1-fi-mbart-6-epoch-1.7/pytorch_model.bin"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECIAL_TOKENS= [">>>SEP<<<"]


def get_tokenizer_and_model():
    tokenizer = MBartTokenizer.from_pretrained(
        BASE_MODEL,
        src_lang="fi_FI",
        tgt_lang="fi_FI",
        additional_special_tokens=SPECIAL_TOKENS,
    )

    logging.info("Loading base model {}".format(BASE_MODEL))
    model = MBartForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.decoder_start_token_id = tokenizer.lang_code_to_id["fi_FI"]

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
    if len(poem_state) > 2:
        source = poem_state[-3].text + ">>>SEP<<<" + poem_state[-2].text + ">>>SEP<<<" + poem_state[-1].text
    if len(poem_state) < 2:
        source = poem_state[-1].text
    elif len(poem_state) < 3:
        source = poem_state[-2].text + ">>>SEP<<<" + poem_state[-1].text

    encoded = tokenizer.encode(
        source, padding="max_length", max_length=32, truncation=True
    )
    encoded = torch.tensor(encoded).unsqueeze(0).to(DEVICE)

    sample_outputs = model.generate(
        encoded,
        do_sample=True,
        max_length=16,
        num_beams=5,
        repetition_penalty=5.0,
        early_stopping=True,
        num_return_sequences=5,
    )

    candidates = [
        tokenizer.decode(sample_output, skip_special_tokens=True)
        for sample_output in sample_outputs
    ]
    logging.info("Generated candidates {}".format(candidates))

    return PoemLineList(
        [PoemLine(text=candidate) for candidate in filter_candidates(candidates)]
    )
