import logging

import torch
from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
)

from poem_generator.io.candidates import PoemLine, PoemLineList
from poem_generator.utils import filter_candidates

BASE_MODEL = "facebook/mbart-large-50"
MODEL_FILE = "models/poetry-generation-nextline-mbart-ws-sv-multi/pytorch_model.bin"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_tokenizer_and_model():
    tokenizer = MBart50TokenizerFast.from_pretrained(
        BASE_MODEL,
        src_lang="sv_SE",
        tgt_lang="sv_SE",
        # FIXME: the model should be trained with the special token >>>SEP<<< and the following line uncommented
        # additional_special_tokens=[">>>SEP<<<"],
    )

    logging.info("Loading base model {}".format(BASE_MODEL))
    model = MBartForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.decoder_start_token_id = tokenizer.lang_code_to_id["sv_SE"]

    model.resize_token_embeddings(len(tokenizer))
    logging.info("Model vocab size is {}".format(model.config.vocab_size))
    model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device(DEVICE)))
    model.to(DEVICE)

    return tokenizer, model


def generate(poem_state: PoemLineList, tokenizer, model) -> PoemLineList:
    """
    Implementation of next line poem generator using mbart for finnish language
    :return:
    """
    source = ">>>SEP<<<".join(poem_line.text for poem_line in poem_state.to_list()[-3:])
    encoded = tokenizer.encode(
        source, padding="max_length", max_length=32, truncation=True
    )
    encoded = torch.tensor([encoded] * 10).to(DEVICE)

    sample_outputs = model.generate(
        encoded,
        do_sample=True,
        max_length=32,
        temperature=2.0,
        top_k=50,
    )

    candidates = [
        tokenizer.decode(sample_output, skip_special_tokens=True)
        for sample_output in sample_outputs
    ]
    logging.info("Generated candidates {}".format(candidates))

    return PoemLineList(
        [PoemLine(text=candidate) for candidate in filter_candidates(candidates)]
    )
