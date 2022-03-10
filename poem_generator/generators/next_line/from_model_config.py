import logging

import torch
from transformers import (
    MBartTokenizer,
    MBartForConditionalGeneration,
)

from poem_generator.io.candidates import PoemLine, PoemLineList
from poem_generator.utils import filter_candidates
from poem_generator.io.config import ModelConfig, GenerationConfig


# This file contains the code for generating the next line with the wikisource-fi-mbart.pytorch_model.bin model.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LANGUAGE_CODES = {"en": "en_XX"}


def get_tokenizer_and_model(model_config: ModelConfig):

    tokenizer = MBartTokenizer.from_pretrained(
        model_config.base_model,
        src_lang=LANGUAGE_CODES[model_config.lang],
        tgt_lang=LANGUAGE_CODES[model_config.lang],
        additional_special_tokens=model_config.special_tokens
        if model_config.special_tokens
        else [],
    )

    logging.info("Loading base model {}".format(model_config.base_model))
    model = MBartForConditionalGeneration.from_pretrained(model_config.base_model)
    model.config.decoder_start_token_id = tokenizer.lang_code_to_id[
        LANGUAGE_CODES[model_config.lang]
    ]

    if model_config.special_tokens:
        model.resize_token_embeddings(len(tokenizer))  # is this really necessary here?
    logging.info("Model vocab size is {}".format(model.config.vocab_size))
    model.load_state_dict(
        torch.load(model_config.model_file, map_location=torch.device(DEVICE))
    )
    model.to(DEVICE)

    return tokenizer, model


def generate(
    poem_state: PoemLineList, tokenizer, model, generation_config: GenerationConfig
) -> PoemLineList:
    """Generates line candidates

    :return: list of line candidates
    """
    source = generation_config.src_builder(poem_state)
    encoded = tokenizer.encode(
        source,
        padding="max_length",
        max_length=generation_config.src_max_length,
        truncation=generation_config.truncation,
    )
    if generation_config.batch_multiply:
        encoded = torch.tensor([encoded] * generation_config.batch_multiply).to(DEVICE)
        sample_outputs = model.generate(
            encoded,
            do_sample=True,
            max_length=generation_config.out_max_length,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k,
        )
    else:
        encoded = torch.tensor(encoded).unsqueeze(0).to(DEVICE)

        sample_outputs = model.generate(
            encoded,
            do_sample=generation_config.do_sample,
            max_length=generation_config.out_max_length,
            num_beams=generation_config.num_beams,
            # repetition_penalty=5.0,
            temperature=generation_config.temperature,
            # no_repeat_ngram_size=2,
            early_stopping=generation_config.early_stopping,
            num_return_sequences=generation_config.num_return_sequences,
            decoder_start_token_id=tokenizer.lang_code_to_id[LANGUAGE_CODES["en"]],
            # use_cache=False,
            # num_beam_groups=5,
            top_k=0,
        )

    candidates = [
        tokenizer.decode(sample_output, skip_special_tokens=True)
        for sample_output in sample_outputs
    ]
    logging.info("Generated candidates {}".format(candidates))

    return PoemLineList(
        [
            PoemLine(text=candidate)
            for candidate in filter_candidates(
                candidates,
                remove_duplicates=generation_config.remove_duplicate_candidates,
            )
        ]
    )
