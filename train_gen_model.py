########################################################################################################################
########################################################################################################################
# Training script for poetry generation
########################################################################################################################
########################################################################################################################


########################################################################################################################
# Import libraries
########################################################################################################################

import argparse
import ast
import configparser
import logging
import os

import numpy as np
import torch
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_error
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    MBartTokenizer,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    IntervalStrategy,
)
from transformers import Trainer, TrainingArguments, set_seed

os.environ["WANDB_DISABLED"] = "true"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# torch.backends.cudnn.enabled = False

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Training config file")
    args = parser.parse_args()

    # args = argparse.Namespace(
    #     config_file="training_config/mbart_debug.ini"
    # )

    ####################################################################################################################
    # Reading config file
    ####################################################################################################################

    logging.info("Reading config file {}".format(args.config_file))
    config = configparser.ConfigParser(os.environ, interpolation=configparser.ExtendedInterpolation())
    config.read(args.config_file)

    config_section_general = "General"
    DEBUG = config[config_section_general]["debug"].lower() == "true"
    DISABLE_TQDM = config[config_section_general]["disable_tqdm"].lower() == "true"
    CHECK_MAX_SEQ_LEN = (
        config[config_section_general]["check_max_sequence_length"].lower() == "true"
    )
    DO_TRAIN = config[config_section_general]["do_train"].lower() == "true"
    SEED = int(config[config_section_general]["seed"])

    config_section_model_specs = "Model Specs"
    MODEL = config[config_section_model_specs]["model"]
    LANGUAGE = config[config_section_model_specs]["language"]
    MAX_SEQ_LEN_SRC = int(config[config_section_model_specs]["max_sequence_length_src"])
    MAX_SEQ_LEN_TRG = int(config[config_section_model_specs]["max_sequence_length_trg"])
    MODEL_OUT_DIR = config[config_section_model_specs]["out_dir"]

    config_section_data_specs = "Dataset Specs"
    DATA_TRAIN = config[config_section_data_specs]["data_train"]
    DATA_VAL = config[config_section_data_specs]["data_validation"]
    COLUMN_SRC = config[config_section_data_specs]["data_column_name_src"]
    COLUMN_TRG = config[config_section_data_specs]["data_column_name_trg"]
    SPECIAL_TOKENS = ast.literal_eval(config[config_section_data_specs]["special_tokens"])

    config_section_training_args = "Training Args"
    EPOCHS = int(config[config_section_training_args]["epochs"])
    BATCH_SIZE_TRAIN = int(config[config_section_training_args]["batch_size_train"])
    BATCH_SIZE_EVAL = int(config[config_section_training_args]["batch_size_eval"])
    WARMUP = int(config[config_section_training_args]["warmup"])
    WEIGHT_DECAY = float(config[config_section_training_args]["weight_decay"])
    STEPS_SAVE = int(config[config_section_training_args]["steps_save"])
    STEPS_EVAL = int(config[config_section_training_args]["steps_eval"])
    STEPS_LOGGING = int(config[config_section_training_args]["steps_logging"])

    logging.info(
        "Config file variables: {}".format(
            {
                "DEBUG": DEBUG,
                "DISABLE_TQDM": DISABLE_TQDM,
                "CHECK_MAX_SEQ_LEN": CHECK_MAX_SEQ_LEN,
                "DO_TRAIN": DO_TRAIN,
                "SEED": SEED,
                "MODEL": MODEL,
                "LANGUAGE": LANGUAGE,
                "MAX_SEQ_LEN_SRC": MAX_SEQ_LEN_SRC,
                "MAX_SEQ_LEN_TRG": MAX_SEQ_LEN_TRG,
                "MODEL_OUT_DIR": MODEL_OUT_DIR,
                "DATA_TRAIN": DATA_TRAIN,
                "DATA_VAL": DATA_VAL,
                "COLUMN_SRC": COLUMN_SRC,
                "COLUMN_TRG": COLUMN_TRG,
                "SPECIAL_TOKENS": SPECIAL_TOKENS,
                "EPOCHS": EPOCHS,
                "BATCH_SIZE_TRAIN": BATCH_SIZE_TRAIN,
                "BATCH_SIZE_EVAL": BATCH_SIZE_EVAL,
                "WARMUP": WARMUP,
                "WEIGHT_DECAY": WEIGHT_DECAY,
                "STEPS_SAVE": STEPS_SAVE,
                "STEPS_EVAL": STEPS_EVAL,
                "STEPS_LOGGING": STEPS_LOGGING,
            }
        )
    )

    set_seed(SEED)
    if DISABLE_TQDM:
        set_verbosity_error()

    ####################################################################################################################
    # check cuda
    ####################################################################################################################
    cuda_available = torch.cuda.is_available()
    logging.info("cuda available: {}".format(cuda_available))

    ####################################################################################################################
    # load model and tokenizer
    ####################################################################################################################

    logging.info("loading tokenizer and model")
    if MODEL == "facebook/mbart-large-cc25":
        language_codes = {
            "en": "en_XX",
            "fi": "fi_FI",
        }
        tokenizer = MBartTokenizer.from_pretrained(
            MODEL,
            src_lang=language_codes[LANGUAGE],
            tgt_lang=language_codes[LANGUAGE],
            additional_special_tokens=SPECIAL_TOKENS,
        )
        model = MBartForConditionalGeneration.from_pretrained(MODEL)
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[
            language_codes[LANGUAGE]
        ]
    elif MODEL == "facebook/bart-large":
        if LANGUAGE != "en":
            logging.warning(
                "Bart model works for English language only. Expect poor performance when fine-tuning for {} language".format(
                    LANGUAGE
                )
            )
        tokenizer = BartTokenizer.from_pretrained(
            MODEL, additional_special_tokens=SPECIAL_TOKENS
        )
        model = BartForConditionalGeneration.from_pretrained(MODEL)
    elif MODEL == "facebook/mbart-large-50":
        language_codes = {
            "en": "en_XX",
            "fi": "fi_FI",
            "sv": "sv_SE"
        }
        tokenizer = MBart50TokenizerFast.from_pretrained(
            MODEL,
            src_lang=language_codes[LANGUAGE],
            tgt_lang=language_codes[LANGUAGE],
            additional_special_tokens=SPECIAL_TOKENS,
            # local_files_only=True,
        )
        model = MBartForConditionalGeneration.from_pretrained(MODEL)
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[
            language_codes[LANGUAGE]
        ]
    else:
        raise NotImplementedError

    logging.info("Pre-trained model vocab size: {}".format(model.config.vocab_size))
    model.resize_token_embeddings(len(tokenizer))
    logging.info("Vocab size after adding special tokens: {}".format(model.config.vocab_size))

    ####################################################################################################################
    # Load data from csv file
    ####################################################################################################################

    logging.info("loading data")
    dataset = load_dataset(
        "csv", data_files={"train": DATA_TRAIN, "validation": DATA_VAL}, delimiter=",",
    )
    if DEBUG:
        dataset["train"] = dataset["train"].select(range(100000))
        dataset["validation"] = dataset["validation"].select(range(25000))

    if CHECK_MAX_SEQ_LEN:
        # find max sequence length after tokenizing
        logging.info("determining length of tokenized sequences (sample 100000 and 25000 examples from train and val)")
        sequence_lengths_X = [
            len(tokenizer.tokenize(example[COLUMN_SRC])) for example in dataset["train"].select(range(100000))
        ]
        sequence_lengths_y = [
            len(tokenizer.tokenize(example[COLUMN_TRG])) for example in dataset["train"].select(range(25000))
        ]

        logging.info("Average lengths")
        logging.info(np.mean(sequence_lengths_X))
        logging.info(np.mean(sequence_lengths_y))
        logging.info("95th percentile")
        logging.info(np.percentile(sequence_lengths_X, 95))
        logging.info(np.percentile(sequence_lengths_y, 95))
        logging.info("99th percentile")
        logging.info(np.percentile(sequence_lengths_X, 99))
        logging.info(np.percentile(sequence_lengths_y, 99))

    ####################################################################################################################
    # Data preparation
    ####################################################################################################################

    if DO_TRAIN:

        def convert_to_features(example_batch):
            inp_enc = tokenizer.batch_encode_plus(
                example_batch[COLUMN_SRC],
                padding="max_length",
                max_length=MAX_SEQ_LEN_SRC,
                truncation=True,
            )
            with tokenizer.as_target_tokenizer():
                trg_enc = tokenizer.batch_encode_plus(
                    example_batch[COLUMN_TRG],
                    padding="max_length",
                    max_length=MAX_SEQ_LEN_TRG,
                    truncation=True,
                )

            lbs = np.array(trg_enc["input_ids"])
            lbs[lbs[:, :] == model.config.pad_token_id] = -100

            encodings = {
                "input_ids": inp_enc["input_ids"],
                "attention_mask": inp_enc["attention_mask"],
                # 'decoder_input_ids': decoder_input_ids,
                "labels": lbs,
            }

            return encodings

        logging.info("converting text to features")
        dataset["train"] = dataset["train"].map(convert_to_features, batched=True)
        dataset["validation"] = dataset["validation"].map(convert_to_features, batched=True)

        columns = [
            "input_ids",
            "labels",
            #  'decoder_input_ids',
            "attention_mask",
        ]
        dataset.set_format(type="torch", columns=columns)

        ################################################################################################################
        # Model Fine-Tuning
        ################################################################################################################

        logging.info("setup trainer")
        training_args = TrainingArguments(
            output_dir=MODEL_OUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE_TRAIN,
            per_device_eval_batch_size=BATCH_SIZE_EVAL,
            warmup_steps=WARMUP,
            weight_decay=WEIGHT_DECAY,
            logging_dir="./logs",
            save_steps=STEPS_SAVE,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=STEPS_EVAL,
            logging_strategy=IntervalStrategy.STEPS,
            logging_steps=STEPS_LOGGING,
            disable_tqdm=DISABLE_TQDM,
        )
        logging.info("training_args.n_gpu = {}".format(training_args.n_gpu))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
        )

        logging.info("trainer device set to {}".format(training_args._setup_devices))

        # train model
        logging.info("fine-tuning the model")
        trainer.train()

        # save model
        logging.info("saving the fine-tuned model to {}".format(MODEL_OUT_DIR))
        trainer.save_model()

        logging.info("done")
