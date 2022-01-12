# README

This folder contains a lot of stuff that is outdated or useless. It should be cleaned up at a certain point...

The most relevant files and folders are:

 * `train_gen_model.py` script to train a generative model for next line generation
 * `training_config` contains example configuration files for the `train_gen_model.py` script  
 * `run_training.sh` wrapper to run python scripts in a virtual environment (useful especially for batch jobs)
 * `predict_mbart.py` script to generate poems with the trained models. does not support config files, needs to be tweaked to each specific model
 * `job_mbart_*.sh` example batch job scripts (for Turso, probably needs to be modified to run on CSC clusters)

## Dependencies

* pytorch
* transformers
* datasets
* protobuf
* sentencepiece


## How to fine-tune a seq2seq model

To run the `train_gen_model.py` script, you need to install all the required packages, and write your own configuration 
file. The file is parsed by the [configparser](https://docs.python.org/3/library/configparser.html) package, and example files are given in the folder `training_config`.

Once you have your configuration file, you can start the fine-tuning by running
```shell
python train_gen_model.py training_config/<YOUR_CONFIG_FILE>.ini
```

If you want to run it in a batch job, you will probably want to activate a virtual environment before running the script.
If you are using anaconda, you can use the wrapper script `run_training.sh`, which activates the specified virtual 
environment before running python. You probably have to modify the script according to your environment settings.

Example batch job scripts (tested in Turso) can be found in the folder, named as `job_mbart_*.sh`.

# Some Notes

## Useful stuff

Useful links:
 * https://towardsdatascience.com/teaching-bart-to-rap-fine-tuning-hugging-faces-bart-model-41749d38f3ef (fine-tuning BART for rap generation)
 * https://colab.research.google.com/drive/1n45bHMFw5dSTEUxcGYrQxhApPaltqGsS
 * https://colab.research.google.com/drive/1Cy27V-7qqYatqMA7fEqG2kgMySZXw9I4?usp=sharing&pli=1
 * https://gutenberg.org/ebooks/ project gutenberg (where to get finnish poetry)


Related work:
 * https://helda.helsinki.fi//bitstream/handle/10138/308738/nlg_poetry_finnish.pdf?sequence=1 (finnish poetry - NLG pipeline)
 * https://aclanthology.org/W19-8637.pdf (poetry generation with genetic algo)

## Files and folders

#### `build_gutenberg_cache.py`

Script that must be run to generate the cache before launching the `build_finnish_poetry_corpus.py` script.

#### `build_finnish_poetry_corpus.py`

Script to get poems from Project-Gutenberg. It can be used for different languages, and I used it to get poetry books for
Finnish, English and Swedish. Books are stored in `corpus_raw/[LANG]`.

#### `gutenberg-poetry-v001.ndjson.gz`

Corpus downloaded from https://github.com/aparrish/gutenberg-poetry-corpus

#### `read_poems.py`

Tentative script to separate actual poetry in the books... Not usable, must be improved.

## Conda virtual environments

 * `finnishPoetryGeneration` python 3.7 (to generate the poetry corpus)

## TODOs

Problem: quite often, the output sequence is just a copy of the input sequence - or very similar.
Could this be due to the presence of songs in the training data? If so, there are probably many examples taken from
choruses. Potentially, this could lead to a high number of training examples where the source sentence is the same as 
the output sentence, explaining the problem.

I should check the overlapping between source and target sentences, and try to deduplicate them,e.g. by
 * check when they are ==
 * check number of overlapping token ngrams
 * using a dedicate package (onion?)
