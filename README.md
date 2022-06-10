# README

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/25dc644dae5a45c3a19261181f5b8b3e)](https://app.codacy.com/gh/bmichele/poetry_generation?utm_source=github.com&utm_medium=referral&utm_content=bmichele/poetry_generation&utm_campaign=Badge_Grade_Settings)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains a simple implementation and some examples to generate poems with python by using neural network models.  
To contribute to this project, please send us a pull request from your fork of this repository. If you want to contribute, more information are given in the [contribution guideline](https://github.com/bmichele/poetry_generation/blob/main/CONTRIBUTING.md).

## Setting up the Poem Generator

First, clone the repository and install the required packages:
```shell
$ git clone https://github.com/bmichele/poetry_generation.git
$ cd poetry_generation/
$ pip install -r requirements.txt
```
Second, download the fine-tuned models as follows:
```shell
$ cd models
$ bash get_models.sh
$ cd ..
```

### Using the Library to Generate Poems

At this point, you can use the `generator` class to generate a poem as follows.  
Disclaimer: the generator might not be able to write down the Kalevala :sweat_smile:
```shell
$ python
>>>
>>> from poem_generator.generator import PoemGenerator
>>> from poem_generator.io.config import PoemGeneratorConfiguration
>>>
>>> # initialize the generator
>>> config = PoemGeneratorConfiguration(lang="fi", style="modern")
>>> generator = PoemGenerator(config)
>>>
>>> # give keywords
>>> keywords = "rakkaus anarkia"
>>>
>>> # generate candidates for the first line with the `PoemGenerator.get_first_line_candidates` method
>>> first_line_candidates = generator.get_first_line_candidates(keywords)
>>> print(first_line_candidates.plain_text())
>>> Voi rakkaus, joka ei ole anarkia!
>>>
>>> # select the line by index and add it to the poem with the `PoemGenerator.add_line` method
>>> generator.add_line(first_line_candidates[0])
>>>
>>> # generate candidates for the next line
>>> line_candidates = generator.get_line_candidates()
>>> print(line_candidates.plain_text())
>>> Ja liikehuolissansa
>>> Ja liikehuolissani
>>> Ja liikehuolten maljassa ottakaa!
>>>
>>> # add the selected line to the poem with the `PoemGenerator.add_line` method
>>> generator.add_line(line_candidates[0])
>>>
>>> # print the poem
>>> print(generator.state.plain_text())
>>> Voi rakkaus, joka ei ole anarkia!
>>> Ja liikehuolissansa
```

A simple command-line utilities to generate poems can be found in `example.py`. You can run the python script as follows:
```shell
python example.py
```
Then you can just follow the instructions from the prompt!

### Web Server

To run the web server, run the following:
```shell
export API_KEY="supersecret"
uvicorn app:app --host 0.0.0.0
```

At the moment the API is not documented, but we provide example scripts in the `scripts/api_test` folder.

## Repository Files and Folders

 * `training_config` contains example configuration files for the `train_gen_model.py` script  
 * `jobs` stuff necessary to run batch jobs in Turso (probably needs changes to be use don CSC)
 * `models` a place where to store the text generation models to be used by the generator
 * `poem_generator` poem generator implementation
 * `legacy` old stuff - you should't care about this
 * `train_gen_model.py` script to train a generative model for next line generation
 * `example.py` example implementation of interactive poem generator
 * `test.py` unit tests
 * `train_data_check.py` utility to check for duplicates in training dataset

## How to fine-tune a seq2seq model

To run the `train_gen_model.py` script, you need to install all the required packages, and write your own configuration 
file. The file is parsed by the [configparser](https://docs.python.org/3/library/configparser.html) package, and example files are given in the folder `training_config`.  
Once you have your configuration file, you can start the fine-tuning by running
```shell
python train_gen_model.py training_config/<YOUR_CONFIG_FILE>.ini
```

If you want to run it in a batch job, you will probably want to activate a virtual environment before running the script.
If you are using anaconda, you can use the wrapper script `jobs/run_training.sh`, which activates the specified virtual 
environment before running python. You probably have to modify the script according to your environment settings.  
Example batch job scripts (tested in Turso) can be found in the `jobs` folder.

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

## Legacy stuff and utilities to produce the training dataset (not in this repo!)

#### `build_gutenberg_cache.py`

Script that must be run to generate the cache before launching the `build_finnish_poetry_corpus.py` script.

#### `build_finnish_poetry_corpus.py`

Script to get poems from Project-Gutenberg. It can be used for different languages, and I used it to get poetry books for
Finnish, English and Swedish. Books are stored in `corpus_raw/[LANG]`.

#### `gutenberg-poetry-v001.ndjson.gz`

Corpus downloaded from https://github.com/aparrish/gutenberg-poetry-corpus

#### `read_poems.py`

Tentative script to separate actual poetry in the books... Not usable, must be improved.

## Some TODOs

Problem: quite often, the output sequence is just a copy of the input sequence - or very similar.
Could this be due to the presence of songs in the training data? If so, there are probably many examples taken from
choruses. Potentially, this could lead to a high number of training examples where the source sentence is the same as 
the output sentence, explaining the problem.

I should check the overlapping between source and target sentences, and try to deduplicate them,e.g. by
 * check when they are ==
 * check number of overlapping token ngrams
 * using a dedicate package (onion?)
