import numpy as np
import spacy
from quality_estimation.coherence_estimator import SyntacticCoherenceEstimator
from quality_estimation.coherence_estimator import SyntacticAnnotator
import os
import logging
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

########################################################################################################################
# Look into data
########################################################################################################################

DATADIR = os.path.join(os.environ["DATADIR_UNI"], "kaggle_poemsdataset/forms")

# # check how many files per folder
# for directory in os.listdir(DATADIR):
#     print("#############")
#     print(directory)
#     print("number of files: {}".format(len(os.listdir(os.path.join(DATADIR, directory)))))

# let's use the italia sonnets to begin with
# load the data
poems = []
for category in [dir for dir in os.listdir(os.path.join(DATADIR)) if dir != ".DS_Store"]:
# for category in ["italian-sonnet", "pastoral", "cinquain"]:
# for category in ["italian-sonnet"]:
    for file in os.listdir(os.path.join(DATADIR, category)):
        file_path = os.path.join(DATADIR, category, file)
        with open(file_path, "r") as f:
            poem = f.read()
        if poem:
            poems.append(poem)


# a bit of cleaning
def clean_poem(poem: str) -> str:
    poem_lines = [line.strip() for line in poem.split("\n")]
    poem_lines = [line for line in poem_lines if line]
    filtered_lines = []
    for line in poem_lines:
        logging.debug("#{}#".format(line))
        if (line[0] in ["(", "["]) or (line[-1] in [")", "]"]) or ("©" in line):
            logging.debug("Removing poem line: {}".format(line))
        else:
            filtered_lines.append(line)
    return "\n".join(filtered_lines)


poems = [clean_poem(poem) for poem in poems]

random.seed(42)
# keep = [random.random() > 0.5 for _ in range(len(poems))]
# poems = [poem for i, poem in enumerate(poems) if keep[i]]
keep_for_train = [random.random() < 0.6 for _ in range(len(poems))]
train_poems = [poem for i, poem in enumerate(poems) if keep_for_train[i]]
test_poems = [poem for i, poem in enumerate(poems) if not keep_for_train[i]]

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_entities")
annotator = SyntacticAnnotator(nlp)
estimator = SyntacticCoherenceEstimator(annotator, 20)

keep_stopwords = False
estimator.train(train_poems, stopwords=keep_stopwords)

max_ngram_prediction = 10
predictions = []
for i, poem in enumerate(train_poems):
    try:
        prediction = estimator.predict(poem, stopwords=keep_stopwords, max_ngram=max_ngram_prediction)
        predictions.append(prediction)
    except KeyError:
        logging.debug("Prediction failed for poem index {}".format(i))
average_log_probability = np.mean(predictions)
variance = np.var(predictions)
logging.info("train set: average log prob {} {} ({} poems)".format(average_log_probability, variance, len(predictions)))


predictions = []
for i, poem in tqdm(enumerate(test_poems)):
    try:
        prediction = estimator.predict(poem, stopwords=keep_stopwords, max_ngram=max_ngram_prediction)
        predictions.append(prediction)
    except:
        logging.debug("Prediction failed for poem index {}".format(i))
average_log_probability = np.mean(predictions)
variance = np.var(predictions)
logging.info("test set: average log prob {} {} ({} poems)".format(average_log_probability, variance, len(predictions)))


# shuffle lines in the poems
def shuffle_lines(poem: str)-> str:
    poem_lines = poem.split("\n")
    random.shuffle(poem_lines)
    return "\n".join(poem_lines)


shuffled_poems = [shuffle_lines(poem) for poem in test_poems]

predictions = []
for i, poem in tqdm(enumerate(shuffled_poems)):
    try:
        prediction = estimator.predict(poem, stopwords=keep_stopwords, max_ngram=max_ngram_prediction)
        predictions.append(prediction)
    except:
        logging.debug("Prediction failed for poem index {}".format(i))
average_log_probability = np.mean(predictions)
variance = np.var(predictions)
logging.info("shuffled lines: average log prob {} {} ({} poems)".format(average_log_probability, variance, len(predictions)))


# shuffle poems by mixing lines from different poems
poem_lengths = [len(poem.split("\n")) for poem in test_poems]
all_lines = [line for poem in test_poems for line in poem.split("\n")]
random.shuffle(all_lines)
mixed_poems = []
for length in poem_lengths:
    mixed_poem_lines = [all_lines.pop(0) for _ in range(length)]
    mixed_poems.append("\n".join(mixed_poem_lines))

predictions = []
for i, poem in tqdm(enumerate(mixed_poems)):
    try:
        prediction = estimator.predict(poem, stopwords=keep_stopwords, max_ngram=max_ngram_prediction)
        predictions.append(prediction)
    except:
        logging.debug("Prediction failed for poem index {}".format(i))
average_log_probability = np.mean(predictions)
variance = np.var(predictions)
logging.info("mixed lines: average log prob {} {} ({} poems)".format(average_log_probability, variance, len(predictions)))

# for grid in estimator.grids:
#     print(grid.array.transpose())
#
# from quality_estimation.coherence_estimator import EntityGrid
#
# tmp_grid = EntityGrid(annotator)
# tmp_grid.train(train_poems[-1])
