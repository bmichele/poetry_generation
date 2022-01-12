import logging
import os
import random
from typing import List
from utils import remove_punct, tokenize

import torch
from transformers import (
    MBartTokenizer,
    MBartForConditionalGeneration,
    BartTokenizer,
    MBart50TokenizerFast,
    BartForConditionalGeneration,
)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)

SEQ_TO_SEQ_MODEL = "mbart"
LANGUAGE = "fi"
DATASET = "wikisource"
SPECIAL_TOKENS = []
CHECKPOINT = "checkpoint-13500"
MODEL = "facebook/mbart-large-cc25"

language_codes = {
    "en": "en_XX",
    "fi": "fi_FI",
    "sv": "sv_SE",
}

########################################################################################################################
# load model and tokenizer
########################################################################################################################

MODEL_DIR = os.path.join(
    os.environ["MODELDIR_UNI"],
    "finnishPoetryGeneration",
    "{}-{}-{}".format(DATASET, LANGUAGE, SEQ_TO_SEQ_MODEL),
)
if CHECKPOINT:
    MODEL_DIR = os.path.join(MODEL_DIR, CHECKPOINT)

if SEQ_TO_SEQ_MODEL == "mbart":
    if MODEL == "facebook/mbart-large-cc25":

        tokenizer = MBartTokenizer.from_pretrained(
            MODEL,
            src_lang=language_codes[LANGUAGE],
            tgt_lang=language_codes[LANGUAGE],
            additional_special_tokens=SPECIAL_TOKENS,
        )

    elif MODEL == "facebook/mbart-large-50":
        tokenizer = MBart50TokenizerFast.from_pretrained(
            MODEL,
            src_lang=language_codes[LANGUAGE],
            tgt_lang=language_codes[LANGUAGE],
            additional_special_tokens=SPECIAL_TOKENS,
        )
    else:
        raise NotImplementedError("MODEL {} not supported".format(MODEL))

    model = MBartForConditionalGeneration.from_pretrained(MODEL)
    model.config.decoder_start_token_id = tokenizer.lang_code_to_id[
        language_codes[LANGUAGE]
    ]
elif SEQ_TO_SEQ_MODEL == "bart" and MODEL == "facebook/bart-large":

    tokenizer = BartTokenizer.from_pretrained(
        MODEL, additional_special_tokens=SPECIAL_TOKENS
    )
    model = BartForConditionalGeneration.from_pretrained(MODEL)
else:
    raise NotImplementedError

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.resize_token_embeddings(len(tokenizer))
logging.info("Model vocab size is {}".format(model.config.vocab_size))
model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "pytorch_model.bin"), map_location=torch.device(device)
    )
)
model.to(device)


########################################################################################################################
# Helper functions
########################################################################################################################


def get_ngrams(text: str, n: int) -> List[str]:
    text_chars = "".join(c for c in text if c.isalpha())
    return [text_chars[i : i + n] for i in range(len(text_chars) - n)]


def ngram_similarity(string1: str, string2: str, n: int) -> float:

    ngrams_1 = set(get_ngrams(string1, n))
    ngrams_2 = set(get_ngrams(string2, n))

    common_ngrams = ngrams_1.intersection(ngrams_2)
    all_ngrams = ngrams_1.union(ngrams_2)

    return len(common_ngrams) / len(all_ngrams)


def token_similarity(string1: str, string2: str) -> float:

    tokens_1 = set([t.lower() for t in tokenize(remove_punct(string1))])
    tokens_2 = set([t.lower() for t in tokenize(remove_punct(string2))])

    common_tokens = tokens_1.intersection(tokens_2)
    all_tokens = tokens_1.union(tokens_2)

    return len(common_tokens) / len(all_tokens)


########################################################################################################################
# Generate text
########################################################################################################################


def get_next_line_candidates(
    input_line: str, keywords: List[str] = None, separator: str = ">>>SEP<<<"
) -> List[str]:
    if keywords:
        source = (
            " ".join(random.sample(keywords, max(len(keywords) - 1, 1)))
            + " "
            + separator
            + " "
            + input_line
        )
    else:
        source = input_line
    logging.debug(source)
    encoded = tokenizer.encode(
        source, padding="max_length", max_length=32, truncation=True
    )
    encoded = torch.tensor(encoded).unsqueeze(0).to(device)

    sample_outputs = model.generate(
        encoded,
        do_sample=True,
        max_length=16,
        num_beams=5,
        # repetition_penalty=5.0,
        early_stopping=True,
        num_return_sequences=5,
    )

    candidates = [
        tokenizer.decode(sample_output, skip_special_tokens=True)
        for sample_output in sample_outputs
    ]
    logging.info("Generated candidates {}".format(candidates))
    return candidates


def get_next_line(
    input_line: str, keywords: List[str] = None, separator: str = ">>>SEP<<<"
) -> str:

    candidates = get_next_line_candidates(input_line, keywords, separator)

    candidates = [
        candidate for candidate in candidates if remove_punct(candidate)
    ]  # make sure that lines containing only punctuation are excluded

    if not candidates:
        return input_line

    # compute character n-gram similarity and token similarity with the input line for every candidate
    scored_candidates = [
        (
            candidate,
            token_similarity(input_line, candidate),
            ngram_similarity(input_line, candidate, 3),
        )
        for candidate in candidates
    ]

    candidates = [candidate for candidate in scored_candidates if candidate[1] < 0.4]

    if not candidates:
        logging.debug(
            "No candidates with token similarity lower than threshold value. Returning candidate with lower "
            "similarity"
        )
        return sorted(scored_candidates, key=lambda x: x[1])[0][0]

    return sorted(candidates, key=lambda x: x[2])[0][0]


def iterative_generation(
    input_line: str,
    lines: int,
    keywords: List[str] = None,
    separator: str = ">>>SEP<<<",
) -> str:

    out = input_line
    last_line = input_line
    counter = 0
    stanza_length = 0
    while True:
        next_line = get_next_line(last_line, keywords, separator)
        if last_line[-1] == "!" and next_line[-1] == "!":
            next_line = next_line[:-1] + "."
        elif last_line[-1] == "." and next_line[-1] == ".":
            next_line = next_line[:-1] + ","
        logging.info("Generated poetry line '{}'".format(next_line))
        counter += 1
        out += "\n"
        out += next_line
        stanza_length += 1
        if next_line.strip()[-1] in [".", "!", "?"]:
            if counter >= lines:
                break
            elif stanza_length > 1:
                out += "\n"
                stanza_length = 0
        last_line = next_line

    return out


# Finnish example promts
prompts = [
    # presented in last email
    "Toivo aikaa armaampaa,",
    "Ei ole sanoja kuvaamaan sitä,",
    "Nauti luonnon ihanuutta,",
    # from seed model
    "Oi, amerikkalainen, onnettomuus sun hartioillas,",
    "Eläin on parempi, onnettomuus –",
    "Se on laki. Ase on lakiin uudistus,",
    "Ja ase, korvaus elämältä,",
    "Pelkästään vain on korvaus tuo!",
    "Päätyä elon, jäädä pelkästään muistostain?",
    "Sitt ’ on yksilön heikko päätyä,",
]

# # Swedish example prompts
# prompts = [
#     "Hoppas tiden är mer söt",
#     "Det finns inga ord för att beskriva det,"
#     "Njut av naturens skönhet,",
#     "Dimman på de branta kullarna,",
# ]

for prompt in prompts:
    out = iterative_generation(prompt, 7)
    with open("poetry_gen_fi_2.txt", "a") as f:
        f.write("#" * len(prompt) + "####\n")
        f.write("# {} #\n".format(prompt))
        f.write("#" * len(prompt) + "####\n\n")
        f.write(out)
        f.write("\n\n\n")

# English test, beam_size = 5
# out = iterative_generation("When the evening cloud prevails,", 10)
# print(out)
# # mbart
# # When the evening cloud prevails,
# # And the morning cloud decays,
# # And the moon is gone.
# # There's nothing left to do,
# # But the sun is shining bright,
# # And the nightingales are singing
# # In the silence of the night.
# # "I'm tired of the moonlight," he said,
# # "I'm tired of the moonlight."
# # "The moonlight," he said.
# # "And the moonlight," he said.
#
# out = iterative_generation("I have to cook my lunch,", 5)
# print(out)
# # mbart
# # I have to cook my lunch,
# # I have to bake my bread,
# # And bake my bread.
# # I've got to go and buy some bread,
# # And I've got to go and buy some bread,
# # And I've got to go and buy some bread
#
# out = iterative_generation("Generating nonsense poetry,", 10)
# print(out)
# # mbart
# # Generating nonsense poetry,
# # Into a kind of poetry,--
# # And the poet's voice, like a whisper
# # Of a bird's voice in the night,
# # And the voice of a bird's voice
# # Of a lion's voice.
# # "O thou, who in the fields of battle
# # Thou shalt find me," quoth he, "alone;
# # Thou shalt find me," quoth he, "alone."
# # And in the shadows of my bed,
# # I'll lay thee down to rest.
#
# out = iterative_generation("One day the winter will come,", 10)
# print(out)
# # mbart
# # One day the winter will come,
# # And the snow will fall.
# # We'll have to go up the hill,
# # And go up the valley,
# # And I'll tell you all.
# # I'd like to know what you are about,
# # And what you are saying.
# # You are not a fool.
# # I am a fool.
# # I am a fool.
# # I am a fool.
#
# out = iterative_generation("When Januar' wind was blawing cauld,", 5)
# print(out)
# # mbart
# # When Januar' wind was blawing cauld,
# # And winter's snow was fallin' o'er,
# # And the wind was blowin' far away,
# # And the moon was in the sky,
# # And the sun was in the sky,
# # And the stars were in the sky,

# Finnish test, beam_size = 20
# out = iterative_generation("Ei ole sanoja kuvaamaan sitä,", 6)
# print(out)
# Ei ole sanoja kuvaamaan sitä,
# Niinkuin muinoin runoelmassakaan.
# Niinkuin vanha Väinämöinenkin,
# Niinkuin kaunis Kaukomieli,
# laulaja iän-ikuinen,
# sanan virkkoi, noin nimesi:
# "Voi poloinen, päiviäni!

# out = iterative_generation("Ellei itse esiin nousiskaan,", 5)
# print(out)
# # ellei itse esiin nousiskaan,
# # ja kentiesi nousta tahdon.
# # Se nousee, kasvaa ja vaurastuu.
# # Vaan työläisnaisten on aika se,
# # mut työläisnaisten on aika se,
# # ja työläisnaisten on aika se,

# out = iterative_generation("Tässä istun, teen ihmisiä", 3)
# print(out)
# Tässä istun, teen ihmisiä
# Yhteen aikaan, yht'äkkiä.
# Vaan ei se nyt niin tyyntä ollut,
# Joss' ei ollut tyyntä ja rauhaisaa,
# Ja laulajain laulua kuultiin,
# Ja laulua kuultiin ja soittiin,
# Ja laulua kuultiin ja soittiin,
# Ja laulua kuultiin ja soittiin,
# Ja laulua kuultiin ja soittiin,
# Ja laulua kuultiin ja soittiin,
# Ja laulua kuultiin ja soittiin,

# out = iterative_generation(
#     "Nauti luonnon ihanuutta,", 5, keywords=["johannus", "koulu"],
# )
# print(out)
# without keywords
# Nauti luonnon ihanuutta,
# Jota nautit ainiaan.
#
# Vaan jos kerran, kerran valitustas
# Sä heräjät, vaikenee valitus,
# Ja turhat huolet häipyy hiljalleen.
#
# Vaan turhat on murheet ja murheellinen
# Ja turhat on surut ja toivottomuus.

# keywords=["johannus", "koulu"]
# Nauti luonnon ihanuutta,
# kun on kaunis kesäilta.
#
# Niin kaunis onpi johannus,
# kun kukkii puisto, koulu.
#
# Ja siellä on niin onnekasta,
# niin hilpeä, herttainen.
