import os
from typing import List

from fastapi import Depends, FastAPI, Security, HTTPException
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel

from poem_generator.generator import (
    PoemGenerator,
    PoemGeneratorConfiguration,
    PoemLineList,
    PoemLine,
)

app = FastAPI()

api_key = APIKeyHeader(name="key")


class Token(BaseModel):
    key: str
    is_valid: bool


class GeneratorRequest(BaseModel):
    # poem_state: List[str]
    language: str
    style: str


class GeneratorFirstLineRequest(GeneratorRequest):
    keywords: str


class GeneratorNextLineRequest(GeneratorRequest):
    poem_state: List[str]


class Candidate(BaseModel):
    poem_line: str
    poem_state: List[str]


class GeneratorResponse(BaseModel):
    candidates: List[Candidate]


class Error(BaseModel):
    status: str
    title: str


def is_authorized(oauth_header: str = Security(api_key)):
    token = Token(key=oauth_header, is_valid=oauth_header == os.environ["API_KEY"])
    return token


# Initialize poem generator
logger.info("Initializing en generator")
config_en = PoemGeneratorConfiguration(lang="en", style="")
generator_en = PoemGenerator(config_en)
# logger.info("Initializing fi generator")
# config_fi = PoemGeneratorConfiguration(lang="fi", style="")
# generator_fi = PoemGenerator(config_fi)
# logger.info("Initializing sv generator")
# config_sv = PoemGeneratorConfiguration(lang="sv", style="")
# generator_sv = PoemGenerator(config_sv)

lang_generator = {
    "en": generator_en,
    # "fi": generator_fi,
    # "sv": generator_sv,
}


@app.get("/health")
def health():
    return "OK!"


@logger.catch
@app.post(
    "/generator/first_line",
    response_model=GeneratorResponse,
    description="Generate candidates for first line",
)
def get_first_line_candidates(
    data: GeneratorFirstLineRequest, auth: Token = Depends(is_authorized)
):
    if not auth.is_valid:
        logger.error(f"Unauthorized token: {auth.key}")
        raise HTTPException(status_code=403, detail="Unauthorized")

    logger.debug(f"Input keywords: {data.keywords}")
    candidates = lang_generator[data.language].get_first_line_candidates(
        keywords=data.keywords
    )
    return GeneratorResponse(
        candidates=[
            Candidate(poem_line=candidate.text, poem_state=[candidate.text])
            for candidate in candidates
        ]
    )


@logger.catch
@app.post(
    "/generator/next_line",
    response_model=GeneratorResponse,
    description="Generate candidates for next line",
)
def get_next_line_candidates(
    data: GeneratorNextLineRequest, auth: Token = Depends(is_authorized)
):
    if not auth.is_valid:
        logger.error(f"Unauthorized token: {auth.key}")
        raise HTTPException(status_code=403, detail="Unauthorized")

    logger.info(f"Input poem state: {data.poem_state}")
    lang_generator[data.language].state = PoemLineList(
        [PoemLine(line) for line in data.poem_state]
    )
    candidates = lang_generator[data.language].get_line_candidates()
    return GeneratorResponse(
        candidates=[
            Candidate(
                poem_line=candidate.text, poem_state=data.poem_state + [candidate.text]
            )
            for candidate in candidates
        ]
    )
