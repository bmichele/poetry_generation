from typing import Callable, List, Optional
from poem_generator.io.candidates import PoemLineList


class ModelConfig:
    def __init__(
        self,
        base_model: str,
        model_file: str,
        lang: str,
        special_tokens: Optional[List[str]] = None,
    ):
        self.base_model = base_model
        self.model_file = model_file
        self.lang = lang
        self.special_tokens = special_tokens


class GenerationConfig:
    def __init__(
        self,
        src_builder: Callable[[PoemLineList], str],
        src_max_length: Optional[int] = 32,
        truncation: Optional[bool] = True,
        out_max_length: Optional[int] = 32,
        do_sample: Optional[bool] = True,
        num_beams: Optional[int] = 5,
        batch_multiply: Optional[int] = None,
        early_stopping: Optional[bool] = False,
        num_return_sequences: Optional[int] = 5,
    ):
        self.src_builder = src_builder
        self.src_max_length = src_max_length
        self.truncation = truncation
        self.out_max_length = out_max_length
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.batch_multiply = batch_multiply
        self.early_stopping = early_stopping
        self.num_return_sequences = num_return_sequences


class PoemGeneratorConfiguration:
    def __init__(
        self,
        lang: Optional[str] = None,
        style: Optional[str] = None,
        next_line_model_config: Optional[ModelConfig] = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        self.lang = lang
        self.style = style
        self.next_line_model_config = next_line_model_config
        self.generation_config = generation_config
