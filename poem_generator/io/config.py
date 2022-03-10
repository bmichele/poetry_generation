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
        """
        Configuration of neural model to be used for generation of candidates.

        :param base_model: base model used for fine-tuning
        :param model_file: checkpoint file of fine-tuned model
        :param lang: language
        :param special_tokens: special tokens added when fine-tuning the model
        """
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
        temperature: Optional[float] = 1.,
        top_k: Optional[int] = 50,
        num_beams: Optional[int] = 5,
        batch_multiply: Optional[int] = None,
        early_stopping: Optional[bool] = False,
        num_return_sequences: Optional[int] = 5,
        remove_duplicate_candidates: Optional[bool] = True,
    ):
        """
        Configuration for candidate generation.

        :param src_builder: Function used to obtain the seq2seq model source from the poem state
        :param src_max_length: Max source langth supported by the model
        :param truncation:
        :param out_max_length:
        :param do_sample:
        :param temperature:
        :param top_k:
        :param num_beams:
        :param batch_multiply:
        :param early_stopping:
        :param num_return_sequences:
        :param remove_duplicate_candidates:
        """
        self.src_builder = src_builder
        self.src_max_length = src_max_length
        self.truncation = truncation
        self.out_max_length = out_max_length
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.num_beams = num_beams
        self.batch_multiply = batch_multiply
        self.early_stopping = early_stopping
        self.num_return_sequences = num_return_sequences
        self.remove_duplicate_candidates = remove_duplicate_candidates


class PoemGeneratorConfiguration:
    def __init__(
        self,
        lang: Optional[str] = None,
        style: Optional[str] = None,
        next_line_model_config: Optional[ModelConfig] = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        """Configuration for PoemGenerator objects."""
        self.lang = lang
        self.style = style
        self.next_line_model_config = next_line_model_config
        self.generation_config = generation_config
