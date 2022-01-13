from poem_generator.generators.next_line import mbart_fi_single_line
from poem_generator.io.candidates import PoemLine, PoemLineList
from poem_generator.io.config import PoemGeneratorConfiguration


class PoemGenerator:
    def __init__(self, config: PoemGeneratorConfiguration):
        self.config = config
        self.state = PoemLineList()
        self.tokenizer, self.model = self.get_tokenizer_and_model()

    def get_tokenizer_and_model(self):
        if self.config.lang == "fi":
            return mbart_fi_single_line.get_tokenizer_and_model()
        else:
            raise NotImplementedError

    def get_first_line_candidates(self) -> PoemLineList:
        # TODO: implement
        return PoemLineList([PoemLine(text="La nebbia agli irti colli,")])

    def get_line_candidates(self) -> PoemLineList:
        if self.config.lang == "fi":
            return mbart_fi_single_line.generate(self.state, self.tokenizer, self.model)
        else:
            raise NotImplementedError

    def add_line(self, line: PoemLine):
        self.state.add_lines(line)
