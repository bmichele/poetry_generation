from poem_generator.generator import PoemGenerator
from poem_generator.io.config import PoemGeneratorConfiguration
import logging
import argparse

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-lang", type=str, help="Language [fi, sv, en]")
    parser.add_argument("-style", type=str, help="Style of poem")
    args = parser.parse_args()
    # TODO: add check on parsed arguments
    config = PoemGeneratorConfiguration(lang=args.lang, style=args.style)

    generator = PoemGenerator(config=config)
    return generator.get_first_line_candidates()


if __name__ == "__main__":
    out = main()
    print(out.plain_text())
