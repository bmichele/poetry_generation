from poem_generator.generator import PoemGenerator
from poem_generator.io.config import PoemGeneratorConfiguration
from poem_generator.io.candidates import PoemLine, PoemLineList


def main():
    # initialize the poem generator
    config = PoemGeneratorConfiguration(lang="fi", style="text")
    generator = PoemGenerator(config)

    # get candidates for the first line
    line_candidates = generator.get_first_line_candidates()

    while True:
        # print line candidates
        print("\nLINE CANDIDATES:\n{}".format(
            "\n".join(["\t" + str(i) + " " + candidate.text for i, candidate in enumerate(line_candidates)])))

        # ask the user for the best one
        allowed_choices = list(range(len(line_candidates)))
        print("\nPLEASE CHOOSE CANDIDATE (integer in {}, -1 to stop.)".format(allowed_choices))
        selection = int(input())
        while selection not in allowed_choices and selection != -1:
            print("PLEASE SELECT A CANDIDATE INDEX AMONG THESE: {} (-1 to stop)".format(allowed_choices))
            selection = int(input())
        if selection == -1:
            break

        # add the selected line
        generator.add_line(line_candidates[selection])

        # print current state of the poem
        print("\nCURRENT POEM STATE:")
        print("\t", generator.state.plain_text(separator="\n\t "))

        line_candidates = generator.get_line_candidates()

    print("\nGENERATED POEM:")
    print("\t", generator.state.plain_text(separator="\n\t "))


if __name__ == "__main__":
    main()
