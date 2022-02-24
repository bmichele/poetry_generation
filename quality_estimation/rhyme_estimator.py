import pronouncing
import editdistance

class RhymeEstimator:
    def rhyming(self, line1, line2):
        return line1.split()[-1] in pronouncing.rhymes(line2.split()[-1])
