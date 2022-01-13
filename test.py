import logging
from unittest import TestCase

from poem_generator.io.candidates import PoemLine
from poem_generator.io.candidates import PoemLineList

logger = logging.getLogger(__name__)


class TestPoemLine(TestCase):
    def test_text(self):
        my_candidate = PoemLine()
        my_candidate.text = "test"
        self.assertEqual(my_candidate.text, "test")
        del my_candidate
        my_candidate = PoemLine(text="test 2")
        self.assertEqual(my_candidate.text, "test 2")

    def test__eq__(self):
        my_candidate = PoemLine("this")
        my_candidate_eq = PoemLine("this")
        my_candidate_diff = PoemLine("another")
        self.assertEqual(my_candidate, my_candidate_eq)
        self.assertNotEqual(my_candidate, my_candidate_diff)


class TestPoemLineList(TestCase):
    def test_candidates(self):
        my_line_list = PoemLineList(poem_lines=[PoemLine("a"), PoemLine("b"), PoemLine("test")])
        self.assertEqual(my_line_list[0], PoemLine("a"))
        self.assertEqual(my_line_list[1], PoemLine("b"))
        self.assertEqual(my_line_list[2], PoemLine("test"))

    def test__eq__(self):
        my_line_list = PoemLineList(poem_lines=[PoemLine("a"), PoemLine("b")])
        my_line_list_eq = PoemLineList(poem_lines=[PoemLine("a"), PoemLine("b")])
        my_line_list_diff = PoemLineList(poem_lines=[PoemLine("a"), PoemLine("c")])
        self.assertEqual(my_line_list, my_line_list_eq)
        self.assertNotEqual(my_line_list, my_line_list_diff)

    def test_to_list(self):
        my_line_list = PoemLineList(poem_lines=[PoemLine("a"), PoemLine("b")])
        self.assertEqual(my_line_list.to_list(), [PoemLine("a"), PoemLine("b")])

    def test_add_candidates(self):
        my_line_list = PoemLineList(poem_lines=[PoemLine("a"), PoemLine("b")])
        my_line_list.add_lines(PoemLine("c"))
        final_line_list = PoemLineList([PoemLine("a"), PoemLine("b"), PoemLine("c")])
        self.assertEqual(my_line_list, final_line_list)
