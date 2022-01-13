from __future__ import annotations

from typing import List, Optional, Union


class PoemLine:
    def __init__(self, text: Optional[str] = None):
        self.text = text

    def __str__(self):
        return "PoemLine({})".format(self.text)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if type(other) == PoemLine and str(other) == str(self):
            return True
        return False


class PoemLineListIterator:
    """Iterator class"""
    def __init__(self, candidate_list: PoemLineList):
        self._line_list = candidate_list
        self._index = 0

    def __next__(self):
        if self._index < len(self._line_list.to_list()):
            result = self._line_list.to_list()[self._index]
            self._index += 1
            return result
        raise StopIteration


class PoemLineList:
    def __init__(self, poem_lines: Optional[List[PoemLine]] = None):
        self._lines = poem_lines if poem_lines else list()

    def __len__(self):
        return len(self._lines)

    def __getitem__(self, item):
        if item < len(self):
            return self._lines[item]
        else:
            raise IndexError

    def __iter__(self):
        return PoemLineListIterator(self)

    def __str__(self):
        return "PoemLineList([{}])".format(", ".join(str(line) for line in self._lines))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if type(other) == PoemLineList and str(other) == str(self):
            return True
        return False

    def to_list(self):
        return list(self._lines)

    def add_lines(self, new_lines: Union[PoemLine, PoemLineList]):
        if type(new_lines) == PoemLine:
            self._lines.append(new_lines)
        elif type(new_lines) == PoemLineList:
            self._lines += new_lines
        else:
            raise TypeError

    def plain_text(self, separator: Optional[str] = "\n"):
        return separator.join(line.text for line in self._lines)
