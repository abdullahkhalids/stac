"""Provide module for blocks of circuit operations."""
from typing import Iterator, Union, Any


class InstructionBlock:
    """Class for creating and manipulating blocks of circuit instructions."""

    def __init__(self):
        self.elements: list = []

    def __repr__(self) -> str:
        """Return a representation of the block."""
        return '\n'.join([str(el) for el in self.elements])

    def __str__(self) -> str:
        """Return a string representation of the block."""
        return self.__repr__()

    def __iter__(self) -> Iterator:
        """Return iterator of the block."""
        return self.elements.__iter__()

    def __getitem__(self, ind) -> Union[Any, list[Any]]:
        """Make Timepoint subscriptable."""
        return self.elements.__getitem__(ind)

    def __len__(self) -> int:
        """Return number of operations in the block."""
        return len(self.elements)

    def insert(self, i, ins) -> int:
        """Insert instruction at particular index."""
        self.elements.insert(i, ins)

    def copy(self) -> 'InstructionBlock':
        """Return a copy of the block."""
        copied_ib = InstructionBlock()
        for el in self.elements:
            copied_ib.append(el.copy())

        return copied_ib

    def append(self, obj) -> None:
        """Append object."""
        self.elements.append(obj)


class AnnotationBlock(InstructionBlock):
    """Class to create blocks that hold annotations."""

    pass


class RepetitionBlock(InstructionBlock):
    """Class to create blocks of repeating instructions."""

    def __init__(self,
                 repetitions: int
                 ) -> None:
        self.repetitions = repetitions
        super().__init__()


class IfBlock(InstructionBlock):
    """Class to store conditional instructions."""

    def __init__(self):
        pass
