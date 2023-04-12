"""Module to provide annotations."""
from typing import Union, Iterator
from .instruction import Instruction


class Annotation(Instruction):
    """Class to represent circuit annotations."""

    def __init__(self,
                 name: str,
                 targets: list = []
                 ) -> None:
        """
        Construct annotation object.

        Parameters
        ----------
        name : str
            Name of annotation.
        targets : list, optional
            Any targets this annotation has. The default is [].

        """
        self.name = name
        self.targets = targets.copy()

    def __repr__(self) -> str:
        """Return a representation of the object."""
        s = self.name
        s += ' ' + ' '.join([str(t) for t in self.targets])
        return s

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return self.__repr__()

    def copy(self) -> 'Annotation':
        """Return copy of object."""
        copied_ann = Annotation.__new__(Annotation)

        copied_ann.name = self.name
        copied_ann.targets = self.targets

        return copied_ann


class AnnotationSlice():
    """Class to create and manipulate annotation slices."""

    def __init__(self,
                 new_ann: Annotation = None) -> None:
        """
        Construct an AnnotationSlice.

        Parameters
        ----------
        new_ann : Annotation, optional
            This annotation will be appended to this slice. The default is
            None.
        """
        self.elements: list[Annotation] = []

        if new_ann is not None:
            self.append(new_ann)

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return '\n'.join([str(ann) for ann in self.elements])

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return self.__repr__()

    def __iter__(self) -> Iterator[Annotation]:
        """Return iterator of the AnnotationSlice."""
        return self.elements.__iter__()

    def __getitem__(self, ind) -> Union[Annotation, list[Annotation]]:
        """Make Timepoint subscriptable."""
        return self.elements.__getitem__(ind)

    def __len__(self) -> int:
        """Return number of annotations in the AnnotationSlice."""
        return len(self.elements)

    def copy(self) -> 'AnnotationSlice':
        """Return a copy of the AnnotationSlice."""
        copied_anns = AnnotationSlice()
        for ann in self.elements:
            copied_anns.append(ann.copy())

        return copied_anns

    def append(self,
               new_ann: Annotation) -> None:
        """
        Append operation to this AnnotationSlice.

        Parameters
        ----------
        new_ann : Annotation
            Annotation to append.

        """
        self.elements.append(new_ann.copy())

    def __add__(self,
                other: 'AnnotationSlice') -> 'AnnotationSlice':
        """
        Create sum of this AnnotationSlice and other AnnotationSlice.

        Parameters
        ----------
        other : AnnotationSlice
            AnnotationSlice to be added.

        Returns
        -------
        anns : AnnotationSlice
            Summed AnnotationSlice.

        """
        anns = self.copy()

        for ann in other:
            anns.append(ann.copy())

        return anns

    def __iadd__(self,
                 other: 'AnnotationSlice') -> 'AnnotationSlice':
        """
        Add other AnnotationSlice to this AnnotationSlice.

        Parameters
        ----------
        other : AnnotationSlice
            AnnotationSlice to be added.

        Returns
        -------
        AnnotationSlice
            Summed AnnotationSlice.

        """
        for ann in other:
            self.append(ann.copy())

        return self
