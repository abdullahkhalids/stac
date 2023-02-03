"""Provides collection of some codes."""
import numpy as np
from .code import Code


class CommonCodes:
    """Class to provide some common codes."""

    def __init__(self):
        """Use the generate_code method to create codes."""
        pass

    @classmethod
    def generate_code(cls,
                      codename: str) -> Code:
        """
        Generate an internally stored Code.

        Parameters
        ----------
        codename : str
            Can be one of the following.
                * [[7,1,3]]
                * [[5,1,3]]
                * [[4,2,2]]
                * [[8,3,3]]
                * [[6,4,2]]

        Raises
        ------
        Exception
            If codename is not recognized.

        Returns
        -------
        Code
            The corresponding code.

        """
        if codename == '[[7,1,3]]':
            return cls._Steane()
        elif codename == '[[5,1,3]]':
            return cls._Code513()
        elif codename == '[[4,2,2]]':
            return cls._Code422()
        elif codename == '[[8,3,3]]':
            return cls._Code833()
        elif codename == '[[6,4,2]]':
            return cls._Code642()
        else:
            raise Exception("Code not found. See method help.")

    @classmethod
    def _Steane(cls) -> Code:
        hamming = np.array([
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 1, 0, 1]
        ], dtype=int)

        zeroM = np.zeros(hamming.shape, dtype=int)

        Sx = np.concatenate((hamming, zeroM))
        Sz = np.concatenate((zeroM, hamming))

        c = Code(Sx, Sz)
        c.distance = 3

        return c

    @classmethod
    def _Code513(cls) -> Code:

        Sx = np.array([
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0]
        ], dtype=int)

        Sz = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1]
        ], dtype=int)

        c = Code(Sx, Sz)
        c.distance = 3

        return c

    @classmethod
    def _Code833(cls) -> Code:
        Sx = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0, 0, 1],
        ], dtype=int)
        Sz = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ], dtype=int)

        c = Code(Sx, Sz)
        c.distance = 3

        return c

    @classmethod
    def _Code422(cls) -> Code:

        Sx = np.array([
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ], dtype=int)

        Sz = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ], dtype=int)

        c = Code(Sx, Sz)
        c.distance = 2

        return c

    @classmethod
    def _Code642(cls) -> Code:

        Sx = np.array([
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0]
        ], dtype=int)

        Sz = np.array([
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1]
        ], dtype=int)

        c = Code(Sx, Sz)
        c.distance = 2

        return c
