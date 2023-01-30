"""Provide a module to concatenate quantum codes."""
from typing import Any
import numpy as np
from .code import Code


class ConcatCode(Code):
    """Class to create concatenated codes."""

    def __init__(self, *args: Any) -> None:
        """
        Construct a concatenated code.

        Parameters
        ----------
        *args:
            Can be on eof the following
        tuple[Code]
            A tuple of codes that will be concatenated in order.
        tuple[Code, int]
        The Code will be concatenated with itself.

        Raises
        ------
        TypeError
            DESCRIPTION.


        """
        if len(args) == 1 and type(args[0]) == tuple:
            self.concat_sequence = args[0]
        elif (len(args) == 2
              and type(args[0]) == Code
              and type(args[1]) == int):
            self.concat_sequence = tuple([args[0]]*args[1])
        else:
            raise TypeError

        for cd in self.concat_sequence:
            if cd.logical_xs is None or cd.logical_zs is None:
                cd.construct_logical_operators()

        # create generator matrix
        self._construct_generator_matrix()

        # after you have constructed the generator matrix for the concat
        # code, we can call the parent init
        super().__init__(self.generator_matrix)

    def _construct_generator_matrix(self):

        # assign the first two codes as code 1 and 2
        code1 = self.concat_sequence[0]

        for code2 in self.concat_sequence[1:]:
            # code2 = self.concat_sequence[1]

            # construct the concat generator matrix
            if code1.num_physical_qubits % code2.num_logical_qubits == 0:
                concat_generator_matrix = \
                    self._construct_generator_matrix_concat_k2_divides_n1(
                        code1, code2)
            else:
                concat_generator_matrix = \
                    self._construct_generator_matrix_concat_k2_not_divides_n1(
                        code1, code2)

            # create a code using the concat_generator_matrix
            code1 = Code(concat_generator_matrix)

        self.generator_matrix = concat_generator_matrix
        return self.generator_matrix

    def _construct_generator_matrix_concat_k2_divides_n1(self,
                                                         code1: Code,
                                                         code2: Code
                                                         ) -> Any:
        """
        Construct concatenatenated generators when k_2 divides n_1.

        Parameters
        ----------
        code1 : Code
            First code.
        code2 : Code
            Second code.

        Returns
        -------
        numpy.ndarray
            The generator matrix of the concatenated code.

        """
        n1 = code1.num_physical_qubits
        k1 = code1.num_logical_qubits
        m1 = code1.num_generators
        n2 = code2.num_physical_qubits
        k2 = code2.num_logical_qubits
        m2 = code2.num_generators
        # physical qubits of concatenated code
        n = int(n1*n2/k2)

        new_gens_x = np.zeros((int(n - k1), n), dtype=int)
        new_gens_z = np.zeros((int(n - k1), n), dtype=int)

        # First for each block of qubits, we associate the generators of code 2
        # number of blocks
        nB = int(n1/k2)
        # size of each block
        sB = n2
        # for each block
        for b in range(nB):
            # for each generator in C2
            for i in range(m2):
                # place it in a shifted manner
                new_gens_x[m2*b + i, sB*b:sB*(b+1)] = code2.generators_x[i]
                new_gens_z[m2*b + i, sB*b:sB*(b+1)] = code2.generators_z[i]

        # create the incomplete generator matrix
        new_gens = np.concatenate((new_gens_x, new_gens_z), axis=1)

        # Now we want to add the encoded generators of code1
        nB = int(n1/k2)
        sB = k2
        # for each generator
        for i in range(m1):
            g = code1.generator_matrix[i]

            encoded_g = np.zeros(2*n, dtype=int)

            # break into blocks
            for b in range(nB):
                gb_x = g[sB*b:sB*(b+1)]
                gb_z = g[n1 + sB*b:n1 + sB*(b+1)]

                # iterate over the entries
                for j in range(sB):
                    # first create shifted logical operators for each block
                    shifted_op_x = np.zeros(2*n, dtype=int)
                    shifted_op_x[n2*b:n2*(b+1)] = code2.logical_xs[j][:n2]
                    shifted_op_x[n+n2*b:n+n2*(b+1)] = code2.logical_xs[j][n2:]

                    shifted_op_z = np.zeros(2*n, dtype=int)
                    shifted_op_z[n2*b:n2*(b+1)] = code2.logical_zs[j][:n2]
                    shifted_op_z[n+n2*b:n+n2*(b+1)] = code2.logical_zs[j][n2:]

                    # Depending on what operator is at g[j], we include the
                    # correct logical operator into the encoded operator
                    if gb_x[j] and gb_z[j]:
                        encoded_g = (encoded_g + shifted_op_x +
                                     shifted_op_z) % 2
                    elif gb_x[j]:
                        encoded_g = (encoded_g + shifted_op_x) % 2
                    else:
                        encoded_g = (encoded_g + shifted_op_z) % 2

            new_gens[n-n1+i] = encoded_g

        return new_gens

    def _construct_generator_matrix_concat_k2_not_divides_n1(
            self,
            code1: Code,
            code2: Code
            ) -> Any:
        """
        Construct concatenatenated generators when k_2 does not divide n_1.

        Parameters
        ----------
        code1 : Code
            First code.
        code2 : Code
            Second code.

        Returns
        -------
        numpy.ndarray
            The generator matrix of the concatenated code.

        """
        n1 = code1.num_physical_qubits
        k1 = code1.num_logical_qubits
        m1 = code1.num_generators
        n2 = code2.num_physical_qubits
        k2 = code2.num_logical_qubits
        m2 = code2.num_generators

        n = int(n1*n2)
        k = int(k1*k2)

        new_gens_x = np.zeros((n-k, n), dtype=int)
        new_gens_z = np.zeros((n-k, n), dtype=int)

        nB = n1
        sB = n2
        # for each block
        for b in range(nB):
            # for each generator in C2
            for i in range(m2):
                # print(b*cd2.num_generators + i, sB*b,sB*(b+1)-1 )
                new_gens_x[b*m2 + i, sB*b:sB*(b+1)] = code2.generators_x[i]
                new_gens_z[b*m2 + i, sB*b:sB*(b+1)] = code2.generators_z[i]

        new_gens = np.concatenate((new_gens_x, new_gens_z), axis=1)

        for c in range(k2):
            for j in range(m1):
                g = code1.generator_matrix[j]
                encoded_g = np.zeros(2*n, dtype=int)
                for k in range(n1):
                    shifted_op_x = np.zeros(2*n, dtype=int)
                    shifted_op_x[n2*k:n2*(k+1)] = code2.logical_xs[c][:n2]
                    shifted_op_x[n+n2*k:n+n2*(k+1)] = code2.logical_xs[c][n2:]

                    shifted_op_z = np.zeros(2*n, dtype=int)
                    shifted_op_z[n2*k:n2*(k+1)] = code2.logical_zs[c][:n2]
                    shifted_op_z[n+n2*k:n+n2*(k+1)] = code2.logical_zs[c][n2:]

                    if g[k] and g[n1+k]:
                        encoded_g = (encoded_g + shifted_op_x +
                                     shifted_op_z) % 2
                    elif g[k]:
                        encoded_g = (encoded_g + shifted_op_x) % 2
                    elif g[n1+k]:
                        encoded_g = (encoded_g + shifted_op_z) % 2

                new_gens[n-n1*k2+c*code1.num_generators+j] = encoded_g

        return new_gens
