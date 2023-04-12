# stac
Stac allows you to generate and simulate quantum stabilizer codes. If you give
it a generator matrix, it will generate a code. You can then,

* check if the code is valid
* convert to the standard form of the generator matrix
* generate an encoding circuit
* generate a decoding circuit
* generate syndrome measurment circuit

All circuits are built using the provided Circuit class. Circuits have a
number of useful functions. You can

* draw the circuits.
* annotate the circuit with errors. This is useful for reasoning about how
  errors effect the circuits of quantum codes.
* simulate circuits using either Qiskit or Stim.
* The above can be harnessed to study the effects of errors on circuits.
* circuits can be exported to qasm, stim or quirk style circuits.

Stac is currently under active development and subject to rapid change.

## Getting started
Please refer to my [recent blog posts](https://abdullahkhalid.com/blog/) which
illustrate its usage. 

## Credits

Thanks for Unitary Fund for funding part of the development of this project.
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](https://unitary.fund)
