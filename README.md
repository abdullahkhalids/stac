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

## Development version
Stac is currently undergoing a complete overwrite to make it suitable for
constructing fault-tolerant circuits. The goals and progress (18 Jan 2023)

[x] There is an intrinsic notion of encoded qubits at any concatenation
    level in the circuit. User can create registers of such qubits.

[x] The `append` function can apply a logical operation to qubits at any level
    of concatentation. The resultant operation is automatically compiled down
    to the physical qubits.

[ ] The user can construct a fault-tolerant circuit for any stabilizer 
    code using a few lines of code. (Works for codes with k=1 currently)

[ ] Provide the user with a rich assembly language to construct custom 
    fault-tolerant circuits (basic functionality present but needs improvement)

[ ] Export a stim circuit that can be used to compute the threshold of the 
    code.

Documentation will be made available once the developmental version is ready
to be merged into `main`. 

## Getting started
Please refer to my [recent blog posts](https://abdullahkhalid.com/blog/) which
illustrate its usage. 


