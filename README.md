# stac
Stac allows you to generate and simulate quantum stabilizer codes. It comes with
its own quantum circuit library that has been designed to make the process of
algorithmically creating quantum error-correction circuits and fault-tolerence
circuits as painless as possible. 

Stac also includes a stabilizer code library. If you give it a generator matrix
of any stabilizer code, you construct a code object. Code objects include 
algorithms for generating

* encoding circuits,
* decoding circuits,
* syndrome measurment circuits,

for the code, among other useful functions.

For these circuits or any other circuits you build using the circuit library,
you can

* draw the circuits.
* annotate the circuit with errors. This is useful for reasoning about how
  errors effect the circuits of quantum codes.
* simulate circuits using stim.
* export to qasm, stim or quirk.

One of the goals of stac (not there yet) is to compute the fault-tolerance
thresholds of simple stabilizer codes in "one-click".

## Getting started
To install stac, run

```
pip install git+https://github.com/abdullahkhalids/stac
```

Please refer to my [mini-book](https://abdullahkhalid.com/qecft/index.html) which
illustrates basic usage in action.

A short guide on more advanced Stac circuits is available 
[here](https://github.com/abdullahkhalids/stac/wiki/guide).

## Citation
If you have used stac in your work, please cite it as follows:

```
@misc{stac2024,
      author = {Abdullah Khalid},
      title = {Stac},
      year = {2024},
      howpublished = {\url{https://github.com/abdullahkhalids/stac}}
}
```

## Credits
Thanks for Unitary Fund for funding part of the development of this project.
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](https://unitary.fund)
