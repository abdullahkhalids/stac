# Operations in stac

Stac uses stim to sample from circuits. Currently, the operations accepted by stac are a subset of those [accepted by stim](https://github.com/quantumlib/Stim/blob/main/doc/gates.md).

## Single-qubit gates with no parameters

Most of the following the self-evident Clifford operations.

* 'I'

* 'X'

* 'Y'

* 'Z'

* 'H'

* 'S'

* 'T'

* 'CAT'
  This is a special operation that is used in template circuits. Typically, if a code is being concatenated, the 'H' gate that appears at level 0 is not a 'H' gate but a cat state creation operation, something that becomes apparent at level 1 and higher.

## Single-qubit parameterized operations
All of the following take one `float` parameter as input.

* 'RX'
  Rotate qubit around the x-axis on the Bloch sphere.

* 'RY'
  Rotate qubit around the y-axis on the Bloch sphere.

* 'RZ'
  Rotate qubit around the z-axis on the Bloch sphere.

## Single-qubit measurements and resets
These operations do not take in any parameters

* 'R'
  Resets the qubit to the zero state.

* 'M'
  Measures the qubit.

* 'MR'
  An 'M' operation followed by a 'R' operation.

## Two-qubit gates without parameters
These operations take two targets.

* 'CX'

* 'CY'

* 'CZ'

## Annotations
Stac also supports stim style annotations.

* 'TICK'
  This has two purposes. You can divide a circuit into distinct parts by using TICK as a separator. When the circuit is drawn, TICKs will be drawn as dotted lines. Additionally, in the `Circuit.simulate` function, TICKs can be used to progressively simulate parts of the circuit.




