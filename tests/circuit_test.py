import stac

def test_circuit_init():
    circ = stac.Circuit()
    assert type(circ) == stac.circuit.Circuit


def test_circuit_init2():
    circ = stac.Circuit.simple(3)
    assert type(circ) == stac.circuit.Circuit
