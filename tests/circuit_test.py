import stac

def test_circuit_init():
    circ = stac.Circuit()
    assert type(circ) == stac.circuit.Circuit
