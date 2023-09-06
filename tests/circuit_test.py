import stac


def test_circuit_init():
    circ = stac.Circuit()
    assert type(circ) == stac.circuit.Circuit


def test_circuit_init2():
    circ = stac.Circuit.simple(3)
    assert type(circ) == stac.circuit.Circuit


cd = stac.CommonCodes.generate_code('[[7,1,3]]')
circ = stac.Circuit()
circ.append_register(cd.construct_data_register(0))
circ.append_register(cd.construct_syndrome_measurement_register(0, 'non_ft'))


def test_append_basic():
    circ.append('H', (0, 0, 0))
    circ.append('CX', (0, 0, 1), (0, 1, 0, 0))
    assert circ.__repr__() == '0 H (0, 0, 0)\n  CX (0, 0, 1) (0, 1, 0, 0)'


def test_append_next_timepoint():
    circ.append('H', (0, 0, 1))
    circ.append('X', (0, 0, 2), time=[1])
    assert circ.__repr__(
    ) == '0 H (0, 0, 0)\n  CX (0, 0, 1) (0, 1, 0, 0)\n1 H (0, 0, 1)\n2 X (0, 0, 2)'


def test_append_past():
    circ.append('Y', (0, 0, 3), time=1)
    assert circ.__repr__() == '0 H (0, 0, 0)\n  CX (0, 0, 1) (0, 1, 0, 0)\n1 H (0, 0, 1)\n  Y (0, 0, 3)\n2 X (0, 0, 2)'


def test_append_future():
    circ.append('Z', (0, 0, 4), time=6)
    assert circ.__repr__() == '0 H (0, 0, 0)\n  CX (0, 0, 1) (0, 1, 0, 0)\n1 H (0, 0, 1)\n  Y (0, 0, 3)\n2 X (0, 0, 2)\n3\n4\n5\n6 Z (0, 0, 4)'


def test_append_set_cur_time():
    circ.cur_time = 4
    circ.append('CY', (0, 0, 2), (0, 0, 5))
    assert circ.__repr__() == '0 H (0, 0, 0)\n  CX (0, 0, 1) (0, 1, 0, 0)\n1 H (0, 0, 1)\n  Y (0, 0, 3)\n2 X (0, 0, 2)\n3\n4 CY (0, 0, 2) (0, 0, 5)\n5\n6 Z (0, 0, 4)'


def test_append_parameterized_gate():
    circ.append('RX', (0, 0, 5), 0.3)
    assert circ.__repr__() == '0 H (0, 0, 0)\n  CX (0, 0, 1) (0, 1, 0, 0)\n1 H (0, 0, 1)\n  Y (0, 0, 3)\n2 X (0, 0, 2)\n3\n4 CY (0, 0, 2) (0, 0, 5)\n5 RX(0.3) (0, 0, 5)\n6 Z (0, 0, 4)'
