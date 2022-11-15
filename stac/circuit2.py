from .operation import Operation
from .timepoint import Timepoint
from .qubit import PhysicalQubit, CircuitQubit
from .register import Register, QubitRegister, RegisterRegister


class timepoint_Circuit:
    def __init__(self):
        """Construct a quantum circuit."""
        self._timepoints = []
        self.num_qubits = 0

        self.registers = [
            []
        ]

    def __str__(self):
        return '\n'.join([str(tp) for tp in self._timepoints])

    def __iter__(self):
        """Return iterator for the quantum circuit."""
        for tp in self._timepoints:
            for op in tp:
                yield op

    def append(self, *args):
        if len(args) == 1 and type(args[0]) is Operation:
            op = args[0]
        elif len(args) == 2:
            op = Operation(args[0], [args[1]])
        else:
            op = Operation(args[0], [args[2]], [args[1]])

        if len(self._timepoints) == 0 or (not self._timepoints[-1].append(op)):
            tp = Timepoint(op)
            self._timepoints.append(tp)

        mq = max(op.affected_qubits)
        if mq + 1 > self.num_qubits:
            self.num_qubits = mq+1

    def _append_tp(self, tp):
        self._timepoints.append(tp)
        mq = max(tp.affected_qubits)
        if mq + 1 > self.num_qubits:
            self.num_qubits = mq+1

    def append_level0_qubitregister(self, register_type, num_qubits):
        index = len(self.registers[0])
        qr = QubitRegister(index, register_type, 0, num_qubits)
        self.registers[0].append(qr)

    def compose(self, circuit2, *args):
        qubit_map = dict()
        if args[0] == 'below':
            self._compose_below(circuit2)
            return
        elif type(args[0]) is int:
            for i in range(circuit2.num_qubits):
                qubit_map[i] = i + args[0]
            if circuit2.num_qubits + args[0] > self.num_qubits:
                self.num_qubits = circuit2.num_qubits + args[0]

        elif type(args[0]) is list:
            if len(args[0]) != circuit2.num_qubits:
                raise ValueError

            for i in range(circuit2.num_qubits):
                qubit_map[i] = args[0][i]
                if args[0][i] + 1 > self.num_qubits:
                    self.num_qubits = args[0][i] + 1
        else:
            raise ValueError

        # fix to append timepoints to respect user layout
        for tp in circuit2._timepoints:
            self._append_tp(tp.remap_qubits(qubit_map))

    def _compose_below(self, circuit2):
        qubit_map = dict()
        for i in range(circuit2.num_qubits):
            qubit_map[i] = i + self.num_qubits

        self.num_qubits += circuit2.num_qubits

        k = 0
        for tp2 in circuit2._timepoints:
            new_tp2 = tp2.remap_qubits(qubit_map)
            if k == len(self._timepoints):
                self._timepoints.append(Timepoint())
            tp = self._timepoints[k]
            while not tp.add(new_tp2):
                k += 1
                if k == len(self._timepoints):
                    self._timepoints.append(Timepoint())
                tp = self._timepoints[k]
            k += 1

    def draw(self, filename=None):
        dash = '─'
        space = ' '
        vert = '│'
        init_num_len = len(str(self.num_qubits))
        circ_disp = [list(str(i).rjust(init_num_len, space)+space)
                     for i in range(self.num_qubits)]
        circ_disp2 = [list(space*(init_num_len+1))
                      for _ in range(self.num_qubits)]

        for tp in self._timepoints:

            slices = [[]]
            slices_touched_qubits = [[]]
            for op_id, op in enumerate(tp.operations):

                if not op.is_controlled:
                    touched_by_op = [op.targets[0]]
                else:
                    touched_by_op = list(range(op.controls[0], op.targets[0]))\
                        + list(range(op.targets[0], op.controls[0]))
                    touched_by_op.append(touched_by_op[-1]+1)

                for s in range(len(slices)):
                    if len(
                            set(touched_by_op).intersection(
                                set(slices_touched_qubits[s]))) == 0:
                        slices[s].append(op)
                        slices_touched_qubits[s] += touched_by_op
                        break
                else:
                    slices.append([op])
                    slices_touched_qubits.append(touched_by_op)

            for sl in slices:
                touched_places = []

                for op in sl:
                    if not op.is_controlled:
                        s = dash + op.name + dash
                        circ_disp[op.targets[0]].append(s)
                        circ_disp2[op.targets[0]].append(space*3)
                        touched_places.append(op.targets[0])

                    elif op.is_controlled:
                        vert_places = list(range(
                            op.controls[0], op.targets[0])) \
                            + list(range(op.targets[0], op.controls[0]))
                        for i in range(self.num_qubits):
                            if i == op.controls[0]:
                                circ_disp[i].append(dash + '●' + dash)
                                if i == vert_places[0]:
                                    circ_disp2[i].append(space + vert + space)
                                else:
                                    circ_disp2[i].append(space*3)
                                touched_places.append(i)
                            elif i == op.targets[0]:
                                circ_disp[i].append(dash + '⊕' + dash)
                                if i == vert_places[0]:
                                    circ_disp2[i].append(space + vert + space)
                                else:
                                    circ_disp2[i].append(space*3)
                                touched_places.append(i)
                            elif i in vert_places[1:]:
                                circ_disp[i].append(dash + '┼' + dash)
                                circ_disp2[i].append(space + vert + space)
                                touched_places.append(i)
                for i in range(self.num_qubits):
                    if i not in set(touched_places):
                        circ_disp[i].append(dash*3)
                        circ_disp2[i].append(space*3)

        circ_disp_str = ''
        for line1, line2 in zip(circ_disp, circ_disp2):
            circ_disp_str += ''.join(line1) + '\n'
            circ_disp_str += ''.join(line2) + '\n'

        if filename is None:
            print(circ_disp_str)
        else:
            with open(filename, 'w') as f:
                f.write(circ_disp_str)
