import copy


class Operation:
    def __init__(self,
                 name,
                 targets,
                 controls=None,
                 classical_control=None):
        self.name = name
        self.targets = targets.copy()
        self.num_affected_qubits = len(targets)
        self.affected_qubits = set(targets.copy())

        if controls is not None:
            self.is_controlled = True
            self.controls = controls
            self.control_state = '1'*len(controls)
            self.num_affected_qubits += len(controls)
            self.affected_qubits |= set(controls.copy())
            self.draw_str_control = '●'
        else:
            self.is_controlled = False

        self.draw_str_target = None

    def __str__(self):
        if self.is_controlled:
            return ' '.join([self.name,
                            str(self.controls[0]),
                            str(self.targets[0])])
        else:
            return ' '.join([self.name,
                            str(self.targets[0])])

    def remap_qubits(self, qubit_map):
        new_targets = [qubit_map.get(q, q) for q in self.targets]
        if self.is_controlled:
            new_controls = [qubit_map.get(q, q) for q in self.controls]
        else:
            new_controls = None
        return Operation(self.name, new_targets, controls=new_controls)
