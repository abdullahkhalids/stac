import copy


class Operation:

    _draw_str_target_dic = {'CX': '⊕',
                            'CY': 'Y',
                            'CZ': '●',
                            'MR': 'm'}

    def __init__(self,
                 name,
                 targets,
                 controls=None,
                 classical_control=None):
        self.name = name
        self.targets = targets.copy()
        self.num_affected_qubits = len(targets)
        self.affected_qubits = set(targets.copy())

        self.draw_str_target = self._draw_str_target_dic.get(name, name[0])

        if controls is not None:
            self.is_controlled = True
            self.controls = controls
            self.control_state = '1'*len(controls)
            self.num_affected_qubits += len(controls)
            self.affected_qubits |= set(controls.copy())
            self.draw_str_control = '●'
            self.draw_str_target = self._draw_str_target_dic.get(name, name[1])
        else:
            self.is_controlled = False

    def __repr__(self):
        if self.is_controlled:
            return ' '.join([self.name,
                            str(self.controls[0]),
                            str(self.targets[0])])
        else:
            return ' '.join([self.name,
                            str(self.targets[0])])

    def __str__(self):
        return self.__repr__()

    def copy(self):
        copied_op = Operation(self.name,
                              [],
                              controls=None,
                              classical_control=None)

        copied_op.targets = self.targets
        copied_op.num_affected_qubits = self.num_affected_qubits
        copied_op.affected_qubits = self.affected_qubits

        copied_op.draw_str_target = self.draw_str_target

        if self.is_controlled:
            copied_op.is_controlled = self.is_controlled
            copied_op.controls = self.controls
            copied_op.control_state = self.control_state
            copied_op.draw_str_control = self.draw_str_control
        else:
            copied_op.is_controlled = self.is_controlled

        return copied_op

    def rebase_qubits(self, new_base):
        L = len(new_base)
        new_targets = [new_base + q[L:] for q in self.targets]
        if self.is_controlled:
            new_controls = [new_base + q[L:] for q in self.controls]
        else:
            new_controls = None
        return Operation(self.name, new_targets, controls=new_controls)
