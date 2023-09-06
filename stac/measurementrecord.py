"""Provides class for measurement records."""

class MeasurementRecord:
    def __init__(self,
                 address: tuple,
                 index: int
                 ) -> None:
        self.address = address
        self.index = index

    def __repr__(self) -> str:
        return f'MR[{self.address}, {self.index}]'
