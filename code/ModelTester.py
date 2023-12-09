from dataclasses import dataclass


@dataclass(repr=False)
class ModelTester():
    specs: list[object]