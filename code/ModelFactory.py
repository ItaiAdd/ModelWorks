from dataclasses import dataclass
from sklearn.model_selection import ParameterGrid


@dataclass(repr=False)
class ModelWorks():
    specs: #instance of ModelSpec
    data: #the data to be fit
    sampling: str #'grid' or 'TPE'

    def prepreprocess(self, spec):
        data = self.data
        for transform in spec.preprocessing.values:
            data = transform(data)
        return data
    

    @staticmethod
    def param_grid(spec):
        return ParameterGrid(spec.params)
    

    # TODO Make individual model tuner