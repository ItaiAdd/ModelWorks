from dataclasses import dataclass
from sklearn.model_selection import ParameterGrid
from FitPredBase import FitPredBase 


@dataclass(repr=False)
class ModelWorks():
    specs: object #instance of ModelSpec
    data: list #the data to be fit

    def prepreprocess(self, spec):
        # TODO add conditional for supervised/unsupervised
        data = self.data
        for transform in spec.preprocessing.values:
            data = transform(data)
        return data
    

    @staticmethod
    def param_grid(spec):
        return ParameterGrid(spec.params)
    

    def grid_tune(self, spec, X, y=None):
        if spec.supervised:
            X, y = 
        param_grid = self.param_grid(spec)

        for trial in param_grid:



    # TODO Make individual model tuner
    # TODO Make overall model tuner
    # TODO Implement TPE sampler