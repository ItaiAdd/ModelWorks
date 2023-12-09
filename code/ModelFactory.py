from dataclasses import dataclass
from sklearn.model_selection import ParameterGrid
from FitPredBase import FitPredBase


@dataclass(repr=False)
class ModelWorks():
    specs: list[object] #instance of ModelSpec
    X: dict #the data to be fit
    y: list = None


    def prepreprocess(self, spec):
        X = self.X
        if self.y:
            y = self.y
            for transform in spec.preprocessing.values:
                X, y = transform(X, y)
        else:
            for transform in spec.preprocessing.values:
                X = transform(X)
        return X, y
    

    @staticmethod
    def param_grid(spec):
        return ParameterGrid(spec.params)
    

# TODO Add train_test_split --> train/val sets
    def grid_tune(self, spec):
        if spec.preprocessing:
            X, y = self.preprocess(spec)
        
        param_grid = self.param_grid(spec)

        for trial in param_grid:
            self.fit(spec, params, X, y)
            




    # TODO Make individual model tuner
    # TODO Implement k-fold cv
    # TODO Make overall model tuner
    # TODO Implement TPE sampler