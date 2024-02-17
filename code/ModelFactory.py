from dataclasses import dataclass
from sklearn.model_selection import ParameterGrid, train_test_split
from FitPredBase import FitPredBase


@dataclass(repr=False)
class ModelWorks(FitPredBase):
    specs: list[object] #instance of ModelSpec
    X: dict #the data to be fit
    y: list 


    def prepreprocess(self, spec):
        X = self.X
        y = self.y
        for transform in spec.preprocessing.values:
            X, y = transform(X, y)
        return X, y
        

    @staticmethod
    def param_grid(spec):
        return ParameterGrid(spec.params)
    

    def grid_tune(self, spec):
        if spec.preprocessing:
            X, y = self.prepreprocess(spec)
            X_train, X_test, y_train, y_test = train_test_split(X,y)
        param_grid = self.param_grid(spec)
        for params in param_grid:
            if spec.cv:
                result = self.k_fold_cv(spec, params, X, y)
            else:
                self.fit(spec, params, X_train, y_train)
                pred = self.predict(spec, X_test)
                result = self.compute_metrics(spec, pred, y_test)
            trial = {**params, **result}
            spec.update_history(trial)
            



    # TODO Make individual model tuner
    # TODO Make overall model tuner
    # TODO Implement TPE sampler