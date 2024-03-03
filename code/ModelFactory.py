from dataclasses import dataclass
from sklearn.model_selection import ParameterGrid, train_test_split
from FitPredBase import FitPredBase
from ModelSpec import ModelSpec


@dataclass(repr=False)
class ModelFactory(FitPredBase):
    specs: list[object] #instance of ModelSpec
    X: dict #the data to be fit
    y: list 


    def prepreprocess(self, spec):
        X = self.X
        y = self.y
        for transform in spec.preprocessing.values():
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


    def tune(self, spec):
        tuning_message = f"{spec.name} being tuned."
        print(tuning_message)
        if spec.sampler == 'grid':
                self.grid_tune(spec)
    

    def tune_all(self):
        for spec in self.specs:
            if isinstance(spec, ModelSpec):
                self.tune(spec)
                tuned_message = f"{spec.name} tuning complete."
                print(tuned_message)
            else:
                print('Provided specification is not an instance of ModelSpec.')