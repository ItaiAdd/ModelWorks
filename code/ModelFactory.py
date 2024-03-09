from dataclasses import dataclass
from sklearn.model_selection import ParameterGrid, train_test_split
from optuna import create_study
from functools import partialmethod
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
        else:
            X = self.X
            y = self.y
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


    def optuna_objective(self, spec, X, y, trial):
        params = {}
        for name, value in spec.params.items():
            if type(value[0]) is int:
                params[name] = trial.suggest_int(name, value[0], value[1])
            elif type(value[0]) is float:
                params[name] = trial.suggest_float(name, value[0], value[1])
            else:
                params[name] = trial.suggest_categorical(name, value)
        if spec.cv:
            result = self.k_fold_cv(spec, params, X, y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            self.fit(spec, params, X_train, y_train)
            pred = self.predict(spec, X_test)
            result = self.compute_metrics(spec, pred, y_test)
        trial_record = {**params, **result}
        spec.update_history(trial_record)
        return result[spec.key_metric]
    

    def optuna_tune(self, spec):
        if spec.preprocessing:
            X, y = self.prepreprocess(spec)
        else:
            X = self.X
            y = self.y
        objective = partialmethod(self.optuna_objective, spec=spec, X=X, y=y)
        study = create_study()
        study.optimize(objective, n_trials=spec.n_trials)


    def tune(self, spec):
        tuning_message = f"{spec.name} being tuned."
        print(tuning_message)
        if spec.sampler == 'grid':
            self.grid_tune(spec)
        elif spec.sampler == 'TPE':
            self.optuna_tune(spec)
    

    def tune_all(self):
        for spec in self.specs:
            if isinstance(spec, ModelSpec):
                self.tune(spec)
                tuned_message = f"{spec.name} tuning complete."
                print(tuned_message)
            else:
                print('Provided specification is not an instance of ModelSpec.')