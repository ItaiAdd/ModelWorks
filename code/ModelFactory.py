from sklearn.model_selection import ParameterGrid, train_test_split
from optuna import create_study
from functools import partial
from FitPredBase import FitPredBase
from ModelSpec import ModelSpec



class ModelFactory(FitPredBase):
    """A class for tuning machine learning models using grid search or Optuna's TPE algorithm.

    Attributes:
        specs (list[object]): A list of model specifications.
        X (dict): The predictors data to be fit.
        y (list): The response data.
    """
    def __init__(self, specs, X, y):
        self.specs = specs
        self.X = X
        self.y = y

    def prepreprocess(self, spec):
        """Applies preprocessing steps stored in the input model specification to instantiated data X and y.

        Args:
            spec (ModelSpec): An instance of ModelSpec instantiated with preprocessing methods.

        Returns:
            tuple: A tuple containing preprocessed predictors (X) and response (y).
        """
        X = self.X
        y = self.y
        for transform in spec.preprocessing.values():
            X, y = transform(X, y)
        return X, y

    @staticmethod
    def param_grid(spec):
        """Creates a combinatoric parameter grid for grid tuning.

        Args:
            spec (ModelSpec): An instance of ModelSpec instantiated with full list of options for each parameter.

        Returns:
            ParameterGrid: An iterable representing the parameter grid.
        """
        return ParameterGrid(spec.params)

    def grid_tune(self, spec):
        """Performs grid tuning for the specified model specification.

        Args:
            spec (ModelSpec): An instance of ModelSpec containing model configuration.

        Returns:
            None
        """
        if spec.preprocessing:
            X, y = self.prepreprocess(spec)
        else:
            X = self.X
            y = self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y)
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
        """Defines the objective function for Optuna optimization.

        Args:
            spec (ModelSpec): An instance of ModelSpec containing model configuration.
            X (array-like): The predictors data.
            y (array-like): The response data.
            trial (optuna.Trial): An Optuna trial object for optimization.

        Returns:
            float: The value of the key metric specified in the model specification.
        """
        params = {}
        for name, value in spec.params.items():
            if isinstance(value[0], int):
                params[name] = trial.suggest_int(name, value[0], value[1])
            elif isinstance(value[0], float):
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
        """Performs tuning using Optuna's TPE algorithm.

        Args:
            spec (ModelSpec): An instance of ModelSpec containing model configuration.

        Returns:
            None
        """
        if spec.preprocessing:
            X, y = self.prepreprocess(spec)
        else:
            X = self.X
            y = self.y
        objective = partial(self.optuna_objective, spec, X, y)
        study = create_study()
        study.optimize(objective, n_trials=spec.n_trials)

    def tune(self, spec):
        """Tunes the model according to the specified sampler.

        Args:
            spec (ModelSpec): An instance of ModelSpec containing model configuration.

        Returns:
            None
        """
        tuning_message = f"{spec.name} being tuned."
        print(tuning_message)
        if spec.sampler == 'grid':
            self.grid_tune(spec)
        elif spec.sampler == 'TPE':
            self.optuna_tune(spec)

    def tune_all(self):
        """Tunes all models specified in the specs attribute.

        Returns:
            None
        """
        for spec in self.specs:
            if isinstance(spec, ModelSpec):
                self.tune(spec)
                tuned_message = f"{spec.name} tuning complete."
                print(tuned_message)
            else:
                print('Provided specification is not an instance of ModelSpec.')
