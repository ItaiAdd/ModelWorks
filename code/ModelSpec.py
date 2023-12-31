from dataclasses import dataclass
from pandas import DataFrame


@dataclass(repr=False)
class ModelSpec():
    """Class to standardise fit/predict interface of model objects and
    and, store training and tuning related information and specify
    data preprocessing.
    
    Attributes:
        name: Identifier for this specification.
        model: Model object.
        params: Dictionary of parameter name: options items.
        metrics: Dictionary of callables which compute performance 
                 metrics, metric name : callable.
        task: Modelling task ('cls' or 'reg').
        origin: Specifies fit/predict interface.
        sampler: 'grid' or 'TPE'.
        trials: If sampler = 'TPE', number of parameter combinations to try.
        needs_proba: Whether or not to predict class probabilities.
        fit_model: Strores currentx trained model object.
        supervised: Whether or not the learning is supervised.
        preprocessing: Dictionary of preprocessing callables,
                       step name : callable.
        fit_params: Parameters for fit method.
        pred_params: Parameters for prediction method.
        custom_fit: Custom fit method.
        custom_predict Custom predict method.

    Methods:
        update_history: Adds trial to tuning history.
        history_df: Returns Pandas DataFrame of tuning_history.
    """

    name: str
    model: object
    params: dict
    task: str
    origin: str
    sampler: str
    metrics: dict[object] = None
    trials: int = None
    needs_proba: bool = False
    fit_model: object = None
    supervised: bool = False
    preprocessing: dict[object] = None
    fit_params: dict = None
    pred_params: dict = None
    custom_fit: callable = None
    custom_predict: callable = None


    def __post_init__(self):
        self.tuning_history = {col:[] for col in 
                               list({**self.params, **self.metrics}.keys)}


    def update_history(self,trial):
        """Method to add a set of trialed parameters and the computed metrics"""
        for col in self.tuning_history.keys():
            self.tuning_history[col].append(trial[col])
    

    def history_df(self):
        """Returns Pandas DataFrame of tuning_history"""
        return DataFrame(self.tuning_history)