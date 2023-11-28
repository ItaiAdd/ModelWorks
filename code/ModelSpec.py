from dataclasses import dataclass
from pandas import DataFrame


@dataclass
class ModelSpec():
    """Class to standardise fit/predict interface of model objects and
    and, store training and tuning related information and specify
    data preprocessing"""

    name: str
    model: object
    params: dict
    metrics: dict[object]
    task: str
    origin: str
    preprocessing: dict[object] = None
    custom_fit: callable = None
    custom_predict: callable = None
    tuning_history: dict = {col:[] for col in list({**params, **metrics}.keys)}


    def update_history(self,trial):
        """Method to add a set of trialed parameters and the computed metrics"""
        for col in self.tuning_history.keys():
            self.tuning_history[col].append(trial[col])
    

    def history_df(self):
        """Returns Pandas DataFrame of tuning_history"""
        return DataFrame(self.tuning_history)