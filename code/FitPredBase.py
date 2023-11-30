class FitPredBase():
    """Base class for fit/predict methods used by ModelFactory and
    ModelTester."""

    def fit_sklearn(spec, X, y=None):
        if y:
            model = spec.model.fit(X, y)
        else:
            model = spec.model.fit(X)
    

    def predict_sklearn(spec, X, proba = False):
        if proba == True:
            return spec.model.predict_proba(X)
        else:
            return spec.predict(X)
    
    # TODO Make Keras fit
    # TODO Make Keras predict

    # TODO Make PyTorch predict
    # TODO Make PyTorch fit

    # TODO Make XGBoost fit
    # TODO Make XGBoost predict

    # TODO Make CatBoost fit
    # TODO Make CatBoost predict

    # TODO Make general fit: model/origin --> fits with correct method --> returns fit model
    # TODO Make general predict: fit model/origin --> predicts with correct method --> returns y_pred