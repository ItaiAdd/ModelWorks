class FitPredBase():
    """Base class for fit/predict methods used by ModelFactory and
    ModelTester."""

    def fit_sklearn(spec, X, y=None):
        spec.fit_model = spec.model
        spec.fit_model.fit(X, y, **spec.fit_params)
    

    def predict_sklearn(spec, X):
        if spec.needs_proba:
            return spec.fit_model.predict_proba(X, **spec.pred_params)
        else:
            return spec.fit_model.predict(X, **spec.pred_params)
    


    # TODO Make PyTorch predict
    # TODO Make PyTorch fit

    # TODO Make XGBoost fit
    # TODO Make XGBoost predict

    # TODO Make CatBoost fit
    # TODO Make CatBoost predict

    # TODO Make general fit: model/origin --> fits with correct method --> returns fit model
    # TODO Make general predict: fit model/origin --> predicts with correct method --> returns y_pred