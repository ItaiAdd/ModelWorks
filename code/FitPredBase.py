class FitPredBase():
    """Base class for fit/predict methods used by ModelFactory and
    ModelTester."""

    @staticmethod
    def fit_sklearn(spec, params, X, y=None):
        spec.fit_model = spec.model(**params)
        if y:
            spec.fit_model.fit(X, y, **spec.fit_params)
        else:
            spec.fit_model.fit(X, **spec.fit_params)
    

    @staticmethod
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

    def fit(self, spec, X, params, y=None):
        if spec.origin == 'sklearn':
            self.fit_sklearn(spec, params, X, y)

    
    def predict(self, spec, X):
        if spec.origin == 'sklearn':
            return self.predict_sklearn(spec, X)
