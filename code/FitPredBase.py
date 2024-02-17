from sklearn.model_selection import KFold
from numpy import mean

#TODO Implement hasattr to know whether to use predict or score
class FitPredBase():
    """Base class for fit/predict methods used by ModelFactory and
    ModelTester."""

    @staticmethod
    def compute_metrics(spec, pred, y):
        results = {}
        for name, metric in spec.metrics.items():
            results[name] = metric(pred, y)
        return results


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
    

    @staticmethod
    def fit_xgb(spec, params, X, y=None):
        spec.fit_model = spec.model(**params)
        spec.fit_model.fit(X, y, **spec.fit_params)
        
    
    @staticmethod
    def predict_xgb(spec, X):
        if spec.needs_proba:
            return spec.fit_model.predict_proba(X, **spec.pred_params)
        else:
            return spec.fit_model.predict(X, **spec.pred_params)


    # TODO Make CatBoost fit
    # TODO Make CatBoost predict


    def fit(self, spec, params, X, y=None):
        if spec.origin == 'sklearn':
           self.fit_sklearn(spec, params, X, y)
        elif spec.origin == 'xgb':
            self.fit_xgb(spec, params, X, y)

            
    def predict(self, spec, X):
        if spec.origin == 'sklearn':
            return self.predict_sklearn(spec, X)


    def k_fold_cv(self, spec, params, X, y):
        kfold = KFold(n_splits=spec.cv, shuffle=True)
        inds = kfold.split(X, y)
        results = {k:[] for k in list(spec.metrics.keys())}

        for ind in inds:
            X_train = X[ind[0]]
            X_test = X[ind[1]]
            y_train = y[ind[0]]
            y_test = y[ind[1]]
            self.fit(spec, params, X_train, y_train)
            pred = self.predict(spec, X_test)
            result = self.compute_metrics(spec, pred, y_test)
            for key in list(results.keys()):
                results[key].append(result[key])
        cv_results = {key:mean(val) for key,val in results.items()}
        return cv_results