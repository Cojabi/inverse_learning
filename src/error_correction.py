import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


import sys
sys.path.append('src/')
from expectation_xgb.constants import RANDOM_SEED


class ErrorRegressor():
    """Class to regress out the conditional average model error based on participants covariates. 
    Fits a linear regression model to estimate average model error based on the residuals from
    cross-validation and then substracts the predicted avg error from the residuals predicted for 
    an individual. """

    def __init__(self, numerical, imp_iter=20) -> None:
        self.model = None
        self.preprocess = None
        self.imp_iter = imp_iter
        self.numerical = numerical

    def fit(self, X, y):
        # scale numerical features and impute missing values
        scale = ColumnTransformer(transformers=[("scaler", StandardScaler(), self.numerical)], 
                                    remainder="passthrough")
        imputer = IterativeImputer(estimator=RandomForestRegressor(), random_state=0, max_iter=self.imp_iter)
        self.preprocess = make_pipeline(scale, imputer).set_output(transform="pandas")
        X_pre = self.preprocess.fit_transform(X)
        self.model = LinearRegression().fit(X_pre, y)
    
    def cv_performance(self, X, y, cv_folds=5):
        """Estimate the model performance using cross-validation."""
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
        from sklearn.model_selection import KFold

        results = {"rmse_train" : [],
                   "mape_train" : [],
                   "R2_train" : [],
                   "rmse_val" : [],
                   "mape_val" : [],
                   "R2_val" : []}
        configs = {fold:{} for fold in range(1, cv_folds+1)}
        all_predictions = pd.DataFrame(columns=["Score", "Fold"], index=X.index)
        feature_importances = pd.DataFrame(index=X.columns)
        
        skf_out = KFold(cv_folds, shuffle=True, random_state=RANDOM_SEED)
        for cur_fold, (train_inds, test_inds) in enumerate(skf_out.split(X, y)):
            
            X_train = X.iloc[train_inds]
            y_train = y.iloc[train_inds]
            X_val = X.iloc[test_inds]    
            y_val = y.iloc[test_inds]

            # scale numerical features and impute missing values
            scale = ColumnTransformer(transformers=[("scaler", StandardScaler(), self.numerical)], remainder="passthrough")
            imputer = IterativeImputer(estimator=RandomForestRegressor(), random_state=0, max_iter=self.imp_iter)
            preprocess = make_pipeline(scale, imputer).set_output(transform="pandas")
            X_train = preprocess.fit_transform(X_train)
            X_val = preprocess.transform(X_val)
            # train model
            model = LinearRegression().fit(X_train, y_train)

            # training score
            pred_train = model.predict(X_train)
            results["rmse_train"].append(mean_squared_error(y_train, pred_train))
            results["mape_train"].append(mean_absolute_percentage_error(y_train, pred_train)/100)
            results["R2_train"].append(r2_score(y_train, pred_train))
            # validation score
            pred = model.predict(X_val)
            results["rmse_val"].append(mean_squared_error(y_val, pred)) 
            results["mape_val"].append(mean_absolute_percentage_error(y_val, pred)/100)
            results["R2_val"].append(r2_score(y_val, pred))
            
            # Get feature importance per outer fold
            feature_importances[f"{cur_fold}"] = model.coef_
            # store predictions
            all_predictions.iloc[test_inds, 0] = pred
            all_predictions.iloc[test_inds,1] = cur_fold
            # store model and CV configs
            configs[cur_fold+1]["test_inds"] = test_inds.tolist()

        # re-arrange prediction table columns for easier reading
        all_predictions['Label'] = y
        all_predictions["Residual"] = all_predictions["Label"] - all_predictions["Score"]
        all_predictions = all_predictions[["Label", "Score", "Residual", "Fold"]]

        return (results, feature_importances, all_predictions, configs)
    
    def predict(self, X):
        preds = self.model.predict(X)
        return preds

    def save(self, save_path):
        pickle.dump(self.model, open(save_path, 'wb'))

    def load(self, save_path):
        self.model = pickle.load(open(save_path, 'rb'))

    def clean_residuals(self, X, residuals, model_path=None):
        """Regress model error estimated in training data out of the residuals gained from test data."""
        assert (self.model is not None) or (model_path is not None), "Either call .train() or provide model_path."
        if (self.model is None) or (model_path is not None):
            self.load(model_path)
            
        X_pre = self.preprocess.transform(X)
        results = pd.DataFrame({"Residual": residuals})
        results['Pred_Error'] = self.predict(X_pre)
        results['Corrected'] = results['Residual'] - results['Pred_Error']
        results['Corrected_SD'] = results['Corrected'] / results['Corrected'].std()
        return results