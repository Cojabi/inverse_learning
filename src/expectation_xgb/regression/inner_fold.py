from typing import Any, Dict
from optuna.trial import Trial
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
from copy import deepcopy
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

import sys
sys.path.append('src/')
from utils import store_scores, store_metrics
from expectation_xgb.constants import RANDOM_SEED


class InnerLoop_Regressor(object):
    """Inner loop for nested cross-validation of a XGBoost regressor.
    
    param_distributions : dict. Dictionary with hyperparameter names as keys and
        distributions as values.
    X : pd.DataFrame. Data.
    y : pd.Series. Labels.
    scoring : str. Metric to optimize (either "rmse" or "mape").
    n_estimators : int. Number of trees to fit.
    cv : int. Number of folds for the inner cross-validation.
    early_stopping : int. Number of rounds without improvement before stopping.
    nthread : int. Number of threads (XGboost).
    """
    
    def __init__(
        self,
        param_distributions: dict,
        X,
        y,
        scoring : str = "rmse",
        cv: int = 5,
        n_estimators: int = 10000,
        early_stopping: int = 30,
        nthread: int = 4,
        ) -> None:

        self.X = X
        self.y = y
        self.cv = cv
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        self.nthread = nthread
        self.fit_params = None
        self.param_distributions = param_distributions
        self.scoring = scoring

    def __call__(self, trial: Trial) -> float:
        """callable for optuna search"""
        
        early_stop = EarlyStopping(rounds=self.early_stopping, save_best=True)
        fit_params = self._get_params(trial) # sample suggested hyperparams
        cv = KFold(n_splits = self.cv, shuffle=True)

        # containers 
        scores = {
           # train
           "train_rmse" : [],
           "train_mape" : [],
           "train_r2" : [],
           # val
           "val_rmse" : [],
           "val_mape" : [],
           "val_r2" : []
           }
        metrics = {
            "n_estimators" : []
            }
        
        # get cross-validated estimate for one particular hyperparameter set
        for fold, (train_idx, val_idx) in enumerate(cv.split(self.X, self.y)):
        
            X_train = self.X.iloc[train_idx]
            y_train = self.y.iloc[train_idx]
            X_val = self.X.iloc[val_idx]    
            y_val = self.y.iloc[val_idx]
            
            # initialize model
            estimator = XGBRegressor(n_estimators=self.n_estimators, eval_metric=["mape", "rmse"], 
                                      nthread=self.nthread, callbacks=[deepcopy(early_stop)])
            estimator.set_params(**fit_params)
            
            # fit with early stopping on validation set and keep best model
            estimator.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # predict
            y_train_pred = estimator.predict(X_train, iteration_range=(0, estimator.best_iteration+1))
            y_val_pred = estimator.predict(X_val, iteration_range=(0, estimator.best_iteration+1))

            # evaluate model performance
            ## train
            scores["train_rmse"].append(mean_squared_error(y_train, y_train_pred, squared=False))
            scores["train_mape"].append(mean_absolute_percentage_error(y_train, y_train_pred))
            scores["train_r2"].append(r2_score(y_train, y_train_pred))
            ## val
            scores["val_rmse"].append(mean_squared_error(y_val, y_val_pred, squared=False))
            scores["val_mape"].append(mean_absolute_percentage_error(y_val, y_val_pred))
            scores["val_r2"].append(r2_score(y_val, y_val_pred))

            # choose scoring to return for optuna to optimize
            if self.scoring == "rmse":
                scores["train_score"] = scores["train_rmse"]
                scores["val_score"] = scores["val_rmse"]
            elif self.scoring == "mape":
                scores["train_score"] = scores["train_mape"]
                scores["val_score"] = scores["val_mape"]
            else:
                raise ValueError("hp_objective must be either 'rmse' or 'mape'")
            
            # metrics
            # get early stopping rounds
            metrics["n_estimators"].append(estimator.best_iteration)
            

        # save scores in trial object
        store_scores(trial, scores)
        store_metrics(trial, metrics)
        
        # the 'mean_' part is added in the store_scores function
        return trial.user_attrs["mean_val_score"]
    
    def _get_params(self, trial: Trial) -> Dict[str, Any]:
        """Get parameters from trial history"""

        return {name:trial._suggest(name, distribution) 
                for name, distribution in self.param_distributions.items()}