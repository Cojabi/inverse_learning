import pandas as pd
import numpy as np
import optuna
import os
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from copy import deepcopy

import sys
sys.path.append('src/')
from expectation_xgb.regression.inner_fold import InnerLoop_Regressor
from plotting import plot_training_diagnostics
from expectation_xgb.constants import RANDOM_SEED

class NestedCV_Regressor():
    """Nested cross-validation for a XGBoost regression.

    exp_name : str. Name of the experiment.
    cv_folds : int. Number of folds for the outer and inner cross-validation.
    n_estimators : int. Number of trees to fit.
    early_stopping : int. Number of rounds without improvement before stopping.
    n_trials : int. Number of trials for the optuna hyperparameter search.
    n_jobs : int. Number of parallel jobs (Optuna).
    nthread : int. Number of threads (XGboost).
    hp_objective : str. Objective for the hyperparameter search.
    hp_space : dict. Hyperparameter space for the hyperparameter search.
    plot_pr : bool. Whether to plot the PR curve.
    """

    def __init__(self,
                 exp_name: str,
                 data: pd.DataFrame,
                 labels: pd.Series,
                 cv_folds: int,
                 nthread: int,
                 n_estimators: int,
                 early_stopping: int,
                 hp_objective: str,
                 hp_space: dict,
                 n_trials: int,
                 n_jobs: int,
                 ) -> None:
        self.exp_name = exp_name
        self.data = data
        self.labels = labels
        self.cv_folds = cv_folds
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        self.hp_objective = hp_objective
        self.hp_space = hp_space
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.nthread = nthread


    def run(self) -> None:
        results = {"rmse_train" : [],
                   "mape_train" : [],
                   "R2_train" : [],
                   "rmse_val" : [],
                   "mape_val" : [],
                   "R2_val" : []}
        configs = {fold:{} for fold in range(0, self.cv_folds)}
        all_predictions = pd.DataFrame(columns=["Score", "Fold"], index=self.data.index)
        feature_importances = pd.DataFrame(index=self.data.columns)
        
        skf_out = KFold(self.cv_folds, shuffle=True)
        for cur_fold, (train_inds, test_inds) in enumerate(skf_out.split(self.data, self.labels)):
            
            X_train = self.data.iloc[train_inds]
            y_train = self.labels.iloc[train_inds]
            X_val = self.data.iloc[test_inds]    
            y_val = self.labels.iloc[test_inds]
            
            # logging
            print(f"\nRunning outer fold {cur_fold}")

            # inner loop hp tuning
            objective = InnerLoop_Regressor(self.hp_space,
                                        X_train,
                                        y_train,
                                        scoring = self.hp_objective,
                                        cv=self.cv_folds,
                                        n_estimators=self.n_estimators,
                                        early_stopping=self.early_stopping,
                                        nthread=self.nthread,)
            study = optuna.create_study(study_name='xGBoost',
                                        storage=f'sqlite:///{self.exp_name}/optuna__xgb_{cur_fold}.db', 
                                        direction = "minimize", 
                                        load_if_exists = True)
            study.optimize(objective, 
                           n_trials = self.n_trials,
                           n_jobs = self.n_jobs)
            
            #### outer loop predictions
            best_params = study.best_params
            early_stop = EarlyStopping(rounds=self.early_stopping, metric_name="rmse", 
                                       maximize=False, save_best=True)
            model = XGBRegressor(n_estimators=self.n_estimators, eval_metric="rmse", 
                                nthread=self.nthread, callbacks=[deepcopy(early_stop)])
            model.set_params(**best_params)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

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
            feature_importances[f"{cur_fold}"] = model.feature_importances_ 
            # store predictions
            all_predictions.iloc[test_inds, 0] = pred
            all_predictions.iloc[test_inds,1] = cur_fold
            # store model and CV configs
            configs[cur_fold]["best_params"] = study.best_params
            configs[cur_fold]["test_inds"] = test_inds.tolist()
            # plot training diagnostics
            diagnostic_plot_path = f"{self.exp_name}/diagnostics/fold_{cur_fold}"
            if not os.path.isdir(f'{self.exp_name}/diagnostics'):
                os.makedirs(f'{self.exp_name}/diagnostics')
            plot_training_diagnostics(model.evals_result(), diagnostic_plot_path)

        
        # re-arrange prediction table columns for easier reading
        all_predictions['Label'] = self.labels
        all_predictions["Residual"] = all_predictions["Label"] - all_predictions["Score"]
        all_predictions = all_predictions[["Label", "Score", "Residual", "Fold"]]
        
        return(results, feature_importances, all_predictions, configs)