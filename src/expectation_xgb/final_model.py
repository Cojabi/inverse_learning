import pandas as pd
import numpy as np
import os
import optuna
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from joblib import dump, load
from glob import glob
from tqdm.auto import tqdm

import sys
sys.path.append('src/')
from plotting import plot_training_diagnostics
from expectation_xgb.regression.inner_fold import InnerLoop_Regressor
from utils import get_best_params


class FinalXGB():
    """Nested cross-validation for a XGBoost regression.

    cv_folds : int. Number of folds for the outer and inner cross-validation.
    data : pd.DataFrame. Dataframe containing the predictors.
    labels : pd.Series. Series containing the labels.
    """

    def __init__(self, save_path, X, y, numerical, categorical, cv_folds, 
                 hp_objective, hp_space, optuna__n_trials, optuna__n_jobs,
                 xgb__nthread, n_estimators, early_stopping_rounds, repeats=5
                 ) -> None:
        
        self.models = []
        self.save_path = save_path
        self.X = X
        self.y = y
        self.numerical = numerical
        self.categorical = categorical
        self.cv_folds = cv_folds
        self.hp_objective = hp_objective
        self.hp_space = hp_space
        self.optuna__n_trials = optuna__n_trials
        self.optuna__n_jobs = optuna__n_jobs
        self.xgb__nthread = xgb__nthread
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.repeats = repeats

    def train(self, hp_opt_mode="cv_results", exp_glob_pattern=None) -> None:
        
        assert self.X is not None and self.y is not None, \
        "data and labels must be provided in class construction."
        for repeat in tqdm(range(self.repeats)):
            save_path = f'{self.save_path}/repeats/' + f'_{repeat}'
            # create directory if it does not exist
            os.makedirs(save_path, exist_ok=True)

            # containers
            results = {}
            configs = {}
            if hp_opt_mode == "bayesian":
                import warnings
                warnings.warn("The Bayesian HPO for the final model is prone to overfit. \
                              Use with caution and try 'cv_results' if it fails.")
                # Inner CV
                objective = InnerLoop_Regressor(self.hp_space,
                                            self.X,
                                            self.y,
                                            scoring = self.hp_objective,
                                            cv=self.cv_folds,
                                            n_estimators=self.n_estimators,
                                            early_stopping=self.early_stopping_rounds,
                                            nthread=self.xgb__nthread,)
                study = optuna.create_study(study_name='xGBoost',
                                            storage=f'sqlite:///{save_path}/optuna__xgb.db', 
                                            direction = "minimize", 
                                            load_if_exists = True)
                study.optimize(objective, 
                            n_trials = self.optuna__n_trials,
                            n_jobs = self.optuna__n_jobs)
                # outer CV
                best_params = study.best_params
                best_n_estimartors = int(study.best_trial.user_attrs["median_n_estimators"])
                # set n_estimators to the median of the inner CV
                best_params["n_estimators"] = int(np.round(best_n_estimartors / (1 - 0.05)))

            elif hp_opt_mode == "cv_results":
                assert exp_glob_pattern is not None, "exp_glob_pattern must be provided."
                best_params = get_best_params(exp_glob_pattern)


            model = XGBRegressor(use_label_encoder=False, eval_metric="rmse", 
                                nthread=self.xgb__nthread)
            model.set_params(**best_params)
            model.fit(self.X, self.y,
                      eval_set=[(self.X, self.y)],
                      verbose=False)
            self.models.append(model)

            print(f'Done training model {repeat}')

            # training score
            pred_train = model.predict(self.X)
            results["rmse_train"] = mean_squared_error(self.y, pred_train)
            results["mape_train"] = mean_absolute_percentage_error(self.y, pred_train)
            results["R2_train"] = r2_score(self.y, pred_train)

            # Get feature importance per outer fold
            feature_importances = pd.Series(model.feature_importances_, index=self.X.columns)

            # store model and configs
            configs["best_params"] = {"HPs": best_params}

            dump(model, f"{save_path}/model.joblib")
            feature_importances.to_csv(f"{save_path}/feature_importances.csv")
            pd.DataFrame(results, index=[0]).to_csv(f"{save_path}/training_performance.csv")
            pd.DataFrame.from_dict(configs).to_csv(f"{save_path}/configs.csv")

            # plot training diagnostics
            diagnostic_plot_path = f"{save_path}/diagnostics/fold_{repeat}"
            if not os.path.isdir(f'{save_path}/diagnostics'):
                os.makedirs(f'{save_path}/diagnostics')
            plot_training_diagnostics(model.evals_result(), diagnostic_plot_path)
        # set up folder to store results and plots
        if not os.path.exists(f'{self.save_path}/plots'):
                os.mkdir(f'{self.save_path}/plots')
                os.mkdir(f'{self.save_path}/results')

    def test(self, X, y, save=True):
        """Evaluate the performance of the models on the test set."""
        assert self.save_path is not None, "save_path must be provided."
        # containers
        results = {}
        all_predictions = pd.DataFrame(index=X.index)
        all_predictions['Label'] = y

        for i, model in enumerate(self.models):

            result = {}

            # validation scores
            preds = model.predict(X)
            result["rmse_val"] = mean_squared_error(y, preds)
            result["mape_val"] = mean_absolute_percentage_error(y, preds)
            result["R2_val"] = r2_score(y, preds)
            results[f"model_{i}"] = result

            all_predictions[f"Pred_m{i}"] = preds
            all_predictions[f"Res_m{i}"] = all_predictions["Label"] - all_predictions[f"Pred_m{i}"]
        
        # get mean residuals and prediction values
        all_predictions["Prediction_mean"] = all_predictions[
            all_predictions.columns[all_predictions.columns.str.contains("Pred")]
            ].mean(axis=1)
        all_predictions["Residual_mean"] = all_predictions["Label"] - all_predictions["Prediction_mean"]
        # save
        if save:
            all_predictions.to_csv(f"{self.save_path}/results/test_predictions.csv")
            pd.DataFrame.from_dict(results).to_csv(f"{self.save_path}/results/test_performance.csv")
        return all_predictions, results

    
    def predict(self, X):
        """Predict the target variable for the provided data."""
        assert len(self.models) > 0, "No models loaded."

        predictions = pd.DataFrame(index=X.index)
        for i, model in enumerate(self.models):
            predictions[f"Pred_m{i}"] = model.predict(X)
        predictions["Prediction_mean"] = predictions[
            predictions.columns[predictions.columns.str.contains("Pred")]
            ].mean(axis=1)
        return predictions

    def load_models(self, glob_pattern):
        """Load the models from the specified path. 
        glob_pattern : str. Path to the directory holding the subdirectories with the model.joblib files.
        """
        files = glob(f'{glob_pattern}')
        assert len(files) > 0, "No files found."
        for file in files:
            self.models.append(load(f"{file}/model.joblib"))
        

if __name__ == "__main__":
    ## Builds a test model to debugg the code
    import sys
    sys.path.append('src/')
    from data_loader import load_data
    from constants import DATA_PATH, NUMERICAL, CATEGORICAL, LABEL, SUBGROUPING_VAL, CV_FOLDS, \
        OPTUNA__N_TRIALS, OPTUNA__N_JOBS, XGB__NTHREAD, N_ESTIMATORS, EARLY_STOPPING_ROUNDS, OBJECTIVE, \
        XGB_HP, SUBGROUPING
    
    save_path = "sent/final_models/xgboost/NoYrsEd"

    # create directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # X, y = load_data(DATA_PATH, label=LABEL, numerical_vars=NUMERICAL, categorical_vars=CATEGORICAL,
    #             subgrouping=SUBGROUPING)
    test, labels = load_data(DATA_PATH, label=LABEL, numerical_vars=NUMERICAL, categorical_vars=CATEGORICAL,
                 subgrouping=SUBGROUPING_VAL)

    final = FinalXGB(save_path, test, labels, NUMERICAL, CATEGORICAL, CV_FOLDS, 
                 OBJECTIVE, XGB_HP, OPTUNA__N_TRIALS, OPTUNA__N_JOBS,
                 XGB__NTHREAD, N_ESTIMATORS, EARLY_STOPPING_ROUNDS, repeats=5)
    #final.train()
    final.load_models('sent/final_models/xgboost/NoYrsEd/repeats/_*')
    final.test(test, labels) # GRADE IST DAS HIER PFUSCH WEIL ICH TRAIN = TEST GEMACHT HABE
    