import os
import sys
import pandas as pd
import json
from expectation_xgb.regression.outer_fold import NestedCV_Regressor


class XGBoostExperiment:
    def __init__(self, save_path, X, y, numerical, categorical, cv_folds, 
                 hp_objective, hp_space, optuna__n_trials, 
                 optuna__n_jobs, xgb__nthread, n_estimators, early_stopping_rounds):
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

    def _save_results(self, save_path, results, feature_importances, predictions, configs):
        """Save the results of the experiment."""
        
        results_df = pd.DataFrame(results)
        results_df.loc["mean"] = results_df.mean()
        results_df.loc["std"] = results_df.std()
        results_df.to_csv(f"{save_path}/results.out")
        feature_importances.to_csv(f"{save_path}/feature_importances.out")
        predictions.to_csv(f"{save_path}/predictions.out")
        json.dump(configs, open(f"{save_path}/params.json", "w"), indent=4)

    def run(self, repeats=5):
        for i in range(repeats):
            
            save_path = os.path.join(self.save_path, f'repeat-{i}')
            # Create the directory for the results
            if not os.path.isdir(save_path):
                os.mkdir(save_path)

            # Run nested cross-validation
            assert self.hp_objective in ["rmse", "mape"], "Invalid hp tuning objective."
            experiment = NestedCV_Regressor(save_path, self.X, self.y, self.cv_folds, self.xgb__nthread, 
                                            self.n_estimators, self.early_stopping_rounds,
                                            self.hp_objective, self.hp_space, 
                                            self.optuna__n_trials, self.optuna__n_jobs)
            results, feature_importances, predictions, configs = experiment.run()
            self._save_results(save_path, results, feature_importances, predictions, configs)


if __name__ == "__main__":
    sys.path.append('src/')
    from data_loader import load_data
    from utils import aggregate_results, aggregate_feature_importances
    from plotting import feature_importance_plot, plot_reg_perf
    from evaluation import groupwise_reg_eval
    from constants import DATA_PATH, NUMERICAL, CATEGORICAL, LABEL, SUBGROUPING, CV_FOLDS, \
        OPTUNA__N_TRIALS, OPTUNA__N_JOBS, XGB__NTHREAD, N_ESTIMATORS, EARLY_STOPPING_ROUNDS, OBJECTIVE, \
        XGB_HP, GROUP_EVAL

    REPEATS = 5
    predictors = NUMERICAL + CATEGORICAL

    #### Experiment parameters ####
    exp_name = f"{str.join('_', [str(x) for x in SUBGROUPING])}_{LABEL}"
    save_path = f"sent/experiments/{exp_name}/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # Save the configs
    exp_config = {"label": LABEL, "predictors": NUMERICAL, 
                    "cv_folds": CV_FOLDS, "subgroup": SUBGROUPING}
    json.dump(exp_config, open(f"{save_path}/exp_params.json", "w"), indent=4)

    ### Load the data
    X, y = load_data(DATA_PATH, label=LABEL, numerical_vars=NUMERICAL, categorical_vars=CATEGORICAL,
                    subgrouping=SUBGROUPING, cross_sectional=True)
    # data for groupwise evaluation
    eval_data = load_data(DATA_PATH)
    
    # Create the directory for the xgboost results
    xgb_save_path = os.path.join(save_path, 'xgboost')
    if not os.path.isdir(xgb_save_path):
        os.mkdir(xgb_save_path)

    exp = XGBoostExperiment(xgb_save_path, X, y, NUMERICAL, CATEGORICAL, CV_FOLDS,
                            OBJECTIVE, XGB_HP, OPTUNA__N_TRIALS, OPTUNA__N_JOBS,
                            XGB__NTHREAD, N_ESTIMATORS, EARLY_STOPPING_ROUNDS)
    exp.run(repeats=REPEATS)

    # aggregate the results across repeats
    mean_result = aggregate_results(xgb_save_path)
    mean_result.to_csv(f"{xgb_save_path}/mean_results.out")
    feat_imps = aggregate_feature_importances(xgb_save_path)
    groupwise_reg_eval(eval_data, GROUP_EVAL, xgb_save_path)
    # plots
    feature_importance_plot(feat_imps, xgb_save_path)
    plot_reg_perf(mean_result, xgb_save_path)


