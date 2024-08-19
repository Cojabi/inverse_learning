import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as aucfunc
from optuna.trial import Trial
import json
import glob

# utility functions to logg results in optuna study object
def store_scores(trial:Trial, scores:dict) -> None:
        """store scores in trial object, compute mean and std
        the array holds the values across folds"""
        
        for name, array in scores.items():
            
            if len(array) > 0:
                for i, score in enumerate(array):
                    trial.set_user_attr(f"split{i}_{name}", score)
            
            # creates the 'mean_val_score' that will be optimized.
            trial.set_user_attr(f"mean_{name}", np.nanmean(array))
            trial.set_user_attr(f"std_{name}", np.nanstd(array))
            
def store_metrics(trial:Trial, metrics:dict) -> None:
    """store metrics in trial object, compute mean and std. 
    the array holds the values across folds"""

    for name, array in metrics.items():
        
        if len(array) > 0:
            for i, metric in enumerate(array):
                    trial.set_user_attr(f"split{i}_{name}", metric)

            trial.set_user_attr(f"mean_{name}", np.nanmean(array))
            trial.set_user_attr(f"std_{name}", np.nanstd(array))

            if name in ["epochs","n_estimators"]:

                trial.set_user_attr(f"median_{name}", np.nanmedian(array))
                iqr = np.nanquantile(array, 0.75) - np.nanquantile(array,0.25) 
                trial.set_user_attr(f"iqr_{name}", iqr)

def calc_aucPR(y_true, y_pred):
    """Calculate the area under the precision recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return aucfunc(recall, precision)

def aggregate_results(exp_path):
    """Aggregate the results of multiple experiments."""
    files = glob.glob(f"{exp_path}/repeat-*/results.out")
    dfs = [pd.read_csv(file, index_col=0).loc["mean"] for file in files]
    df = pd.concat(dfs, axis=1)
    return df.transpose()

def aggregate_residuals(exp_path):
    """Aggregate the residuals of repeated experiments."""
    files = glob.glob(f"{exp_path}/repeat-*/predictions.out")
    dfs = [pd.read_csv(file, index_col=0) for file in files]
    # combine labels with residuals of every model
    concat = [dfs[0]["Label"]] + \
                [df.rename(columns={"Residual":f"Residual_{i}"})[f"Residual_{i}"] for i, df in enumerate(dfs)]
    df = pd.concat(concat, axis=1)
    df["Residual_mean"] = df[df.columns[df.columns.str.contains("Res")]].mean(axis=1)
    return df

def aggregate_feature_importances(exp_path):
    """Aggregate the feature importances of multiple experiments."""
    files = glob.glob(f"{exp_path}/repeat-*/feature_importances.out")
    dfs = [pd.read_csv(f, index_col=0) for f in files]
    df = pd.concat(dfs, axis=1)
    return df

def get_best_params(glob_pattern, CV_FOLDS=5):
    """Get the mean of the best xgboost parameters from the Bayesian HP optimization."""
    files = glob.glob(glob_pattern)

    l = []
    for file in files:
        with open(file, 'r') as f:
            x = json.load(f)
            for i in range(CV_FOLDS):
                l.append(x[str(i)]["best_params"])
    params = pd.DataFrame(l).mean().to_dict()
    params['max_depth'] = int(params['max_depth'])
    return params

def run_IL_lin(X_train_reduced, X_test_reduced, y_train, y_test, r2_target=None, with_correction=False):
    from standard_approach.std_lin_approach import LinearApproach
    from sklearn.metrics import explained_variance_score
    r = LinearApproach(numerical=X_train_reduced.columns, imp_iter=25)
    r.fit(X_train_reduced, y_train)

    if r2_target is not None:
        r2 = explained_variance_score(r2_target, r.predict(X_test_reduced))
        return r2
    
    results, feature_importances, residuals_my_lin, configs = r.cv_performance(X_train_reduced, y_train, 
                                                                        cv_folds=5, repeats=1)

    ## Test data
    results_my_lin = r.get_residuals(X_test_reduced, y_test, exp_var=False)

    if with_correction:
        from error_correction import ErrorRegressor
        residual_regressor = ErrorRegressor(numerical=X_train_reduced.columns, imp_iter=25)
        residual_regressor.fit(X_train_reduced, residuals_my_lin["Residual"])
        mylin_corr = residual_regressor.clean_residuals(X_test_reduced, results_my_lin["Residual"])
        results_my_lin["Corrected"] = mylin_corr["Corrected"]

    return results_my_lin, results["ExpVar_val"]

def run_std_approach(X_train, y_train, X_test, y_test, r2_target=None, combine=True):
    from standard_approach.std_lin_approach import LinearApproach
    from sklearn.metrics import explained_variance_score
    
    if combine:
        std_train = pd.concat((X_train, X_test))
        std_labels = pd.concat((y_train, y_test))
    else:
        std_train = X_test
        std_labels = y_test

    r = LinearApproach(numerical=X_train.columns, imp_iter=25)
    r.fit(std_train, std_labels)

    if r2_target is not None:
        preds = r.predict(X_test)
        r2 = explained_variance_score(r2_target, preds)
        return r2

    results_std_app = r.get_residuals(X_test, y_test)
    return results_std_app

def mix_data_splits(X_train, y_train, res_coefs, 
                    fraction, random_seed, pass_ids=False, pass_res_prop=False):
    # get samples to replace and samples to replace with
    train_sample_ids = X_train.sample(frac=fraction, random_state=random_seed).index
    
    y_train_new = y_train.copy()
    res_proportion = X_train.loc[train_sample_ids] @ res_coefs.values
    y_train_new[train_sample_ids] = y_train[train_sample_ids] + res_proportion

    if pass_ids and pass_res_prop:
        return y_train_new, train_sample_ids, res_proportion
    elif pass_ids:
        return y_train_new, train_sample_ids
    elif pass_res_prop:
        return y_train_new, res_proportion