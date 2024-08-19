import pandas as pd

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
        from src.error_correction import ErrorRegressor
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