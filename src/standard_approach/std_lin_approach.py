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


class LinearApproach():
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
    
    def cv_performance(self, X, y, cv_folds=5, repeats=3):
        """Estimate the model performance using cross-validation."""
        from sklearn.metrics import mean_squared_error, r2_score, \
            mean_absolute_percentage_error, explained_variance_score
        from sklearn.model_selection import KFold

        configs = {fold:{} for fold in range(1, cv_folds+1)}
        all_predictions = pd.DataFrame(columns=["Score", "Fold"], index=X.index)
        feature_importances = pd.DataFrame(index=X.columns)
        results = pd.DataFrame(columns=["rmse_train", "mape_train", "R2_train", "ExpVar_train",
                                        "rmse_val", "mape_val", "R2_val", "ExpVar_val"],
                               index=range(repeats))

        for i in range(repeats):
            results_i = {"rmse_train" : [],
                    "mape_train" : [],
                    "R2_train" : [],
                    "ExpVar_train" : [],
                    "rmse_val" : [],
                    "mape_val" : [],
                    "R2_val" : [],
                    "ExpVar_val" : []}
            
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
                results_i["rmse_train"].append(mean_squared_error(y_train, pred_train))
                results_i["mape_train"].append(mean_absolute_percentage_error(y_train, pred_train)/100)
                results_i["R2_train"].append(r2_score(y_train, pred_train))
                results_i["ExpVar_train"].append(explained_variance_score(y_train, pred_train))
                # validation score
                pred = model.predict(X_val)
                results_i["rmse_val"].append(mean_squared_error(y_val, pred)) 
                results_i["mape_val"].append(mean_absolute_percentage_error(y_val, pred)/100)
                results_i["R2_val"].append(r2_score(y_val, pred))
                results_i["ExpVar_val"].append(explained_variance_score(y_val, pred))
                
                # Get feature importance per outer fold
                feature_importances[f"{cur_fold}"] = model.coef_
                # store predictions
                all_predictions.iloc[test_inds, 0] = pred
                all_predictions.iloc[test_inds,1] = cur_fold
                # store model and CV configs
                configs[cur_fold+1]["test_inds"] = test_inds.tolist()
            
            results.loc[i] = pd.DataFrame(results_i).mean()
            # results = pd.concat([results, pd.DataFrame(results).mean()], axis=1)

        # re-arrange prediction table columns for easier reading
        all_predictions['Label'] = y
        all_predictions["Residual"] = all_predictions["Label"] - all_predictions["Score"]
        all_predictions = all_predictions[["Label", "Score", "Residual", "Fold"]]

        return (results, feature_importances, all_predictions, configs)
    
    def predict(self, X):
        X_pre = self.preprocess.transform(X)
        preds = self.model.predict(X_pre)
        return preds

    def save(self, save_path):
        pickle.dump(self.model, open(save_path, 'wb'))

    def load(self, save_path):
        self.model = pickle.load(open(save_path, 'rb'))

    def get_residuals(self, X, y, exp_var=True, model_path=None):
        """Regress model error estimated in training data out of the residuals gained from test data."""
        assert (self.model is not None) or (model_path is not None), "Either call .train() or provide model_path."
        if (self.model is None) or (model_path is not None):
            self.load(model_path)
        results = pd.DataFrame({"Label": y})
        results['Prediction'] = self.predict(X)
        results['Residual'] = results['Label'] - results['Prediction']

        if exp_var:
            from sklearn.metrics import explained_variance_score
            exp_var = explained_variance_score(results['Label'], results['Prediction'])
            return results, exp_var
        return results


if __name__ == "__main__":
    from data_loader import load_data
    from constants import DATA_PATH, STDAPP_NUMERICAL, STDAPP_CATEGORICAL, LABEL, \
    SUBGROUPING, SUBGROUPING_VAL

    ## Training
    # load training data used for the cross-validation experiment
    X_train, y_train = load_data(DATA_PATH, label=LABEL, numerical_vars=STDAPP_NUMERICAL, categorical_vars=STDAPP_CATEGORICAL,
                    subgrouping=SUBGROUPING)

    r = LinearApproach(numerical=STDAPP_NUMERICAL, imp_iter=25)

    # test CV performance
    results, feature_importances, predictions, configs = r.cv_performance(X_train, y_train, cv_folds=5, repeats=1)
    pd.DataFrame(results).to_csv("sent/final_models/reduced_features_stdapp/standardApp_cv_performance.out")
    print("Regression fitted. CV performance:", results)

    ## Test data
    # load_test_data_(different subgroup selection than in the cross-validation / model training)
    X_test, y_test = load_data(DATA_PATH, label=LABEL, numerical_vars=STDAPP_NUMERICAL, categorical_vars=STDAPP_CATEGORICAL,
                    subgrouping=SUBGROUPING_VAL)

    r.fit(X_test, y_test)
    results, exp_var = r.get_residuals(X_test, y_test)
    results.to_csv(f"sent/final_models/reduced_features_stdapp/standardApp_residuals_expvar:{exp_var*100}.out")
    
    #res_plot_save_path = "sent/final_models/xgboost/plots"
