import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_friedman1, make_friedman2
from sklearn.preprocessing import MinMaxScaler

import torch
from scipy import stats

class DataGenerator():
    def __init__(self, random_seed=None) -> None:
        if random_seed is not None:
            self.random_seed = random_seed
            np.random.seed(random_seed)
    
    def sample_from_normal(self, n_samples, corr, mean=0., std=1):
        """Sample from a multivariate normal distribution with a given correlation matrix."""
        # turn correlation matrix into covariance matrix, assuming std is shared
        df_std = [std]*corr.shape[0]
        cov = np.outer(df_std, df_std) * corr

        dist = stats.multivariate_normal(mean=[mean]*cov.shape[0], cov=cov, seed=self.random_seed)
        samples = dist.rvs(n_samples)
        return samples
    
    def generate_linear_data(self, n_samples, n_informative, effective_rank, train_split=.33, res_coef_value=50, 
                      gt_coef_value=50, noise=0, n_common=0, feature_range=(-3, 3), method="mr", 
                      deducted_frac=0, return_res_coefs=True,):
        
        split = np.round(n_samples * train_split).astype(int)
        n_feats = n_informative * 3 + n_common

        # get data matrix
        if method == 'gaussian':
            corr = self.fill_corr_matrix(n_informative, deducted_frac=deducted_frac)
            X = self.sample_from_normal(n_samples, corr)
        elif method == 'f1':
            X, y = make_friedman1(n_samples=n_samples, n_features=n_feats, noise=noise, 
                                  random_state=self.random_seed)
            y = torch.tensor(y, dtype=torch.float64)
            return_res_coefs = False
        elif method == 'f2':
            X, y = make_friedman2(n_samples=n_samples, noise=noise, random_state=self.random_seed)
            y = torch.tensor(y, dtype=torch.float64)
            return_res_coefs = False
            n_feats = 4
            n_informative = 2
        else:
            X, _ = make_regression(n_samples=n_samples, effective_rank=effective_rank,  n_features=n_feats,
                                    n_informative=n_informative, noise=noise, random_state=self.random_seed)
        # min max scale so that different generations are on the same numerical scale for plots
        n = MinMaxScaler(feature_range=feature_range)
        X = n.fit_transform(X)
        X = torch.tensor(X, dtype=torch.float64)

        # get outcomes
        if not method == 'f1' and not method == 'f2':
            coefs = torch.zeros(n_feats, dtype=torch.float64)
            coefs[:n_informative] = gt_coef_value
            coefs[n_informative:n_informative+n_common] = gt_coef_value
            y = X @ coefs
            coefs = pd.Series(coefs)

        X_train = X[:split].clone().detach()
        y_train = y[:split].clone().detach()
        X_test = X[split:].clone().detach()
        y_test_no_coi = y[split:].clone().detach()

        res_coefs = torch.zeros(n_feats, dtype=torch.float64)
        res_coefs[-n_informative:] = res_coef_value
        res_coefs[n_informative:n_informative+n_common] = res_coef_value
        res_proportion = (X_test @ res_coefs).detach().numpy()
        y_test = y_test_no_coi + res_proportion

        X_train = pd.DataFrame(X_train, 
                               columns=[f'{i}' for i in range(X_train.shape[1])],
                               index=[i for i in range(X_train.shape[0])])
        X_test = pd.DataFrame(X_test, 
                              columns=[f'{i}' for i in range(X_train.shape[1])],
                              index=[i for i in range(X_train.shape[0], X_train.shape[0]+X_test.shape[0])])
        y_train = pd.Series(y_train, 
                            index=[i for i in range(X_train.shape[0])])
        y_test = pd.Series(y_test, 
                           index=[i for i in range(X_train.shape[0], X_train.shape[0]+X_test.shape[0])]).map(float)
        y_test_no_coi = pd.Series(y_test_no_coi, 
                                  index=[i for i in range(X_train.shape[0], X_train.shape[0]+X_test.shape[0])])
        res_proportion = pd.Series(res_proportion, 
                                  index=[i for i in range(X_train.shape[0], X_train.shape[0]+X_test.shape[0])])
        res_coefs = pd.Series(res_coefs)

        if return_res_coefs:
            return X_train, y_train, X_test, y_test, y_test_no_coi, res_proportion, coefs, res_coefs
        return X_train, y_train, X_test, y_test, y_test_no_coi, res_proportion
    
    @staticmethod
    def fill_corr_matrix(n_informative, n_common=0, deducted_frac=0):
        """Function to generate a correlation matrix that can be manipulated for different experiments."""
        n = 3*n_informative + n_common
        across_inform_corr = (0.9999 / n_informative) * (1 - deducted_frac)
        cov = np.diag([1.0]*n)
        cov[:n_informative, -n_informative:] = across_inform_corr # corr between gt and COI preds
        cov[-n_informative:, :n_informative] = across_inform_corr # same as above for symmetry
        # cov[:n_informative, :n_informative] = n_inform_corr # gt pred correlation
        # cov[-n_informative:, -n_informative:] = n_inform_corr # COI pred correlation
        # cov[n_informative:n_informative+n_common, n_informative:n_informative+n_common] = n_common_corr
        # cov += np.random.normal(-1, 1, (n, n))
        cov = np.maximum(cov, cov.transpose())
        np.fill_diagonal(cov, 1.0)
        return cov