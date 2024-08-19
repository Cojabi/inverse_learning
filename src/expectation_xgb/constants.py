from optuna.distributions import FloatDistribution, IntDistribution

RANDOM_SEED = 42

### XGBoost settings
CV_FOLDS = 5
REPEATS = 2
OPTUNA__N_TRIALS = 20
OPTUNA__N_JOBS = 2
XGB__NTHREAD = 4
N_ESTIMATORS = 8000
EARLY_STOPPING_ROUNDS = 30
OBJECTIVE = "rmse"

XGB_HP = {"gamma":FloatDistribution(1, 8),
         "learning_rate":FloatDistribution(1e-7, 1, log=True),
         "max_depth":IntDistribution(3,8),
         "subsample":FloatDistribution(0.7, 1, 0.1)}
