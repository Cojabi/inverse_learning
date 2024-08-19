import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from simulated_data_experiments.constants_sim import IL_COLOR

# feature importance plot
def feature_importance_plot(feature_importances: pd.DataFrame, save_path: str) -> None:
    """Plot feature importances.

    feature_importances : pd.DataFrame. Feature importances.
    save_path : str. Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    feature_importances["Importance"] = feature_importances.mean(axis=1)
    feature_importances["StErr"] = feature_importances.sem(axis=1)
    feature_importances = feature_importances.sort_values("Importance", ascending=False)
    feature_importances["Importance"].plot.bar(ax=ax, yerr=feature_importances["StErr"], capsize=3)
    
    ax.set_title("Feature importance")
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{save_path}/feature_importances.png")
    plt.close()

def plot_reg_perf(results: pd.DataFrame, save_path: str, show=False, close=True):
    """Plot the performance scores of a regression model."""
    sns.boxplot(data=results, palette="Set2")
    plt.title("Performance")
    plt.ylabel("Value")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{save_path}/performance.png")
    plt.close()

def plot_training_diagnostics(eval_results, save_path: str):
    """Plot the training curves. 
    Only works for the first metric in eval_results. Plus only for 2 eval sets."""
    fig, ax = plt.subplots(figsize=(5, 5))

    # get the metric name
    key = next(iter(eval_results))
    eval_metric = next(iter(eval_results[key]))
    # plot metric for each eval set (Train, Test)
    keys = iter(eval_results)
    for i in range(len(eval_results)):
        ax.plot(eval_results[next(keys)][eval_metric], label=f"{['Train', 'Val'][i]}")

    ax.set_title(f"Training curves ({eval_metric})")
    ax.set_ylabel("Objective")
    ax.set_xlabel("Iteration (Boosting round)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}_training_curve.png")
    plt.close()

def plot_residual_correction(results, save_path=None):
    """Plot the residuals vs the corrected residuals."""
    # color patients that flip sign during correction
    flipped = results[np.sign(results['Corrected']) != np.sign(results['Residual'])].index
    unflipped = results.index.difference(flipped)
    plt.figure(figsize=(7, 7))
    plt.plot([-1.8, 2.5], [-1.8, 2.5], 'k--', label='x=y')
    plt.plot(results.loc[unflipped, 'Residual'], results.loc[unflipped, 'Corrected'], 'o', alpha=0.5)
    plt.plot(results.loc[flipped, 'Residual'], results.loc[flipped, 'Corrected'], 'o', 
             color='#faae20', label='Flipped sign', alpha=0.5)
    plt.title('Residuals vs Corrected Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Corrected Residuals')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_residual_error(results, savepath=None):
    """Diagnostic plot to see how the predicted model error 
    relates to the value of the residuals."""
    # find patients that flip their residual sign
    flipped = results[np.sign(results['Corrected']) != np.sign(results['Residual'])].index
    unflipped = results.index.difference(flipped)

    plt.plot(results.loc[unflipped, "Residual"], results.loc[unflipped, 'Pred_Error'], 'o', alpha=0.5)
    plt.plot(results.loc[flipped, 'Residual'], results.loc[flipped, 'Pred_Error'], 'o', 
             color='#faae20', label='Flipped sign', alpha=0.5)
    plt.xlabel('Residual')
    plt.ylabel('Estimated model error')
    plt.title('Residuals vs. estimated model error')
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals(residuals, savepath=None):
    """Plot the residuals.
    residuals : pd.DataFrame. Residual table gained from the residual_regressor.clean_residuals function
    or run_IL_lin function."""
    ids = residuals[residuals['Corrected'] > 0].index

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.violinplot(residuals['Corrected'], color="#0ce3eb", orient='h', alpha=0.3, inner=None)
    sns.swarmplot(residuals.drop(ids)['Corrected'], color='Grey', orient='h')
    sns.swarmplot(x='Corrected', y=None, hue='Corrected', data=residuals.loc[ids], palette='cool', orient='h')
    plt.title('Corrected residuals', weight='bold')
    plt.xlabel('Residual value')
    # remove legend
    plt.legend([],[], frameon=False)
    # remove y-axis ticks
    plt.yticks([])
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()

def plot_residual_vs_outcome(test_residual, label, outcome_str="Outcome", savepath=None):
    """Plot the corrected residuals vs the true outcome."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # get regression line
    regr = LinearRegression()
    regr.fit(label.values.reshape(-1, 1), test_residual["Corrected"].values.reshape(-1, 1))
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.plot(label, test_residual["Corrected"], 'o', alpha=0.5)
    sns.lineplot(x=label, y=regr.predict(label.values.reshape(-1, 1)).reshape(-1),
                linewidth=3, color="#f2bd2c")
    upper = np.max([label.max(), test_residual["Corrected"].max()])
    lower = np.min([label.min(), test_residual["Corrected"].min()])
    plt.xlim(lower-0.05*lower, upper+0.05*upper)
    plt.ylim(lower-0.05*lower, upper+0.05*upper)
    plt.xlabel(f"True {outcome_str}")
    plt.ylabel("Corrected Residual")
    plt.title(f"Corrected Residual vs True {outcome_str}", weight='bold')
    plt.text(.02, .98, f"R^2 = {r2_score(label, test_residual['Corrected']):.2f}", 
             ha='left', va='top', transform=ax.transAxes)
    # save
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_residual_distribution(test_residual_table, train_residual_table, savepath=None):
    fig, axes = plt.subplots(1,1, figsize=(6,4))
    sns.histplot(x=train_residual_table["Residual_mean"], bins=20, alpha=0.5)
    sns.histplot(x=test_residual_table["Residual"], bins=20, alpha=0.5)
    plt.legend(['Training data (CogOut=1)', 'Test data (CogOut=0)'])
    plt.xlabel('Residual')
    plt.title('Residual distribution')
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

def residual_plot_wrapper(test_residual_table, label, 
                          train_residual_table, outcome_str, savepath):
    """Wrapper to create all the residual plots."""
    plot_residuals(test_residual_table, f"{savepath}/residuals.png")
    plot_residual_distribution(test_residual_table, train_residual_table, f"{savepath}/error_distribution.png")
    plot_residual_error(test_residual_table, f"{savepath}/residual_error.png")
    plot_residual_correction(test_residual_table, f"{savepath}/residual_correction.png")
    plot_residual_vs_outcome(test_residual_table, label, outcome_str, f"{savepath}/residual_vs_{outcome_str}.png")

def performance_comparison(non_lin_perf_path, lin_perf_path, save_path=None):
    """Plot a comparison of the performance of the standard linear approach and our new, non-linear approach."""
    non_lin_perf = pd.read_csv(non_lin_perf_path, index_col=0)
    lin_perf = pd.read_csv(lin_perf_path, index_col=0)

    non_lin_perf['Model'] = 'Non-linear'
    lin_perf['Model'] = 'Linear'
    rename = {
        "mape_train": "MAPE Train",
        "mape_val": "MAPE Validation",
        "rmse_train": "RMSE Train",
        "rmse_val": "RMSE Validation",
        "R2_train": "R² Train",
        "R2_val": "R² Validation",
    }
    perf = pd.concat([non_lin_perf, lin_perf]).rename(columns=rename).melt(id_vars=["Model"])

    fig = plt.figure(figsize=(11, 5))
    sns.boxplot(x='variable', y='value', data=perf, hue="Model", palette="Set2", width=0.5)
    plt.ylim((0, 1))
    plt.ylabel("Value")
    plt.xlabel("Performance Metric")
    plt.title("Comparison of ground-truth model performance", weight="bold")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def residual_concordance_plot(non_lin_res_path, lin_res_path, save_path=None):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    lin_res = pd.read_csv(lin_res_path, index_col=0)
    non_lin_res = pd.read_csv(non_lin_res_path, index_col=0)

    # get regression line
    regr = LinearRegression()
    regr.fit(lin_res['Residual'].values.reshape(-1, 1), non_lin_res["Corrected"].values.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    upper_lim = max(max(lin_res['Residual']), max(non_lin_res['Corrected']))
    lower_lim = min(min(lin_res['Residual']), min(non_lin_res['Corrected']))
    plt.plot([lower_lim, upper_lim], [lower_lim, upper_lim], '--', lw=2, color='grey', alpha=0.8)

    plt.plot(lin_res['Residual'], non_lin_res['Corrected'], 'o', alpha=0.5)
    sns.lineplot(x=lin_res['Residual'], y=regr.predict(lin_res['Residual'].values.reshape(-1, 1)).reshape(-1),
                    linewidth=3, color="#f2bd2c")

    plt.ylim(lower_lim-0.1*lower_lim, upper_lim+0.1*upper_lim) 
    plt.xlim(lower_lim-0.1*lower_lim, upper_lim+0.1*upper_lim)
    plt.xlabel('Standard approach residuals')
    plt.ylabel('Predictive approach residuals')
    plt.title('Residuals Comparison')

    plt.text(.02, .98, f"R² = {r2_score(lin_res['Residual'], non_lin_res['Corrected']):.2f}", 
                ha='left', va='top', transform=ax.transAxes)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def approach_comparison_plot(res_proportion, std_app_table, my_lin_table, xgb_table,
                             title="True vs extracted residuals", 
                             plot_titles=["Linear model", "Linear model + correction", 
                                          "XGBoost", "XGBoost + correction"], 
                             save_path=None):
    from sklearn.metrics import mean_squared_error
    fig, axes = plt.subplots(1, 5, figsize=(14, 3), sharex=True, sharey=True)
    plt.suptitle(title, y=1)

    axes[0].set_title("Std. approach")
    axes[0].scatter(res_proportion, std_app_table["Residual"], alpha=0.5, s=20)
    axes[0].plot(res_proportion, res_proportion, "-", color='grey')
    plt.text(.02, .98, f"MSE: {mean_squared_error(res_proportion, std_app_table['Residual']):.2f}", 
                    ha='left', va='top', transform=axes[0].transAxes)
    axes[0].set_xlabel("True residual")

    for to_plot, plot_title, ax in zip([my_lin_table["Residual"], my_lin_table['Corrected'], 
                                        xgb_table["Residual"], xgb_table['Corrected']], 
                                        plot_titles, axes[1::]):
        ax.set_title(plot_title)
        ax.scatter(res_proportion, to_plot, alpha=0.5, s=20, color=IL_COLOR)
        ax.plot(res_proportion, res_proportion, "-", color='grey')
        plt.text(.02, .98, f"MSE: {mean_squared_error(res_proportion, to_plot):.2f}", 
                        ha='left', va='top', transform=ax.transAxes)
        ax.set_xlabel("True residual")
        
    axes[0].set_ylabel("Extracted residual")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()