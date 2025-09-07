"""
EECS 445 Winter 2025

This script should contain most of the work for the project. You will need to fill in every TODO comment.
"""


import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
import helper


__all__ = [
    "generate_feature_vector",
    "impute_missing_values",
    "normalize_feature_matrix",
    "get_classifier",
    "performance",
    "cv_performance",
    "select_param_logreg",
    "select_param_RBF",
    "plot_weight",
]


# load configuration for the project, specifying the random seed and variable types
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
seed = config["seed"]
np.random.seed(seed)


def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.

    Args:
        df: pd.Dataframe, with columns [Time, Variable, Value]

    Returns:
        a python dictionary of format {feature_name: feature_value}
        for example, {"Age": 32, "Gender": 0, "max_HR": 84, ...}
    """
    timeseries_variables = config['timeseries']
    # TODO: 1) Replace unknown values with np.nan
    # NOTE: pd.DataFrame.replace() may be helpful here, refer to documentation for details
    df_replaced = df.replace(-1, np.nan)

    # Extract time-invariant and time-varying features (look into documentation for pd.DataFrame.iloc)
    static, timeseries = df_replaced.iloc[0:5], df_replaced.iloc[5:]

    feature_dict = {}
    # TODO: 2) extract raw values of time-invariant variables into feature dict
    for _, row in static.iterrows():
        feature_dict[row["Variable"]] = row["Value"]
    # TODO  3) extract max of time-varying variables into feature dict
    for variable in timeseries_variables:
        values = timeseries[timeseries["Variable"] == variable]["Value"]
        feature_dict['max_' + variable] = values.max() 
    return feature_dict


def impute_missing_values(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, impute missing values (np.nan) with the population mean for that feature.

    Args:
        X: array of shape (N, d) which could contain missing values
        
    Returns:
        X: array of shape (N, d) without missing values
    """
    # TODO: implement
    col_means = np.nanmean(X, axis=0)
    nan_indices = np.isnan(X)
    X[nan_indices] = np.take(col_means, np.where(nan_indices)[1])
    return X

def normalize_feature_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: array of shape (N, d).

    Returns:
        X: array of shape (N, d). Values are normalized per column.
    """
    # NOTE: sklearn.preprocessing.MinMaxScaler may be helpful
    # TODO: implement
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


def get_classifier(
    loss: str = "logistic",
    penalty: str | None = None,
    C: float = 1.0,
    class_weight: dict[int, float] | None = None,
    kernel: str = "rbf",
    gamma: float = 0.1,
) -> KernelRidge | LogisticRegression:
    """
    Return a classifier based on the given loss, penalty function and regularization parameter C.

    Args:
        loss: Specifies the loss function to use.
        penalty: The type of penalty for regularization.
        C: Regularization strength parameter.
        class_weight: Weights associated with classes.
        kernel : Kernel type to be used in Kernel Ridge Regression.
        gamma: Kernel coefficient.

    Returns:
        A classifier based on the specified arguments.
    """
    # TODO (optional, but highly recommended): implement function based on docstring
    if loss == "logistic":
        return LogisticRegression(penalty=penalty, C=C, class_weight=class_weight, solver='liblinear', fit_intercept=False, random_state=seed)
    elif loss == "squared_error":
        return KernelRidge(alpha=1/(2*C), kernel=kernel, gamma=gamma)

def get_sample_performance(y_true: npt.NDArray[np.int64],
                           y_pred: npt.NDArray[np.int64],
                           y_score: npt.NDArray[np.float64],
                           metric: str = "accuracy") -> np.float64:
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == "precision":
        return metrics.precision_score(y_true, y_pred, zero_division=0)
    elif metric == "f1_score":
        return metrics.f1_score(y_true, y_pred, zero_division=0)
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_score)
    elif metric == "average_precision":
        return metrics.average_precision_score(y_true, y_score)
    elif metric == "sensitivity":
        return metrics.recall_score(y_true, y_pred, pos_label=1)
    elif metric == "specificity":
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    

def performance(
    clf_trained: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y_true: npt.NDArray,
    metric: str = "accuracy"
) -> float:
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted scores from clf_trained and X.
    Returns single sample performance as specified by the user. Note: you may
    want to implement an additional helper function to reduce code redundancy.

    Args:
        clf_trained: a fitted instance of sklearn estimator
        X : (n,d) np.array containing features
        y_true: (n,) np.array containing true labels
        metric: string specifying the performance metric (default='accuracy'
                other options: 'precision', 'f1-score', 'auroc', 'average_precision',
                'sensitivity', and 'specificity')
    Returns:
        peformance for the specific metric
    """
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    # Get binary predictions
    y_pred = clf_trained.predict(X)
    if hasattr(clf_trained, "decision_function"):
        y_score = clf_trained.decision_function(X)  
    else:
        y_score=y_pred
        y_pred=np.where(y_score>=0,1,-1)

    return get_sample_performance(y_true, y_pred, y_score, metric)



def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y: npt.NDArray,
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.

    Args:
        clf: an instance of a sklearn classifier
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) vector of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy'
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')

    Returns:
        a tuple containing (mean, min, max) cross-validation performance across the k folds
    """
    # NOTE: you may find sklearn.model_selection.StratifiedKFold helpful
    skf = StratifiedKFold(n_splits=k, shuffle=False)  
    performances = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        performance_score = performance(clf, X_test, y_test, metric)
        performances.append(performance_score)
    return np.mean(performances), np.min(performances), np.max(performances)


def select_param_logreg(
    X: npt.NDArray,
    y: npt.NDArray,
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    penalties: list[str] = ["l2", "l1"],
) -> tuple[float, str]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression, calculating the k-fold CV
    performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric for which to optimize (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        penalties: a list of strings specifying the type of regularization penalties to be searched over

    Returns:
        The hyperparameters for a logistic regression model that maximizes the
        average k-fold CV performance.
    """
    # NOTE: use your cv_performance function to evaluate the performance of each classifier
    best_C, best_penalty = None, None
    highest_score = float("-inf")
    for penalty in penalties:
        for C in C_range:
            model = get_classifier(C=C, penalty=penalty)
            mean_score, _, _ = cv_performance(model, X, y, metric, k)
            if mean_score > highest_score:
                best_C, best_penalty, highest_score = C, penalty, mean_score
    return best_C, best_penalty


def select_param_RBF(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    gamma_range: list[float] = [],
) -> tuple[float, float]:
    """
    Sweeps different settings for the hyperparameter of a RBF Kernel Ridge Regression,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')
        C_range: an array with C values to be searched over
        gamma_range: an array with gamma values to be searched over

    Returns:
        The parameter values for a RBF Kernel Ridge Regression that maximizes the
        average k-fold CV performance.
    """
    # NOTE: this function should be similar to your implementation of select_param_logreg
    best_C, best_gamma = None, None
    highest_score = float("-inf")
    for C in C_range:
        for gamma in gamma_range:
            clf = get_classifier(loss='squared_error', C=C, kernel='rbf', gamma=gamma)
            mean_score, _, _ = cv_performance(clf, X, y, metric=metric, k=k)
            if mean_score > highest_score:
                best_C, best_gamma, highest_score = C, gamma, mean_score
    return best_C, best_gamma



def plot_weight(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    penalties: list[str],
) -> None:
    """
    The funcion takes training data X and labels y, plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier, and saves the plot.
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}

    Returns:
        None
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")

    for penalty in penalties:
        norm0 = []
        for C in C_range:
            # TODO: initialize clf with C and penalty
            clf = get_classifier(loss='logistic', C=C, penalty=penalty)
            
            # TODO: fit clf to X and y
            clf.fit(X, y)
            # TODO: extract learned coefficients from clf into w
            # NOTE: the sklearn.linear_model.LogisticRegression documentation will be helpful here
            w = clf.coef_.flatten()
            
            # TODO: count the number of nonzero coefficients and append the count to norm0
            non_zero_count = np.count_nonzero(w)
            norm0.append(non_zero_count)

        # This code will plot your L0-norm as a function of C
        plt.plot(C_range, norm0)
        plt.xscale("log")
    plt.legend([penalties[0], penalties[1]])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")

    plt.savefig("L0_Norm.png", dpi=200)
    plt.show()  

# 1(d)
def report_feature_statistics(X, feature_names):
    means = np.mean(X, axis=0)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    feature_summary = pd.DataFrame({
        "Feature Name": feature_names,
        "Mean Value": means,
        "Interquartile Range (IQR)": iqr
    })
    print(feature_summary)

# 2(c)
def find_best_hyperparameters_df(X_train, y_train, metric_list):
    C_range = np.geomspace(1e-3, 1e3, num=7) 
    penalties = ['l1', 'l2']
    results = []
    for metric in metric_list:
        best_C, best_penalty = select_param_logreg(X_train, y_train, metric, 5, C_range, penalties)
        mean_score, min_score, max_score = cv_performance(
            LogisticRegression(penalty=best_penalty, C=best_C, solver='liblinear', fit_intercept=False, random_state=42),
            X_train, y_train, metric, 5
        )
        results.append([metric, best_C, best_penalty, f"{mean_score:.4f} ({min_score:.4f}, {max_score:.4f})"])
    df_results = pd.DataFrame(results, columns=["Performance Measure", "Best C", "Penalty", "Mean (Min, Max) CV Performance"])
    print(df_results) 
    return df_results

# 2(d)
def evaluate_test_performance(X_train, y_train, X_test, y_test, metric_list):
    best_C, best_penalty = select_param_logreg(X_train, y_train, metric="auroc", k=5, 
                                               C_range=np.geomspace(1e-3, 1e3, num=7), penalties=['l1', 'l2'])
    model = get_classifier(penalty=best_penalty, C=best_C)
    model.fit(X_train, y_train)
    results = [[metric, performance(model, X_test, y_test, metric)] for metric in metric_list]
    df_results = pd.DataFrame(results, columns=["Performance Measures", "Test Performance"])
    print(f"C = {best_C}, penalty = {best_penalty}")
    print(df_results) 
    return df_results 

# 2(f)
def find_most_predictive_features(X_train, y_train, feature_names):
    model = get_classifier(penalty='l1', C=1)
    model.fit(X_train, y_train)
    coefficients = model.coef_.flatten()
    top_positive_indices = np.argsort(coefficients)[-4:][::-1]  
    top_negative_indices = np.argsort(coefficients)[:4] 
    df = pd.DataFrame({
        "Positive Coefficient": coefficients[top_positive_indices],
        "Feature Name (Positive)": [feature_names[i] for i in top_positive_indices],
        "Negative Coefficient": coefficients[top_negative_indices],
        "Feature Name (Negative)": [feature_names[i] for i in top_negative_indices]
    })
    print(df)
    return df

# 3.1(b)
def train_weighted_logreg(X_train, y_train, X_test, y_test, metric_list):
    print("class_weight={-1: 1, 1: 50}")
    class_weight = {-1: 1, 1: 50}
    clf = get_classifier(C=1.0, penalty='l2', class_weight={-1: 1, 1: 50})
    clf.fit(X_train, y_train)
    results = []
    for metric in metric_list:
        median_perf = performance(clf, X_test, y_test, metric)
        results.append([metric, f"{median_perf:.4f}"])
    df_results = pd.DataFrame(results, columns=["Performance Measure", "Test Performance"])
    print(df_results)

# 3.2(a)
def find_best_class_weights(X_train, y_train, metric="f1_score"):
    class_ratios = [1, 5, 10, 20, 50, 100]  
    best_f1 = 0
    best_Wn, best_Wp = None, None
    for Wp in class_ratios:
        class_weights = {-1: 1, 1: Wp} 
        clf = get_classifier(C=1.0, penalty='l2', class_weight=class_weights)
        mean_f1, _, _ = cv_performance(clf, X_train, y_train, metric, k=5)  
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_Wn, best_Wp = 1, Wp
    print(f"Best class weights: Wn = {best_Wn}, Wp = {best_Wp} with {metric} = {best_f1:.4f}")
    print(best_Wn, best_Wp) 
    return best_Wn, best_Wp

# 3.2(b)
def evaluate_class_weights(X_train, y_train, X_test, y_test, metric_list):
    print("class_weight={-1: 1, 1: 5}")
    clf = get_classifier(C=1.0, penalty='l2', class_weight={-1: 1, 1: 5})
    clf.fit(X_train, y_train)
    results = []
    for metric in metric_list:
        median_perf = performance(clf, X_test, y_test, metric)
        results.append([metric, f"{median_perf:.4f}"])
    df_results = pd.DataFrame(results, columns=["Performance Measure", "Test Performance"])
    print(df_results)

# 3.3(a)
def plot_roc_curves(X_train, y_train, X_test, y_test):
    class_weights = [{-1: 1, 1: 1}, {-1: 1, 1: 5}]
    labels = ["Wn=1, Wp=1", "Wn=1, Wp=5"]
    plt.figure(figsize=(8, 6))
    for class_weight, label in zip(class_weights, labels):
        clf = get_classifier(loss='logistic', penalty='l2', C=1.0, class_weight=class_weight)
        clf.fit(X_train, y_train)
        y_score = clf.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUROC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Different Class Weights")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("ROC_Curves.png", dpi=200)
    plt.show()

# 4.1(b)
def compare_models(X_train, y_train, X_test, y_test, metrics_list):
    print("C = 0.1, penalty = 'l2'")
    log_reg = get_classifier(loss='logistic', C=1.0, penalty='l2')
    ridge_reg = get_classifier(loss='squared_error', C=1.0, kernel='linear')
    log_reg.fit(X_train, y_train)
    ridge_reg.fit(X_train, y_train)
    results = {"Metric": [], "Logistic Regression": [], "Ridge Regression": []}
    for metric in metrics_list:
        results["Metric"].append(metric)
        results["Logistic Regression"].append(performance(log_reg, X_test, y_test, metric))
        results["Ridge Regression"].append(performance(ridge_reg, X_test, y_test, metric))
    df_results = pd.DataFrame(results)
    print(df_results)
    return df_results

# 4.2(b)
def report_cv_performance(X_train, y_train, gamma_range=[0.001, 0.01, 0.1, 1, 10, 100]):
    results = []
    for gamma in gamma_range:
        clf = get_classifier(loss='squared_error', C=1.0, kernel='rbf', gamma=gamma)  
        mean_score, min_score, max_score = cv_performance(clf, X_train, y_train, metric="auroc", k=5)  
        results.append((gamma, mean_score, min_score, max_score))
    df_results = pd.DataFrame(results, columns=["Gamma", "Mean", "Min", "Max"])
    print(df_results)
    return df_results

# 4.2(c)
def report_test_performance(X_train, y_train, X_test, y_test):
    C_range = [0.01, 0.1, 1.0, 10, 100]
    gamma_range = [0.01, 0.1, 1, 10]
    best_C, best_gamma = select_param_RBF(X_train, y_train, metric="auroc", k=5, C_range=C_range, gamma_range=gamma_range)
    clf = get_classifier(loss="squared_error", C=best_C, kernel="rbf", gamma=best_gamma)
    clf.fit(X_train, y_train)
    metrics_list = ["accuracy", "precision", "f1_score", "auroc", "average_precision", "sensitivity", "specificity"]
    results = {metric: performance(clf, X_test, y_test, metric) for metric in metrics_list}
    print(f"\nBest Parameters: C = {best_C}, γ = {best_gamma}")
    df_results = pd.DataFrame(list(results.items()), columns=["Performance Measure", "Test Performance"])
    print(df_results)
    return df_results

###############################################################################################################################
# Challenge
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def generate_feature_vector_advanced(df: pd.DataFrame, use_subset=False) -> dict:
    """
    For a single patient's raw data (DataFrame with columns [Time, Variable, Value]),
    extract an advanced feature vector with several methods:
      - For static variables: treat numeric and categorical variables differently.
        For numeric variables, store the original value and (if > 0) its log transform.
        For categorical variables such as ICUType, perform one-hot encoding.
      - For time-series variables:
            * Compute full 48-hour statistics: mean, std, min, max, median, IQR, and count.
            * Split the 48 hours into the first 24 hours and the last 24 hours, compute the mean in each,
              and compute a 'change' feature: (last 24h mean - first 24h mean) to reflect deterioration.
            * Also, compute the mean and std over the last 12 hours to highlight recent changes.
      - If use_subset is True, only retain the last-12-hour features.
    """
    df_replaced = df.replace(-1, np.nan)
    
    static_df = df_replaced.iloc[0:5].copy()
    timeseries_df = df_replaced.iloc[5:].copy()
    
    feature_dict = {}
    
    for _, row in static_df.iterrows():
        var = row["Variable"]
        val = row["Value"]
        if var == "ICUType":
            possible_types = ["Med-Surg", "Cardiac", "Neuro", "Surgical"]
            for cat in possible_types:
                feature_dict[f"{var}_{cat}"] = 1.0 if val == cat else 0.0
        elif isinstance(val, str):
            feature_dict[f"{var}_{val}"] = 1.0
        else:
            feature_dict[var] = val
            if pd.notna(val) and val > 0:
                feature_dict[f"{var}_log"] = np.log(val)
            else:
                feature_dict[f"{var}_log"] = np.nan

    grouped = timeseries_df.groupby("Variable")
    for var, group in grouped:
        times = pd.to_numeric(group["Time"], errors="coerce")
        values = group["Value"]
        stats_full = {
            f"{var}_mean": values.mean(),
            f"{var}_std": values.std(),
            f"{var}_min": values.min(),
            f"{var}_max": values.max(),
            f"{var}_median": values.median(),
            f"{var}_iqr": values.quantile(0.75) - values.quantile(0.25),
            f"{var}_count": values.count()
        }
        
        mask_0_24 = times < 24
        values_0_24 = values[mask_0_24]
        mean_0_24 = values_0_24.mean() if len(values_0_24) > 0 else np.nan
        
        mask_24_48 = times >= 24
        values_24_48 = values[mask_24_48]
        mean_24_48 = values_24_48.mean() if len(values_24_48) > 0 else np.nan
        
        change_24 = mean_24_48 - mean_0_24 if pd.notna(mean_0_24) and pd.notna(mean_24_48) else np.nan
        stats_24_change = {
            f"{var}_mean_0_24": mean_0_24,
            f"{var}_mean_24_48": mean_24_48,
            f"{var}_change_24": change_24
        }
        
        mask_last12 = times >= (48 - 12)
        values_last12 = values[mask_last12]
        stats_last12 = {
            f"{var}_mean_last12": values_last12.mean() if len(values_last12) > 0 else np.nan,
            f"{var}_std_last12": values_last12.std() if len(values_last12) > 0 else np.nan
        }
        
        if use_subset:
            feature_dict.update(stats_last12)
        else:
            feature_dict.update(stats_full)
            feature_dict.update(stats_24_change)
            feature_dict.update(stats_last12)
            
    return feature_dict

def impute_missing_values_advanced(X: np.ndarray, strategy="median") -> np.ndarray:
    col_all_nan = np.all(np.isnan(X), axis=0)
    if np.any(col_all_nan):
        print("Dropping", np.sum(col_all_nan), "columns that are entirely NaN.")
        X = X[:, ~col_all_nan]
    
    if strategy == "median":
        fill_values = np.nanmedian(X, axis=0)
    elif strategy == "mean":
        fill_values = np.nanmean(X, axis=0)
    else:
        raise ValueError("Unsupported strategy. Use 'median' or 'mean'.")
    
    fill_values = np.where(np.isnan(fill_values), 0, fill_values)
    missing_indicator = np.isnan(X).astype(float)
    X_imputed = np.where(np.isnan(X), fill_values, X)
    X_augmented = np.hstack([X_imputed, missing_indicator])
    return X_augmented

def normalize_feature_matrix_advanced(X: np.ndarray, scaling_method="minmax") -> np.ndarray:
    if scaling_method == "standard":
        scaler = StandardScaler()
    elif scaling_method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported scaling_method. Use 'standard' or 'minmax'.")
        
    X_scaled = scaler.fit_transform(X)
    return X_scaled

    
def main():
    print(f"Using Seed = {seed}")
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED IMPLEMENTING generate_feature_vector,
    #       fill_missing_values AND normalize_feature_matrix!
    # NOTE: Only set debug=True when testing your implementation against debug.txt. DO NOT USE debug=True when
    #       answering the project questions! It only loads a small sample (n = 100) of the data in debug mode,
    #       so your performance will be very bad when working with the debug data.
    X_train, y_train, X_test, y_test, feature_names = helper.get_project_data(debug=False)

    metric_list = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    # TODO: Questions 1, 2, 3, 4
    # NOTE: It is highly recomended that you create functions for each
    #       sub-question/question to organize your code!
    
    

    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       helper.save_challenge_predictions to save your predicted labels
    X_challenge, y_challenge, X_heldout, feature_names = helper.get_challenge_data(use_advanced=True)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_challenge, y_challenge, test_size=0.20, stratify=y_challenge, random_state=seed
    )
    
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_val)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_val)
    best_class_weights = {-1: 1, 1: 5}
    print("Computed class weights (from validation data):", best_class_weights)
    
    C_range = np.geomspace(1e-2, 1e2, num=5)
    penalties = ['l1', 'l2']
    best_C, best_penalty = select_param_logreg(X_train_split, y_train_split, metric="f1_score", k=5, 
                                               C_range=C_range, penalties=penalties)
    print(f"Best hyperparameters from CV: C = {best_C}, penalty = {best_penalty}")
    
    tuned_model = get_classifier(loss='logistic', penalty=best_penalty, C=best_C, class_weight=best_class_weights)
    tuned_model.set_params(max_iter=1000)
    tuned_model.fit(X_train_split, y_train_split)
    
    f1_val = performance(tuned_model, X_val, y_val, metric="f1_score")
    auroc_val = performance(tuned_model, X_val, y_val, metric="auroc")
    print(f"Validation Performance (default threshold) -- F1-score: {f1_val:.4f}, AUROC: {auroc_val:.4f}")
    
    if hasattr(tuned_model, "decision_function"):
        decision_scores_val = tuned_model.decision_function(X_val)
    else:
        decision_scores_val = tuned_model.predict_proba(X_val)[:, 1]
    
    thresholds = np.linspace(np.min(decision_scores_val), np.max(decision_scores_val), num=50)
    best_threshold = thresholds[0]
    best_f1_threshold = 0
    for t in thresholds:
        y_val_pred_thresh = np.where(decision_scores_val >= t, 1, -1)
        current_f1 = metrics.f1_score(y_val, y_val_pred_thresh, zero_division=0)
        if current_f1 > best_f1_threshold:
            best_f1_threshold = current_f1
            best_threshold = t
    print(f"Optimal decision threshold on validation set: {best_threshold:.4f} with F1 = {best_f1_threshold:.4f}")
    
    final_model = get_classifier(loss='logistic', penalty='l2', C=100.0, class_weight=best_class_weights)
    final_model.set_params(max_iter=1000)
    final_model.fit(X_challenge, y_challenge)
    
    y_train_final_pred = final_model.predict(X_challenge)
    if hasattr(final_model, "decision_function"):
        y_train_final_score = final_model.decision_function(X_challenge)
    else:
        y_train_final_score = final_model.predict_proba(X_challenge)[:, 1]
    f1_final = performance(final_model, X_val, y_val, metric="f1_score")
    auroc_final = performance(final_model, X_val, y_val, metric="auroc")
    cm_final = confusion_matrix(y_challenge, y_train_final_pred, labels=[-1, 1])
    print("Final Model Performance on validation set:")
    print(f"F1-score: {f1_final:.4f}, AUROC: {auroc_final:.4f}")
    print("Confusion Matrix on Challenge Training Data:")
    print(cm_final)
    
    if hasattr(final_model, "decision_function"):
        decision_scores_heldout = final_model.decision_function(X_heldout)
    else:
        decision_scores_heldout = final_model.predict_proba(X_heldout)[:, 1]
    
    y_label = np.where(decision_scores_heldout >= best_threshold, 1, -1)
    
    uniqname = "yanshun"  
    helper.save_challenge_predictions(y_label, decision_scores_heldout, uniqname)
    print(f"Predictions saved to {uniqname}.csv")
if __name__ == "__main__":
    main()
