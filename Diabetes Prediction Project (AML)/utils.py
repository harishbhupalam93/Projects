import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)

# function for getting the feature names in FeatureTransformer
def col_out(transformer, feature_names):
    return [f'{col}_log1p' for col in feature_names]

# function to print the model coefficients
def model_summary(pipeline, model):
    if model == 'linear':
        feature_names = ["Constant"] + pipeline[0].get_feature_names_out().tolist()
        coef = [pipeline[-1].intercept_] + pipeline[-1].coef_.tolist()
        model_summary = pd.DataFrame({"Feature_Names": feature_names, "Coefficient": coef})
    elif model == 'logistic':
        feature_names = ["Constant"] + pipeline[0].get_feature_names_out().tolist()
        coef = [pipeline[-1].intercept_[0]] + pipeline[-1].coef_[0].tolist()
        model_summary = pd.DataFrame({"Feature_Names": feature_names, "Coefficient": coef})
        model_summary["odds"] = np.exp(model_summary["Coefficient"])
        model_summary["percent_change_odds"] = (np.exp(model_summary["Coefficient"]) - 1) * 100
    return model_summary

# function to compute adjusted R-squared
def adj_r2_score(predictors, targets, predictions):
    r2 = r2_score(targets, predictions)
    n = predictors.shape[0]
    k = predictors.shape[1]
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

# function to compute different metrics to check performance of a regression model
def model_performance_regression(predictors, target, predictions):
    r2 = r2_score(target, predictions)  # to compute R-squared
    adjr2 = adj_r2_score(predictors, target, predictions)  # to compute adjusted R-squared
    rmse = np.sqrt(mean_squared_error(target, predictions))  # to compute RMSE
    mae = mean_absolute_error(target, predictions)  # to compute MAE

    # creating a dataframe of metrics
    df_perf = pd.DataFrame({"RMSE": rmse,"MAE": mae,"R-squared": r2,"Adj. R-squared": adjr2,},index=[0],)

    return df_perf

# defining a function to plot the confusion_matrix of a classification model
def confusion_matrix_helper(model, predictors, target, threshold=0.5):
    y_pred = model.predict_proba(predictors)[:,1] > threshold
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)
    
    return cm, labels

# defining a function to compute different metrics to check performance of a classification model built using statsmodels
def model_performance_classification_sklearn(model, predictors, target, threshold=0.5):
    # predicting using the independent variables
    pred = model.predict_proba(predictors)[:,1] > threshold

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame({"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1}, index=[0],)

    return df_perf

