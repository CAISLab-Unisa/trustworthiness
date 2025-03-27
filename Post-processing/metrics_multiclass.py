import numpy as np
import pandas as pd

def compute_metrics(conf_matrix):
    conf_matrix = np.array(conf_matrix)
    num_classes = conf_matrix.shape[0]

    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    TN = np.sum(conf_matrix) - (TP + FP + FN)

    # Metrics
    accuracy = np.sum(TP) / np.sum(conf_matrix)
    TP_total = np.sum(TP)
    FP_total = np.sum(FP)
    FN_total = np.sum(FN)

    # Micro
    precision_micro = TP_total / (TP_total + FP_total)
    recall_micro = TP_total / (TP_total + FN_total)
    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)

    # Macro
    precision_macro = np.mean(TP / (TP + FP + 1e-10))
    recall_macro = np.mean(TP / (TP + FN + 1e-10))
    specificity_macro = np.mean(TN / (TN + FP + 1e-10))
    f1_macro = np.mean(2 * TP / (2 * TP + FP + FN + 1e-10))

    # FPR e FNR 
    fpr_macro = np.mean(FP / (FP + TN + 1e-10))  # False Positive Rate
    fnr_macro = np.mean(FN / (FN + TP + 1e-10))  # False Negative Rate

    # Output 
    metrics = {
        "ACC (%)": accuracy * 100,
        "SN micro (%)": recall_micro * 100,
        "SP macro (%)": specificity_macro * 100,
        "PRE micro (%)": precision_micro * 100,
        "F1 micro (%)": f1_micro * 100,
        "SN macro (%)": recall_macro * 100,
        "PRE macro (%)": precision_macro * 100,
        "F1 macro (%)": f1_macro * 100,
        "FPR macro (%)": fpr_macro * 100,
        "FNR macro (%)": fnr_macro * 100
    }

    df = pd.DataFrame([metrics])
    print(df.to_string(index=False))

# confusion matrix
conf_matrix = [
    [31,25,14,1,3],
    [26,127,20,17,10],
    [8,2,350,0,1],
    [5,25,6,17,6],
    [0,20,3,10,6]
]

compute_metrics(conf_matrix)
