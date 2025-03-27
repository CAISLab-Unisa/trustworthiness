def calculate_metrics(tp, tn, fp, fn):
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
    specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0
    f1_score = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
    fnr = (fn / (fn + tp) * 100) if (fn + tp) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2

    # Stampa i risultati in percentuale
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall (Sensitivity): {recall:.2f}%")
    print(f"Specificity: {specificity:.2f}%")
    print(f"F1-Score: {f1_score:.2f}%")
    print(f"False Positive Rate (FPR): {fpr:.2f}%")
    print(f"False Negative Rate (FNR): {fnr:.2f}%")
    print(f"Balanced Accuracy: {balanced_accuracy:.2f}%")

#confusion matrix
tp = 105  # True Positives
tn = 28  # True Negatives
fp = 33  # False Positives
fn = 6  # False Negatives


calculate_metrics(tp, tn, fp, fn)

