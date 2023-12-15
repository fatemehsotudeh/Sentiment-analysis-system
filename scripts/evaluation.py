from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, \
    roc_curve, auc


def evaluate_model(model, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='micro',
                                                                     zero_division='warn')
    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    return results
