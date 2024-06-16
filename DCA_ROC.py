'''
Decision Curve Analysis and ROC Curves, and so many utils
Author: Yihang Wu
'''
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.preprocessing import label_binarize, OneHotEncoder


def net_benefit_all(y_true):
    encoder = OneHotEncoder(sparse=False)
    y_true_one_hot = encoder.fit_transform(y_true.reshape(-1, 1))
    n_classes = y_true_one_hot.shape[ 1 ]
    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefits = []
    for threshold in thresholds:
        net_benefit_for_class = []
        for class_idx in range(n_classes):
            prevalence = np.mean(y_true_one_hot[:, class_idx ])  # prevalence is the portions of this class among all the samples.
            net_benefit_for_class.append([
                prevalence - (1 - prevalence) * (threshold / (1 - threshold))])  # [100] samples, one dimension.
        net_benefits.append(np.mean(net_benefit_for_class))
    return net_benefits

def net_benefit_none():
    thresholds = np.linspace(0.01, 0.99, 100)
    return [0 for _ in thresholds]


def calculate_net_benefit_multiclass(y_true, y_proba):
    classes = set(y_true)
    num_classes = len(classes)
    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefits = [ ]
    for threshold in thresholds:
        class_net_benefit = np.zeros(num_classes)
        for class_idx in range(num_classes):
            w = threshold / (1 - threshold)  # Weight for false positives
            predictions = y_proba[ :, class_idx ] >= threshold
            tp = np.sum((predictions == 1) & (y_true == class_idx))
            fp = np.sum((predictions == 1) & (y_true != class_idx))
            epsilon = 1e-6
            class_net_benefit[ class_idx ] = (tp - fp * w) / len(y_true)

        net_benefits.append(np.mean(class_net_benefit))

    return net_benefits


def roc(y_true, pred):
    classes = []
    real = set(y_true)
    for i in real:
        classes.append(i)
    y_true = label_binarize(y_true, classes=classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[ i ], tpr[ i ], _ = roc_curve(y_true[ :, i ], pred[ :, i ])
        roc_auc[ i ] = auc(fpr[ i ], tpr[ i ])


    all_fpr = np.unique(np.concatenate([ fpr[ i ] for i in range(len(classes)) ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr[ i ], tpr[ i ])
    mean_tpr /= len(classes)
    roc_auc[ 'macro' ] = auc(all_fpr, mean_tpr)
    interp_fpr = np.linspace(0.01, 1, 100)
    interp_tpr = interp(interp_fpr, all_fpr, mean_tpr)

    return interp_tpr, interp_fpr