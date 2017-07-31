import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             roc_curve, roc_auc_score,
                             precision_recall_curve,
                             confusion_matrix, auc, brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


def score_clf(clf, X, y, X_train=None, y_train=None):

    result = {}

    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    confmatrix = confusion_matrix(y, y_pred)

    aps = average_precision_score(y, y_proba)
    fpr, tpr, thresholds = roc_curve(y, y_proba, drop_intermediate=False)

    try:
        importances = clf.feature_importances_
        result['feature_importances'] = importances
    except:
        pass

    scores = {'accuracy': accuracy,
              'average_precision_score': aps,
              'roc_curve': [fpr, tpr, thresholds],
              'confusion_matrix': confmatrix}

    result['scores'] = scores

    if (X_train is not None) & (y_train is not None):
        y_train_pred = clf.predict(X_train)
        y_train_proba = clf.predict_proba(X_train)[:, 1]

        training_accuracy = accuracy_score(y_train, y_train_pred)
        confmatrix_train = confusion_matrix(y_train, y_train_pred)

        aps_train = average_precision_score(y_train, y_train_proba)
        fpr_train, tpr_train, thresholds_train = \
                    roc_curve(y_train, y_train_proba, drop_intermediate=False)

        scores_train = {'accuracy': training_accuracy,
                        'average_precision_score': aps_train,
                        'roc_curve': [fpr_train, tpr_train, thresholds_train],
                        'confusion_matrix': confmatrix_train}

        result['scores_train'] = scores_train

    return result

def print_clf_scores(result, trainset=False,
                     feature_labels=None, roc_plot=False):

    #print('--------------- Accuracy ---------------')
    if trainset:
        print("Accuracy on training dataset: %f" % result['scores_train']['accuracy'])
    print("Accuracy on test dataset:       %f" % result['scores']['accuracy'])

    #print('--------------- Precision ---------------')
    if trainset:
        print("Average Precision Score on training dataset: %f" % result['scores_train']['average_precision_score'])
    print("Average Precision Score on test dataset:       %f" % result['scores']['average_precision_score'])

    #print('--------------- Confusion Matrix ---------------')
    if trainset:
        print('Confusion Matrix on train set: ', result['scores_train']['confusion_matrix'])
    print('Confusion Matrix on test set : ', result['scores']['confusion_matrix'])

    if feature_labels is not None:
        importances = result['feature_importances']
        indices = np.argsort(importances)[::-1]
        print('--------------- Feature Importances ---------------')
        for f in range(len(feature_labels)):
            print('%2d) %-*s %f' % (f + 1, 20, feature_labels[indices[f]], importances[indices[f]]))
        print('\n')

    if roc_plot:
        fpr = result['scores']['roc_curve'][0]
        tpr = result['scores']['roc_curve'][1]
        roc_auc = auc(fpr, tpr)
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr, ls='-', label='ROC Curve (area = %0.2f)' % roc_auc)
        plt.plot(x, y, ls='--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')

        plt.show()

    return

def plot_reliability_curve(clf, data, label=None):

    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]

    y = np.concatenate([y_train, y_test], axis=0)

    # Calculate probability for the class label == 1
    prob_pos = clf.predict_proba(X_test)[:, 1]
    # Calculate Brier score
    clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
    # Create values for reliability curve
    fraction_of_positives, mean_predicted_value = \
                        calibration_curve(y_test, prob_pos, n_bins=10)

    data_label = label + ('(%1.3f)' % clf_score)

    plt.figure(figsize=(5,5))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(mean_predicted_value, fraction_of_positives,
             "s-", label=data_label)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")

    return

def plot_calibration_curve(est, name, fig_index, data):
    """Plot calibration curve for est w/o and with calibration.
    Adopted from: http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py

    Plots reliability curves with and withou calibrations, along with Logistic
    regression fits. Isotonic and sigmoid calibrations are included.
    """

    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]

    y = np.concatenate([y_train, y_test], axis=0)

    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(1, figsize=(15, 10))
    ax1 = plt.subplot2grid((4, 6), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((4, 6), (0, 2), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((4, 6), (0, 4), colspan=2, rowspan=2)
    ax4 = plt.subplot2grid((4, 6), (2, 0), colspan=6,  rowspan=2)

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
            y_proba = prob_pos.copy()
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            y_proba = prob_pos.copy()
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f" % f1_score(y_test, y_pred))
        print("\tAve. Precision Score: %1.3f\n" % \
                            average_precision_score(y_test, y_proba))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        fpr, tpr, thresholds = roc_curve(y_test, y_proba, drop_intermediate=False)
        roc_auc = roc_auc_score(y_test, y_proba)
        ax2.plot(fpr, tpr, ls='-', label="%s (%1.3f)" % (name, roc_auc))

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ax3.plot(recall, precision)

        ax4.hist(prob_pos, range=(0, 1), bins=10,
                        label='%s' % name, histtype="step", lw=2)

    ax1.set_xlabel("Score", fontsize=14)
    ax1.set_ylabel("Fraction of positives", fontsize=14)
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)', fontsize=16)

    ax2.set_xlabel("False Positive Rate", fontsize=14)
    ax2.set_ylabel("True Positive Rate", fontsize=14)
    ax2.set_ylim([-0.05, 1.05])
    ax2.legend(loc="lower right")
    ax2.set_title('ROC Curve', fontsize=16)

    ax3.set_xlabel("Recall", fontsize=14)
    ax3.set_ylabel("Precision", fontsize=14)
    ax3.set_ylim([-0.05, 1.05])
    ax3.legend(loc="lower center")
    ax3.set_title('Precision-Recall Curve', fontsize=16)

    ax4.set_xlabel("Mean predicted value", fontsize=14)
    ax4.set_ylabel("Count", fontsize=14)
    ax4.legend(loc="upper center")
    ax4.set_title('Classification Result', fontsize=16)

    plt.tight_layout()

    plt.show()

    return

def score_clf2(clf, X, y, X_train=None, y_train=None):

    result = {}

    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    confmatrix = confusion_matrix(y, y_pred)

    aps = average_precision_score(y, y_proba)
    fpr, tpr, thresholds = roc_curve(y, y_proba, drop_intermediate=False)
    roc_auc = roc_auc_score(y, y_proba)

    prec, rec, thresh = precision_recall_curve(y, y_proba)

    brier = brier_score_loss(y, y_proba)

    try:
        importances = clf.feature_importances_
        result['feature_importances'] = importances
    except:
        pass

    scores = {'accuracy': accuracy,
              'precision': precision,
              'recall': recall,
              'f1': f1,
              'average_precision_score': aps,
              'roc_curve': [fpr, tpr, thresholds],
              'roc_auc': roc_auc,
              'pr_curve': [prec, rec, thresh],
              'confusion_matrix': confmatrix,
              'brier': brier}

    result['scores'] = scores

    if (X_train is not None) & (y_train is not None):
        y_train_pred = clf.predict(X_train)
        y_train_proba = clf.predict_proba(X_train)[:, 1]

        accuracy_train = accuracy_score(y_train, y_train_pred)
        precision_train  = precision_score(y_train, y_train_pred)
        recall_train  = recall_score(y_train, y_train_pred)
        f1_train  = f1_score(y_train, y_train_pred)
        confmatrix_train = confusion_matrix(y_train, y_train_pred)

        aps_train = average_precision_score(y_train, y_train_proba)
        fpr_train, tpr_train, thresholds_train = \
                    roc_curve(y_train, y_train_proba, drop_intermediate=False)
        roc_auc_train = roc_auc_score(y_train, y_train_proba)
        prec_train, rec_train, thresh_train = \
                    precision_recall_curve(y_train, y_train_proba)
        brier_train = brier_score_loss(y_train, y_train_proba)

        scores_train = {'accuracy': accuracy_train,
                        'precision': precision_train,
                        'recall': recall_train,
                        'f1': f1_train,
                        'average_precision_score': aps_train,
                        'roc_curve': [fpr_train, tpr_train, thresholds_train],
                        'roc_auc': roc_auc_train,
                        'pr_curve': [prec_train, rec_train, thresh_train],
                        'confusion_matrix': confmatrix_train,
                        'brier': brier_train}

        result['scores_train'] = scores_train

    return result

def print_clf_scores2(result, trainset=False,
                     feature_labels=None, roc_plot=False, clf_name=None):

    if clf_name:
        clf_name = clf_name
    else:
        clf_name = 'Classifier'
    print('==== %s Metrics on Test Set ====' % clf_name)
    print('Accuracy: %f' % result['scores']['accuracy'])
    print('Precision: %f' % result['scores']['precision'])
    print('Recall: %f' % result['scores']['recall'])
    print('F-Score: %f' % result['scores']['f1'])
    print('Brier Score: %f' % result['scores']['brier'])
    print('Average Precision Score: %f' \
                        % result['scores']['average_precision_score'])
    print('Confusion Matrix: ', result['scores']['confusion_matrix'])


    if trainset:
        print('==== %s Metrics on Train Set ====' % clf_name)
        print('Accuracy: %f' % result['scores_train']['accuracy'])
        print('Precision: %f' % result['scores_train']['precision'])
        print('Recall: %f' % result['scores_train']['recall'])
        print('F-Score: %f' % result['scores_train']['f1'])
        print('Brier Score: %f' % result['scores_train']['brier'])
        print('Average Precision Score: %f' \
                            % result['scores_train']['average_precision_score'])
        print('Confusion Matrix: ', result['scores_train']['confusion_matrix'])

    if feature_labels is not None:
        importances = result['feature_importances']
        indices = np.argsort(importances)[::-1]
        print('--------------- Feature Importances ---------------')
        for f in range(len(feature_labels)):
            print('%2d) %-*s %f' % (f + 1, 20,
                                    feature_labels[indices[f]],
                                    importances[indices[f]]))
        print('\n')

    if roc_plot:
        fpr = result['scores']['roc_curve'][0]
        tpr = result['scores']['roc_curve'][1]
        roc_auc = result['scores']['roc_auc']
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr, ls='-', label='ROC Curve (area = %0.2f)' % roc_auc)
        plt.plot(x, y, ls='--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')

        plt.show()

    return
