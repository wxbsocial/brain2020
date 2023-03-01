import numpy as np

from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def get_confusion_matrix(preds, labels):
    labels = labels.data.cpu().numpy()
    preds = preds.data.cpu().numpy()
    matrix = [[0, 0], [0, 0]]
    for index, pred in enumerate(preds):
        if np.amax(pred) == pred[0]:
            if labels[index] == 0:
                matrix[0][0] += 1
            if labels[index] == 1:
                matrix[0][1] += 1
        elif np.amax(pred) == pred[1]:
            if labels[index] == 0:
                matrix[1][0] += 1
            if labels[index] == 1:
                matrix[1][1] += 1
    return matrix


def matrix_sum(A, B): 
    return [[A[0][0]+B[0][0], A[0][1]+B[0][1]],
            [A[1][0]+B[1][0], A[1][1]+B[1][1]]]


def get_accu(matrix):
    return float(matrix[0][0] + matrix[1][1])/ float(sum(matrix[0]) + sum(matrix[1]))


def get_MCC(matrix):
    TP, TN, FP, FN = float(matrix[0][0]), float(matrix[1][1]), float(matrix[0][1]), float(matrix[1][0])
    upper = TP * TN - FP * FN
    lower = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    return upper / (lower**0.5 + 0.000000001)




def draw_roc(actuals,  probabilities, n_classes,title,label):
    """
    compute ROC curve and ROC area for each class in each fold

    """
    print("actuals", actuals)
    print("probabilities", probabilities)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(actuals[i], probabilities[i],pos_label=1)
        #fpr[i], tpr[i], _ = roc_curve([0,1,0,1,1], [0.2,0.3,0.52,0.6,0.8])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # fpr, tpr, _ = roc_curve(actuals, probabilities)
    # roc_auc = auc(fpr, tpr)

    print("roc_auc", roc_auc)
    #plt.figure(figsize=(6,6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='{0}({1:0.2f})'
                                    ''.format(label, roc_auc[i]))  # roc_auc_score

    #plt.plot(fpr, tpr, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))  # roc_auc_score
    plt.plot([0, 1], [0, 1], 'k--')
    # plt.grid()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    # plt.tight_layout()
    plt.show()
    