import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

def plotAUC(df):
    labels_no = df['label']
    scores_no = df['score']
    labels_no = [int(e)   for e in labels_no]
    scores_no = [float(e) for e in scores_no]
    auc_value_no = roc_auc_score(labels_no, scores_no)
    fpr_no, tpr_no, thresholds_no = roc_curve(labels_no, scores_no, pos_label=1)

    plt.figure()
    lw = 2
    plt.plot(fpr_no,     tpr_no,     color='black', lw=lw, label='AUC = %0.4f' % auc_value_no)
    plt.plot([0, 1], [0, 1], color='darkorange', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc="lower right")
    plt.show()
    return