import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import auc, roc_curve


def get_roc_data(dataloader, network):
    preds = np.array([])
    labels = np.array([])

    for input_cols, labs in dataloader:
        p = network(input_cols.float())
        
        p = p.detach().cpu().numpy()
        labs = labs.detach().cpu().numpy()
        
        preds = np.append(preds, p)
        labels = np.append(labels, labs)

    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    best_id = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_id]

    return tpr, fpr, roc_auc, best_threshold


def display_roc_curve(dataloader, network):
    tpr, fpr, roc_auc, best_threshold = get_roc_data(dataloader, network)

    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(fpr, tpr, lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curves')
    plt.legend(loc="lower right")
    plt.show()

    print(f"OPTIMAL THRESHOLD AT {best_threshold}")