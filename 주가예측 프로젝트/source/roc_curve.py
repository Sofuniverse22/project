import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

class_F = np.array([1, 0])
proba_F = np.array([0.559262097,0.345985, ])

false_positive_rate_F, true_positive_rate_F, thresholds_F =roc_curve(class_F, proba_F)
roc_auc_F=auc(false_positive_rate_F,true_positive_rate_F)

plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate(1- Specificity')
plt.ylabel('True Positive Rate(Sensitivity')
plt.plot(false_positive_rate_F, true_positive_rate_F, 'b', label='Model F (AUC = %0.12f)'%roc_auc_F)
plt.plot([0,1],[1,1],'y--')
plt.plot([0,1],[0,1],'r--')

plt.legend(loc='lower right')
plt.show()