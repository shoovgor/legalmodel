import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

classes = ['denied', 'granted', 'granted_uncontested', 'partial', 'settled', 'withdrawn']

cm = np.array([
    [  21,   40,   0,   53,   3,   3],
    [  46, 2409,  56,  195,  84,  95],
    [   0,  131, 320,    3,  17,  12],
    [  14,  138,   4,  289,  13,   2],
    [   1,  145,  23,   10, 637,  12],
    [   4,   71,   6,    3,  13, 280],
])

tp = np.diag(cm)
recall = tp / cm.sum(axis=1)
precision = tp / cm.sum(axis=0)
f1 = 2 * precision * recall / (precision + recall)

metrics = pd.DataFrame({'Class': classes, 'Precision': precision, 'Recall': recall, 'F1': f1})
display(metrics.style.format({'Precision': '{:.2f}', 'Recall': '{:.2f}', 'F1': '{:.2f}'}).hide(axis='index'))
print(f"Macro precision {precision.mean():.3f} | macro recall {recall.mean():.3f} | macro F1 {f1.mean():.3f}")

plot_df = metrics.melt(id_vars='Class', var_name='Metric', value_name='Score')
plt.figure(figsize=(10, 4.5))
sns.barplot(x='Class', y='Score', hue='Metric', data=plot_df, palette='Set2')
plt.title('Per-Class Precision, Recall and F1 (HistGBM)')
plt.xlabel('Class')
plt.ylim(0, 1.0)
plt.xticks(rotation=20)
plt.legend(loc='upper right', ncol=3)
plt.tight_layout()
plt.show()